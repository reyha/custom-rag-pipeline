import pandas as pd
import numpy as np
from numpy.linalg import norm
import torch
import tensorflow as tf
from sentence_transformers import SentenceTransformer
from sentence_transformers.cross_encoder import CrossEncoder
from rouge_score import rouge_scorer
from streamlit_app import get_qna_response

STS_EMBEDDING_MODEL = 'sentence-transformers/multi-qa-MiniLM-L6-cos-v1'
SAS_CROSS_ENCODER = 'sentence-transformers/stsb-roberta-large'

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
#
sts_model = SentenceTransformer(f'{STS_EMBEDDING_MODEL}')
sas_scorer = CrossEncoder(f'{SAS_CROSS_ENCODER}')


def vector_similarity(vector_1, vector_2):
    """
    Returns the cosine similarity between two vectors.
    """
    dot_product = np.dot(np.array(vector_1), np.array(vector_2))
    l2_norm = norm(np.array(vector_1)) * norm(np.array(vector_2))
    return dot_product / l2_norm


def get_sts_embedding(sentences):
    embeddings = sts_model.encode(sentences)
    return embeddings


def compute_STS_metrics(generated_answers, gold_answers):
    answer_embedding = get_sts_embedding(generated_answers)
    gold_embedding = get_sts_embedding(gold_answers)
    scores = []
    for i in range(len(answer_embedding)):
        scores.append(vector_similarity(answer_embedding[i], gold_embedding[i]))
    return list(scores)


def compute_ROUGE_metrics(generated_answers, gold_answers):
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    scores = []
    for i in range(len(generated_answers)):
        scores.append(scorer.score(gold_answers[i], generated_answers[i])["rougeL"][2])
    return list(scores)


def compute_SAS_metrics(generated_answers, gold_answers):
    with torch.no_grad():
        scores = sas_scorer.predict(list(zip(generated_answers, gold_answers)))
    return list(scores)


def compute_metrics(generated_answers, gold_answers):
    """
    Computes 4 metrics:
    1. STS
    2. ROUGE
    3. SAS
    4. Weighted Metric - customized function to give max weightage to STS
    followed by equal weightage to SAS and ROUGE. These weights are chosen
    randomly and are subject to change based on findings.
    """
    rouge_scores = compute_ROUGE_metrics(generated_answers, gold_answers)
    sts_scores = compute_STS_metrics(generated_answers, gold_answers)
    sas_scores = compute_SAS_metrics(generated_answers, gold_answers)
    weighted_metrics = [rouge_scores[i] * 0.25 +
                        sts_scores[i] * 0.50 +
                        sas_scores[i] * 0.25 for i in range(len(rouge_scores))]
    metrics = pd.DataFrame(zip(rouge_scores, sts_scores, sas_scores, weighted_metrics),
                           columns=['ROUGE', 'STS', 'SAS', 'Aggregated_Metric'])
    return metrics


import argparse

if __name__ == "__main__":
    # 1. Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_csv", help="answer answer ", type=str, default="")
    args = parser.parse_args()

    # 2. Load eval data in dataframe
    data = pd.read_csv(args.response_csv)

    # 3. Compute metrics only if reference i.e. gold present
    assert 'gold_answer' in data

    # 4. Add qa responses for all test queries
    data['final_response'] = data['query'].apply(lambda x: get_qna_response(x))

    # 5. Compute metrics only if candidate present
    assert 'final_response' in data

    # 5. Compute metrics
    generated_answers = data['final_response']
    gold_labels = data['gold_answer']
    metrics = compute_metrics(generated_answers, gold_labels)

    # 6. Update csv with metrics
    combined_df = pd.concat((data, metrics), axis=1)
    combined_df.to_csv(args.test_csv.replace('.csv', '_metrics.csv'), index=False)