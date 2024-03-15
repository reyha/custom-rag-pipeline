# Custom RAG Pipeline 

A RAG pipeline built using open-source models for retrieving answers from specific sources. For this project, we use pdf
as our primary data source, `Llama-13b` as our query responder model and Sentence-Transformers `all-mpnet-base-v2` as embedding model. 

## Installation 
- Clone this repo 
- You can install dependencies using by running following under project root directory, <br>
```
pip install -r requirements.txt
```

## Quick Start


### Input 
This pipeline takes in 2 arguments:
1. user_query (mandatory): query that user wants model to respond to 
2. model_id (optional): id of the model user wants this query to use for responding. We currently support only `llama-13b` and 
this field defaults to this model.
```
{
   "user_query": "What is biology"
   "model_id": "oss_llama-13b"
}
```

### Output
[Insert architecture diagram here] <br>
[Insert class digram here]

### Assumption


## Evaluation 

## Run UI using Streamlit

## Potential Issues

## Future Improvements
