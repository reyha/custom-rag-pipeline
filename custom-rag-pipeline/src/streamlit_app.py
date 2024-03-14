import streamlit as st
import requests
from requests.adapters import HTTPAdapter
from urllib3 import Retry


st.title("Biology Guide")
user_input = st.text_input("Enter your question about biology:", "")


def get_qna_response(query):
    # Call Search API with retries
    session = requests.Session()
    adapter = HTTPAdapter(max_retries=Retry(total=2, backoff_factor=1, allowed_methods=None,
                                            status_forcelist=[429, 500, 502, 503, 504]))
    session.mount("https://", adapter)

    url = 'http://127.0.0.1:5550/v1/custom_rag_qna'
    request = {
        'user_query': f"{query}"
    }

    response = session.post(url=url, json=request, timeout=60, verify=False)
    json_response = response.json()
    result = json_response.get('response', " ")

    return result


if st.button("Submit"):
    try:
        # response = query_engine.query(user_input)
        response = get_qna_response(user_input)
        st.write(response)
    except Exception as e:
        st.write(f"An error occurred: {e}")