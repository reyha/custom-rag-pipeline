"""
UI app for QA pipeline powered by streamlit
"""
import streamlit as st
import requests
from requests.adapters import HTTPAdapter
from urllib3 import Retry

# =================== Initializations ===================
# 0. Initialize streamlit elements
st.title("Biology Guide")
user_input = st.text_input("Enter your question about biology:", "")


def get_qna_response(query):
    """
    Call QA pipeline's flask endpoint with retries
    """
    session = requests.Session()
    adapter = HTTPAdapter(max_retries=Retry(total=2, backoff_factor=1, allowed_methods=None,
                                            status_forcelist=[429, 500, 502, 503, 504]))
    session.mount("https://", adapter)

    url = 'http://127.0.0.1:5550/v1/custom_rag_qna'
    request = {
        'user_query': f"{query}"
    }

    response = session.post(url=url, json=request, verify=False)
    json_response = response.json()
    result = json_response.get('response', " ")

    return result


if st.button("Submit"):
    try:
        # 1. Get output from qa service
        response = get_qna_response(user_input)
        # 2. Retrieve response from output
        json_response = response.json()
        answer = json_response.get("response", "")
        # 3. Display to user
        st.write(answer)
    except Exception as e:
        st.write(f"An error occurred: {e}")