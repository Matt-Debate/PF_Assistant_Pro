import requests
import httpx
from requests.exceptions import SSLError, RequestException
import streamlit as st

# Constants
PERPLEXITY_API_KEY = st.secrets["PERPLEXITY_API_KEY"]
PERPLEXITY_ENDPOINT = "https://api.perplexity.ai/chat/completions"

# --- Prompt Builder ---
def build_perplexity_prompt(claim):
    prompt = f"produce 3 evidence cards that support this claim: {claim}."

    prompt += (
        " 1 evidence card can only have 1 quote and 1 source. "
        "Before outputting to the user, verify that the cards match the claim. "
        "If no matching quotes are found, respond 'no cards found'. "
        "If only 1 or 2 quotes are found, only provide the matching quotes. "
        "Find quotes with statistical data. "
    )

    prompt += (
        "Format cards exactly like this, and provide no additional commentary or markdown:\n\n"
        "Author, Year of Publication\n"
        "URL\n"
        "\"Direct Quote\""
    )

    return prompt

# --- Perplexity API Query ---
def query_perplexity(prompt):
    headers = {
        "Authorization": f"Bearer {PERPLEXITY_API_KEY}",
        "Accept": "application/json",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "sonar",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.5
    }

    try:
        response = requests.post(PERPLEXITY_ENDPOINT, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
    except SSLError:
        with httpx.Client(http2=True, verify=True, timeout=30) as client:
            response = client.post(PERPLEXITY_ENDPOINT, headers=headers, json=payload)
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]
    except RequestException as e:
        raise RuntimeError(f"Request to Perplexity failed: {e}") from e
