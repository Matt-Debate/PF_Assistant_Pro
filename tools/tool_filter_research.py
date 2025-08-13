import openai
from tools.tool_log_and_retry import log_and_retry
import streamlit as st

openai.api_key = st.secrets["OPENAI_API_KEY"]

def filter_and_format_evidence(contention_text: str, evidence_json_str: str) -> str:
    """
    Given a contention and a JSON string of evidence cards,
    generate a plain text list of evidence cards used in the contention
    following the precise formatting rules.

    Args:
        contention_text: The finalized contention text.
        evidence_json_str: JSON string containing the "evidence_cards" list.

    Returns:
        Plain text formatted evidence cards as per instructions.
    """

    prompt = f"""
You are tasked with producing the evidence cards for a public forum debate contention.

Inputs Provided:
1. Contention:  
{contention_text}

2. Evidence Cards:  
{evidence_json_str}

Instructions:

1. Output Structure:

   - Evidence Cards:  
     List each evidence card that was used in the contention. Format each evidence card as follows:
     ```
     Author, Year
     URL
     "Quote"
     ```
     - Author, Year: Combine the "author" and "year" fields, separated by a comma.
     - URL: Present the "url" field on the next line.
     - Quote: Enclose the "quote" field in quotation marks on the following line.
     - Spacing: Insert a blank line between each evidence card for readability.

2. Formatting Requirements:
   - No Asterisks or Markdown Formatting:  
     Do not include any asterisks (*) or other Markdown symbols. The output should be plain text.

   - No Additional Text:  
     Do not add any headers, footers, or explanatory text outside of the contention and the evidence cards.

   - Consistent Formatting:  
     Ensure that all evidence cards follow the exact formatting as specified without deviations.

3. Accuracy and Consistency:
   - Use Only Provided Evidence:  
     Only include evidence cards listed in 'Evidence Cards'. Do not introduce any external sources or references.

   - Maintain Quote Integrity:  
     Ensure that the quotes are presented exactly as provided, without alterations or paraphrasing.

Output only the formatted evidence cards.
"""

    def call_model():
        response = openai.chat.completions.create(
            model="gpt-4o",  # or your preferred ChatGPT model
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=700,
        )
        return response.choices[0].message.content

    formatted_evidence = log_and_retry(call_model)

    return formatted_evidence
