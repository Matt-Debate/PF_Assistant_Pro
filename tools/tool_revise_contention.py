import openai
from tools.tool_log_and_retry import log_and_retry
import streamlit as st

openai.api_key = st.secrets["OPENAI_API_KEY"]

def revise_contention(raw_contention: str, evidence_json_str: str) -> str:
    """
    Refine the provided contention to match the tone, style, and formatting required.
    
    Inputs:
    - raw_contention: The contention text output from Step 5.
    - evidence_json_str: JSON string of evidence (to verify accuracy).

    Returns:
    - Polished contention string.
    """

    prompt = f"""
You are tasked with reviewing and refining the following contention for a public forum debate to ensure it matches the specified tone, style, headings, and formatting. Make sure all evidence is accurately represented and that the terminology aligns with the provided example.

Original Contention:
"{raw_contention}"

Evidence:
{evidence_json_str}

Instructions:

1. Terminology Adjustments:
   - Replace all instances of "supporting claims" with "links".
   - Introduce the supporting links with phrases like "We have X links:" where X is the number of links.
   - Introduce impact with "Impact:".

2. Structural Modifications:
   - Remove headers such as "Thesis Statement" and "claim".
   - Ensure that the contention starts with the title, followed by the main argument, the links, and the impact ***without additional section headers***.

3. Formatting Consistency:
   - Use numbering for the links as shown in the example.
   - Ensure each link includes:
     - Claim Title: A brief statement of the claim.
     - Supporting reasoning and evidence: The reasoning, quote and the reference (author and year) without introducing new sources.
   - Omit markdown like "*" and "#".

4. Tone and Style:
   - Maintain a clear, concise, and formal tone appropriate for a public forum debate.
   - Ensure the language is assertive and the arguments are logically structured.
   - Avoid redundacy. 

5. Accuracy Check:
   - Verify that all evidence cited in the contention matches the provided evidence.
   - Ensure no external sources are introduced.

6. Word count check:
   - Keep the contention roughly 300 words.

7. Impact check:
   - Ensure the impact describes how people are helped or harmed rather than society generally.
     For example, if the impact is "GDP increase," clarify it as "reducing poverty which saves lives."

Output only the revised contention, no extra commentary.
"""

    def call_model():
        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=500,
        )
        return response.choices[0].message.content

    polished_contention = log_and_retry(call_model)

    return polished_contention
