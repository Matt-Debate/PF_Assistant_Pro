import openai
from tools.tool_log_and_retry import log_and_retry

def write_contention(topic: str, side: str, area: str, raw_evidence_list: list[str]) -> str:
    """
    Generate a full contention for a public forum debate using raw evidence text.
    """

    # Concatenate all raw evidence into one big text block, separated by line breaks or separators
    combined_evidence = "\n\n---\n\n".join(raw_evidence_list)

    prompt = f"""
You are tasked with writing a final contention for a public forum debate based on the provided information.

**Topic:** {topic}  
**Side:** {side}  
**Topic Area:** {area}  
**Evidence:**  
{combined_evidence}

Instructions:

1. Contentions Title:  
Start with "Contention _______: {area}".

2. Thesis Statement:  
Craft a brief thesis statement summarizing the main argument for the **{side}** side on the topic "{topic}".

3. Links:  
Develop up to 4 supporting claims derived **directly** from the provided evidence. Each claim should:  
- State the Claim Clearly  
- Explain the Reasoning  
- Reference the Provided Evidence
- Be distinct from the other supporting claims. It's ok to have less than 4. 

4. Impact:  
Conclude with an impact statement directly supported by the evidence provided.

5. Formatting:  
- Follow the above structure  
- Keep it organized and clear  
- No outside sources
-no markdown such as "**"
"""

    def call_model():
        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=700,
        )
        return response.choices[0].message.content

    return log_and_retry(call_model)