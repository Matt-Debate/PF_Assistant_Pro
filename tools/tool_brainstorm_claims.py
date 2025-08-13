import json
import streamlit as st
import openai

openai.api_key = st.secrets["OPENAI_API_KEY"]

def build_prompt(topic, side, area):
    """
    Builds the prompt text for the model based on topic, side, and area.
    """
    return f"""
You are tasked with creating a contention outline for a public forum debate.

Topic: {topic}
Side: {side}
Topic Area: {area}

Instructions:

1. Contention Title: Begin with “Contention _______: {area}”.
2. Thesis Statement: Provide a brief thesis statement summarizing the main argument for the {side} side on the topic “{topic}”.
3. Supporting Claims: List up to 6 specific supporting claims that need to be researched. For each claim:
   - Write a concise description of the claim.
   - Indicate the type of evidence needed (e.g., statistical data, expert opinion, case studies).
   - Do not include fabricated evidence; instead, specify that these claims require research.
4. Impact Claim: Identify the impact of the contention as a claim that also needs to be researched. Provide:
   - A concise description of the impact.
   - Try to quantify the potential number of lives saved or lost.
   - The impact must be directly connected to the topic and links.
   - Do not include the link in the impact statement.
   - Write the claim so it is web-search friendly.
5. Formatting:
   - Output ONLY a valid JSON object.
   - Do not include any text, commentary, or explanations outside the JSON.
   - All keys and string values must be enclosed in double quotes.
   - Escape special characters inside strings.
   - Follow the exact JSON structure and key names shown in the example.

Example (follow this exact structure):

{{
  "contention_title": "Contention 3: Environment",
  "thesis_statement": "AVs reduce traffic congestion and replace gas-powered cars, reducing air pollution.",
  "claims": [
    {{
      "claim_number": 1,
      "description": "Replacing Gas-Powered Cars: Autonomous vehicles (AVs) will replace gas-powered cars, potentially reducing greenhouse gas emissions by a significant percentage.",
      "evidence_needed": "Statistical data on emission reductions from transitioning to AVs."
    }},
    {{
      "claim_number": 2,
      "description": "Increased Efficiency: The advanced technology in autonomous cars allows for smoother driving, leading to more efficient fuel use and decreased overall air pollution.",
      "evidence_needed": "Expert analyses on fuel efficiency improvements due to AV technology."
    }},
    {{
      "claim_number": 3,
      "description": "Reduced Traffic Congestion: AVs can improve traffic flow and minimize idle time at intersections, reducing fuel consumption and carbon dioxide emissions.",
      "evidence_needed": "Studies demonstrating how AVs affect traffic patterns and emissions."
    }}
  ],
  "impact_claim": {{
    "description": "Reduction of Air Pollution: Decreasing air pollution through the adoption of AVs can lead to significant public health benefits and lower mortality rates associated with pollution-related diseases in China.",
    "evidence_needed": "Statistics on health impacts of air pollution in China, data on mortality rates related to pollution, expert opinions on potential improvements from reduced emissions."
  }}
}}
END_OF_JSON

Now create the contention outline using the provided topic, side, and area.
"""

def clean_json_output(raw_output):
    """
    Extract JSON block from the model output, trimming stray text outside the JSON braces.
    """
    if not raw_output:
        return ""
    try:
        start = raw_output.index("{")
        end = raw_output.rindex("}") + 1
        return raw_output[start:end].strip()
    except ValueError:
        # If braces not found, return raw output stripped
        return raw_output.strip()

def generate_claims(topic, side, area):
    """
    Generate contention claims by calling OpenAI's chat completion API.
    Returns the parsed JSON object or None on failure.
    Displays intermediate outputs using Streamlit.
    """

    prompt = build_prompt(topic, side, area)

    try:
        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
            max_completion_tokens=1500,
        )
    except Exception as e:
        st.error(f"OpenAI API call failed: {e}")
        return None

    raw_claim_json_str = response.choices[0].message.content.strip()

   

    cleaned_claim_json_str = clean_json_output(raw_claim_json_str)

    
    try:
        claims_data = json.loads(cleaned_claim_json_str)
        return claims_data
    except json.JSONDecodeError as e:
        st.error(f"Failed to parse brainstorm claims JSON after formatting: {e}")
        return None
