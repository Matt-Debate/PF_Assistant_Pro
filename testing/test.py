import json
import openai
import streamlit as st
from tools.tool_format_json import format_json
from tools.tool_log_and_retry import log_and_retry
from auth import get_username, log_query

TOOL_NAME = "brainstorm_claims"

openai.api_key = st.secrets["OPENAI_API_KEY"]

def build_prompt(topic, side, area):
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
   - The goal is to quantify the potential number of lives saved or lost.
   - Important: The impact must be directly connected to the topic and links.
   - Write each claim to be web-search friendly.
5. Formatting:
   - Present the output as **valid JSON only**.
   - All keys and values must use double quotes.
   - Escape any special characters.
   - Do not include any text outside of the JSON object.
"""

def strip_code_fences(text):
    text = text.strip()
    if text.startswith("```"):
        parts = text.split("```")
        if len(parts) > 1:
            inner = parts[1].strip()
            if inner.lower().startswith("json"):
                inner = inner[4:].strip()
            return inner
    return text

def generate_claims(topic: str, side: str, area: str) -> dict:
    def call_model():
        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": build_prompt(topic, side, area)}],
            temperature=0.7,
            max_tokens=500,
        )
        return response.choices[0].message.content.strip()

    try:
        raw_output = log_and_retry(call_model)
        # Try direct JSON parse
        try:
            return json.loads(raw_output)
        except json.JSONDecodeError:
            pass

        # Try after stripping code fences
        cleaned = strip_code_fences(raw_output)
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass

        # Fallback to formatter tool
        formatted_output = format_json(raw_output)
        return json.loads(formatted_output)

    except Exception as e:
        log_and_retry(lambda: (_ for _ in ()).throw(e))
        raise RuntimeError(f"{TOOL_NAME} failed to produce valid JSON: {e}")
