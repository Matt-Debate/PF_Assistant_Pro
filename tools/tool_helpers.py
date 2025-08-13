import streamlit as st
from contextlib import contextmanager
import json
import ast
import re


def set_wide_centered_layout(max_width=1000):
    st.set_page_config(layout="wide")
    st.markdown(
        f"""
        <style>
        /* Center all headers */
        h1, h2, h3, h4, h5, h6 {{
            text-align: center !important;
        }}
        /* Center main content container */
        main > div.block-container {{
            margin-left: auto !important;
            margin-right: auto !important;
            max-width: {max_width}px !important;  /* wider container */
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )

def strip_code_fences(text: str) -> str:
    """Remove markdown-style code fences like ```json ... ``` and return inner text."""
    if not text:
        return ""
    text = text.strip()
    if text.startswith("```"):
        parts = text.split("```")
        if len(parts) >= 3:
            inner = parts[1]
            # If the inner starts with "json" language tag remove it
            if inner.lower().startswith("json"):
                inner = inner.split("\n", 1)[1] if "\n" in inner else ""
            return inner.strip()
        # fallback: strip any backticks
        return text.replace("```", "").strip()
    return text


def parse_claims_output(raw_output):
    """
    Accept a model output (str or dict). Return a Python dict with claims.
    Tries:
      - If already dict, return it.
      - strip code fences + BOM, then json.loads
      - attempt to extract first {...} JSON block
      - ast.literal_eval as a last resort
    Raises ValueError on failure.
    """
    if isinstance(raw_output, dict):
        return raw_output

    if raw_output is None:
        raise ValueError("generate_claims returned None")

    if not isinstance(raw_output, str):
        raise ValueError(f"Unexpected type from generate_claims: {type(raw_output)}")

    # Clean BOM and fences
    text = raw_output.lstrip("\ufeff").strip()
    text = strip_code_fences(text)

    # Try JSON
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try JSON after finding the first {...} block
    try:
        start = text.find('{')
        end = text.rfind('}')
        if start != -1 and end != -1 and end > start:
            snippet = text[start:end + 1]
            return json.loads(snippet)
    except Exception:
        pass

    # Try Python literal (single-quoted dicts)
    try:
        return ast.literal_eval(text)
    except Exception as e:
        raise ValueError(f"Could not parse claims output as JSON or Python literal: {e}")


@contextmanager
def step_ui(step_number: int, title: str, description: str, auto_advance=True, clear_on_finish=False):
    """
    Display step UI with title, description, spinner, and success message.
    
    Args:
        step_number (int): Step number.
        title (str): Step title.
        description (str): What the step is doing (shown during spinner).
        auto_advance (bool): If True, the step will auto move on after spinner finishes.
        clear_on_finish (bool): If True, clears description and success message after spinner (e.g. final step).
        
    Usage:
        with step_ui(2, "Generate Claims", "Generating claims..."):
            # do work here
    """
    st.subheader(f"Step {step_number}: {title}")
    desc_placeholder = st.empty()
    desc_placeholder.markdown(description)
    with st.spinner(f"{title}..."):
        yield
    success_placeholder = st.empty()
    success_placeholder.success(f"{title} complete ✅")

    if clear_on_finish:
        # clear both description and success messages after a brief pause
        import time
        time.sleep(1)  # optional delay so user sees success briefly
        desc_placeholder.empty()
        success_placeholder.empty()


def _parse_single_perplexity_block(raw_block: str) -> dict:
    """
    Parse a single Perplexity-style block:
      Line 1: Author, Year (source)
      Line 2: URL (optional)
      Line 3+: "Quote"
    Returns a dict with keys: source, url (optional), quote (optional).
    """
    lines = [ln.strip() for ln in raw_block.splitlines() if ln.strip()]
    if not lines:
        return {}

    card = {}
    card["source"] = lines[0]

    # second line is URL if looks like URL
    if len(lines) >= 2 and re.match(r"https?://", lines[1]):
        card["url"] = lines[1]
        quote_lines = lines[2:]
    else:
        quote_lines = lines[1:]

    # Filter out "Current date" lines and combine the rest as quote
    quote_lines = [ln for ln in quote_lines if not ln.lower().startswith("current date")]
    if quote_lines:
        card["quote"] = " ".join(quote_lines)
    else:
        card["quote"] = ""

    return card


def merge_evidence_dicts(raw_evidence_json_strings):
    """
    Merge multiple evidence inputs into a single dict with 'evidence_cards'.
    Accepts each input as:
      - JSON string with {"evidence_cards": [...]}
      - JSON-serializable dict with "evidence_cards"
      - Plain text (Perplexity style) blocks separated by blank lines

    Deduplicates by (quote, url, source) tuple.
    """
    all_cards = []
    seen = set()

    for idx, raw in enumerate(raw_evidence_json_strings):
        if raw is None:
            continue

        # if it's already a dict/list
        if isinstance(raw, dict):
            data = raw
        elif isinstance(raw, list):
            # flatten lists of cards
            data = {"evidence_cards": raw}
        elif isinstance(raw, str):
            raw_str = raw.strip()
            if not raw_str:
                continue

            # try to parse as JSON first
            try:
                data = json.loads(raw_str)
            except json.JSONDecodeError:
                # fallback: parse as Perplexity-style blocks (double-newline separated)
                blocks = re.split(r"\n\s*\n", raw_str)
                cards = []
                for b in blocks:
                    parsed = _parse_single_perplexity_block(b)
                    if parsed:
                        cards.append(parsed)
                data = {"evidence_cards": cards} if cards else None
        else:
            data = None

        if not data:
            # nothing parsed from this input; continue
            continue

        cards = data.get("evidence_cards", [])
        if not isinstance(cards, list):
            continue

        for card in cards:
            if not isinstance(card, dict):
                continue
            quote = (card.get("quote") or "").strip()
            url = (card.get("url") or "").strip()
            source = (card.get("source") or "").strip()
            key = (quote, url, source)
            if key not in seen:
                seen.add(key)
                all_cards.append(card)

    return {"evidence_cards": all_cards}


def clean_json_text(text: str) -> str:
    """
    Remove BOM if present and trim whitespace.
    """
    if not text:
        return text
    if text.startswith("\ufeff"):
        text = text[1:]
    return text.strip()