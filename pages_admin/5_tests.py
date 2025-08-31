import json
import streamlit as st
import pandas as pd
from typing import List, Dict, Any, Tuple
from io import BytesIO

from auth import get_username, log_query, get_remaining_queries
from openai import OpenAI

# --- Constants ---
TOOL_NAME = "evidence_machine"
DEFAULT_MODEL = "gpt-5"  # Planning call uses Responses API without response_format


# --- Helpers ---
def _init_state():
    ss = st.session_state
    ss.setdefault("evm_intake", {})
    ss.setdefault("evm_plan_text", None)
    ss.setdefault("evm_plan_approved", False)
    ss.setdefault("evm_batches", [])  # list of list[card]
    ss.setdefault("evm_seen_set", set())  # (url||quote) strings
    ss.setdefault("evm_current_batch", 1)
    ss.setdefault("evm_model", DEFAULT_MODEL)


def _get_oai() -> OpenAI:
    api_key = st.secrets.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Missing OPENAI_API_KEY in Streamlit secrets.")
    return OpenAI(api_key=api_key)


def _get_output_text(resp) -> str:
    """
    Best-effort extraction for both Responses API objects and Chat Completions objects.
    - Responses API: prefer resp.output_text if available.
    - Chat Completions: resp.choices[0].message.content (or .parsed if available).
    Fallback: str(resp).
    """
    if hasattr(resp, "output_text"):
        text = getattr(resp, "output_text")
        if isinstance(text, str) and text.strip():
            return text

    try:
        choices = getattr(resp, "choices", None)
        if choices and len(choices) > 0:
            msg = getattr(choices[0], "message", None)
            if msg is not None:
                parsed = getattr(msg, "parsed", None)
                if parsed is not None:
                    try:
                        return json.dumps(parsed, ensure_ascii=False)
                    except Exception:
                        pass
                content = getattr(msg, "content", None)
                if isinstance(content, str) and content.strip():
                    return content
    except Exception:
        pass

    try:
        return str(resp)
    except Exception:
        return ""


def _get_parsed_json(resp) -> Any:
    """
    Prefer the SDK's validated parsed object; otherwise JSON-load the content.
    """
    try:
        choices = getattr(resp, "choices", None)
        if choices and len(choices) > 0:
            msg = getattr(choices[0], "message", None)
            if msg is not None:
                parsed = getattr(msg, "parsed", None)
                if parsed is not None:
                    return parsed
                content = getattr(msg, "content", None)
                if isinstance(content, str) and content.strip():
                    return json.loads(content)
    except Exception:
        pass
    text = _get_output_text(resp)
    try:
        return json.loads(text)
    except Exception:
        return None


def _ensure_cards_list(obj: Any) -> List[Dict[str, Any]]:
    """
    Accept either:
      - {"cards": [ ... ]}  (preferred wrapped schema)
      - [ ... ]             (legacy unwrapped; we normalize to list)
    """
    if isinstance(obj, dict) and "cards" in obj and isinstance(obj["cards"], list):
        return obj["cards"]
    if isinstance(obj, list):
        return obj
    raise ValueError("Model did not return JSON with a 'cards' array.")


def _seen_key(card: Dict[str, Any]) -> str:
    try:
        url = (card.get("citation") or {}).get("url", "").strip()
        quote = (card.get("quote") or "").strip()
        return f"{url}||{quote}"
    except Exception:
        return ""


def _propose_plan(intake: Dict[str, Any], model: str) -> str:
    client = _get_oai()

    resolution = intake.get("resolution", "").strip()
    area = intake.get("area", "").strip()
    sides = ", ".join(intake.get("sides", ["Pro", "Con"]))
    total_cards = int(intake.get("total_cards", 240))
    batch_size = int(intake.get("batch_size", 20))

    sys = "You are a debate coach planning an evidence pack across batches."
    user = f"""
OBJECTIVE
Produce an expansive evidence pack of {total_cards} cards using {batch_size}-card batches.

RESOLUTION: {resolution}
AREA: {area or '(none)'}
SIDES: {sides}

Task: Propose a numbered plan of 10–15 batches (each {batch_size} cards; sums to {total_cards}). For each batch, state:
- Side(s) (Pro or Con; balance across plan)
- Thematic focus (functions/angles you’ll target)
- Quant target per batch so we exceed 30–40% quantitative cards overall
- Coverage goals (countries/subregions, sectors, mechanisms) so the whole set is broad and non-duplicative

Use the taxonomy: Definitions & frameworks; Economy/exports/productivity; Border/customs/single window/authorized operator; Domestic transport (roads/rail/waterways); Maritime connectivity/ports/shipping; Energy corridors/interconnection; Agriculture/cold chain/food loss; SME inclusion/competitiveness; Digital trade/logistics; Climate resilience & disruptions; Finance/PPP; Legal/regulatory/integration blocs; Subregional case studies; and for Con: fiscal trade-offs, distributional/inequality effects, environment/externalities, dependency/lock-in, governance capacity risks.

Output: A concise, numbered plan in prose. Do not produce any cards yet.
"""

    resp = client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": sys},
            {"role": "user", "content": user}
        ],
        reasoning={"effort": "high"},
    )
    return _get_output_text(resp).strip()


def _batch_prompt_header(intake: Dict[str, Any]) -> str:
    resolution = intake.get("resolution", "").strip()
    area = intake.get("area", "").strip()
    sides = ", ".join(intake.get("sides", ["Pro", "Con"]))
    return f"RESOLUTION: {resolution}\nAREA: {area or '(none)'}\nSIDES: {sides}"


def _schema_block() -> str:
    return (
        """
STRICT CARD SCHEMA (Return JSON ONLY; wrapped as {"cards":[...]} )
{
  "cards": [
    {
      "side": "Pro" | "Con",
      "function": "Definitions" | "Economy Links" | "Agriculture Links" | "Impact Cards" | "Border/Customs" | "Maritime Connectivity" | "Energy Corridors" | "Resilience" | "Digital Trade/Logistics" | "Finance/PPP" | "Legal/Regulatory" | "Case Studies",
      "tag": "short claim",
      "citation": {
        "authors": "Last & Last" | "Org Name",
        "year": "YYYY",
        "date": "YYYY-MM-DD",
        "title": "Article or Report Title",
        "source": "Publisher or Journal",
        "url": "https://..."
      },
      "quote": "Full, direct, multi-sentence paragraph(s) from a single location on the page.",
      "underline_phrases": ["phrase1","phrase2","phrase3"],
      "flow_sentence": "One clean sentence that reads if the underlined phrases are stitched."
    }
  ]
}
"""
    ).strip()


def _quality_rules_block() -> str:
    return (
        """
QUALITY RULES
- Canonical dates: use the page’s stated publication date; if missing, use the official press-release/report PDF date.
- De-dupe: no duplicate quotes; if two cards share a URL, use distinct passages. Maintain a private seen set keyed by (url, exact quote) across batches.
- Quant hygiene: prefer stats with denominators/baselines and clear timeframes.
- Con rebuttal seeds: for Con cards, favor quotes that frame trade-offs (fiscal costs, regional disparities, environmental externalities, governance capacity).
- Quotes must be verbatim multi-sentence paragraphs from one location on the page (no stitching far-apart parts).
- Prefer authoritative, recent, accessible sources; avoid broken links or paywalled abstracts without visible text. If text is not present at the URL, replace the card before delivering the batch.
- Diversity: vary publishers/countries/years/functions; avoid over-reliance on any single institution.
"""
    ).strip()


def _cards_envelope_schema(item_schema: Dict[str, Any], min_items: int = 1, max_items: int | None = None) -> Dict[str, Any]:
    arr_schema: Dict[str, Any] = {
        "type": "array",
        "items": item_schema,
        "minItems": min_items,
    }
    if max_items is not None:
        arr_schema["maxItems"] = max_items
    return {
        "type": "object",
        "properties": {
            "cards": arr_schema
        },
        "required": ["cards"],
        "additionalProperties": False,
    }


def _card_schema() -> Dict[str, Any]:
    """
    Card object schema with strict nested object rules (additionalProperties = false).
    """
    return {
        "type": "object",
        "properties": {
            "side": {"type": "string", "enum": ["Pro", "Con"]},
            "function": {"type": "string"},
            "tag": {"type": "string"},
            "citation": {
                "type": "object",
                "properties": {
                    "authors": {"type": "string"},
                    "year": {"type": "string"},
                    "date": {"type": "string"},
                    "title": {"type": "string"},
                    "source": {"type": "string"},
                    "url": {"type": "string"},
                },
                "required": ["authors", "year", "date", "title", "source", "url"],
                "additionalProperties": False,  # strict
            },
            "quote": {"type": "string"},
            "underline_phrases": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 2,
                "maxItems": 5,
            },
            "flow_sentence": {"type": "string"},
        },
        "required": [
            "side",
            "function",
            "tag",
            "citation",
            "quote",
            "underline_phrases",
            "flow_sentence",
        ],
        "additionalProperties": False,  # strict
    }


def _generate_batch(
    intake: Dict[str, Any], plan_text: str, batch_num: int, model: str, seen: List[Tuple[str, str]]
) -> List[Dict[str, Any]]:
    client = _get_oai()

    header = _batch_prompt_header(intake)
    schema = _schema_block()
    rules = _quality_rules_block()

    seen_pairs_text = "\n".join([f"- URL: {u} | QUOTE: {q[:140]}" for (u, q) in seen]) if seen else "(none yet)"

    user = f"""
{header}

BATCH PLANNING (approved by user):
{plan_text}

Now produce BATCH {batch_num} as JSON ONLY, following the plan and taxonomy.
Return JSON in the wrapped form {{ "cards": [ ... ] }} — do not include any extra prose.

{rules}

Use this seen set of (url, exact quote) already used in prior batches; do not repeat any: 
{seen_pairs_text}

{schema}
"""

    card_schema = _card_schema()
    wrapped_schema = _cards_envelope_schema(card_schema, min_items=1)

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You cut debate evidence cards with meticulous sourcing and formatting."},
            {"role": "user", "content": user},
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "cards_object_envelope",
                "schema": wrapped_schema,
                "strict": True,
            },
        },
    )

    obj = _get_parsed_json(resp)
    if obj is None:
        raw = _get_output_text(resp).strip()
        try:
            obj = json.loads(raw)
        except Exception:
            raise ValueError("Model did not return valid JSON.")
    cards = _ensure_cards_list(obj)
    if not isinstance(cards, list) or not cards:
        raise ValueError("Model returned empty or invalid 'cards' array.")
    return cards


def _merge_to_excel(all_cards: List[Dict[str, Any]]) -> BytesIO | None:
    # Build DataFrames
    rows_cards = []
    for c in all_cards:
        cit = c.get("citation", {}) or {}
        rows_cards.append(
            {
                "Side": c.get("side", ""),
                "Function": c.get("function", ""),
                "Tag": c.get("tag", ""),
                "Authors": cit.get("authors", ""),
                "Year": cit.get("year", ""),
                "Date": cit.get("date", ""),
                "Title": cit.get("title", ""),   # <- from citation
                "Source": cit.get("source", ""), # <- from citation
                "URL": cit.get("url", ""),       # <- from citation
                "Quote": c.get("quote", ""),
                "Flow Sentence": c.get("flow_sentence", ""),
            }
        )

    df_cards = pd.DataFrame(rows_cards)
    df_index = pd.DataFrame([
        {
            "#": i + 1,
            "Side": r.get("Side", ""),
            "Function": r.get("Function", ""),
            "Tag": r.get("Tag", ""),
            "Year": r.get("Year", ""),
            "Source": r.get("Source", ""),
            "Title": r.get("Title", ""),
            "URL": r.get("URL", ""),
        }
        for i, r in enumerate(rows_cards)
    ])

    bio = BytesIO()
    last_err = None

    # Try available engines: let pandas choose (None), then openpyxl, then xlsxwriter
    for eng in (None, "openpyxl", "xlsxwriter"):
        try:
            bio.seek(0); bio.truncate(0)
            with pd.ExcelWriter(bio, engine=eng) as writer:
                df_cards.to_excel(writer, sheet_name="Cards", index=False)
                df_index.to_excel(writer, sheet_name="Quick Index", index=False)
            bio.seek(0)
            return bio
        except Exception as e:
            last_err = e
            continue

    # If we get here, no Excel engines were available
    return None

# --- UI ---
st.title("📚 Evidence Pack Machine (Planner + Batches)")
_init_state()

username = get_username()
remaining = get_remaining_queries(username, tool=TOOL_NAME)
if remaining is None:
    remaining = 0

st.caption(f"Remaining today for {TOOL_NAME}: {remaining}")

# Intake
with st.form("evm_intake_form"):
    st.subheader("Intake")
    resolution = st.text_area("RESOLUTION (required)", key="evm_resolution", height=100)
    area = st.text_input("AREA (optional)", key="evm_area")
    sides = st.multiselect("SIDES", options=["Pro", "Con"], default=["Pro", "Con"], key="evm_sides")
    total_cards = st.slider("TOTAL_CARDS", min_value=200, max_value=300, value=240, step=10, key="evm_total_cards")
    batch_size = st.number_input("BATCH_SIZE", min_value=10, max_value=50, value=20, step=1, key="evm_batch_size")
    model = st.text_input("Model", value=st.session_state.get("evm_model", DEFAULT_MODEL))
    submit_intake = st.form_submit_button("Propose Batch Plan")

if submit_intake:
    if not resolution.strip():
        st.error("RESOLUTION is required.")
        st.stop()
    st.session_state.evm_intake = {
        "resolution": resolution.strip(),
        "area": area.strip(),
        "sides": sides or ["Pro", "Con"],
        "total_cards": int(total_cards),
        "batch_size": int(batch_size),
    }
    st.session_state.evm_model = (model or DEFAULT_MODEL).strip()

    with st.spinner("Drafting batch plan..."):
        try:
            plan_text = _propose_plan(st.session_state.evm_intake, st.session_state.evm_model)
            st.session_state.evm_plan_text = plan_text
            st.session_state.evm_plan_approved = False
            st.session_state.evm_batches = []
            st.session_state.evm_seen_set = set()
            st.session_state.evm_current_batch = 1
            log_query(username, TOOL_NAME, True)
        except Exception as e:
            log_query(username, TOOL_NAME, False, feedback=str(e))
            st.error(f"Failed to generate plan: {e}")

# Show plan if present
if st.session_state.get("evm_plan_text"):
    st.subheader("Proposed Batch Plan")
    st.text_area("Plan (review)", st.session_state.evm_plan_text, height=300)
    c1, c2 = st.columns(2)
    with c1:
        if st.button("Approve Plan"):
            st.session_state.evm_plan_approved = True
            st.success("Plan approved. Ready to generate batches.")
    with c2:
        if st.button("Regenerate Plan"):
            st.session_state.evm_plan_text = None
            st.session_state.evm_plan_approved = False

# Batch generation controls
if st.session_state.get("evm_plan_approved"):
    st.subheader("Batch Production")
    current_n = st.session_state.get("evm_current_batch", 1)
    total_cards = int(st.session_state.evm_intake.get("total_cards", 240))
    batch_size = int(st.session_state.evm_intake.get("batch_size", 20))
    total_batches = max(1, (total_cards + batch_size - 1) // batch_size)
    st.caption(f"Planned batches: {total_batches} x {batch_size} (target {total_cards} cards)")

    if current_n <= total_batches:
        if st.button(f"Generate Batch {current_n}"):
            try:
                seen_list: List[Tuple[str, str]] = []
                for batch in st.session_state.evm_batches:
                    for card in batch:
                        k = _seen_key(card)
                        if not k:
                            continue
                        url, quote = k.split("||", 1)
                        seen_list.append((url, quote))

                with st.spinner(f"Generating batch {current_n}..."):
                    batch_cards = _generate_batch(
                        st.session_state.evm_intake,
                        st.session_state.evm_plan_text,
                        current_n,
                        st.session_state.evm_model,
                        seen_list,
                    )

                new_batch = []
                dups = 0
                for c in batch_cards:
                    key = _seen_key(c)
                    if not key:
                        continue
                    if key in st.session_state.evm_seen_set:
                        dups += 1
                        continue
                    st.session_state.evm_seen_set.add(key)
                    new_batch.append(c)

                if dups > 0:
                    st.warning(f"Removed {dups} duplicate card(s) that repeated a prior (url, quote).")

                st.session_state.evm_batches.append(new_batch)
                st.session_state.evm_current_batch = current_n + 1
                st.success(f"Batch {current_n} generated with {len(new_batch)} cards.")
            except Exception as e:
                st.error(f"Batch generation failed: {e}")

    if st.session_state.evm_batches:
        st.markdown("### Latest Batch JSON")
        last_batch = st.session_state.evm_batches[-1]
        st.code(json.dumps(last_batch, ensure_ascii=False, indent=2))
        st.download_button(
            label="Download Last Batch JSON",
            data=json.dumps(last_batch, ensure_ascii=False, indent=2),
            file_name=f"batch_{len(st.session_state.evm_batches)}.json",
            mime="application/json",
        )

        # Under: if st.session_state.evm_batches:
        last_batch = st.session_state.evm_batches[-1]
        last_idx = len(st.session_state.evm_batches)
        current_n = st.session_state.get("evm_current_batch", 1)
        total_cards = int(st.session_state.evm_intake.get("total_cards", 240))
        batch_size = int(st.session_state.evm_intake.get("batch_size", 20))
        total_batches = max(1, (total_cards + batch_size - 1) // batch_size)

        c_next1, c_next2 = st.columns(2)
        with c_next1:
            if current_n <= total_batches and st.button(f"✅ Accept Batch {last_idx} & Generate Batch {current_n}"):
                try:
                    # Build seen list
                    seen_list: List[Tuple[str, str]] = []
                    for b in st.session_state.evm_batches:
                        for c in b:
                            k = _seen_key(c)
                            if not k:
                                continue
                            url, quote = k.split("||", 1)
                            seen_list.append((url, quote))

                    with st.spinner(f"Generating batch {current_n}..."):
                        next_cards = _generate_batch(
                            st.session_state.evm_intake,
                            st.session_state.evm_plan_text,
                            current_n,
                            st.session_state.evm_model,
                            seen_list,
                        )

                    # De-dup vs global seen set
                    new_batch = []
                    dups = 0
                    for c in next_cards:
                        key = _seen_key(c)
                        if not key:
                            continue
                        if key in st.session_state.evm_seen_set:
                            dups += 1
                            continue
                        st.session_state.evm_seen_set.add(key)
                        new_batch.append(c)

                    if dups > 0:
                        st.warning(f"Removed {dups} duplicate card(s) that repeated a prior (url, quote).")

                    st.session_state.evm_batches.append(new_batch)
                    st.session_state.evm_current_batch = current_n + 1
                    st.success(f"Batch {current_n} generated with {len(new_batch)} cards.")
                except Exception as e:
                    st.error(f"Batch generation failed: {e}")

        with c_next2:
            if st.button("🔁 Regenerate This Batch Instead"):
                # Remove last batch and decrement current_n so you can rerun it
                if st.session_state.evm_batches:
                    removed = st.session_state.evm_batches.pop()
                    # Also clean their seen keys
                    for c in removed:
                        k = _seen_key(c)
                        if k and k in st.session_state.evm_seen_set:
                            st.session_state.evm_seen_set.remove(k)
                    st.session_state.evm_current_batch = max(1, current_n - 1)
                    st.info("Last batch removed. Click Generate to redo it.")

        # Keep “Fix a Card” visible at this level (not nested inside a button)
        st.markdown("#### Fix a Card in Last Batch")
        fix_idx = st.number_input(
            "Card # to replace (1-based)",
            min_value=1,
            max_value=max(1, len(last_batch)),
            value=1
        )
        if st.button("Fix #N in Last Batch"):
            try:
                client = _get_oai()

                header = _batch_prompt_header(st.session_state.evm_intake)
                rules = _quality_rules_block()
                schema = _schema_block()

                # Build context (exclude last batch so we can replace within it freely)
                seen_list: List[Tuple[str, str]] = []
                for b in st.session_state.evm_batches[:-1]:
                    for c in b:
                        k = _seen_key(c)
                        if k:
                            u, q = k.split("||", 1)
                            seen_list.append((u, q))

                user = f"""
{header}

We are correcting one card in the most recent batch. Replace the card with index {int(fix_idx)} in that batch with a new, valid card that obeys all rules. Do not duplicate any (url, exact quote) from this seen set across prior batches:
{"\n".join([f"- URL: {u} | QUOTE: {q[:140]}" for (u,q) in seen_list]) or '(none)'}

Return JSON with exactly ONE card in the wrapped form {{ "cards": [ ... ] }}. Do not include any prose.

{rules}
{schema}
"""

                card_schema = _card_schema()
                wrapped_schema = _cards_envelope_schema(card_schema, min_items=1, max_items=1)

                resp = client.chat.completions.create(
                    model=st.session_state.evm_model,
                    messages=[
                        {"role": "system", "content": "You cut debate evidence cards with meticulous sourcing and formatting."},
                        {"role": "user", "content": user},
                    ],
                    response_format={
                        "type": "json_schema",
                        "json_schema": {
                            "name": "one_card_object_envelope",
                            "schema": wrapped_schema,
                            "strict": True,
                        },
                    },
                )

                obj = _get_parsed_json(resp)
                if obj is None:
                    raw = _get_output_text(resp).strip()
                    try:
                        obj = json.loads(raw)
                    except Exception:
                        raise ValueError("Model did not return valid JSON.")

                repl_cards = _ensure_cards_list(obj)
                if not isinstance(repl_cards, list) or len(repl_cards) != 1:
                    raise ValueError("Model did not return exactly one card in 'cards'.")

                replacement = repl_cards[0]

                # Update seen set (remove old key, add new)
                old_key = _seen_key(last_batch[int(fix_idx) - 1])
                if old_key and old_key in st.session_state.evm_seen_set:
                    st.session_state.evm_seen_set.remove(old_key)
                new_key = _seen_key(replacement)
                if new_key in st.session_state.evm_seen_set:
                    raise ValueError("Replacement duplicates an existing (url, quote). Try again.")
                st.session_state.evm_seen_set.add(new_key)
                last_batch[int(fix_idx) - 1] = replacement
                st.success(f"Replaced card #{int(fix_idx)} in last batch.")
            except Exception as e:
                st.error(f"Fix failed: {e}")

# Merge/export
st.markdown("### Merge All Batches → Excel")
if st.button("Merge"):
    try:
        all_cards = [c for b in st.session_state.evm_batches for c in b]
        seen = set()
        merged: List[Dict[str, Any]] = []
        for c in all_cards:
            k = _seen_key(c)
            if k and k not in seen:
                seen.add(k)
                merged.append(c)
        bio = _merge_to_excel(merged)
        if bio is not None:
            st.download_button(
                label="Download Evidence.xlsx",
                data=bio,
                file_name="evidence_pack.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
        else:
            st.error(
                "Could not create Excel file because no Excel writer engine is installed. "
                "Add either 'openpyxl' or 'xlsxwriter' to your environment (requirements) and try again."
            )
import json, glob
import pandas as pd

def load_batches(pattern='batch_*.json'):
    cards = []
    for path in sorted(glob.glob(pattern)):
        with open(path, 'r', encoding='utf-8') as f:
            batch = json.load(f)
            cards.extend(batch)
    return cards

cards = load_batches()
# De-dupe by (url, quote)
seen = set()
merged = []
for c in cards:
    url = (c.get('citation') or {}).get('url','').strip()
    quote = (c.get('quote') or '').strip()
    k = f"{url}||{quote}"
    if k and k not in seen:
        seen.add(k)
        merged.append(c)

rows = []
for c in merged:
    cit = c.get('citation') or {}
    rows.append({
        'Side': c.get('side',''),
        'Function': c.get('function',''),
        'Tag': c.get('tag',''),
        'Authors': cit.get('authors',''),
        'Year': cit.get('year',''),
        'Date': cit.get('date',''),
        'Title': cit.get('title',''),
        'Source': cit.get('source',''),
        'URL': cit.get('url',''),
        'Quote': c.get('quote',''),
        'Flow Sentence': c.get('flow_sentence',''),
    })

df_cards = pd.DataFrame(rows)
df_index = pd.DataFrame([
    {'#': i+1, 'Side': r['Side'], 'Function': r['Function'], 'Tag': r['Tag'], 'Year': r['Year'], 'Source': r['Source'], 'Title': r['Title'], 'URL': r['URL']} 
    for i, r in enumerate(rows)
])

with pd.ExcelWriter('evidence_pack.xlsx', engine='xlsxwriter') as w:
    df_cards.to_excel(w, sheet_name='Cards', index=False)
    df_index.to_excel(w, sheet_name='Quick Index', index=False)
print('Wrote evidence_pack.xlsx')
"""
            st.code(snippet, language="python")
    except Exception as e:
        st.error(f"Merge failed: {e}")
