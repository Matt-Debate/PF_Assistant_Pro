import os
import json
import uuid
import random
import time
import streamlit as st
import pandas as pd
from typing import List, Dict, Any, Tuple, Optional
from io import BytesIO

from auth import get_username, log_query, get_remaining_queries
from openai import OpenAI

# ---------------------------
# Constants / Paths
# ---------------------------
TOOL_NAME = "evidence_machine"
DEFAULT_MODEL = "gpt-5"
RUNS_DIR = "evm_runs"  # persisted checkpoints live here (create if missing)
os.makedirs(RUNS_DIR, exist_ok=True)

# ---------------------------
# Session init
# ---------------------------
def _init_state():
    ss = st.session_state
    ss.setdefault("evm_run_id", None)            # active run id
    ss.setdefault("evm_run", None)               # active run dict
    ss.setdefault("evm_model", DEFAULT_MODEL)    # model
    ss.setdefault("evm_logs", [])                # UI logs (derived from run)

# ---------------------------
# OpenAI client (singleton)
# ---------------------------
@st.cache_resource(show_spinner=False)
def _get_oai() -> OpenAI:
    api_key = st.secrets.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Missing OPENAI_API_KEY in Streamlit secrets.")
    return OpenAI(api_key=api_key)

# ---------------------------
# Retry wrapper
# ---------------------------
def _with_retries(make_call, *, retries: int = 4, base_delay: float = 1.0, max_delay: float = 8.0):
    last_err = None
    for attempt in range(retries):
        try:
            return make_call()
        except Exception as e:
            last_err = e
            if attempt >= retries - 1:
                raise
            sleep_s = min(max_delay, base_delay * (2 ** attempt)) + random.uniform(0, 0.25)
            time.sleep(sleep_s)
    if last_err:
        raise last_err

# ---------------------------
# Persistence
# ---------------------------
def _run_path(username: str, run_id: str) -> str:
    return os.path.join(RUNS_DIR, f"{username}_{run_id}.json")

def _save_run(username: str, run: Dict[str, Any]) -> None:
    path = _run_path(username, run["run_id"])
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(run, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)  # atomic

def _load_run(username: str, run_id: str) -> Optional[Dict[str, Any]]:
    path = _run_path(username, run_id)
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def _list_runs_for_user(username: str) -> List[str]:
    prefix = f"{username}_"
    ids = []
    for name in sorted(os.listdir(RUNS_DIR), reverse=True):
        if name.startswith(prefix) and name.endswith(".json"):
            rid = name[len(prefix):-5]
            ids.append(rid)
    return ids

# ---------------------------
# Model helpers (extraction)
# ---------------------------
def _get_output_text(resp) -> str:
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

# ---------------------------
# Evidence logic
# ---------------------------
def _seen_key(card: Dict[str, Any]) -> str:
    try:
        url = (card.get("citation") or {}).get("url", "").strip()
        quote = (card.get("quote") or "").strip()
        return f"{url}||{quote}"
    except Exception:
        return ""

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
      "quote": "Full, direct, multi-sentence paragraph(s) from one location on the page.",
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
- Quotes must be verbatim multi-sentence paragraphs from one location on the page (no stitching far-apart parts). Quotes must be in English.
- Prefer authoritative, recent, accessible sources; avoid broken links or paywalled abstracts without visible text. If text is not present at the URL, replace the card before delivering the batch.
- Diversity: vary publishers/countries/years/functions; avoid over-reliance on any single institution.
"""
    ).strip()

def _card_schema() -> Dict[str, Any]:
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
                "additionalProperties": False,
            },
            "quote": {"type": "string"},
            "underline_phrases": {"type": "array", "items": {"type": "string"}, "minItems": 2, "maxItems": 5},
            "flow_sentence": {"type": "string"},
        },
        "required": ["side", "function", "tag", "citation", "quote", "underline_phrases", "flow_sentence"],
        "additionalProperties": False,
    }

def _cards_envelope_schema(item_schema: Dict[str, Any], min_items: int = 1, max_items: Optional[int] = None) -> Dict[str, Any]:
    arr_schema: Dict[str, Any] = {"type": "array", "items": item_schema, "minItems": min_items}
    if max_items is not None:
        arr_schema["maxItems"] = max_items
    return {"type": "object", "properties": {"cards": arr_schema}, "required": ["cards"], "additionalProperties": False}

def _propose_plan(intake: Dict[str, Any], model: str) -> str:
    client = _get_oai()
    resolution = intake.get("resolution", "").strip()
    area = intake.get("area", "").strip()
    sides = ", ".join(intake.get("sides", ["Pro", "Con"]))
    total_cards = int(intake.get("total_cards", 100))

    sys = "You are a debate coach planning an evidence pack across batches."
    user = f"""
OBJECTIVE
Build an evidence pack of about {total_cards} total cards.

RESOLUTION: {resolution}
AREA: {area or '(none)'}
SIDES: {sides}

Task: Propose a numbered batch plan where YOU choose both:
- the number of batches, and
- the typical cards per batch,
based on the topic’s complexity and the {total_cards}-card target. The plan should total ≈{total_cards} cards overall (±10%).

For each batch, include:
- Side(s) (Pro or Con; balance across plan)
- Thematic focus (functions/angles you’ll target)
- Cards-per-batch target (your choice; can vary by batch)
- Coverage goals to keep the full set broad and non-duplicative
- Approximate share of quantitative cards so overall we exceed 30–40% quant

Use the taxonomy: Definitions & frameworks; Economy/exports/productivity; Border/customs/single window/authorized operator;
Domestic transport (roads/rail/waterways); Maritime connectivity/ports/shipping; Energy corridors/interconnection;
Agriculture/cold chain/food loss; SME inclusion/competitiveness; Digital trade/logistics; Climate resilience & disruptions;
Finance/PPP; Legal/regulatory/integration blocs; Subregional case studies; and for Con: fiscal trade-offs, distributional/inequality
effects, environment/externalities, dependency/lock-in, governance capacity risks.

Output: A concise, numbered plan in prose. Do not produce any cards yet.
"""
    resp = _with_retries(lambda: _get_oai().responses.create(
        model=model,
        input=[{"role": "system", "content": sys}, {"role": "user", "content": user}],
        reasoning={"effort": "high"},
    ))
    return _get_output_text(resp).strip()

def _generate_batch(intake: Dict[str, Any], plan_text: str, batch_num: int, model: str, seen_pairs: List[Tuple[str, str]]) -> List[Dict[str, Any]]:
    header = _batch_prompt_header(intake)
    schema = _schema_block()
    rules = _quality_rules_block()
    seen_text = "\n".join([f"- URL: {u} | QUOTE: {q[:140]}" for (u, q) in seen_pairs]) if seen_pairs else "(none yet)"

    user = f"""
{header}

BATCH PLAN (approved):
{plan_text}

Now produce BATCH {batch_num} as JSON ONLY, following the plan and taxonomy.
Return JSON in the wrapped form {{ "cards": [ ... ] }} — no extra prose.

{rules}

Do not repeat any (url, exact quote) from previous batches:
{seen_text}

{schema}
"""
    card_schema = _card_schema()
    wrapped_schema = _cards_envelope_schema(card_schema, min_items=1)

    resp = _with_retries(lambda: _get_oai().chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You cut debate evidence cards with meticulous sourcing and formatting."},
            {"role": "user", "content": user},
        ],
        response_format={"type": "json_schema", "json_schema": {"name": "cards_object_envelope", "schema": wrapped_schema, "strict": True}},
    ))

    obj = _get_parsed_json(resp)
    if not obj or not isinstance(obj, dict) or "cards" not in obj or not isinstance(obj["cards"], list):
        raw = _get_output_text(resp).strip()
        obj = json.loads(raw)

    cards = obj["cards"]
    if not isinstance(cards, list) or not cards:
        raise ValueError("Model returned empty or invalid 'cards' array.")
    return cards

# ---------------------------
# Excel underline helpers
# ---------------------------
def _find_all_occurrences(text: str, phrase: str) -> List[Tuple[int, int]]:
    hits = []
    if not phrase:
        return hits
    t_low = text.lower()
    p_low = phrase.lower()
    start = 0
    while True:
        idx = t_low.find(p_low, start)
        if idx == -1:
            break
        hits.append((idx, idx + len(phrase)))
        start = idx + len(phrase)
    return hits

def _merge_overlaps(ranges: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    if not ranges:
        return []
    ranges = sorted(ranges, key=lambda x: (x[0], x[1]))
    merged = []
    cur_s, cur_e = ranges[0]
    for s, e in ranges[1:]:
        if s <= cur_e:
            cur_e = max(cur_e, e)
        else:
            merged.append((cur_s, cur_e))
            cur_s, cur_e = s, e
    merged.append((cur_s, cur_e))
    return merged

def _rich_parts_for_quote(quote: str, phrases: List[str], underline_format) -> List[Any] | None:
    all_ranges: List[Tuple[int, int]] = []
    for ph in phrases or []:
        all_ranges.extend(_find_all_occurrences(quote, ph))
    if not all_ranges:
        return None
    merged = _merge_overlaps(all_ranges)
    parts: List[Any] = []
    pos = 0
    for s, e in merged:
        if s > pos:
            parts.append(quote[pos:s])
        parts.extend([underline_format, quote[s:e]])
        pos = e
    if pos < len(quote):
        parts.append(quote[pos:])
    return parts

def _merge_to_excel(all_cards: List[Dict[str, Any]]) -> BytesIO | None:
    rows_cards = []
    phrases_list: List[List[str]] = []
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
                "Title": cit.get("title", ""),
                "Source": cit.get("source", ""),
                "URL": cit.get("url", ""),
                "Quote": c.get("quote", ""),
                "Flow Sentence": c.get("flow_sentence", ""),
            }
        )
        phrases_list.append(list(c.get("underline_phrases", []) or []))

    df_cards = pd.DataFrame(rows_cards)
    df_index = pd.DataFrame(
        [
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
        ]
    )

    bio = BytesIO()
    # prefer xlsxwriter for underline formatting
    for eng in ("xlsxwriter", None):
        try:
            bio.seek(0); bio.truncate(0)
            with pd.ExcelWriter(bio, engine=eng) as writer:
                df_cards.to_excel(writer, sheet_name="Cards", index=False)
                df_index.to_excel(writer, sheet_name="Quick Index", index=False)

                if eng == "xlsxwriter":
                    workbook = writer.book
                    ws = writer.sheets["Cards"]
                    underline_fmt = workbook.add_format({"underline": True})

                    qcol = None
                    try:
                        qcol = df_cards.columns.get_loc("Quote")
                    except Exception:
                        pass

                    if qcol is not None:
                        for i, row in enumerate(rows_cards):
                            quote = row.get("Quote", "")
                            parts = _rich_parts_for_quote(quote, phrases_list[i], underline_fmt)
                            excel_row = i + 1
                            if parts and len(parts) >= 2:
                                ws.write_rich_string(excel_row, qcol, *parts)
                            else:
                                ws.write(excel_row, qcol, quote)

            bio.seek(0)
            return bio
        except Exception:
            continue
    return None

# ---------------------------
# Generation orchestration (resumable)
# ---------------------------
def _append_log(run: Dict[str, Any], msg: str):
    run.setdefault("logs", [])
    run["logs"].append(msg)

def _rebuild_seen_set(run: Dict[str, Any]) -> set:
    seen = set()
    for batch in run.get("batches", []):
        for c in batch:
            k = _seen_key(c)
            if k:
                seen.add(k)
    return seen

def _resume_or_start_generation(username: str, run: Dict[str, Any], auto_continue: bool):
    """
    Drive generation until target or until one batch (if auto_continue is False).
    Always saves after each batch so a disconnect won't lose progress.
    """
    intake = run["intake"]
    model = run["model"]
    plan_text = run.get("plan_text") or ""
    target_total = int(intake["total_cards"])

    # ensure seen_set based on saved batches
    run["seen_set"] = list(_rebuild_seen_set(run))

    # If no plan yet, create one
    if not plan_text:
        _append_log(run, "Drafting batch plan…")
        plan_text = _propose_plan(intake, model)
        run["plan_text"] = plan_text
        _append_log(run, "Batch plan ready.")
        _save_run(username, run)

    # Generate batches
    MAX_BATCHES = 80
    while run["total_generated"] < target_total and len(run["batches"]) < MAX_BATCHES:
        batch_num = len(run["batches"]) + 1
        _append_log(run, f"Generating batch {batch_num}…")

        # Build seen_pairs from seen_set
        seen_pairs: List[Tuple[str, str]] = []
        for sk in run["seen_set"]:
            try:
                u, q = sk.split("||", 1)
                seen_pairs.append((u, q))
            except Exception:
                continue

        cards = _generate_batch(intake, run["plan_text"], batch_num, model, seen_pairs)

        # de-dupe vs global seen_set
        new_batch = []
        dups = 0
        for c in cards:
            k = _seen_key(c)
            if k and k not in run["seen_set"]:
                run["seen_set"].append(k)
                new_batch.append(c)
            else:
                dups += 1

        run["batches"].append(new_batch)
        run["total_generated"] += len(new_batch)

        _append_log(run, f"Batch {batch_num} complete ({len(new_batch)} cards; total {run['total_generated']}/{target_total}).")
        _save_run(username, run)

        if not auto_continue:
            break  # stop after one batch if user chose manual stepping

# ---------------------------
# UI
# ---------------------------
st.title("📚 Evidence Packet Machine")
_init_state()

username = get_username()
remaining = get_remaining_queries(username, tool=TOOL_NAME) or 0
st.caption(f"Remaining today for {TOOL_NAME}: {remaining}")

with st.sidebar:
    st.subheader("Resume an existing run")
    existing = _list_runs_for_user(username)
    selected_run = st.selectbox("Pick a Run ID", ["(none)"] + existing, index=0)
    auto_continue_sidebar = st.toggle("Auto-continue generation", value=True, key="auto_continue_sidebar")
    if selected_run != "(none)":
        if st.button("Load Run"):
            run = _load_run(username, selected_run)
            if run:
                st.session_state.evm_run_id = selected_run
                st.session_state.evm_run = run
                st.session_state.evm_model = run.get("model", DEFAULT_MODEL)
                st.session_state.evm_logs = run.get("logs", [])
                st.success(f"Loaded run {selected_run}")
            else:
                st.error("Could not load selected run (file missing or corrupted).")

# Intake
with st.form("evm_intake_form"):
    resolution = st.text_area("RESOLUTION (required)", key="evm_resolution", height=100)
    area = st.text_input("AREA (optional)", key="evm_area")
    sides = st.multiselect("SIDES", options=["Pro", "Con"], default=["Pro", "Con"], key="evm_sides")
    total_cards = st.slider("TOTAL_CARDS", min_value=40, max_value=200, value=100, step=10, key="evm_total_cards")
    model = st.text_input("Model", value=st.session_state.get("evm_model", DEFAULT_MODEL))
    auto_continue = st.toggle("Auto-continue after each batch", value=True, key="auto_continue")
    submit_intake = st.form_submit_button("Start New Run")

if submit_intake:
    if not resolution.strip():
        st.error("RESOLUTION is required.")
        st.stop()

    run_id = uuid.uuid4().hex[:8]
    new_run = {
        "run_id": run_id,
        "username": username,
        "model": (model or DEFAULT_MODEL).strip(),
        "intake": {
            "resolution": resolution.strip(),
            "area": area.strip(),
            "sides": sides or ["Pro", "Con"],
            "total_cards": int(total_cards),
        },
        "plan_text": "",
        "batches": [],
        "seen_set": [],
        "logs": [],
        "total_generated": 0,
        "created_at": int(time.time()),
    }
    _save_run(username, new_run)

    st.session_state.evm_run_id = run_id
    st.session_state.evm_run = new_run
    st.session_state.evm_model = new_run["model"]
    st.session_state.evm_logs = []

    # kick off first steps immediately
    try:
        _resume_or_start_generation(username, st.session_state.evm_run, auto_continue=auto_continue)
        log_query(username, TOOL_NAME, True)
    except Exception as e:
        log_query(username, TOOL_NAME, False, feedback=str(e))
        st.error(f"Error during generation: {e}")

# If there's an active run in session, show console/progress + actions
if st.session_state.evm_run:
    run = st.session_state.evm_run
    intake = run["intake"]
    target_total = intake["total_cards"]

    st.markdown(f"**Run ID:** `{run['run_id']}`  •  **Target:** {target_total} cards  •  **Generated so far:** {run['total_generated']}")

    # show logs (persisted)
    st.divider()
    st.subheader("Progress")
    logs_container = st.container()
    with logs_container:
        for line in run.get("logs", []):
            st.write("• " + line)

    # Action buttons
    st.divider()
    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("▶️ Generate Next Batch"):
            try:
                _resume_or_start_generation(username, run, auto_continue=False)
                st.session_state.evm_logs = run.get("logs", [])
                st.success("Next batch generated.")
            except Exception as e:
                st.error(f"Error: {e}")

    with c2:
        if st.button("⏩ Auto-Continue Until Done"):
            try:
                _resume_or_start_generation(username, run, auto_continue=True)
                st.session_state.evm_logs = run.get("logs", [])
                st.success("Auto-continue step completed.")
            except Exception as e:
                st.error(f"Error: {e}")

    with c3:
        if st.button("🧹 Finish & Merge Now"):
            pass  # just a UI affordance; merge button is below and always available

    # Show last batch JSON for quick inspection
    if run.get("batches"):
        last_batch = run["batches"][-1]
        st.markdown("**Latest Batch (JSON preview):**")
        st.code(json.dumps(last_batch, ensure_ascii=False, indent=2))

    # Merge/export (available anytime)
    st.subheader("Download")
    all_cards = [c for b in run.get("batches", []) for c in b]
    bio = _merge_to_excel(all_cards)
    if bio is not None:
        st.download_button(
            label="⬇️ Download Evidence.xlsx",
            data=bio,
            file_name=f"evidence_{run['run_id']}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
    else:
        st.error(
            "Could not create Excel file because no Excel writer engine is installed. "
            "Add 'xlsxwriter' to your environment for best results."
        )
else:
    st.info("Start a new run above, or use the sidebar to load an existing run.")
