import streamlit as st
import json
from tools.tool_brainstorm_claims import generate_claims
from tools.tool_get_research import query_perplexity, build_perplexity_prompt
from tools.tool_write_contention import write_contention
from tools.tool_revise_contention import revise_contention
from tools.tool_filter_research import filter_and_format_evidence
from tools.tool_combine_contention_research import combine_contention_and_evidence
from tools.tool_log_and_retry import log_and_retry  
from tools.tool_helpers import (parse_claims_output,merge_evidence_dicts,step_ui)

def main():
    st.title("🧠 AI Argument Writer")

    # Session state defaults
    if "step" not in st.session_state:
        st.session_state.step = 1
    if "content" not in st.session_state:
        st.session_state.content = {}

    # -----------------------
    # Step 1: Input Topic, Side, Area
    # -----------------------
    if st.session_state.step == 1:
        with st.form("step1"):
            writing_type = st.selectbox(
                "What do you want to write?",
                ["Contention", "Rebuttal Block", "Frontline", "Crossfire Questions"],
                key="writing_type",
            )
            topic = st.text_input("Topic:", key="topic")
            side = st.selectbox("Side:", ["Pro", "Con"], key="side")
            area = st.text_input("Area:", key="area")
            submitted = st.form_submit_button("Proceed")

        if submitted:
            st.session_state.content = {
                "topic": topic.strip(),
                "side": side,
                "area": area.strip(),
                "writing_type": writing_type,
            }
            st.session_state.step = 2
            st.rerun()

    # -----------------------
    # Step 2: Generate Contention Claims
    # -----------------------
    elif st.session_state.step == 2:
        content = st.session_state.content
        topic = content.get("topic", "")
        side = content.get("side", "")
        area = content.get("area", "")

        with step_ui(2, "Generate Contention Claims", "Generating claims based on your input..."):
            try:
                raw_brainstorm_output = generate_claims(topic, side, area)
                claims_json = parse_claims_output(raw_brainstorm_output)
            except Exception as e:
                st.error(f"Error generating or parsing claims: {e}")
                # show raw output if available for debugging
                try:
                    st.text_area("Raw model output (debug)", str(raw_brainstorm_output), height=300)
                except Exception:
                    pass
                return

            # sanity check
            if not isinstance(claims_json, dict) or "claims" not in claims_json:
                st.error("Parsed claims output is missing the 'claims' key or is not an object.")
                st.text_area("Parsed output (debug)", json.dumps(claims_json, indent=2, ensure_ascii=False), height=300)
                return

            # store for next steps
            st.session_state.content["claims_json"] = claims_json

            # quick readable summary of the claims for the user
            st.markdown("**Claims generated:**")
            for c in claims_json.get("claims", []):
                num = c.get("claim_number", "")
                desc = c.get("description", "") or c.get("claim", "")
                st.write(f"- **Claim {num}** — {desc}")

        # advance to step 3 automatically
        st.session_state.step = 3
        st.rerun()

    # -----------------------
    elif st.session_state.step == 3:
        claims_data = st.session_state.content.get("claims_json")
        if not claims_data:
            st.error("No claims found in session — please regenerate claims.")
            return

        claims = claims_data.get("claims", [])
        impact_claim = claims_data.get("impact_claim")
        if impact_claim:
            claims.append({
                "claim_number": "impact",
                "description": impact_claim.get("description", ""),
                "evidence_needed": impact_claim.get("evidence_needed", "")
            })

        if not claims:
            st.warning("No claims found to search for.")
            return

        with step_ui(3, "Evidence Search", "Searching for evidence based on your claims..."):
            evidence_results = []
            for i, claim in enumerate(claims, start=1):
                description = claim.get("description", "")
                prompt = build_perplexity_prompt(description)
                if st.session_state.content.get("quant", False):
                    prompt += " Find quotes with statistical data."

                with st.spinner(f"Fetching evidence for claim {i}..."):
                    try:
                        raw_evidence = log_and_retry(lambda: query_perplexity(prompt))
                    except Exception as e:
                        raw_evidence = f"Error retrieving evidence for claim {i}: {e}"

                    evidence_results.append(raw_evidence)

            st.session_state.content["evidence_results"] = evidence_results

        st.session_state.step = 4
        st.rerun()

    # -----------------------
    # Step 4: Review Evidence (show all raw evidence outputs)
    # -----------------------
    elif st.session_state.step == 4:
        st.subheader("Step 4: Review Evidence")
        st.markdown("Review the raw evidence retrieved from the AI sources (one output per claim).")

        raw_evidence_texts = st.session_state.content.get("evidence_results", [])
        if not raw_evidence_texts:
            st.warning("No evidence outputs found. Go back to Step 3.")
        else:
            for i, raw_text in enumerate(raw_evidence_texts, start=1):
                st.markdown(f"### Evidence Output {i}")
                st.text_area(f"Evidence Output {i}", raw_text, height=300)

        # give the user a chance to proceed manually
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Restart Process"):
                st.session_state.step = 1
                st.session_state.content = {}
                st.rerun()

        with col2:
            if st.button("Proceed to Step 5: Write Contention"):
                st.session_state.step = 5
                st.rerun()
        
    # -----------------------
    # Step 5: Write Contention (final draft from LLM)
    # -----------------------
    elif st.session_state.step == 5:
        st.subheader("Step 5: Write Contention")
        st.markdown("Generating initial contention draft using the collected evidence...")

        content = st.session_state.content
        topic = content.get("topic", "")
        side = content.get("side", "")
        area = content.get("area", "")
        raw_evidence_list = content.get("evidence_results", [])

        if not (topic and side and area and raw_evidence_list):
            st.error("Missing topic/side/area/evidence. Please complete prior steps.")
            return

        with step_ui(5, "Write Contention", "Writing the contention from the provided evidence..."):
            try:
                # the write_contention tool should accept the list of raw evidence strings (it will merge/parse)
                raw_contention = write_contention(topic, side, area, raw_evidence_list)
            except Exception as e:
                st.error(f"Error writing contention: {e}")
                return

            st.text_area("Raw Contention Draft", raw_contention, height=350)
            st.session_state.content["raw_contention"] = raw_contention
            st.session_state.step = 6
            st.rerun()

    # -----------------------
    # Steps 6-8: leave mostly unchanged but use st.rerun
    # -----------------------
    elif st.session_state.step == 6:

        raw_contention = st.session_state.content.get("raw_contention", "")
        merged_evidence_json_str = st.session_state.content.get("merged_evidence_json_str", "")

        if not raw_contention:
            st.error("Missing raw contention. Please generate it in Step 5.")
            return

        with step_ui(6, "Revise Contention", "Polishing the contention..."):
            try:
                polished_contention = revise_contention(raw_contention, merged_evidence_json_str)
            except Exception as e:
                st.error(f"Error revising contention: {e}")
                return

            st.text_area("Polished Contention", polished_contention, height=600)
            # Store both keys so final output can use either
            st.session_state.content["contention_text"] = polished_contention
            st.session_state.content["polished_contention"] = polished_contention

            # ... your existing merging of evidence json ...
            if not merged_evidence_json_str:
                merged = merge_evidence_dicts(st.session_state.content.get("evidence_results", []))
                merged_evidence_json_str = json.dumps(merged, ensure_ascii=False)
                st.session_state.content["merged_evidence_json_str"] = merged_evidence_json_str

        st.session_state.step = 7
        st.rerun()

    # Step 7: Filter and Format Evidence
    elif st.session_state.step == 7:
        st.subheader("Step 7: Filter and Format Evidence")
        st.markdown("Choosing relevant evidence cards...")

        contention_text = st.session_state.content.get("contention_text", "")
        evidence_json_str = st.session_state.content.get("merged_evidence_json_str", "")

        if not contention_text:
            st.error("No contention text found. Please complete Step 6 first.")
            return

        if not evidence_json_str:
            st.error("No evidence JSON found. Please complete Step 5 first.")
            return

        try:
            formatted_evidence = filter_and_format_evidence(contention_text, evidence_json_str)
        except Exception as e:
            st.error(f"Error generating final evidence cards: {e}")
            return

        st.session_state.content["formatted_evidence"] = formatted_evidence
        st.success("Evidence successfully Filtered!")
        st.session_state.step = 8
        st.rerun()
    
    # Step 8: Final Argument Assembly

    elif st.session_state.step == 8:
        polished_contention = st.session_state.content.get("polished_contention", "")
        formatted_evidence = st.session_state.content.get("formatted_evidence", "")

        if not (polished_contention and formatted_evidence):
            st.error("Missing content/evidence. Please complete prior steps.")
            return

        with step_ui(8, "Combine Contention and Evidence", "Combining polished contention and formatted evidence cards..."):
            try:
                final_output = combine_contention_and_evidence(polished_contention, formatted_evidence)
            except Exception as e:
                st.error(f"Error combining final output: {e}")
                return
            st.text_area("Final Debate Argument", final_output, height=500)

            if st.button("Restart Process"):
                st.session_state.step = 1
                st.session_state.content = {}
                st.rerun()


if __name__ == "__main__":
    main()
