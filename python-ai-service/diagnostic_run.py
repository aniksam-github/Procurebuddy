"""Deep diagnostic: traces every layer of the ProcureBuddy RAG pipeline."""
import sys
import traceback
import logging

# Show logs on stdout
logging.basicConfig(level=logging.INFO, format="%(name)s | %(levelname)s | %(message)s")

QUERY = "What is the procurement method for Rs. 8 lakh purchase?"

print("=" * 70)
print("STEP 1: Environment & KB check")
print("=" * 70)
try:
    from app.core.config import settings
    print(f"  LLM model      : {settings.llm_model}")
    print(f"  API key set    : {bool(settings.llm_api_key)}")
    print(f"  Top-K          : {settings.top_k}")
except Exception as e:
    print(f"  CONFIG ERROR: {e}")
    sys.exit(1)

print()
print("=" * 70)
print("STEP 2: Knowledge base retrieval check")
print("=" * 70)
try:
    from app.services.knowledge_base import knowledge_base
    from app.core.rag_engine import retrieve_candidates
    matches = retrieve_candidates(QUERY)
    print(f"  Chunks retrieved: {len(matches)}")
    for i, m in enumerate(matches[:3], 1):
        meta = getattr(m, "metadata", {}) or {}
        print(f"  [{i}] score={m.score:.3f} | file={m.file_name[:50]} | rule={meta.get('rule_number','')}")
        print(f"       content_preview: {m.content[:120].strip()!r}")
except Exception as e:
    print(f"  KB RETRIEVAL ERROR: {e}")
    traceback.print_exc()

print()
print("=" * 70)
print("STEP 3: Amount extraction check")
print("=" * 70)
try:
    from app.utils.processors import extract_amount_lakhs
    amt = extract_amount_lakhs(QUERY)
    print(f"  Extracted amount: {amt} lakhs")
    if amt is None:
        print("  [ROOT CAUSE CANDIDATE] Amount extraction returned None — threshold routing broken")
except Exception as e:
    print(f"  AMOUNT PARSE ERROR: {e}")

print()
print("=" * 70)
print("STEP 4: Planner decision check")
print("=" * 70)
try:
    from app.core.orchestrator import plan_query, heuristic_plan
    heuristic = heuristic_plan(QUERY)
    print(f"  Heuristic type     : {heuristic.problem_type}")
    print(f"  Heuristic conf     : {heuristic.confidence}")
    print(f"  needs_rag          : {heuristic.needs_rag}")
    print(f"  needs_threshold    : {heuristic.needs_threshold_logic}")
    planner = plan_query(QUERY, bypass_cache=True)
    print(f"  Final planner type : {planner.problem_type}")
    print(f"  Final tools        : {planner.tools()}")
except Exception as e:
    print(f"  PLANNER ERROR: {e}")
    traceback.print_exc()

print()
print("=" * 70)
print("STEP 5: LLM call check")
print("=" * 70)
try:
    from app.services.llm_service import generate_llm_response, LLM_MAX_TOKENS, LLM_TEMPERATURE
    print(f"  LLM_MAX_TOKENS   : {LLM_MAX_TOKENS}")
    print(f"  LLM_TEMPERATURE  : {LLM_TEMPERATURE}")
    test_prompt = "Say the word HELLO only."
    response = generate_llm_response(test_prompt, system_prompt="You are a test assistant.")
    print(f"  LLM response     : {response!r}")
    if not response:
        print("  [ROOT CAUSE] LLM returned None — API key or rate limit issue")
except Exception as e:
    print(f"  LLM ERROR: {e}")
    traceback.print_exc()

print()
print("=" * 70)
print("STEP 6: STRUCTURED_FORMAT_PROMPT check (formatter)")
print("=" * 70)
try:
    from app.core.constants import STRUCTURED_FORMAT_PROMPT
    print(f"  Prompt type: {'Markdown format' if '## Quick Answer' in STRUCTURED_FORMAT_PROMPT else 'JSON format'}")
    print(f"  Returns JSON: {'Return only valid JSON' in STRUCTURED_FORMAT_PROMPT}")
    print(f"  Returns Markdown: {'## Quick Answer' in STRUCTURED_FORMAT_PROMPT}")
    # This is a critical check
    if 'Return only valid JSON' in STRUCTURED_FORMAT_PROMPT:
        print("  [ROOT CAUSE] STRUCTURED_FORMAT_PROMPT still expects JSON output!")
        print("               But render_verified_answer expects ## Markdown sections")
except Exception as e:
    print(f"  PROMPT CHECK ERROR: {e}")

print()
print("=" * 70)
print("STEP 7: structure_answer() output check")
print("=" * 70)
try:
    from app.core.orchestrator import (
        execute_tools, generate_reasoning_draft, structure_answer,
        plan_query
    )
    planner = plan_query(QUERY, bypass_cache=True)
    tool_state = execute_tools(QUERY, planner, blocked_chunk_ids=[])
    draft, from_llm = generate_reasoning_draft(QUERY, "test", tool_state, bypass_cache=True)
    print(f"  Draft from LLM: {from_llm}")
    print(f"  Draft preview : {str(draft)[:200]!r}")
    structured = structure_answer(QUERY, draft, tool_state)
    print(f"  Structured keys: {list(structured.keys()) if structured else 'None'}")
    if structured:
        print(f"  status        : {structured.get('status')}")
        print(f"  final_decision: {structured.get('final_decision')}")
        print(f"  analysis      : {str(structured.get('analysis',''))[:150]!r}")
        if "## Quick Answer" in str(structured.get("analysis", "")):
            print("  [INFO] LLM already returned structured markdown in analysis field")
        else:
            print("  [ROOT CAUSE CANDIDATE] analysis is compact text, render_verified_answer")
            print("                          will build 8-section format FROM this — check render output")
except Exception as e:
    print(f"  STRUCTURE ANSWER ERROR: {e}")
    traceback.print_exc()

print()
print("=" * 70)
print("STEP 8: render_verified_answer output check")
print("=" * 70)
try:
    from app.core.orchestrator import render_verified_answer
    rendered = render_verified_answer(structured)
    print(f"  Rendered length: {len(rendered)} chars")
    print(f"  Has '## Quick Answer': {'## Quick Answer' in rendered}")
    print(f"  Has '## TL;DR'       : {'## TL;DR' in rendered}")
    print(f"  Has FINAL DECISION   : {'FINAL DECISION' in rendered}")
    print()
    print("  === RENDERED OUTPUT (first 800 chars) ===")
    print(rendered[:800])
except Exception as e:
    print(f"  RENDER ERROR: {e}")
    traceback.print_exc()

print()
print("=" * 70)
print("STEP 9: post_process_structured_output check")
print("=" * 70)
try:
    from app.utils.output_validator import post_process_structured_output, validate_structured_output
    from app.utils.processors import extract_amount_lakhs
    amount_lakhs = extract_amount_lakhs(QUERY)
    processed = post_process_structured_output(rendered, query=QUERY, amount_lakhs=amount_lakhs)
    report = validate_structured_output(processed, amount_lakhs=amount_lakhs)
    print(f"  Validation passed : {report.is_valid}")
    print(f"  Validation errors : {report.errors}")
    print(f"  Processed length  : {len(processed)} chars")
except Exception as e:
    print(f"  POST-PROCESS ERROR: {e}")
    traceback.print_exc()

print()
print("=" * 70)
print("STEP 10: API endpoint check (what frontend actually gets)")
print("=" * 70)
try:
    from app.api.routes import router
    print("  API router loaded OK")
    # Check what field the API returns as the answer
    import inspect
    src = inspect.getsource(router.__class__)
    print("  (Check routes.py manually for response field name)")
except Exception as e:
    pass

try:
    from app.api import routes
    import inspect
    src = inspect.getsource(routes)
    # Look for the answer field
    import re
    fields = re.findall(r'"(answer|generation|response|text|result)"', src)
    print(f"  Response fields in API: {set(fields)}")
except Exception as e:
    print(f"  API SOURCE CHECK ERROR: {e}")

print()
print("=" * 70)
print("DIAGNOSTIC COMPLETE")
print("=" * 70)
