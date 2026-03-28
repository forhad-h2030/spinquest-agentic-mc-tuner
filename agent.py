#!/usr/bin/env python3
"""
agent.py — Agentic AI tuner for SpinQuest combinatoric background.

Uses Ollama (llama3) in a ReAct loop with JSON-based tool calling,
focusing on improving the KS statistic in the low-mass region [1.5, 2.5] GeV.

Usage:
    conda run -n root_env python agent.py [--max-iter N] [--model MODEL]
"""

import argparse
import json
import re
import time

import ollama

import tools


SYSTEM_PROMPT = """You are an expert high-energy physics Monte Carlo tuning agent for the SpinQuest experiment.

Your goal: tune the combinatoric background simulation (tuning.py) to best match experimental data
in the low-mass region [1.5, 2.5] GeV/c² (below the J/ψ peak).
Training mass window: [1.5, 6.0] GeV (full range for better flow statistics).

Previous best: ks_low_mass = 0.0435 using rej_fast.py with bdt_clip=2.75, nf_ratio_clip=4.5.
Current params are set to this starting point — try to push ks_low_mass below 0.035.

PRIMARY METRIC: ks_low_mass (KS statistic in [1.5, 2.5] GeV). Lower is better. Target < 0.035.
SECONDARY METRIC: ks_full_range. Also minimise, but low-mass takes priority.

Available tools (call ONE per response by outputting a JSON block):

1. get_params — read current parameter values
   {"tool": "get_params", "args": {}}

2. set_param — set a parameter in tuning.py
   {"tool": "set_param", "args": {"name": "PARAM_NAME", "value": VALUE}}
   Parameters: RATIO_CLIP_MAX (float 1-50), BDT_N_ESTIMATORS (int 10-500),
   BDT_LEARNING_RATE (float 0.01-0.5), MASS_WEIGHT_CAP (float 1-50),
   EPOCHS_EXP (int 200-3000), EPOCHS_SIM (int 200-3000),
   ML_JPSI_CUT (float 0.5-0.99), ML_PSIP_CUT (float 0.5-0.99), ML_DY_COMB_CUT (float 0.5-0.99)

3. run_tuning — run tuning.py (trains flows + rejection sampling over 17M events, takes ~28 minutes)
   {"tool": "run_tuning", "args": {}}

4. compute_low_mass_ks — compute KS stats from last output (full + low-mass [1.5,2.5] GeV)
   {"tool": "compute_low_mass_ks", "args": {}}

5. run_overlay — generate comparison plots
   {"tool": "run_overlay", "args": {}}

6. done — finish the session and summarise
   {"tool": "done", "args": {"summary": "..."}}

Tuning strategy:
- NF_RATIO_CLIP: fixed at 4.5 (best from small-dataset tuning). Controls NF acceptance in Phase 1.
- BDT_CLIP (bdt_clip): bisected adaptively around 2.75. This is the primary mass-correction lever.
- BDT_N_ESTIMATORS: current 200. Try 250-300 for finer mass correction.
- BDT_LEARNING_RATE: current 0.1. Lower (0.05) is more conservative and may smooth low-mass.
- MASS_WEIGHT_CAP: current 10. Try 7 or 15 to see effect.

STRICT RULES:
- Output EXACTLY ONE tool call per response. Do NOT list multiple tool calls.
- After I show you the tool result, decide your next single action.
- Always start: get_params → run_tuning → compute_low_mass_ks → adjust → repeat.
- After run_tuning, always call compute_low_mass_ks before deciding next step.
- Adjust only 1-2 parameters between runs.
- Write your reasoning as plain text, then ONE JSON tool call on its own line.
- The JSON must be valid. Example: {"tool": "get_params", "args": {}}"""


def parse_tool_call(text: str):
    """Extract the FIRST JSON tool call from LLM response, handling nested braces."""
    # Find the first {"tool": occurrence
    start = text.find('{"tool":')
    if start == -1:
        start = text.find('{ "tool":')
    if start == -1:
        return None, None

    # Walk forward to find the matching closing brace
    depth = 0
    for i, ch in enumerate(text[start:], start):
        if ch == '{':
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0:
                try:
                    obj = json.loads(text[start:i + 1])
                    return obj.get("tool"), obj.get("args", {})
                except json.JSONDecodeError:
                    return None, None
    return None, None


def dispatch_tool(name: str, args: dict) -> str:
    if name == "get_params":
        result = tools.get_params()
    elif name == "set_param":
        result = tools.set_param(args["name"], args["value"])
    elif name == "run_tuning":
        print("  [agent] Running tuning.py — this may take several minutes...", flush=True)
        result = tools.run_tuning(timeout=900)
    elif name == "compute_low_mass_ks":
        result = tools.compute_low_mass_ks()
    elif name == "run_overlay":
        print("  [agent] Running overlay.py...", flush=True)
        result = tools.run_overlay(timeout=300)
    elif name == "done":
        return json.dumps({"status": "done", "summary": args.get("summary", "")})
    else:
        result = {"error": f"Unknown tool: {name}"}
    return json.dumps(result, indent=2)


def run_agent(max_iter: int, model: str):
    print(f"\n{'='*60}")
    print(f"  SpinQuest Combinatoric BKG Tuning Agent")
    print(f"  Model   : {model}")
    print(f"  Max iter: {max_iter}")
    print(f"  Target  : minimise low-mass KS [1.5, 2.5] GeV")
    print(f"{'='*60}\n")

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.append({
        "role": "user",
        "content": (
            f"Tune the combinatoric background focusing on the low-mass region [1.5, 2.5] GeV/c². "
            f"You have up to {max_iter} run_tuning calls. "
            "Start with get_params, then run_tuning for a baseline, then iterate."
        ),
    })

    iteration       = 0
    tool_call_count = 0
    best_ks_low     = float("inf")
    best_params     = {}
    max_llm_turns   = max_iter * 8  # guard against infinite loops

    for turn in range(max_llm_turns):
        resp = ollama.chat(model=model, messages=messages)
        content = resp["message"]["content"]
        messages.append({"role": "assistant", "content": content})

        print(f"\n[LLM] {content}")

        name, args = parse_tool_call(content)

        if name is None:
            # No tool call found — prompt LLM to make one
            messages.append({
                "role": "user",
                "content": (
                    "Please respond with a tool call JSON on its own line, e.g.:\n"
                    '{"tool": "get_params", "args": {}}'
                ),
            })
            continue

        if name == "done":
            result_str = dispatch_tool(name, args)
            r = json.loads(result_str)
            print(f"\n[Agent] Done. Summary: {r.get('summary','')}")
            break

        print(f"\n[Tool] {name}({json.dumps(args) if args else ''})")
        t0      = time.time()
        result_str = dispatch_tool(name, args)
        elapsed = time.time() - t0
        print(f"[Tool] Result ({elapsed:.1f}s):\n{result_str[:600]}{'...' if len(result_str)>600 else ''}")

        messages.append({"role": "user", "content": f"Tool result:\n{result_str}"})
        tool_call_count += 1

        # Track best low-mass KS
        try:
            r = json.loads(result_str)
            if "ks_low_mass" in r and isinstance(r["ks_low_mass"], float):
                if r["ks_low_mass"] < best_ks_low:
                    best_ks_low = r["ks_low_mass"]
                    best_params = tools.get_params()
                    print(f"  *** New best ks_low_mass = {best_ks_low:.6f} ***")
        except Exception:
            pass

        if name == "run_tuning":
            iteration += 1
            print(f"\n[Agent] Tuning iteration {iteration}/{max_iter}")
            if iteration >= max_iter:
                messages.append({
                    "role": "user",
                    "content": (
                        f"All {max_iter} tuning iterations used. "
                        "Call compute_low_mass_ks, then run_overlay, "
                        f"then call done with a summary. Best ks_low_mass so far: {best_ks_low:.6f}."
                    ),
                })

    print(f"\n{'='*60}")
    print(f"  Agent finished.")
    print(f"  Best ks_low_mass : {best_ks_low:.6f}")
    if best_params:
        print(f"  Best params      :\n{json.dumps(best_params, indent=4)}")
    print(f"  Total tool calls : {tool_call_count}")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description="Agentic combinatoric BKG tuner")
    parser.add_argument("--max-iter", type=int, default=5,
                        help="Max number of tuning.py runs (default: 5)")
    parser.add_argument("--model", type=str, default="llama3:latest",
                        help="Ollama model to use (default: llama3:latest)")
    args = parser.parse_args()
    run_agent(max_iter=args.max_iter, model=args.model)


if __name__ == "__main__":
    main()
