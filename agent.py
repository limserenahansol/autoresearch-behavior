"""
agent.py  --  Autonomous Research Agent Loop
==============================================
Implements the karpathy/autoresearch pattern:
  1. Read program.md (human instructions)
  2. Read pipeline.py (current experiment code)
  3. Ask LLM: "propose ONE change to improve per_mouse_acc"
  4. LLM returns new pipeline.py code
  5. Run it, measure metric
  6. If improved -> keep.  If worse -> revert.
  7. Log everything to output/experiment_log.csv
  8. Repeat.

Usage (requires API key):
  Windows PowerShell:
    $env:ANTHROPIC_API_KEY = "sk-ant-..."
    python agent.py --max-iter 50

  Or with OpenAI:
    $env:OPENAI_API_KEY = "sk-..."
    python agent.py --provider openai --max-iter 50

If you don't have an API key, use Cursor AI as the agent instead:
  - Open pipeline.py in Cursor
  - Ask Cursor: "Read program.md. Propose one change to pipeline.py to improve
    per_mouse_acc. Then run: python run_experiments.py --note 'your change'"
  - Cursor acts as the LLM in the loop.
"""
import os
import re
import sys
import csv
import time
import shutil
import argparse
import subprocess
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).parent
PIPELINE_FILE = ROOT / "pipeline.py"
PROGRAM_FILE = ROOT / "program.md"
OUTPUT_DIR = ROOT / "output"
LOG_FILE = OUTPUT_DIR / "experiment_log.csv"
BACKUP_DIR = OUTPUT_DIR / "snapshots"

LOG_HEADER = [
    "exp_id", "timestamp", "per_mouse_acc", "accuracy", "f1_macro",
    "duration_s", "note",
]

MAX_RUNTIME_SEC = 120


def parse_args():
    p = argparse.ArgumentParser(description="Autoresearch agent loop")
    p.add_argument("--max-iter", type=int, default=50)
    p.add_argument("--provider", choices=["anthropic", "openai"], default="anthropic")
    p.add_argument("--model", type=str, default=None)
    return p.parse_args()


def read_file(path):
    return path.read_text(encoding="utf-8")


def write_file(path, content):
    path.write_text(content, encoding="utf-8")


def setup_dirs():
    OUTPUT_DIR.mkdir(exist_ok=True)
    BACKUP_DIR.mkdir(exist_ok=True)


def get_next_id():
    if not LOG_FILE.exists():
        return 0
    with open(LOG_FILE) as f:
        reader = csv.DictReader(f)
        ids = [int(row['exp_id']) for row in reader if row['exp_id'].isdigit()]
    return max(ids) + 1 if ids else 0


def append_log(exp_id, metrics, elapsed, note):
    new_file = not LOG_FILE.exists()
    with open(LOG_FILE, "a", newline="") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(LOG_HEADER)
        w.writerow([
            exp_id,
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            f"{metrics.get('per_mouse_acc', 0):.6f}",
            f"{metrics.get('accuracy', 0):.6f}",
            f"{metrics.get('f1_macro', 0):.6f}",
            f"{elapsed:.1f}",
            note[:200],
        ])


def run_pipeline():
    start = time.time()
    result = subprocess.run(
        [sys.executable, str(PIPELINE_FILE)],
        capture_output=True, text=True, timeout=MAX_RUNTIME_SEC,
        cwd=str(ROOT),
    )
    elapsed = time.time() - start
    output = result.stdout + "\n" + result.stderr

    if result.returncode != 0:
        return None, elapsed, output[-500:]

    metrics = {}
    for line in output.split("\n"):
        m = re.match(r"METRIC\s+(\w+)=([\d.eE+-]+)", line)
        if m:
            metrics[m.group(1)] = float(m.group(2))

    if 'per_mouse_acc' not in metrics:
        return None, elapsed, f"No METRIC found in output:\n{output[-300:]}"

    return metrics, elapsed, output


def extract_code(text):
    for pat in [r"```python\n(.*?)```", r"```\n(.*?)```"]:
        match = re.search(pat, text, re.DOTALL)
        if match:
            return match.group(1).strip()
    return None


def call_llm(system, user, provider, model):
    if provider == "anthropic":
        from anthropic import Anthropic
        r = Anthropic().messages.create(
            model=model or "claude-sonnet-4-20250514", max_tokens=8192,
            system=system, messages=[{"role": "user", "content": user}],
        )
        return r.content[0].text
    else:
        from openai import OpenAI
        r = OpenAI().chat.completions.create(
            model=model or "gpt-4o", max_tokens=8192,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}],
        )
        return r.choices[0].message.content


def main():
    args = parse_args()
    setup_dirs()

    program = read_file(PROGRAM_FILE) if PROGRAM_FILE.exists() else "Maximize per_mouse_acc."

    print("=" * 60)
    print("AUTORESEARCH AGENT")
    print(f"Provider: {args.provider} | Max iterations: {args.max_iter}")
    print("=" * 60)

    # Baseline
    exp_id = get_next_id()
    print(f"\n[Exp #{exp_id}] Running baseline...")
    metrics, elapsed, output = run_pipeline()
    if not metrics:
        print(f"Baseline FAILED: {output[:300]}")
        return

    best = metrics['per_mouse_acc']
    append_log(exp_id, metrics, elapsed, "baseline")
    shutil.copy2(PIPELINE_FILE, BACKUP_DIR / "pipeline_best.py")
    print(f"  per_mouse_acc = {best:.6f} ({elapsed:.1f}s)")

    for i in range(1, args.max_iter + 1):
        exp_id = get_next_id()
        print(f"\n[Exp #{exp_id}] Asking LLM... (best so far: {best:.6f})")

        sys_prompt = (
            "You are an ML research agent. Propose ONE modification to pipeline.py "
            "to improve per_mouse_acc. Return:\n"
            "Rationale: <1 sentence>\n\n```python\n<complete pipeline.py>\n```"
        )
        usr_prompt = (
            f"## Instructions\n{program}\n\n"
            f"## Current pipeline.py\n```python\n{read_file(PIPELINE_FILE)}\n```\n\n"
            f"## Best per_mouse_acc = {best:.6f}\n"
            "Propose ONE change."
        )

        try:
            response = call_llm(sys_prompt, usr_prompt, args.provider, args.model)
        except Exception as e:
            print(f"  LLM error: {e}")
            append_log(exp_id, {}, 0, f"LLM_ERROR: {str(e)[:100]}")
            time.sleep(5)
            continue

        new_code = extract_code(response)
        if not new_code or "def run()" not in new_code:
            print("  Invalid code from LLM, skipping.")
            append_log(exp_id, {}, 0, "INVALID_CODE")
            continue

        rationale = ""
        for line in response.split("\n"):
            if line.strip().lower().startswith("rationale:"):
                rationale = line.strip()[len("rationale:"):].strip()
                break

        # Save backup, write new code
        backup = BACKUP_DIR / f"pipeline_exp{exp_id:03d}.py"
        shutil.copy2(PIPELINE_FILE, backup)
        write_file(PIPELINE_FILE, new_code)

        print(f"  Rationale: {rationale[:80]}")
        print(f"  Running...")
        metrics, elapsed, output = run_pipeline()

        if not metrics:
            print(f"  FAILED ({elapsed:.1f}s)")
            append_log(exp_id, {}, elapsed, f"FAILED: {rationale[:100]}")
            shutil.copy2(BACKUP_DIR / "pipeline_best.py", PIPELINE_FILE)
            continue

        new_val = metrics['per_mouse_acc']
        improved = new_val > best

        if improved:
            print(f"  IMPROVED: {best:.6f} -> {new_val:.6f} (+{new_val - best:.6f})")
            best = new_val
            shutil.copy2(PIPELINE_FILE, BACKUP_DIR / "pipeline_best.py")
            append_log(exp_id, metrics, elapsed, f"IMPROVED: {rationale[:150]}")
        else:
            print(f"  No improvement: {new_val:.6f} <= {best:.6f}")
            shutil.copy2(BACKUP_DIR / "pipeline_best.py", PIPELINE_FILE)
            append_log(exp_id, metrics, elapsed, f"REVERTED: {rationale[:150]}")

        time.sleep(2)

    print(f"\n{'='*60}")
    print(f"DONE. Best per_mouse_acc = {best:.6f}")
    print(f"Log: {LOG_FILE}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
