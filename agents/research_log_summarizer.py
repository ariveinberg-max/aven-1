"""Research log summarizer -- a real local agent, not a demo.

Reads actual project state (latest TASKS.md handoff, live training logs
from both machines) and asks a local, free, open-source model (via Ollama,
no API cost, no Claude involved) to write a short plain-English summary.

This is the kind of task that genuinely benefits from a language model --
unlike most other "roles" on the org page, which are really just scripted
checks. Run manually or on a schedule (see the launchd plist in this
directory).
"""
import json
import re
import subprocess
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "llama3.2:3b"
WINDOWS_HOST = "arive@192.168.68.65"
WINDOWS_LOG = r"C:\Users\arive\aven-1\train_log.txt"


def latest_tasks_handoff():
    text = (ROOT / "research/TASKS.md").read_text(encoding="utf-8")
    sections = re.split(r"\n(?=## )", text)
    handoffs = [s for s in sections if "handoff" in s.lower() or "Handoff" in s]
    return handoffs[-1].strip() if handoffs else "(no handoff found)"


def mac_checkpoint_status():
    try:
        import torch
        d = torch.load(ROOT / "checkpoints/latest.pt", map_location="cpu", weights_only=False)
        return f"step {d.get('step')}, stage {d.get('stage')}"
    except Exception as e:
        return f"(could not read Mac checkpoint: {e})"


def windows_log_tail(n=15):
    try:
        result = subprocess.run(
            ["ssh", "-o", "ConnectTimeout=8", WINDOWS_HOST,
             f'powershell -Command "Get-Content -Path \'{WINDOWS_LOG}\' -Tail {n}"'],
            capture_output=True, text=True, timeout=20,
        )
        return result.stdout.strip() or "(empty)"
    except Exception as e:
        return f"(could not reach Windows PC: {e})"


def ask_local_model(prompt):
    req = urllib.request.Request(
        OLLAMA_URL,
        data=json.dumps({"model": MODEL, "prompt": prompt, "stream": False}).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read())["response"]


def main():
    handoff = latest_tasks_handoff()
    mac_status = mac_checkpoint_status()
    windows_tail = windows_log_tail()

    prompt = f"""You are a research assistant summarizing the status of an AI training project for a busy founder who doesn't have time to read the raw logs. Be concise (under 150 words), plain English, no jargon unless necessary. State facts only from what's given below -- don't guess or add anything not present.

Latest project handoff note:
{handoff[:2000]}

Mac model (Aven-1) checkpoint: {mac_status}

Windows PC (Aven-2) training log, last 15 lines:
{windows_tail}

Write a short status summary covering: (1) what's currently happening on each machine, (2) anything that looks like a problem, (3) one sentence on what to check next."""

    summary = ask_local_model(prompt)

    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    out_path = ROOT / "research/summaries" / f"{timestamp.replace(':', '')}.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(f"# Research summary -- {timestamp}\n\n{summary}\n", encoding="utf-8")

    print(summary)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
