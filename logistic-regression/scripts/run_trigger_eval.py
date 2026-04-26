#!/usr/bin/env python3
"""Evaluate whether the skill is triggered correctly for each query in trigger-eval.json."""

import argparse
import json
import os
import sys
from pathlib import Path

import anthropic


JUDGE_PROMPT = """\
You are evaluating whether a user query should activate a specialized Claude skill.

The skill's trigger criteria (from its description field):
---
{description}
---

User query: "{query}"

Based only on the trigger criteria above, should this skill be activated for this query?
Answer with exactly one word: yes or no."""


def load_skill_description(skill_path: Path) -> str:
    skill_md = (skill_path / "SKILL.md").read_text()
    # Extract the description value from the YAML frontmatter
    in_front = False
    desc_lines = []
    capturing = False
    for line in skill_md.splitlines():
        if line.strip() == "---":
            if not in_front:
                in_front = True
                continue
            else:
                break
        if in_front:
            if line.startswith("description:"):
                rest = line[len("description:"):].strip()
                if rest and rest != ">":
                    desc_lines.append(rest)
                capturing = True
            elif capturing and (line.startswith(" ") or line.startswith("\t")):
                desc_lines.append(line.strip())
            elif capturing:
                break
    return " ".join(desc_lines)


def judge_query(client: anthropic.Anthropic, model: str, description: str, query: str) -> bool:
    resp = client.messages.create(
        model=model,
        max_tokens=5,
        messages=[{"role": "user", "content": JUDGE_PROMPT.format(
            description=description, query=query
        )}],
    )
    answer = resp.content[0].text.strip().lower()
    return answer.startswith("yes")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run trigger evals for a Claude skill.")
    parser.add_argument("--eval-set", required=True, help="Path to trigger-eval.json")
    parser.add_argument("--skill-path", required=True, help="Path to skill directory")
    parser.add_argument("--model", default="claude-haiku-4-5-20251001", help="Claude model to use")
    parser.add_argument("--results-dir", required=True, help="Directory to write results.json")
    args = parser.parse_args()

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("ERROR: ANTHROPIC_API_KEY is not set", file=sys.stderr)
        sys.exit(1)

    skill_path = Path(args.skill_path)
    description = load_skill_description(skill_path)
    if not description:
        print("ERROR: could not extract description from SKILL.md", file=sys.stderr)
        sys.exit(1)

    evals = json.loads(Path(args.eval_set).read_text())
    client = anthropic.Anthropic(api_key=api_key)

    passed = 0
    details = []
    for item in evals:
        query = item["query"]
        should_trigger = item["should_trigger"]
        predicted = judge_query(client, args.model, description, query)
        correct = predicted == should_trigger
        if correct:
            passed += 1
        status = "PASS" if correct else "FAIL"
        print(f"[{status}] should_trigger={should_trigger} predicted={predicted}  {query!r}")
        details.append({
            "query": query,
            "should_trigger": should_trigger,
            "predicted": predicted,
            "correct": correct,
        })

    total = len(evals)
    result = {
        "best_score": f"{passed}/{total}",
        "passed": passed,
        "total": total,
        "pass_rate": passed / total if total else 0,
        "details": details,
    }

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "results.json").write_text(json.dumps(result, indent=2))
    print(f"\nResult: {passed}/{total} ({result['pass_rate']:.0%})")


if __name__ == "__main__":
    main()
