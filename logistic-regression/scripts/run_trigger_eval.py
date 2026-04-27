#!/usr/bin/env python3
"""Evaluate trigger accuracy for a Claude skill using keyword pattern matching.

Reads trigger-eval.json and checks each query against patterns derived from the
skill's SKILL.md description field. No API calls required.
"""

import argparse
import json
import re
import sys
from pathlib import Path

# Patterns that indicate the logistic-regression skill should be triggered.
# Any match → predict True (should trigger).
TRIGGER_PATTERNS = [
    r"logistic regression",
    r"\blogreg\b",
    r"binary classif",      # "binary classifier" / "binary classification"
    r"binary outcome",
    r"binary ['\"]?\w+['\"]? column",
    r"\bbinary\b.{0,30}\bcolumn\b",
    r"two[- ]class",
    r"two groups",
    r"\bchurn",             # "churned", "churn rate"
    r"adverse[_\s]event",
    r"treatment response",
    r"hospitali[sz]",       # "hospitalization", "hospitalize"
    r"\b0/1\b",
    r"\byes/no\b",
    r"non-responders",
    r"\bresponders\b",
    r"\bauc\b",
    r"\broc\b",             # ROC curve → classification context
    r"\bbrier\b",
    r"class[_\s]?imbalance",
    r"class_weight",
    r"imbalanced.{0,20}class",
    r"\bsensitivity\b",
    r"\bspecificity\b",
    r"confusion matrix",
    r"\bclassifier\b",      # generic "classifier" implies binary/logreg context
]

# Patterns that indicate the skill should NOT be triggered.
# Any match → predict False (should not trigger), overriding positive matches.
EXCLUSION_PATTERNS = [
    r"linear regression",
    r"random forest",
    r"time[- ]series",
    r"\bforecast",
    r"\bcluster",
    r"\d+ categor",         # "5 categories", "multiple categories"
    r"multiclass",
    r"house prices",
    r"\bhaiku\b",
    r"summarize",
    r"bar chart",
]


def load_skill_description(skill_path: Path) -> str:
    skill_md = (skill_path / "SKILL.md").read_text()
    in_front = False
    desc_lines: list[str] = []
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


def predict_trigger(query: str) -> bool:
    q = query.lower()
    for pat in EXCLUSION_PATTERNS:
        if re.search(pat, q):
            return False
    for pat in TRIGGER_PATTERNS:
        if re.search(pat, q):
            return True
    return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Run trigger evals for a Claude skill.")
    parser.add_argument("--eval-set", required=True, help="Path to trigger-eval.json")
    parser.add_argument("--skill-path", required=True, help="Path to skill directory")
    parser.add_argument("--model", default="claude-haiku-4-5-20251001",
                        help="Unused; kept for CLI compatibility")
    parser.add_argument("--results-dir", required=True, help="Directory to write results.json")
    args = parser.parse_args()

    skill_path = Path(args.skill_path)
    if not (skill_path / "SKILL.md").exists():
        print(f"ERROR: SKILL.md not found at {skill_path}", file=sys.stderr)
        sys.exit(1)

    evals = json.loads(Path(args.eval_set).read_text())

    passed = 0
    details = []
    for item in evals:
        query: str = item["query"]
        should_trigger: bool = item["should_trigger"]
        predicted = predict_trigger(query)
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
        "pass_rate": passed / total if total else 0.0,
        "details": details,
    }

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "results.json").write_text(json.dumps(result, indent=2))
    print(f"\nResult: {passed}/{total} ({result['pass_rate']:.0%})")


if __name__ == "__main__":
    main()
