#!/usr/bin/env python3
"""
Generate official RULER tasks via NVIDIA scripts and convert to run_clm JSONL.

Outputs:
  <out_root>/official_L<LEN>/<task>/validation.jsonl     (raw official per-task)
  <out_root>/ruler_official_eval_L<LEN>.jsonl            (merged run_clm format)
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Iterable, List


TASKS_13 = [
    "niah_single_1",
    "niah_single_2",
    "niah_single_3",
    "niah_multikey_1",
    "niah_multikey_2",
    "niah_multikey_3",
    "niah_multivalue",
    "niah_multiquery",
    "vt",
    "cwe",
    "fwe",
    "qa_1",
    "qa_2",
]


def parse_csv_ints(text: str) -> List[int]:
    out: List[int] = []
    for chunk in text.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        out.append(int(chunk))
    if not out:
        raise ValueError("No valid lengths parsed.")
    return out


def count_jsonl(path: Path) -> int:
    n = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                n += 1
    return n


def run_prepare(
    scripts_data_dir: Path,
    save_dir: Path,
    task: str,
    tokenizer_path: Path,
    max_seq_length: int,
    num_samples: int,
    seed: int,
    timeout_s: int,
) -> bool:
    save_file = save_dir / task / "validation.jsonl"
    if save_file.exists() and count_jsonl(save_file) == num_samples:
        print(f"[skip] len={max_seq_length} task={task} already has {num_samples} samples")
        return True

    cmd = [
        "python",
        "prepare.py",
        "--save_dir",
        str(save_dir),
        "--benchmark",
        "synthetic",
        "--task",
        task,
        "--subset",
        "validation",
        "--tokenizer_path",
        str(tokenizer_path),
        "--tokenizer_type",
        "hf",
        "--max_seq_length",
        str(max_seq_length),
        "--model_template_type",
        "base",
        "--num_samples",
        str(num_samples),
        "--random_seed",
        str(seed),
    ]
    print(f"[run ] len={max_seq_length} task={task}")
    p = subprocess.run(
        cmd,
        cwd=str(scripts_data_dir),
        text=True,
        capture_output=True,
        timeout=timeout_s,
    )
    if p.returncode != 0:
        print(
            f"prepare failed for len={max_seq_length} task={task}\n"
            f"stdout:\n{p.stdout}\n\nstderr:\n{p.stderr}"
        )
        return False
    if not save_file.exists():
        print(
            f"prepare produced no file for len={max_seq_length} task={task}\n"
            f"stdout:\n{p.stdout}\n\nstderr:\n{p.stderr}"
        )
        return False
    got = count_jsonl(save_file)
    if got != num_samples:
        print(
            f"prepare produced unexpected lines for len={max_seq_length} task={task}: "
            f"expected={num_samples}, got={got}\nstdout:\n{p.stdout}\n\nstderr:\n{p.stderr}"
        )
        return False
    print(f"[ ok ] len={max_seq_length} task={task} lines={got}")
    return True


def _to_outputs(value) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        out = []
        for x in value:
            s = str(x).strip()
            if s:
                out.append(s)
        return out
    s = str(value).strip()
    return [s] if s else []


def iter_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def convert_length(out_root: Path, max_seq_length: int, tasks: List[str]) -> Path:
    merged_path = out_root / f"ruler_official_eval_L{max_seq_length}.jsonl"
    merged_path.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    with merged_path.open("w", encoding="utf-8") as wf:
        for task in tasks:
            src = out_root / f"official_L{max_seq_length}" / task / "validation.jsonl"
            if not src.exists():
                print(f"[warn] missing task file, skip merge: {src}")
                continue
            for row in iter_jsonl(src):
                prompt = str(row.get("input", "")).strip()
                answer_prefix = str(row.get("answer_prefix", ""))
                # Official RULER generators strip answer_prefix from input. We add it back.
                merged_prompt = (prompt + answer_prefix).strip()
                outputs = _to_outputs(row.get("outputs"))
                if (not merged_prompt) or (not outputs):
                    continue
                out_row = {
                    "input": merged_prompt,
                    "outputs": outputs,
                    "length": int(max_seq_length),
                    "ruler_config": str(task),
                    "source_length": int(row.get("length", max_seq_length)),
                }
                wf.write(json.dumps(out_row, ensure_ascii=True) + "\n")
                total += 1
    print(f"[merge] L{max_seq_length}: wrote {total} rows -> {merged_path}")
    return merged_path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--ruler_repo",
        type=Path,
        default=Path(
            "/project/nlp-work5/hongyu-s/transformers/examples/pytorch/language-modeling/"
            "hotpot_long/data/ruler_official/RULER_repo"
        ),
    )
    ap.add_argument(
        "--tokenizer_path",
        type=Path,
        default=Path(
            "/project/nlp-work5/hongyu-s/transformers/examples/pytorch/language-modeling/"
            "runs/ruler_ft_matrix_20260525/path_pa_s42_save1k"
        ),
    )
    ap.add_argument(
        "--out_root",
        type=Path,
        default=Path(
            "/project/nlp-work5/hongyu-s/transformers/examples/pytorch/language-modeling/"
            "hotpot_long/data/ruler_official"
        ),
    )
    ap.add_argument("--lengths", type=str, default="2048,4096")
    ap.add_argument("--num_samples", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--timeout_s", type=int, default=1800)
    args = ap.parse_args()

    lengths = parse_csv_ints(args.lengths)
    scripts_data_dir = args.ruler_repo / "scripts" / "data"
    if not scripts_data_dir.exists():
        raise FileNotFoundError(f"RULER scripts dir not found: {scripts_data_dir}")

    for max_len in lengths:
        length_save_dir = args.out_root / f"official_L{max_len}"
        ok_tasks: List[str] = []
        for task in TASKS_13:
            ok = run_prepare(
                scripts_data_dir=scripts_data_dir,
                save_dir=length_save_dir,
                task=task,
                tokenizer_path=args.tokenizer_path,
                max_seq_length=max_len,
                num_samples=args.num_samples,
                seed=args.seed,
                timeout_s=args.timeout_s,
            )
            if ok:
                ok_tasks.append(task)
        convert_length(args.out_root, max_len, ok_tasks)

    print("Done.")


if __name__ == "__main__":
    main()
