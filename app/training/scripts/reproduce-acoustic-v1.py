"""Local reproduction of the Acoustic v1 pipeline: inventory -> validate-labels -> split -> train
[-> finetune] [-> evaluate/select] [-> export].

Each stage is the normal `python -m whalemix_training <command>` entry point. The script stops at the
first non-zero exit and writes `reproduce-ledger.jsonl` (stage, command, exit code, duration) in the
configured output directory. Training stays setup-only while `training.max_updates` is 0, which is
the setting for this build; nothing here trains weights unless the config asks for it.

  python scripts/reproduce-acoustic-v1.py --config configs/acoustic-v1.example.json --dry-run
  python scripts/reproduce-acoustic-v1.py --config run.json \\
      [--parent CKPT] [--eval-set EVAL.json --predictions PRED.json --select] [--export-checkpoint CKPT]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from whalemix_training.config import ConfigError, load_config, require_path


def plan(args: argparse.Namespace) -> list[tuple[str, list[str]]]:
    base = ["--config", str(args.config)]
    stages: list[tuple[str, list[str]]] = [
        ("inventory", ["inventory", *base]),
        ("validate-labels", ["validate-labels", *base]),
        ("split", ["split", *base]),
        ("train", ["train", *base]),
    ]
    if args.parent:
        stages.append(("finetune", ["finetune", *base, "--parent", str(args.parent)]))
    if args.eval_set and args.predictions:
        cmd = ["evaluate", *base, "--eval-set", str(args.eval_set), "--predictions", str(args.predictions)]
        stages.append(("evaluate", [*cmd, "--select"] if args.select else cmd))
    if args.export_checkpoint:
        stages.append(("export", ["export", *base, "--checkpoint", str(args.export_checkpoint)]))
    return stages


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--parent", type=Path, default=None)
    p.add_argument("--eval-set", type=Path, default=None)
    p.add_argument("--predictions", type=Path, default=None)
    p.add_argument("--select", action="store_true")
    p.add_argument("--export-checkpoint", type=Path, default=None)
    p.add_argument("--dry-run", action="store_true", help="print the stage plan and exit")
    args = p.parse_args(argv)
    stages = plan(args)
    if args.dry_run:
        print(json.dumps({"stages": [{"stage": s, "argv": a} for s, a in stages]}, indent=2))
        return 0
    try:
        out_dir = require_path(load_config(args.config), "output_dir")
    except ConfigError as exc:
        print(f"invalid configuration: {exc}", file=sys.stderr)
        return 2
    out_dir.mkdir(parents=True, exist_ok=True)
    ledger = out_dir / "reproduce-ledger.jsonl"
    for stage, stage_args in stages:
        t0 = time.monotonic()
        code = subprocess.call([sys.executable, "-m", "whalemix_training", *stage_args])
        seconds = round(time.monotonic() - t0, 3)
        entry = {"stage": stage, "argv": stage_args, "exit_code": code, "seconds": seconds}
        with ledger.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry) + "\n")
        if code != 0:
            print(f"stage {stage} exited {code}; stopping (see {ledger})", file=sys.stderr)
            return code
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
