#!/usr/bin/env python3
"""Create a minimal experiment folder scaffold."""

from __future__ import annotations

import argparse
from pathlib import Path


TEMPLATE = """# {exp_id} {title}

- 日期：
- 仓库：
- 数据集：
- 阶段：
- 目标：

## Baseline

- 使用的基线配置：

## 本次唯一变量

- 

## 运行命令

```bash
# command here
```

## 结果

- train loss：
- val loss：
- sample 摘要：

## 观察

- 

## 结论

- 

## 下一步

- 
"""


def main() -> int:
    parser = argparse.ArgumentParser(description="Create a new experiment scaffold.")
    parser.add_argument("exp_id", help="Experiment id, e.g. EXP-001")
    parser.add_argument("title", help="Short experiment title")
    parser.add_argument(
        "--root",
        default="experiments",
        help="Root folder for experiment records (default: experiments)",
    )
    args = parser.parse_args()

    root = Path(args.root)
    exp_dir = root / args.exp_id
    exp_dir.mkdir(parents=True, exist_ok=True)

    for name in ("run.log", "samples.txt"):
        (exp_dir / name).touch(exist_ok=True)

    notes = exp_dir / "notes.md"
    if not notes.exists():
        notes.write_text(TEMPLATE.format(exp_id=args.exp_id, title=args.title), encoding="utf-8")

    print(f"Created experiment scaffold at: {exp_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
