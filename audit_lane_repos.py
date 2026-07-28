#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
对比 LaneRobot 原版与修改版 Ultralytics 仓库，重点排查：

1. 每个 epoch 是否少跑了 batch / sample
2. 多 DataLoader 是否被 zip() / min(len(...)) 截断
3. steps_per_epoch 是否被任务数错误整除
4. optimizer.step()、loss.backward()、zero_grad() 是否改变
5. 新增 head 是否可能未进入 optimizer
6. backbone / head 是否被错误冻结
7. loss 是否被全部任务数错误平均
8. scheduler.step() 调用位置是否改变
9. 单类输出通道、CrossEntropyLoss / BCEWithLogitsLoss 是否不匹配
10. 预训练权重 load_state_dict(strict=False) 是否掩盖未加载问题

脚本只做静态分析，不修改仓库。

使用示例：
python3 audit_lane_repos.py \
  --modified /home/xhm/Desktop/ULTRALYTICS_LANE_ROBOT/ultralytics \
  --original /home/xhm/Desktop/lane_backup/ultralytics \
  --output /home/xhm/Desktop/lane_repo_audit.md

建议随后打开报告：
code /home/xhm/Desktop/lane_repo_audit.md
"""

from __future__ import annotations

import argparse
import difflib
import hashlib
import os
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional


TEXT_SUFFIXES = {
    ".py", ".yaml", ".yml", ".json", ".toml", ".ini", ".cfg",
    ".txt", ".md", ".sh"
}

IGNORE_DIRS = {
    ".git", ".idea", ".vscode", "__pycache__", ".pytest_cache",
    "runs", "wandb", "dist", "build", ".mypy_cache", ".ruff_cache",
    "node_modules"
}

# 训练链路重点文件名/路径关键字
FOCUS_PATH_KEYWORDS = (
    "train", "trainer", "engine", "task", "lane", "model", "head",
    "loss", "dataset", "dataloader", "loader", "optimizer", "scheduler",
    "build", "parse_model", "yaml", "cfg"
)


@dataclass
class Finding:
    severity: str
    category: str
    repo_name: str
    relpath: str
    line_no: int
    line: str
    reason: str

    def sort_key(self):
        rank = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}.get(self.severity, 9)
        return (rank, self.category, self.relpath, self.line_no)


@dataclass
class FileInfo:
    relpath: str
    path: Path
    sha256: str
    size: int
    lines: int


RULES = [
    # 训练步数/数据量风险
    (
        "HIGH",
        "epoch步数可能被截断",
        re.compile(r"\bzip\s*\([^)]*(loader|dataloader)", re.I),
        "使用 zip() 并行多个 DataLoader 时，会在最短的 DataLoader 结束后停止。",
    ),
    (
        "HIGH",
        "epoch步数可能被截断",
        re.compile(r"\bmin\s*\([^)]*len\s*\([^)]*(loader|dataloader)", re.I),
        "使用最短 DataLoader 长度作为 epoch 长度，可能只训练了部分数据。",
    ),
    (
        "HIGH",
        "epoch步数可能被任务数整除",
        re.compile(
            r"(len\s*\([^)]*(loader|dataloader)[^)]*\)|steps?_per_epoch|epoch_size)"
            r"\s*//\s*(num_tasks?|n_tasks?|len\s*\([^)]*tasks?[^)]*\))",
            re.I,
        ),
        "epoch 步数被任务数量整除，可能导致 90 秒变成约 30 秒。",
    ),
    (
        "HIGH",
        "手工限制训练步数",
        re.compile(r"\b(max_steps?|limit_train_batches|steps?_per_epoch|epoch_size)\b\s*[:=]", re.I),
        "存在手工限制 step/epoch 长度的变量，需要核对修改前后的取值。",
    ),
    (
        "MEDIUM",
        "数据子集或采样",
        re.compile(r"\b(Subset|RandomSampler|WeightedRandomSampler|DistributedSampler)\b"),
        "使用子集或自定义采样器，需确认 num_samples、drop_last 和 world_size。",
    ),
    (
        "MEDIUM",
        "DataLoader丢弃尾批",
        re.compile(r"\bdrop_last\s*=\s*True\b", re.I),
        "drop_last=True 会丢弃不足一个 batch 的样本；通常不会造成三倍差异，但应核对。",
    ),

    # optimizer / backward
    (
        "HIGH",
        "优化器更新链路",
        re.compile(r"\boptimizer\s*\.\s*step\s*\(", re.I),
        "记录 optimizer.step() 位置，需确认每个 batch 是否真的执行。",
    ),
    (
        "HIGH",
        "反向传播链路",
        re.compile(r"\b(backward|scaler\s*\.\s*scale)\s*\(", re.I),
        "记录 backward/AMP 反向传播位置，需和原版逐项对比。",
    ),
    (
        "MEDIUM",
        "梯度清零链路",
        re.compile(r"\boptimizer\s*\.\s*zero_grad\s*\(", re.I),
        "记录 zero_grad() 位置，检查梯度累积逻辑是否改变。",
    ),
    (
        "HIGH",
        "参数冻结",
        re.compile(r"\brequires_grad\s*=\s*False\b|\bfreeze\b", re.I),
        "可能冻结了 backbone 或 lane head，导致可训练参数减少。",
    ),
    (
        "HIGH",
        "动态新增模块",
        re.compile(r"\b(ModuleDict|ModuleList|task_heads?|heads?)\b", re.I),
        "多任务 head 的创建时机需早于 optimizer；否则新增参数可能未进入 optimizer。",
    ),

    # loss / 类别
    (
        "HIGH",
        "loss按任务数平均",
        re.compile(
            r"(total_?loss|loss)\s*=.*\/\s*(len\s*\([^)]*tasks?[^)]*\)|num_tasks?|n_tasks?)",
            re.I,
        ),
        "loss 可能除以全部任务数量，而不是当前有效任务数量。",
    ),
    (
        "MEDIUM",
        "loss求平均",
        re.compile(r"\b(mean|average|avg)\s*\([^)]*loss|loss.*\.mean\s*\(", re.I),
        "检查 loss 是否重复求平均，或者不同任务权重被缩小。",
    ),
    (
        "HIGH",
        "分类损失定义",
        re.compile(r"\b(CrossEntropyLoss|BCEWithLogitsLoss|BCELoss|NLLLoss|FocalLoss)\b"),
        "单类任务必须核对输出通道和标签定义：CE 通常需要背景+目标两个通道，BCE 可用一个通道。",
    ),
    (
        "HIGH",
        "类别数量",
        re.compile(r"\b(num_classes|nc|cls_num|class_num|out_channels)\b\s*[:=]", re.I),
        "检查修改版是否把单类任务错误设置成一个 softmax 通道。",
    ),
    (
        "MEDIUM",
        "忽略标签",
        re.compile(r"\bignore_index\b\s*[:=]", re.I),
        "检查是否把大量 lane 标签变成 ignore_index。",
    ),

    # scheduler / checkpoint
    (
        "HIGH",
        "学习率调度",
        re.compile(r"\b(scheduler|lr_scheduler)\s*\.\s*step\s*\(", re.I),
        "确认 scheduler.step() 是按 batch 还是按 epoch 调用，是否和原版一致。",
    ),
    (
        "HIGH",
        "宽松加载权重",
        re.compile(r"load_state_dict\s*\([^)]*strict\s*=\s*False", re.I),
        "strict=False 可能掩盖 backbone/head 大量权重未加载。",
    ),
    (
        "MEDIUM",
        "权重键过滤",
        re.compile(r"(missing_keys|unexpected_keys|state_dict|checkpoint|pretrained)", re.I),
        "检查预训练权重键名是否因新增前缀或结构调整而不匹配。",
    ),

    # 多任务路由
    (
        "HIGH",
        "任务路由",
        re.compile(r"\b(task_id|task_idx|task_name|active_tasks?|current_task)\b", re.I),
        "检查数据任务编号、head 顺序、loss 顺序是否一致。",
    ),
    (
        "MEDIUM",
        "按任务过滤样本",
        re.compile(r"\bif\b.*\btask\b.*==|sample.*task|task.*sample", re.I),
        "可能只保留某个任务的部分样本，需要打印各任务样本数。",
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="静态对比 LaneRobot 原版与修改版 Ultralytics 仓库"
    )
    parser.add_argument(
        "--modified",
        type=Path,
        default=Path("/home/xhm/Desktop/ULTRALYTICS_LANE_ROBOT/ultralytics"),
        help="修改版 ultralytics 目录",
    )
    parser.add_argument(
        "--original",
        type=Path,
        default=Path("/home/xhm/Desktop/lane_backup/ultralytics"),
        help="原版 ultralytics 目录",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("./lane_repo_audit.md"),
        help="输出 Markdown 报告",
    )
    parser.add_argument(
        "--max-diff-files",
        type=int,
        default=35,
        help="报告中最多展示多少个重点文件 diff",
    )
    parser.add_argument(
        "--diff-context",
        type=int,
        default=3,
        help="统一 diff 上下文行数",
    )
    parser.add_argument(
        "--max-diff-lines-per-file",
        type=int,
        default=240,
        help="每个文件最多展示多少行 diff",
    )
    return parser.parse_args()


def should_ignore(path: Path) -> bool:
    return any(part in IGNORE_DIRS for part in path.parts)


def is_text_file(path: Path) -> bool:
    return path.suffix.lower() in TEXT_SUFFIXES


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def safe_read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        return f"# READ_ERROR: {exc}\n"


def collect_files(root: Path) -> dict[str, FileInfo]:
    result: dict[str, FileInfo] = {}

    for path in root.rglob("*"):
        if not path.is_file():
            continue

        rel = path.relative_to(root)

        if should_ignore(rel):
            continue
        if not is_text_file(path):
            continue

        text = safe_read_text(path)
        result[str(rel)] = FileInfo(
            relpath=str(rel),
            path=path,
            sha256=sha256_file(path),
            size=path.stat().st_size,
            lines=text.count("\n") + (1 if text else 0),
        )

    return result


def scan_repo(repo_name: str, root: Path, files: dict[str, FileInfo]) -> list[Finding]:
    findings: list[Finding] = []

    for relpath, info in files.items():
        text = safe_read_text(info.path)
        for line_no, line in enumerate(text.splitlines(), start=1):
            stripped = line.strip()

            if not stripped or stripped.startswith("#"):
                continue

            for severity, category, pattern, reason in RULES:
                if pattern.search(line):
                    findings.append(
                        Finding(
                            severity=severity,
                            category=category,
                            repo_name=repo_name,
                            relpath=relpath,
                            line_no=line_no,
                            line=stripped[:500],
                            reason=reason,
                        )
                    )

    return findings


def path_focus_score(relpath: str) -> int:
    lower = relpath.lower()
    score = sum(1 for key in FOCUS_PATH_KEYWORDS if key in lower)
    if lower.endswith((".yaml", ".yml")):
        score += 2
    if "trainer" in lower or "train" in Path(lower).name:
        score += 4
    if "loss" in lower:
        score += 3
    if "model" in lower or "head" in lower:
        score += 2
    return score


def changed_line_score(original_text: str, modified_text: str) -> tuple[int, int, int]:
    matcher = difflib.SequenceMatcher(
        a=original_text.splitlines(),
        b=modified_text.splitlines(),
        autojunk=False,
    )
    added = 0
    removed = 0
    replaced = 0

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "insert":
            added += j2 - j1
        elif tag == "delete":
            removed += i2 - i1
        elif tag == "replace":
            removed += i2 - i1
            added += j2 - j1
            replaced += max(i2 - i1, j2 - j1)

    return added, removed, replaced


def make_diff(
    relpath: str,
    original_path: Path,
    modified_path: Path,
    context: int,
    max_lines: int,
) -> str:
    original_text = safe_read_text(original_path).splitlines()
    modified_text = safe_read_text(modified_path).splitlines()

    diff = list(
        difflib.unified_diff(
            original_text,
            modified_text,
            fromfile=f"original/{relpath}",
            tofile=f"modified/{relpath}",
            n=context,
            lineterm="",
        )
    )

    if len(diff) > max_lines:
        diff = diff[:max_lines]
        diff.append(
            f"... DIFF TRUNCATED: 该文件 diff 超过 {max_lines} 行 ..."
        )

    return "\n".join(diff)


def find_counterparts(
    only_modified: Iterable[str],
    original_files: dict[str, FileInfo],
) -> dict[str, list[str]]:
    """
    修改版新增文件可能只是原文件被改名/移动。
    用文件名和相似路径给出候选对应关系。
    """
    by_name: dict[str, list[str]] = defaultdict(list)
    for rel in original_files:
        by_name[Path(rel).name].append(rel)

    counterparts: dict[str, list[str]] = {}
    for rel in only_modified:
        name = Path(rel).name
        candidates = by_name.get(name, [])

        if not candidates:
            stem = Path(rel).stem.lower()
            candidates = [
                old_rel
                for old_rel in original_files
                if Path(old_rel).stem.lower() == stem
            ]

        counterparts[rel] = sorted(
            candidates,
            key=lambda p: (
                -path_focus_score(p),
                abs(len(Path(p).parts) - len(Path(rel).parts)),
                p,
            ),
        )[:5]

    return counterparts


def markdown_escape(text: str) -> str:
    return text.replace("|", r"\|").replace("`", r"\`")


def build_report(
    modified_root: Path,
    original_root: Path,
    modified_files: dict[str, FileInfo],
    original_files: dict[str, FileInfo],
    modified_findings: list[Finding],
    original_findings: list[Finding],
    args: argparse.Namespace,
) -> str:
    modified_set = set(modified_files)
    original_set = set(original_files)

    common = sorted(modified_set & original_set)
    same = sorted(
        rel for rel in common
        if modified_files[rel].sha256 == original_files[rel].sha256
    )
    changed = sorted(
        rel for rel in common
        if modified_files[rel].sha256 != original_files[rel].sha256
    )
    only_modified = sorted(modified_set - original_set)
    only_original = sorted(original_set - modified_set)

    # 重点 diff 排序
    changed_ranked = []
    for rel in changed:
        old_text = safe_read_text(original_files[rel].path)
        new_text = safe_read_text(modified_files[rel].path)
        added, removed, replaced = changed_line_score(old_text, new_text)
        focus = path_focus_score(rel)
        total_delta = added + removed
        changed_ranked.append(
            (focus, total_delta, replaced, rel, added, removed)
        )

    changed_ranked.sort(key=lambda x: (-x[0], -x[1], -x[2], x[3]))

    # 修改版中“新增的高风险项”：同一位置原版未出现同类规则，优先展示
    original_index = {
        (f.category, f.relpath, f.line.strip())
        for f in original_findings
    }
    new_risk_findings = [
        f for f in modified_findings
        if (f.category, f.relpath, f.line.strip()) not in original_index
    ]
    new_risk_findings.sort(key=Finding.sort_key)

    severity_count = Counter(f.severity for f in new_risk_findings)
    category_count = Counter(f.category for f in new_risk_findings)

    counterpart_map = find_counterparts(only_modified, original_files)

    lines: list[str] = []
    lines.append("# LaneRobot 原版与修改版仓库静态审计报告")
    lines.append("")
    lines.append("## 1. 审计对象")
    lines.append("")
    lines.append(f"- 修改版：`{modified_root}`")
    lines.append(f"- 原版：`{original_root}`")
    lines.append("")
    lines.append("> 本报告只做静态代码对比，不会修改仓库，也不能替代运行时 batch、sample、梯度和参数更新检查。")
    lines.append("")

    lines.append("## 2. 总体差异")
    lines.append("")
    lines.append("| 指标 | 数量 |")
    lines.append("|---|---:|")
    lines.append(f"| 修改版文本文件 | {len(modified_files)} |")
    lines.append(f"| 原版文本文件 | {len(original_files)} |")
    lines.append(f"| 相同文件 | {len(same)} |")
    lines.append(f"| 同路径但内容变化 | {len(changed)} |")
    lines.append(f"| 仅修改版存在 | {len(only_modified)} |")
    lines.append(f"| 仅原版存在 | {len(only_original)} |")
    lines.append("")

    lines.append("## 3. 优先结论")
    lines.append("")
    lines.append("按你当前的现象，先核对以下四类问题：")
    lines.append("")
    lines.append("1. 修改版是否通过 `zip(loader...)`、`min(len(loader...))`、`steps_per_epoch // num_tasks` 只训练了约三分之一的数据。")
    lines.append("2. 新增任务 head 是否在创建 optimizer 之后才挂到模型上，导致 head 参数未进入 optimizer。")
    lines.append("3. `loss` 是否除以配置中的全部任务数，而不是当前有效任务数。")
    lines.append("4. `scheduler.step()` 是否从按 epoch 调用改成按 batch 调用，导致学习率过快衰减。")
    lines.append("")

    lines.append("### 修改版新增/变化的风险命中统计")
    lines.append("")
    lines.append("| 严重度 | 数量 |")
    lines.append("|---|---:|")
    for sev in ("HIGH", "MEDIUM", "LOW"):
        lines.append(f"| {sev} | {severity_count.get(sev, 0)} |")
    lines.append("")

    lines.append("| 风险类别 | 数量 |")
    lines.append("|---|---:|")
    for category, count in category_count.most_common():
        lines.append(f"| {markdown_escape(category)} | {count} |")
    if not category_count:
        lines.append("| 未发现 | 0 |")
    lines.append("")

    lines.append("## 4. 修改版新增/变化的高风险代码位置")
    lines.append("")
    if not new_risk_findings:
        lines.append("未发现规则层面的新增风险命中。仍需检查后面的重点文件 diff。")
        lines.append("")
    else:
        max_findings = 200
        for idx, finding in enumerate(new_risk_findings[:max_findings], start=1):
            lines.append(
                f"### {idx}. [{finding.severity}] {finding.category}"
            )
            lines.append("")
            lines.append(
                f"- 文件：`{finding.relpath}:{finding.line_no}`"
            )
            lines.append(f"- 原因：{finding.reason}")
            lines.append("")
            lines.append("```python")
            lines.append(finding.line)
            lines.append("```")
            lines.append("")

        if len(new_risk_findings) > max_findings:
            lines.append(
                f"> 其余 {len(new_risk_findings) - max_findings} 条风险命中未展开。"
            )
            lines.append("")

    lines.append("## 5. 重点变化文件")
    lines.append("")
    lines.append("| 排名 | 文件 | 关注度 | 新增行 | 删除行 |")
    lines.append("|---:|---|---:|---:|---:|")
    for idx, (focus, total_delta, replaced, rel, added, removed) in enumerate(
        changed_ranked[:100], start=1
    ):
        lines.append(
            f"| {idx} | `{rel}` | {focus} | {added} | {removed} |"
        )
    if not changed_ranked:
        lines.append("| - | 无同路径内容变化文件 | 0 | 0 | 0 |")
    lines.append("")

    lines.append("## 6. 重点文件统一 diff")
    lines.append("")
    diff_count = 0
    for focus, total_delta, replaced, rel, added, removed in changed_ranked:
        if diff_count >= args.max_diff_files:
            break

        # 优先展示训练相关文件；若没有关键词，仍展示少量高变化文件
        if focus == 0 and diff_count >= min(10, args.max_diff_files):
            continue

        diff_text = make_diff(
            relpath=rel,
            original_path=original_files[rel].path,
            modified_path=modified_files[rel].path,
            context=args.diff_context,
            max_lines=args.max_diff_lines_per_file,
        )

        if not diff_text:
            continue

        lines.append(f"### `{rel}`")
        lines.append("")
        lines.append(
            f"- 关注度：{focus}；新增 {added} 行；删除 {removed} 行。"
        )
        lines.append("")
        lines.append("```diff")
        lines.append(diff_text)
        lines.append("```")
        lines.append("")
        diff_count += 1

    if diff_count == 0:
        lines.append("没有可展示的同路径文件 diff。")
        lines.append("")

    lines.append("## 7. 修改版新增文件")
    lines.append("")
    if only_modified:
        for rel in sorted(
            only_modified,
            key=lambda p: (-path_focus_score(p), p)
        ):
            candidates = counterpart_map.get(rel, [])
            lines.append(f"- `{rel}`")
            if candidates:
                lines.append(
                    "  - 原版可能对应："
                    + "、".join(f"`{x}`" for x in candidates)
                )
    else:
        lines.append("- 无")
    lines.append("")

    lines.append("## 8. 原版中消失的文件")
    lines.append("")
    if only_original:
        for rel in sorted(
            only_original,
            key=lambda p: (-path_focus_score(p), p)
        ):
            lines.append(f"- `{rel}`")
    else:
        lines.append("- 无")
    lines.append("")

    lines.append("## 9. 运行时必须补充的检查")
    lines.append("")
    lines.append("静态分析完成后，建议在原版和修改版训练循环中都加入以下统计。两边只训练 1 个 epoch 后比较：")
    lines.append("")
    lines.append("```python")
    lines.append("epoch_batches = 0")
    lines.append("epoch_samples = 0")
    lines.append("optimizer_steps = 0")
    lines.append("")
    lines.append("for batch_i, batch in enumerate(train_loader):")
    lines.append("    epoch_batches += 1")
    lines.append("    images = batch[0] if isinstance(batch, (list, tuple)) else batch['img']")
    lines.append("    epoch_samples += int(images.shape[0])")
    lines.append("")
    lines.append("    # 原训练代码")
    lines.append("    # loss.backward()")
    lines.append("    # optimizer.step()")
    lines.append("    optimizer_steps += 1")
    lines.append("")
    lines.append("print({")
    lines.append("    'dataset_len': len(train_loader.dataset),")
    lines.append("    'loader_len': len(train_loader),")
    lines.append("    'epoch_batches': epoch_batches,")
    lines.append("    'epoch_samples': epoch_samples,")
    lines.append("    'optimizer_steps': optimizer_steps,")
    lines.append("    'batch_size': train_loader.batch_size,")
    lines.append("    'drop_last': train_loader.drop_last,")
    lines.append("})")
    lines.append("```")
    lines.append("")
    lines.append("如果修改版约为原版的三分之一，优先搜索报告中的：")
    lines.append("")
    lines.append("- `zip(loader...)`")
    lines.append("- `min(len(loader...))`")
    lines.append("- `steps_per_epoch`")
    lines.append("- `num_tasks`")
    lines.append("- `task_id` / `active_tasks`")
    lines.append("- `Subset` / sampler 的 `num_samples`")
    lines.append("")

    lines.append("## 10. 32 张图片过拟合判定")
    lines.append("")
    lines.append("固定 32 张训练图，关闭所有增强，训练 300～1000 step：")
    lines.append("")
    lines.append("- 原版能记住、修改版不能记住：模型 forward、head、loss、梯度或 optimizer 有问题。")
    lines.append("- 两版都能记住，但完整训练修改版差：采样、epoch 步数、scheduler、增强或验证/推理链路有问题。")
    lines.append("- 修改版训练集结果好、验证集差：重点检查数据分布、标签、增强和类别权重。")
    lines.append("")

    return "\n".join(lines) + "\n"


def validate_root(path: Path, name: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{name}目录不存在: {path}")
    if not path.is_dir():
        raise NotADirectoryError(f"{name}不是目录: {path}")


def main() -> int:
    args = parse_args()

    modified_root = args.modified.expanduser().resolve()
    original_root = args.original.expanduser().resolve()
    output_path = args.output.expanduser().resolve()

    validate_root(modified_root, "修改版")
    validate_root(original_root, "原版")

    print("=" * 80)
    print("LaneRobot 仓库静态审计")
    print("=" * 80)
    print("修改版:", modified_root)
    print("原版  :", original_root)
    print("输出  :", output_path)
    print("-" * 80)

    print("[1/4] 收集文本文件...")
    modified_files = collect_files(modified_root)
    original_files = collect_files(original_root)

    print(f"      修改版: {len(modified_files)}")
    print(f"      原版  : {len(original_files)}")

    print("[2/4] 扫描训练链路风险模式...")
    modified_findings = scan_repo("modified", modified_root, modified_files)
    original_findings = scan_repo("original", original_root, original_files)

    print(f"      修改版命中: {len(modified_findings)}")
    print(f"      原版命中  : {len(original_findings)}")

    print("[3/4] 生成仓库 diff 与风险报告...")
    report = build_report(
        modified_root=modified_root,
        original_root=original_root,
        modified_files=modified_files,
        original_files=original_files,
        modified_findings=modified_findings,
        original_findings=original_findings,
        args=args,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding="utf-8")

    print("[4/4] 完成")
    print("-" * 80)
    print("报告路径:", output_path)
    print()
    print("建议查看命令:")
    print(f"  code {output_path}")
    print()
    print("重点先搜索:")
    print("  zip(")
    print("  steps_per_epoch")
    print("  num_tasks")
    print("  optimizer.step")
    print("  requires_grad")
    print("  scheduler.step")
    print("  load_state_dict")
    print("=" * 80)

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\n用户中断。", file=sys.stderr)
        raise SystemExit(130)
    except Exception as exc:
        print(f"\n审计失败: {exc}", file=sys.stderr)
        raise SystemExit(1)
