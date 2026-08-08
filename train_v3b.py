#!/usr/bin/env python3
"""LaneRobotV3B 四槽位多线模型正式训练脚本。

与 train_xhm.py（V2 时代脚本）的区别：
    - 默认模型为 yolo26s-lane-v3b.yaml（V3-B 头，256x448 输入）
    - 支持 --weights 传入 V2 checkpoint，自动只迁移 backbone/neck（L0~L15）
    - 增强配置按 V3-B 定稿值收敛：小 translate/scale、保守 hsv、fliplr 0.5
    - 路径基于脚本所在目录，不再硬编码 /home/xhm/Desktop/...

默认用法：
    python train_v3b.py --weights runs/lane/lane_n_baseline/weights/last.pt

常用覆盖：
    python train_v3b.py --weights <v2.pt> --epochs 300 --patience 60 --batch 16

冒烟测试（本地快速验证管线）：
    python train_v3b.py --weights <v2.pt> --fraction 0.02 --epochs 1 --batch 4 \
        --workers 2 --device cpu
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL = PROJECT_ROOT / "ultralytics/cfg/models/26/yolo26s-lane-v3b.yaml"
DEFAULT_DATA = PROJECT_ROOT / "ultralytics/cfg/datasets/lane-robot.yaml"
DEFAULT_RUNS = PROJECT_ROOT / "runs/lane"

# Non-interactive shells (nohup/tmux) do not source ~/.bashrc; keep the dataset
# root self-contained so training works regardless of the shell environment.
os.environ.setdefault("LANE_ROBOT_DATASETS", str(PROJECT_ROOT / "datasets"))


# V3-B 定稿增强配置：几何只做轻微位姿扰动，颜色保持保守，连续结构增强全部关闭。
AUGMENTATION_CONFIG = {
    # 颜色增强：色相几乎不动，饱和度/亮度轻微扰动。
    "hsv_h": 0.002,
    "hsv_s": 0.05,
    "hsv_v": 0.05,
    "bgr": 0.0,

    # 几何增强：小幅平移/缩放，模拟位姿抖动；大变形关闭。
    "degrees": 2.0,
    "translate": 0.03,
    "scale": 0.05,
    "shear": 0.0,
    "perspective": 0.0,
    "flipud": 0.0,
    "fliplr": 0.5,

    # 拼接/混合增强会破坏连续车道结构，保持关闭。
    "mosaic": 0.0,
    "mixup": 0.0,
    "cutmix": 0.0,
    "copy_paste": 0.0,
    "erasing": 0.0,
    "close_mosaic": 0,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the four-slot LaneRobot V3-B model.")

    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL, help="Lane model YAML path.")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA, help="Dataset YAML path.")
    parser.add_argument("--project", type=Path, default=DEFAULT_RUNS, help="Training output root.")
    parser.add_argument("--name", default="lane_v3b", help="Experiment directory name.")
    parser.add_argument(
        "--weights",
        type=Path,
        default=None,
        help="V2 checkpoint path; backbone/neck (L0~L15) will be transferred, head starts random.",
    )

    parser.add_argument("--epochs", type=int, default=500, help="Maximum training epochs.")
    parser.add_argument(
        "--patience",
        type=int,
        default=100,
        help="Stop after this many epochs without fitness improvement; 0 disables early stopping.",
    )
    parser.add_argument(
        "--batch",
        type=float,
        default=-1,
        help="Batch size: -1 auto-selects largest safe batch; 0<batch<=1 uses a GPU-memory fraction.",
    )
    parser.add_argument("--workers", type=int, default=8, help="DataLoader worker processes.")
    parser.add_argument("--fraction", type=float, default=1.0, help="Fraction of the training set to use (smoke tests).")
    parser.add_argument(
        "--freeze",
        type=str,
        default=None,
        help="Freeze first N layers (int) or specific layer indices (comma list, e.g. '0,1,2').",
    )
    parser.add_argument("--device", default="0", help="CUDA device, e.g. 0, 0,1, or cpu.")
    parser.add_argument("--img-height", type=int, default=256, help="Input image height.")
    parser.add_argument("--img-width", type=int, default=448, help="Input image width.")

    parser.add_argument("--optimizer", default="AdamW", choices=("AdamW", "Adam", "SGD", "auto"))
    parser.add_argument("--lr0", type=float, default=3e-4, help="Initial learning rate.")
    parser.add_argument("--lrf", type=float, default=0.01, help="Final LR fraction: final_lr = lr0*lrf.")
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-epochs", type=float, default=3.0)

    # LaneRobot loss / decode hyperparameters (defaults mirror ultralytics/cfg/default.yaml).
    parser.add_argument("--lane-ce", type=float, default=1.0, help="Row-anchor x-class CE loss gain.")
    parser.add_argument("--lane-loc", type=float, default=2.0, help="Expected-x SmoothL1 loss gain.")
    parser.add_argument("--lane-exist", type=float, default=1.5, help="Lane-existence BCE loss gain.")
    parser.add_argument("--lane-smooth", type=float, default=0.03, help="Adjacent-row smoothness loss gain.")
    parser.add_argument("--lane-curv", type=float, default=0.02, help="Second-order curvature loss gain.")
    parser.add_argument("--lane-offset", type=float, default=3.0, help="Sub-grid offset SmoothL1 loss gain.")
    parser.add_argument("--lane-soft-sigma", type=float, default=1.0, help="Gaussian sigma for soft-label CE.")
    parser.add_argument("--lane-softargmax-topk", type=int, default=5, help="Top-k window for soft-argmax.")
    parser.add_argument("--lane-exist-thr", type=float, default=0.5, help="No-lane probability threshold.")
    parser.add_argument("--lane-end-weight", type=float, default=1.0, help="Extra weight for end-segment rows (r25-34); 1.0 disables.")
    parser.add_argument(
        "--lane-end-weight-tail",
        type=float,
        default=6.0,
        help="End-segment weight at the last row (linear ramp from --lane-end-weight); 1.0 disables.",
    )
    parser.add_argument(
        "--lane-end-no-lane-weight",
        type=float,
        default=1.0,
        help="Multiplier for no-lane supervision inside the end-segment band; 1.0 disables.",
    )

    parser.add_argument("--save-period", type=int, default=100, help="Save checkpoint every N epochs; -1 disables.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", type=Path, default=None, help="Path to last.pt for interrupted-run resume.")
    parser.add_argument("--exist-ok", action="store_true", help="Allow reusing an existing experiment directory.")
    parser.add_argument("--no-amp", action="store_true", help="Disable automatic mixed precision.")
    parser.add_argument("--nondeterministic", action="store_true", help="Disable deterministic training mode.")

    args = parser.parse_args()

    if args.epochs <= 0:
        parser.error("--epochs must be greater than 0")
    if args.patience < 0:
        parser.error("--patience must be 0 or greater")
    if args.batch != -1 and args.batch <= 0:
        parser.error("--batch must be -1 (auto) or a positive value (<=1 means a GPU-memory fraction)")
    if args.batch > 1 and float(args.batch).is_integer():
        args.batch = int(args.batch)
    if args.img_height <= 0 or args.img_width <= 0:
        parser.error("image dimensions must be greater than 0")
    if not (0.0 < args.fraction <= 1.0):
        parser.error("--fraction must be in (0, 1]")
    if args.lane_softargmax_topk < 1:
        parser.error("--lane-softargmax-topk must be >= 1")

    if args.freeze is not None:
        try:
            args.freeze = [int(x) for x in args.freeze.split(",")]
        except ValueError:
            parser.error("--freeze must be an int or a comma-separated list of ints")

    return args


def require_file(path: Path, description: str) -> Path:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{description} does not exist: {path}")
    return path


def _reset_lane1_head(trainer) -> None:
    """Re-initialize lane1 cls/offset heads after pretrained load.

    Lane1's classifier input is conditioned on lane0's predicted geometry (side branch),
    so the old checkpoint's lane1 branch must not carry its previous standalone meaning.
    Called from the on_train_start callback (pretrained weights are already loaded).
    """
    from ultralytics.nn.modules.utils import linear_init
    from ultralytics.nn.modules.head import LaneRobotV3B

    head = next((m for m in trainer.model.modules() if isinstance(m, LaneRobotV3B)), None)
    if head is None:
        print("WARNING: LaneRobotV3B head not found; lane1 branch not reset.")
        return
    if head.num_lanes <= 1:
        return
    linear_init(head.cls_heads[1])
    linear_init(head.offset_heads[1])
    print("Lane1 cls/offset heads reset (side-conditioned residual scheme).")


def main() -> None:
    args = parse_args()

    data_path = require_file(args.data, "Dataset YAML")
    project_path = args.project.expanduser().resolve()
    project_path.mkdir(parents=True, exist_ok=True)

    if args.weights is not None:
        weights_path = require_file(args.weights, "V2 checkpoint")
        pretrained = str(weights_path)
    else:
        pretrained = False

    if args.resume is not None:
        resume_path = require_file(args.resume, "Resume checkpoint")
        model = YOLO(str(resume_path), task="lane")
        resume_value: bool | str = str(resume_path)
        model_source = resume_path
    else:
        model_path = require_file(args.model, "Model YAML")
        model = YOLO(str(model_path), task="lane")
        if args.weights is not None:
            # Pretrained weights are loaded inside trainer; reset lane1 branch afterwards.
            model.add_callback("on_train_start", _reset_lane1_head)
        resume_value = False
        model_source = model_path

    train_config = {
        "data": str(data_path),
        "epochs": args.epochs,
        "patience": args.patience,
        "batch": args.batch,
        "imgsz": [args.img_height, args.img_width],
        "device": args.device,
        "workers": args.workers,
        "optimizer": args.optimizer,
        "lr0": args.lr0,
        "lrf": args.lrf,
        "weight_decay": args.weight_decay,
        "warmup_epochs": args.warmup_epochs,
        "fraction": args.fraction,
        "freeze": args.freeze,
        "cos_lr": True,
        "amp": not args.no_amp,
        "seed": args.seed,
        "deterministic": not args.nondeterministic,
        "pretrained": pretrained,
        "val": True,
        "plots": True,
        "save": True,
        "save_period": args.save_period,
        "project": str(project_path),
        "name": args.name,
        "exist_ok": args.exist_ok,
        "resume": resume_value,
        "lane_ce": args.lane_ce,
        "lane_loc": args.lane_loc,
        "lane_exist": args.lane_exist,
        "lane_smooth": args.lane_smooth,
        "lane_curv": args.lane_curv,
        "lane_offset": args.lane_offset,
        "lane_soft_sigma": args.lane_soft_sigma,
        "lane_softargmax_topk": args.lane_softargmax_topk,
        "lane_exist_thr": args.lane_exist_thr,
        "lane_end_weight": args.lane_end_weight,
        "lane_end_weight_tail": args.lane_end_weight_tail,
        "lane_end_no_lane_weight": args.lane_end_no_lane_weight,
        **AUGMENTATION_CONFIG,
    }

    print("=" * 72)
    print("LaneRobot V3-B training")
    print(f"model/resume : {model_source}")
    print(f"data         : {data_path}")
    print(f"v2 weights   : {weights_path if args.weights is not None else 'None (random init)'}")
    print(f"output       : {project_path / args.name}")
    print(f"epochs       : {args.epochs}")
    print(f"patience     : {args.patience} (0 means disabled)")
    print(f"batch        : {args.batch} (-1 = autobatch, <=1 = memory fraction)")
    print(f"fraction     : {args.fraction} (smoke tests use e.g. 0.02)")
    print(f"freeze       : {args.freeze}")
    print(f"imgsz        : [{args.img_height}, {args.img_width}]")
    print(f"device       : {args.device}")
    print(
        "lane loss    : "
        f"ce={args.lane_ce}, loc={args.lane_loc}, exist={args.lane_exist}, "
        f"smooth={args.lane_smooth}, curv={args.lane_curv}, offset=disabled(plan-A), "
        f"end_weight={args.lane_end_weight}->{args.lane_end_weight_tail}, "
        f"no_lane_w={args.lane_end_no_lane_weight}"
    )
    print(
        "augmentation : "
        f"fliplr={AUGMENTATION_CONFIG['fliplr']}, "
        f"translate={AUGMENTATION_CONFIG['translate']}, "
        f"scale={AUGMENTATION_CONFIG['scale']}, "
        f"degrees={AUGMENTATION_CONFIG['degrees']}, "
        f"hsv=({AUGMENTATION_CONFIG['hsv_h']}, {AUGMENTATION_CONFIG['hsv_s']}, {AUGMENTATION_CONFIG['hsv_v']})"
    )
    print("=" * 72)

    model.train(**train_config)


if __name__ == "__main__":
    main()
