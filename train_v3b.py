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
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL = PROJECT_ROOT / "ultralytics/cfg/models/26/yolo26s-lane-v3b.yaml"
DEFAULT_DATA = PROJECT_ROOT / "ultralytics/cfg/datasets/lane-robot.yaml"
DEFAULT_RUNS = PROJECT_ROOT / "runs/lane"


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
    parser.add_argument("--batch", type=int, default=-1, help="Batch size; -1 auto-selects the largest safe batch.")
    parser.add_argument("--workers", type=int, default=8, help="DataLoader worker processes.")
    parser.add_argument("--device", default="0", help="CUDA device, e.g. 0, 0,1, or cpu.")
    parser.add_argument("--img-height", type=int, default=256, help="Input image height.")
    parser.add_argument("--img-width", type=int, default=448, help="Input image width.")

    parser.add_argument("--optimizer", default="AdamW", choices=("AdamW", "Adam", "SGD", "auto"))
    parser.add_argument("--lr0", type=float, default=3e-4, help="Initial learning rate.")
    parser.add_argument("--lrf", type=float, default=0.01, help="Final LR fraction: final_lr = lr0*lrf.")
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-epochs", type=float, default=3.0)

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
    if args.batch == 0:
        parser.error("--batch must be -1 (auto) or a positive integer")
    if args.img_height <= 0 or args.img_width <= 0:
        parser.error("image dimensions must be greater than 0")

    return args


def require_file(path: Path, description: str) -> Path:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{description} does not exist: {path}")
    return path


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
    print(f"batch        : {args.batch} (-1 = autobatch)")
    print(f"imgsz        : [{args.img_height}, {args.img_width}]")
    print(f"device       : {args.device}")
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
