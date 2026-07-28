#!/usr/bin/env python3
"""Lane Robot 四槽位多线模型正式训练脚本。

默认用法：
    cd /home/xhm/Desktop/ULTRALYTICS_LANE_ROBOT
    conda activate lane_robot
    python train_xhm.py

临时覆盖常用参数：
    python train_xhm.py --epochs 300 --patience 60 --batch 8 --name lane_n_e300

断点续训：
    python train_xhm.py --resume runs/lane/lane_n_baseline/weights/last.pt \
        --epochs 300 --patience 60
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


# 固定为你的项目根目录。脚本应放在该目录下运行。
PROJECT_ROOT = Path("/home/xhm/Desktop/ULTRALYTICS_LANE_ROBOT")
DEFAULT_MODEL = PROJECT_ROOT / "ultralytics/cfg/models/26/yolo26n-lane.yaml"
DEFAULT_DATA = PROJECT_ROOT / "ultralytics/cfg/datasets/lane-robot.yaml"
DEFAULT_RUNS = PROJECT_ROOT / "runs/lane"


# 当前 LaneRobotDataset 只执行读取图片、EXIF 修正和 resize，尚未调用这些增强参数。
# 因此本字典先保持为无增强基线。以后将增强逻辑接入 dataset.py 后，这些值才会生效。
AUGMENTATION_CONFIG = {
    # 颜色增强：黄色通道/绿色背景具有语义，建议保守。
    "hsv_h": 0.0,
    "hsv_s": 0.0,
    "hsv_v": 0.0,
    "bgr": 0.0,

    # 几何增强：当前没有同步变换车道标签的实现，必须关闭。
    "degrees": 0.5,
    "translate": 0.03,
    "scale": 0.05,
    "shear": 0.0,
    "perspective": 0.0,
    "flipud": 0.0,
    "fliplr": 0.5,

    # 拼接/混合增强会破坏连续通道结构，保持关闭。
    "mosaic": 0.0,
    "mixup": 0.0,
    "cutmix": 0.0,
    "copy_paste": 0.0,
    "erasing": 0.0,
    "close_mosaic": 0,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the four-slot Lane Robot model.")

    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL, help="Lane model YAML path.")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA, help="Dataset YAML path.")
    parser.add_argument("--project", type=Path, default=DEFAULT_RUNS, help="Training output root.")
    parser.add_argument("--name", default="lane_n_baseline", help="Experiment directory name.")

    parser.add_argument("--epochs", type=int, default=200, help="Maximum training epochs.")
    parser.add_argument(
        "--patience",
        type=int,
        default=50,
        help="Stop after this many epochs without fitness improvement; 0 disables early stopping.",
    )
    parser.add_argument("--batch", type=int, default=8, help="Batch size; lower to 4 or 2 on CUDA OOM.")
    parser.add_argument("--workers", type=int, default=4, help="DataLoader worker processes.")
    parser.add_argument("--device", default="0", help="CUDA device, e.g. 0, 0,1, or cpu.")
    parser.add_argument("--img-height", type=int, default=256, help="Input image height.")
    parser.add_argument("--img-width", type=int, default=320, help="Input image width.")

    parser.add_argument("--optimizer", default="AdamW", choices=("AdamW", "Adam", "SGD", "auto"))
    parser.add_argument("--lr0", type=float, default=3e-4, help="Initial learning rate.")
    parser.add_argument("--lrf", type=float, default=0.01, help="Final LR fraction: final_lr = lr0*lrf.")
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-epochs", type=float, default=3.0)

    parser.add_argument("--save-period", type=int, default=10, help="Save checkpoint every N epochs; -1 disables.")
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
    if args.batch <= 0:
        parser.error("--batch must be greater than 0")
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
        "pretrained": False,
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
    print("Lane Robot formal training")
    print(f"model/resume : {model_source}")
    print(f"data         : {data_path}")
    print(f"output       : {project_path / args.name}")
    print(f"epochs       : {args.epochs}")
    print(f"patience     : {args.patience} (0 means disabled)")
    print(f"batch        : {args.batch}")
    print(f"imgsz        : [{args.img_height}, {args.img_width}]")
    print(f"device       : {args.device}")
    print(
    "augmentation : "
    f"fliplr={AUGMENTATION_CONFIG['fliplr']}, "
    f"flipud={AUGMENTATION_CONFIG['flipud']}, "
    f"degrees={AUGMENTATION_CONFIG['degrees']}, "
    f"translate={AUGMENTATION_CONFIG['translate']}, "
    f"scale={AUGMENTATION_CONFIG['scale']}"
)
    print("=" * 72)

    model.train(**train_config)


if __name__ == "__main__":
    main()
