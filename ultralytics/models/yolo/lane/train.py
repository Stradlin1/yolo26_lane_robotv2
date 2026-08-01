# Ultralytics 🚀 AGPL-3.0 License

from __future__ import annotations

from copy import copy, deepcopy
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from ultralytics.engine.trainer import BaseTrainer
from ultralytics.models import yolo
from ultralytics.models.yolo.lane.dataset import LaneRobotDataset
from ultralytics.models.yolo.lane.geometry import parse_letterbox_color
from ultralytics.models.yolo.lane.plotting import decode_lane, save_lane_grid
from ultralytics.models.yolo.lane.val import get_lane_head
from ultralytics.nn.tasks import LaneRobotModel, yaml_model_load
from ultralytics.utils import DEFAULT_CFG, LOGGER, RANK, YAML
from ultralytics.utils.torch_utils import unwrap_model


class LaneRobotTrainer(BaseTrainer):
    """Trainer for a single-task YOLO26 LaneRobot row-anchor model."""

    def __init__(self, cfg=DEFAULT_CFG, overrides: dict[str, Any] | None = None, _callbacks: dict | None = None):
        overrides = overrides or {}
        overrides.setdefault("task", "lane")
        super().__init__(cfg, overrides, _callbacks)

    def get_dataset(self):
        data = YAML.load(self.args.data) if isinstance(self.args.data, (str, Path)) else dict(self.args.data)
        root = Path(data.get("path", ".")).expanduser()
        for k in ("train", "val", "test"):
            if data.get(k) and not Path(data[k]).is_absolute():
                data[k] = str(root / data[k])
        # The dataset YAML is the single source of truth for lane structure.
        data["x_grids"] = int(data.get("x_grids", getattr(self.args, "lane_x_grids", 320)))
        data["row_anchors"] = int(data.get("row_anchors", getattr(self.args, "lane_row_anchors", 56)))
        data["num_lanes"] = int(data.get("num_lanes", getattr(self.args, "lane_num_lanes", 1)))
        data["y_start"] = float(data.get("y_start", getattr(self.args, "lane_y_start", 0.67)))
        data["y_end"] = float(data.get("y_end", getattr(self.args, "lane_y_end", 1.0)))
        data["letterbox"] = bool(data.get("letterbox", getattr(self.args, "lane_letterbox", False)))
        data["letterbox_color"] = list(
            parse_letterbox_color(
                data.get("letterbox_color", getattr(self.args, "lane_letterbox_color", [0, 0, 0]))
            )
        )
        data["letterbox_bottom_align"] = bool(
            data.get(
                "letterbox_bottom_align",
                getattr(self.args, "lane_letterbox_bottom_align", True),
            )
        )
        data["channels"] = int(data.get("channels", 3))

        if data["x_grids"] < 2:
            raise ValueError(f"x_grids must be >= 2, got {data['x_grids']}")
        if data["row_anchors"] < 2:
            raise ValueError(f"row_anchors must be >= 2, got {data['row_anchors']}")
        if data["num_lanes"] < 1:
            raise ValueError(f"num_lanes must be >= 1, got {data['num_lanes']}")

        # Keep Ultralytics metadata consistent with the number of fixed lane slots.
        data["nc"] = data["num_lanes"]
        names = data.get("names")
        if isinstance(names, list):
            names = {i: str(name) for i, name in enumerate(names)}
        elif isinstance(names, dict):
            try:
                names = {int(i): str(name) for i, name in names.items()}
            except (TypeError, ValueError):
                names = None

        expected_ids = set(range(data["num_lanes"]))
        if not isinstance(names, dict) or set(names) != expected_ids:
            if names is not None:
                LOGGER.warning(
                    f"Lane names must use IDs 0..{data['num_lanes'] - 1}; generating default names instead."
                )
            names = {i: f"lane_{i}" for i in range(data["num_lanes"])}
        data["names"] = names

        # Mirror the resolved values into args because loss/validation code also reads lane_* options.
        self.args.lane_x_grids = data["x_grids"]
        self.args.lane_row_anchors = data["row_anchors"]
        self.args.lane_num_lanes = data["num_lanes"]
        self.args.lane_y_start = data["y_start"]
        self.args.lane_y_end = data["y_end"]
        self.args.lane_letterbox = data["letterbox"]
        self.args.lane_letterbox_color = data["letterbox_color"]
        self.args.lane_letterbox_bottom_align = data["letterbox_bottom_align"]
        if not data.get("val"):
            data["val"] = data["train"]
            LOGGER.warning("Lane data yaml has no val split; using train split for validation.")
        return data

    def build_dataset(self, img_path: str, mode: str = "train", batch: int | None = None):
        return LaneRobotDataset(img_path, self.args, self.data, mode=mode)

    def get_dataloader(self, dataset_path: str, batch_size: int = 16, rank: int = 0, mode: str = "train"):
        dataset = self.build_dataset(dataset_path, mode, batch_size)
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=mode == "train",
            num_workers=self.args.workers if mode == "train" else max(self.args.workers // 2, 0),
            pin_memory=self.device.type != "cpu",
            collate_fn=LaneRobotDataset.collate_fn,
            drop_last=False,
        )

    def preprocess_batch(self, batch: dict) -> dict:
        batch["img"] = batch["img"].to(self.device, non_blocking=self.device.type == "cuda").float() / 255.0
        batch["lane"] = batch["lane"].to(self.device, non_blocking=self.device.type == "cuda").long()
        if "lane_x" in batch:
            batch["lane_x"] = batch["lane_x"].to(self.device, non_blocking=self.device.type == "cuda").float()
        if "lane_y" in batch:
            batch["lane_y"] = batch["lane_y"].to(self.device, non_blocking=self.device.type == "cuda").float()
        batch["cls"] = batch["cls"].to(self.device, non_blocking=self.device.type == "cuda")
        return batch

    def set_model_attributes(self):
        self.model.args = self.args
        self.model.names = self.data["names"]
        self.model.nc = self.data["nc"]

    def get_model(self, cfg: str | dict | None = None, weights: str | None = None, verbose: bool = True):
        """Build a lane model whose output dimensions are controlled by the dataset YAML."""
        if cfg is None:
            raise ValueError("A lane model YAML or model config dictionary is required.")

        # cfg can be a YAML path during normal training or a dict when resuming/loading a checkpoint.
        model_cfg = deepcopy(cfg) if isinstance(cfg, dict) else yaml_model_load(cfg)

        # Override the model YAML defaults with the resolved dataset structure.
        model_cfg["task"] = "lane"
        model_cfg["x_grids"] = int(self.data["x_grids"])
        model_cfg["row_anchors"] = int(self.data["row_anchors"])
        model_cfg["num_lanes"] = int(self.data["num_lanes"])
        model_cfg["nc"] = int(self.data["num_lanes"])

        LOGGER.info(
            "Building LaneRobot model from dataset structure: "
            f"x_grids={model_cfg['x_grids']}, "
            f"row_anchors={model_cfg['row_anchors']}, "
            f"num_lanes={model_cfg['num_lanes']}"
        )

        model = LaneRobotModel(
            model_cfg,
            ch=int(self.data["channels"]),
            verbose=verbose and RANK == -1,
        )

        # Fail early if a future model YAML/head implementation ignores the overridden values.
        head = get_lane_head(model)
        expected = (
            int(self.data["x_grids"]),
            int(self.data["row_anchors"]),
            int(self.data["num_lanes"]),
        )
        actual = (
            int(head.x_grids),
            int(head.row_anchors),
            int(head.num_lanes),
        )
        if actual != expected:
            raise RuntimeError(
                "Lane model/data structure mismatch: "
                f"head={actual}, data={expected}. "
                "Tuple order is (x_grids, row_anchors, num_lanes)."
            )

        model.names = dict(self.data["names"])
        model.nc = int(self.data["nc"])

        if weights:
            model.load(weights)
        return model

    def get_validator(self):
        self.loss_names = "lane_ce", "lane_loc", "lane_exist", "lane_smooth", "lane_curv", "lane_offset"
        return yolo.lane.LaneRobotValidator(
            self.test_loader, save_dir=self.save_dir, args=copy(self.args), _callbacks=self.callbacks
        )

    def label_loss_items(self, loss_items=None, prefix: str = "train"):
        keys = [f"{prefix}/{x}" for x in self.loss_names]
        if loss_items is not None:
            loss_items = [round(float(x), 5) for x in loss_items]
            return dict(zip(keys, loss_items))
        return keys

    def progress_string(self):
        return ("\n" + "%11s" * (4 + len(self.loss_names))) % (
            "Epoch",
            "GPU_mem",
            *self.loss_names,
            "Instances",
            "Size",
        )

    def plot_training_samples(self, batch: dict[str, Any], ni: int) -> None:
        targets = batch.get("lane_x", batch["lane"].float()).detach().cpu().numpy()
        targets = targets.copy()
        x_grids = int(self.data["x_grids"])
        targets[targets >= x_grids] = -1
        save_lane_grid(
            batch["img"].detach().cpu(),
            targets,
            targets,
            x_grids=x_grids,
            row_anchors=int(self.data["row_anchors"]),
            save_path=self.save_dir / f"train_batch{ni}_lane.jpg",
            row_y=batch.get("lane_y"),
            y_start=float(self.data.get("y_start", 0.67)),
            y_end=float(self.data.get("y_end", 1.0)),
        )

    def plot_training_labels(self):
        # Saved during batch plotting; row-anchor labels do not have a box distribution plot.
        pass

    def final_eval(self):
        super().final_eval()
        # Ensure a final visual artifact exists even when early stopping fires before the final scheduled plot.
        if RANK in {-1, 0} and self.args.plots and hasattr(self, "test_loader"):
            model = self.ema.ema if self.ema else self.model
            model.eval()
            try:
                batch = next(iter(self.test_loader))
                batch = self.preprocess_batch(batch)
                with torch.no_grad():
                    preds = model(batch["img"])
                head = get_lane_head(unwrap_model(model))
                pred_xy = decode_lane(preds, no_lane_idx=head.x_grids, topk=int(getattr(self.args, "lane_softargmax_topk", 5)), exist_thr=float(getattr(self.args, "lane_exist_thr", 0.5)), post_smooth=bool(getattr(self.args, "lane_post_smooth", True)), poly_degree=int(getattr(self.args, "lane_poly_degree", 2)), poly_blend=float(getattr(self.args, "lane_poly_blend", 0.5)))
                tgt = batch.get("lane_x", batch["lane"].float()).detach().cpu().numpy()
                tgt = tgt.copy()
                tgt[tgt >= head.x_grids] = -1
                save_lane_grid(
                    batch["img"].detach().cpu(),
                    pred_xy,
                    tgt,
                    x_grids=head.x_grids,
                    row_anchors=head.row_anchors,
                    save_path=self.save_dir / "lane_final_predictions.jpg",
                    row_y=batch.get("lane_y"),
                    y_start=float(self.data.get("y_start", 0.67)),
                    y_end=float(self.data.get("y_end", 1.0)),
                )
            except Exception as e:
                LOGGER.warning(f"Could not save final lane visualization: {e}")
