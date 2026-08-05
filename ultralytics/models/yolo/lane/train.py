# Ultralytics 🚀 AGPL-3.0 License

from __future__ import annotations

from copy import copy
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from ultralytics.engine.trainer import BaseTrainer
from ultralytics.models import yolo
from ultralytics.models.yolo.lane.dataset import LaneRobotDataset
from ultralytics.models.yolo.lane.plotting import decode_lane, save_lane_grid
from ultralytics.models.yolo.lane.val import get_lane_head
from ultralytics.nn.tasks import LaneRobotModel
from ultralytics.utils import DEFAULT_CFG, LOGGER, RANK, YAML
from ultralytics.utils.torch_utils import unwrap_model


class LaneRobotTrainer(BaseTrainer):
    """Trainer for fixed-task YOLO26 LaneRobot row-anchor models, including independent branches."""

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
        data.setdefault("x_grids", int(getattr(self.args, "lane_x_grids", 640)))
        data.setdefault("row_anchors", int(getattr(self.args, "lane_row_anchors", 56)))
        data.setdefault("num_lanes", int(getattr(self.args, "lane_num_lanes", 2)))
        data.setdefault("y_start", float(getattr(self.args, "lane_y_start", 0.67)))
        data.setdefault("y_end", float(getattr(self.args, "lane_y_end", 1.0)))
        data.setdefault("channels", 3)
        data.setdefault("names", {i: f"lane_{i}" for i in range(int(data["num_lanes"]))})
        data.setdefault("nc", int(data["num_lanes"]))
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

    def get_model(self, cfg: str | None = None, weights: str | None = None, verbose: bool = True):
        model = LaneRobotModel(cfg, ch=self.data["channels"], verbose=verbose and RANK == -1)
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
