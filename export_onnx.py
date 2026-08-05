from pathlib import Path

import torch
from ultralytics import YOLO


# =========================
# 固定配置
# =========================
WEIGHTS = "/home/xhm/Desktop/ULTRALYTICS_LANE_ROBOT/runs/lane/train/weights/best.pt"
OUTPUT = "/home/xhm/Desktop/ULTRALYTICS_LANE_ROBOT/runs/lane/train/weights/best.onnx"

IMG_SIZE = 640
OPSET = 11
DEVICE = "cuda:0"   # 没有 CUDA 会自动切到 CPU
SIMPLIFY = False    # 需要简化 ONNX 时改成 True


class LaneONNXWrapper(torch.nn.Module):
    """
    将 YOLO26-LaneRobotV2Independent 包装成稳定 ONNX 输出。

    期望输出：
        cls_logits: [B, 161, 56, 4]
        offset:     [B, 1,   56, 4]
    """

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, images):
        y = self.model(images)

        if isinstance(y, dict):
            cls_logits = y.get("cls", None)
            offset = y.get("offset", None)

            if cls_logits is None:
                cls_logits = y.get("cls_logits", None)
            if offset is None:
                offset = y.get("lane_offset", None)

            if cls_logits is None or offset is None:
                raise RuntimeError(f"模型 dict 输出中找不到 cls/offset，当前 keys={list(y.keys())}")

            return cls_logits, offset

        if isinstance(y, (tuple, list)):
            if len(y) < 2:
                raise RuntimeError(f"模型输出数量不足，当前 len={len(y)}")
            return y[0], y[1]

        raise RuntimeError(f"模型输出类型异常：{type(y)}，期望 dict/tuple/list")


def main():
    weights = Path(WEIGHTS)
    output = Path(OUTPUT)

    if not weights.exists():
        raise FileNotFoundError(f"权重文件不存在：{weights}")

    output.parent.mkdir(parents=True, exist_ok=True)

    device = DEVICE
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"

    yolo = YOLO(str(weights))
    model = yolo.model.to(device).eval()

    wrapper = LaneONNXWrapper(model).to(device).eval()
    dummy = torch.zeros(1, 3, IMG_SIZE, IMG_SIZE, device=device)

    with torch.no_grad():
        cls_logits, offset = wrapper(dummy)

    print("PyTorch 输出检查：")
    print(f"  cls_logits: shape={tuple(cls_logits.shape)}, dtype={cls_logits.dtype}")
    print(f"  offset:     shape={tuple(offset.shape)}, dtype={offset.dtype}")
    expected_cls = (1, 161, 56, 4)
    expected_offset = (1, 1, 56, 4)
    if tuple(cls_logits.shape) != expected_cls:
        raise RuntimeError(f"cls_logits shape mismatch: expected {expected_cls}, got {tuple(cls_logits.shape)}")
    if tuple(offset.shape) != expected_offset:
        raise RuntimeError(f"offset shape mismatch: expected {expected_offset}, got {tuple(offset.shape)}")

    torch.onnx.export(
        wrapper,
        dummy,
        str(output),
        export_params=True,
        opset_version=OPSET,
        do_constant_folding=True,
        input_names=["images"],
        output_names=["cls_logits", "offset"],
        dynamic_axes=None,
    )

    print(f"ONNX 已导出：{output}")

    if SIMPLIFY:
        import onnx
        from onnxsim import simplify

        onnx_model = onnx.load(str(output))
        onnx_model_simplified, ok = simplify(onnx_model)

        if not ok:
            raise RuntimeError("onnxsim simplify 检查失败")

        sim_output = output.with_name(output.stem + "_sim.onnx")
        onnx.save(onnx_model_simplified, str(sim_output))
        print(f"简化 ONNX 已导出：{sim_output}")


if __name__ == "__main__":
    main()