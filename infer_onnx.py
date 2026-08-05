# infer_onnx.py
import os
import site
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort


# =========================
# 固定配置
# =========================

ONNX_PATH = "/home/xhm/Desktop/ULTRALYTICS_LANE_ROBOT/runs/lane/train/weights/best.onnx"
IMAGE_PATH = "/home/xhm/Desktop/ULTRALYTICS_LANE_ROBOT/test"
SAVE_PATH = "/home/xhm/Desktop/ULTRALYTICS_LANE_ROBOT/test_lmm"

IMG_SIZE = 640

# 与训练配置保持一致
X_GRIDS = 160
ROW_ANCHORS = 56
NUM_LANES = 4
Y_START = 0.67
Y_END = 1.0

NO_LANE_CLASS = X_GRIDS
LANE_NAMES = ["lane_task_0", "lane_task_1", "lane_task_2", "lane_task_3"]
LANE_COLORS = [
    (0, 0, 255),
    (0, 255, 0),
    (255, 0, 0),
    (0, 255, 255),
]

# 后处理参数
CONF_THR = 0.15
NO_LANE_THR = 0.5
TOPK = 5
OFFSET_CLIP = 0.5

# resize 与你当前验证正确的版本一致
PREPROCESS_MODE = "resize"  # "resize" 或 "letterbox"


# =========================
# CUDA / cuDNN 动态库路径
# =========================

def add_nvidia_lib_paths():
    """
    onnxruntime-gpu 1.19.2 没有 ort.preload_dlls。
    这里把 pip 安装的 nvidia 动态库目录加入 LD_LIBRARY_PATH。
    需要在创建 InferenceSession 前执行。
    """
    targets = [
        "nvidia/cudnn/lib",
        "nvidia/cublas/lib",
        "nvidia/cuda_runtime/lib",
        "nvidia/cuda_nvrtc/lib",
    ]

    paths = []
    for sp in site.getsitepackages():
        base = Path(sp)
        for t in targets:
            d = base / t
            if d.exists():
                paths.append(str(d))

    old = os.environ.get("LD_LIBRARY_PATH", "")
    old_parts = [p for p in old.split(":") if p]
    new_parts = paths + old_parts

    seen = set()
    dedup = []
    for p in new_parts:
        if p not in seen:
            dedup.append(p)
            seen.add(p)

    os.environ["LD_LIBRARY_PATH"] = ":".join(dedup)


# =========================
# 基础工具
# =========================

def softmax(x: np.ndarray, axis: int = 0) -> np.ndarray:
    x = x.astype(np.float32)
    x = x - np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def preprocess_resize(img_bgr: np.ndarray):
    """
    直接 resize 到 640x640。
    坐标还原:
      x_ori = x640 / 640 * original_width
      y_ori = y640 / 640 * original_height
    """
    h0, w0 = img_bgr.shape[:2]

    img_640 = cv2.resize(img_bgr, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)

    img_rgb = cv2.cvtColor(img_640, cv2.COLOR_BGR2RGB)
    x = img_rgb.astype(np.float32) / 255.0
    x = np.transpose(x, (2, 0, 1))[None]

    meta = {
        "mode": "resize",
        "orig_w": w0,
        "orig_h": h0,
        "scale_x": w0 / IMG_SIZE,
        "scale_y": h0 / IMG_SIZE,
    }

    return x, img_640, meta


def preprocess_letterbox(img_bgr: np.ndarray):
    """
    letterbox 到 640x640。
    当前你确认 resize 结果正确，默认不用这个分支。
    """
    h0, w0 = img_bgr.shape[:2]

    r = min(IMG_SIZE / h0, IMG_SIZE / w0)
    new_w = int(round(w0 * r))
    new_h = int(round(h0 * r))

    resized = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    canvas = np.full((IMG_SIZE, IMG_SIZE, 3), 114, dtype=np.uint8)
    dw = (IMG_SIZE - new_w) // 2
    dh = (IMG_SIZE - new_h) // 2
    canvas[dh:dh + new_h, dw:dw + new_w] = resized

    img_rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    x = img_rgb.astype(np.float32) / 255.0
    x = np.transpose(x, (2, 0, 1))[None]

    meta = {
        "mode": "letterbox",
        "orig_w": w0,
        "orig_h": h0,
        "ratio": r,
        "pad_x": dw,
        "pad_y": dh,
    }

    return x, canvas, meta


def preprocess(img_bgr: np.ndarray):
    if PREPROCESS_MODE == "resize":
        return preprocess_resize(img_bgr)
    if PREPROCESS_MODE == "letterbox":
        return preprocess_letterbox(img_bgr)
    raise ValueError(f"Unsupported PREPROCESS_MODE: {PREPROCESS_MODE}")


def to_original_xy(x640: float, y640: float, meta: dict):
    """
    640 输入尺度坐标还原到原图尺度。
    """
    if meta["mode"] == "resize":
        x = x640 * meta["scale_x"]
        y = y640 * meta["scale_y"]

    elif meta["mode"] == "letterbox":
        x = (x640 - meta["pad_x"]) / meta["ratio"]
        y = (y640 - meta["pad_y"]) / meta["ratio"]

    else:
        raise ValueError(f"Unsupported preprocess mode: {meta['mode']}")

    x = float(np.clip(x, 0, meta["orig_w"] - 1))
    y = float(np.clip(y, 0, meta["orig_h"] - 1))

    return x, y


# =========================
# ONNX 推理与后处理
# =========================

def create_session(onnx_path: str):
    add_nvidia_lib_paths()

    providers = []
    available = ort.get_available_providers()

    if "CUDAExecutionProvider" in available:
        providers.append("CUDAExecutionProvider")
    providers.append("CPUExecutionProvider")

    session = ort.InferenceSession(onnx_path, providers=providers)
    return session


def decode_lane_outputs(cls_logits: np.ndarray, offset: np.ndarray):
    """
    输入:
      cls_logits: [1, 161, 56, 4]
      offset:     [1, 1, 56, 4]

    输出:
      lanes: list[list[dict]]
      每条 lane 固定输出 56 行记录。

    关键方向:
      训练标签从图像上方到下方写入。
      模型 row 索引和可视化 row 方向相反。
      所以这里使用:
        draw_row = ROW_ANCHORS - 1 - row
    """
    if cls_logits.ndim != 4:
        raise ValueError(f"cls_logits shape error: {cls_logits.shape}")
    if offset.ndim != 4:
        raise ValueError(f"offset shape error: {offset.shape}")

    b, c, r, l = cls_logits.shape

    if b != 1:
        raise ValueError(f"Only batch=1 supported, got batch={b}")
    if c != X_GRIDS + 1:
        raise ValueError(f"Expected cls dim {X_GRIDS + 1}, got {c}")
    if r != ROW_ANCHORS:
        raise ValueError(f"Expected row anchors {ROW_ANCHORS}, got {r}")
    if l != NUM_LANES:
        raise ValueError(f"Expected num lanes {NUM_LANES}, got {l}")

    probs_all = softmax(cls_logits, axis=1)

    visible_probs = probs_all[:, :X_GRIDS, :, :]
    no_lane_probs = probs_all[:, NO_LANE_CLASS, :, :]

    lanes = []

    for lane_i in range(NUM_LANES):
        lane_rows = []

        for row in range(ROW_ANCHORS):
            p_visible = visible_probs[0, :, row, lane_i]
            p_no_lane = float(no_lane_probs[0, row, lane_i])

            cls_id = int(np.argmax(p_visible))
            conf = float(p_visible[cls_id])

            visible = bool((p_no_lane < NO_LANE_THR) and (conf >= CONF_THR))

            if TOPK is not None and TOPK > 1:
                k = min(int(TOPK), X_GRIDS)
                top_idx = np.argpartition(-p_visible, k - 1)[:k]
                top_idx = top_idx[np.argsort(top_idx)]

                top_p = p_visible[top_idx].astype(np.float64)
                denom = float(np.sum(top_p))

                if denom > 1e-12:
                    x_grid_base = float(np.sum(top_p * top_idx) / denom)
                else:
                    x_grid_base = float(cls_id)
            else:
                x_grid_base = float(cls_id)

            off = float(offset[0, 0, row, lane_i])
            off = float(np.clip(off, -OFFSET_CLIP, OFFSET_CLIP))

            x_grid = float(np.clip(x_grid_base + off, 0.0, X_GRIDS - 1.0))
            x640 = x_grid / (X_GRIDS - 1.0) * (IMG_SIZE - 1.0)

            draw_row = ROW_ANCHORS - 1 - row
            y_norm = Y_START + draw_row / (ROW_ANCHORS - 1.0) * (Y_END - Y_START)
            y640 = y_norm * (IMG_SIZE - 1.0)

            lane_rows.append(
                {
                    "row": row,
                    "draw_row": draw_row,
                    "visible": visible,
                    "cls": cls_id,
                    "offset": off,
                    "x_grid_base": x_grid_base,
                    "x_grid": x_grid,
                    "x640": float(x640),
                    "y640": float(y640),
                    "conf": conf,
                    "p_no_lane": p_no_lane,
                }
            )

        # 按图像 y 从上到下打印和绘制
        lane_rows = sorted(lane_rows, key=lambda p: p["y640"])
        lanes.append(lane_rows)

    return lanes


def draw_lanes_on_original(img_bgr: np.ndarray, lanes, meta: dict):
    """
    只在原图上画可见点，不画连线。
    """
    vis = img_bgr.copy()

    for lane_i, points in enumerate(lanes):
        for p in points:
            if not p["visible"]:
                continue

            x, y = to_original_xy(p["x640"], p["y640"], meta)

            xi = int(round(x))
            yi = int(round(y))

            cv2.circle(vis, (xi, yi), 5, LANE_COLORS[lane_i % len(LANE_COLORS)], -1)

    return vis


def print_all_rows(lanes, meta: dict):
    print("\n最终车道线 56 行输出：")

    for lane_i, points in enumerate(lanes):
        valid_count = sum(1 for p in points if p["visible"])
        print(f"{LANE_NAMES[lane_i]}: total_rows={len(points)}, visible_rows={valid_count}")

        for j, p in enumerate(points):
            x0, y0 = to_original_xy(p["x640"], p["y640"], meta)

            visible_flag = 1 if p["visible"] else 0
            x_print = x0 if p["visible"] else -1.0

            print(
                f"  {j:02d}: "
                f"visible={visible_flag}, "
                f"row={p['row']:02d}, draw_row={p['draw_row']:02d}, "
                f"x={x_print:.2f}, y={y0:.2f}, "
                f"x640={p['x640']:.2f}, y640={p['y640']:.2f}, "
                f"conf={p['conf']:.4f}, no_lane={p['p_no_lane']:.4f}, "
                f"cls={p['cls']}, offset={p['offset']:.4f}, "
                f"x_grid={p['x_grid']:.4f}"
            )


# =========================
# 主流程
# =========================

def main():
    if not os.path.exists(ONNX_PATH):
        raise FileNotFoundError(f"ONNX not found: {ONNX_PATH}")

    if not os.path.exists(IMAGE_PATH):
        raise FileNotFoundError(f"Image not found: {IMAGE_PATH}")

    print(f"ONNXRuntime version: {ort.__version__}")
    print(f"Available providers: {ort.get_available_providers()}")

    session = create_session(ONNX_PATH)
    print(f"Session providers: {session.get_providers()}")

    img_bgr = cv2.imread(IMAGE_PATH)
    if img_bgr is None:
        raise RuntimeError(f"Failed to read image: {IMAGE_PATH}")

    h0, w0 = img_bgr.shape[:2]
    print(f"Original image size: {w0}x{h0}")
    print(f"Preprocess mode: {PREPROCESS_MODE}")

    inp, img_640, meta = preprocess(img_bgr)

    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: inp})

    if len(outputs) != 2:
        raise RuntimeError(f"Expected 2 ONNX outputs, got {len(outputs)}")

    cls_logits, offset = outputs

    print("\nONNX 输出：")
    print(f"  cls_logits: {cls_logits.shape} {cls_logits.dtype}")
    print(f"  offset:     {offset.shape} {offset.dtype}")

    lanes = decode_lane_outputs(cls_logits, offset)

    print_all_rows(lanes, meta)

    vis_ori = draw_lanes_on_original(img_bgr, lanes, meta)
    cv2.imwrite(SAVE_PATH, vis_ori)

    print(f"\n原图点结果已保存：{SAVE_PATH}")


if __name__ == "__main__":
    main()