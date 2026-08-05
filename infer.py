# infer_camera_onnx.py
import os
import site
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort


# =========================
# 固定配置
# =========================

ONNX_PATH = "/home/baater/ultralytics/runs/lane/train-10/weights/best.onnx"

CAMERA_ID = "/dev/video2"
CAP_WIDTH = 1920
CAP_HEIGHT = 1080
CAP_FPS = 30

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

# 你前面确认 resize 是正确的
PREPROCESS_MODE = "resize"

# 显示窗口大小
DISPLAY_WIDTH = 640
DISPLAY_HEIGHT = 360

WINDOW_NAME = "Lane ONNX Camera"


# =========================
# CUDA / cuDNN 动态库路径
# =========================

def add_nvidia_lib_paths():
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

    return x, meta


def preprocess_letterbox(img_bgr: np.ndarray):
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

    return x, meta


def preprocess(img_bgr: np.ndarray):
    if PREPROCESS_MODE == "resize":
        return preprocess_resize(img_bgr)
    if PREPROCESS_MODE == "letterbox":
        return preprocess_letterbox(img_bgr)
    raise ValueError(f"Unsupported PREPROCESS_MODE: {PREPROCESS_MODE}")


def to_original_xy(x640: float, y640: float, meta: dict):
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

    available = ort.get_available_providers()
    providers = []

    if "CUDAExecutionProvider" in available:
        providers.append("CUDAExecutionProvider")

    providers.append("CPUExecutionProvider")

    session = ort.InferenceSession(onnx_path, providers=providers)
    return session


def decode_lane_outputs(cls_logits: np.ndarray, offset: np.ndarray):
    """
    cls_logits: [1, 161, 56, 4]
    offset:     [1, 1, 56, 4]
    """
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

            # 保持和你已经验证正确的后处理一致
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
                    "x_grid": x_grid,
                    "x640": float(x640),
                    "y640": float(y640),
                    "conf": conf,
                    "p_no_lane": p_no_lane,
                }
            )

        lane_rows = sorted(lane_rows, key=lambda p: p["y640"])
        lanes.append(lane_rows)

    return lanes


def draw_lanes_on_original(img_bgr: np.ndarray, lanes, meta: dict):
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


def put_info(vis: np.ndarray, fps: float, valid_points: int):
    text1 = f"FPS: {fps:.1f}"
    text2 = f"points: {valid_points}"

    cv2.putText(
        vis,
        text1,
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )

    cv2.putText(
        vis,
        text2,
        (20, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )

    return vis


# =========================
# 摄像头
# =========================

def open_camera():
    cap = cv2.VideoCapture(CAMERA_ID, cv2.CAP_V4L2)

    if not cap.isOpened():
        raise RuntimeError(f"无法打开摄像头: {CAMERA_ID}")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAP_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAP_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, CAP_FPS)

    # 优先 MJPG，很多 USB 摄像头 1080p30 需要 MJPG
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))

    real_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    real_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    real_fps = cap.get(cv2.CAP_PROP_FPS)

    print(f"Camera opened: {CAMERA_ID}")
    print(f"Requested: {CAP_WIDTH}x{CAP_HEIGHT}@{CAP_FPS}")
    print(f"Actual:    {real_w}x{real_h}@{real_fps:.2f}")

    return cap


# =========================
# 主流程
# =========================

def main():
    if not os.path.exists(ONNX_PATH):
        raise FileNotFoundError(f"ONNX not found: {ONNX_PATH}")

    print(f"ONNXRuntime version: {ort.__version__}")
    print(f"Available providers: {ort.get_available_providers()}")

    session = create_session(ONNX_PATH)
    input_name = session.get_inputs()[0].name

    print(f"Session providers: {session.get_providers()}")
    print(f"Input name: {input_name}")
    print(f"Preprocess mode: {PREPROCESS_MODE}")

    cap = open_camera()

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, DISPLAY_WIDTH, DISPLAY_HEIGHT)

    last_time = cv2.getTickCount()
    freq = cv2.getTickFrequency()
    fps_smooth = 0.0

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                print("读取摄像头帧失败")
                break

            inp, meta = preprocess(frame)

            outputs = session.run(None, {input_name: inp})
            if len(outputs) != 2:
                raise RuntimeError(f"Expected 2 ONNX outputs, got {len(outputs)}")

            cls_logits, offset = outputs
            lanes = decode_lane_outputs(cls_logits, offset)

            valid_points = sum(
                1
                for lane in lanes
                for p in lane
                if p["visible"]
            )

            vis = draw_lanes_on_original(frame, lanes, meta)

            now = cv2.getTickCount()
            dt = (now - last_time) / freq
            last_time = now

            fps = 1.0 / dt if dt > 1e-6 else 0.0
            if fps_smooth <= 0:
                fps_smooth = fps
            else:
                fps_smooth = 0.9 * fps_smooth + 0.1 * fps

            vis = put_info(vis, fps_smooth, valid_points)

            display = cv2.resize(
                vis,
                (DISPLAY_WIDTH, DISPLAY_HEIGHT),
                interpolation=cv2.INTER_AREA,
            )

            cv2.imshow(WINDOW_NAME, display)

            key = cv2.waitKey(1) & 0xFF

            # q 或 ESC 退出
            if key == ord("q") or key == 27:
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()