import numpy as np
import onnxruntime as ort

onnx_path = "/home/baater/ultralytics/runs/lane/train-10/weights/best.onnx"

print("ONNXRuntime version:", ort.__version__)
print("ONNXRuntime device:", ort.get_device())
print("Available providers:", ort.get_available_providers())

session = ort.InferenceSession(
    onnx_path,
    providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
)

print("Session providers:", session.get_providers())

img = np.zeros((1, 3, 640, 640), dtype=np.float32)

outputs = session.run(None, {"images": img})

for i, out in enumerate(outputs):
    print(i, out.shape, out.dtype)