import numpy as np
import onnxruntime

onnx_path = r"D:\pythonProject\hirain\CNN_11.onnx"
session = onnxruntime.InferenceSession(onnx_path)
input_name = session.get_inputs()[0].name
print(input_name)
input = np.random.randn(1, 3, 224, 224).astype(np.float32)
output = session.run(None, {input_name: input})
print(output[0])

