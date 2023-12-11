import torch
import onnxruntime

# 加载导出的模型
onnx_file_path = r"D:\pythonProject\hirain\LSTM_hirain\LSTM.onnx"
ort_session = onnxruntime.InferenceSession(onnx_file_path)
x = torch.randn(3, 224, 224).numpy()
input_name = ort_session.get_inputs()[0].name
ort_inputs = {input_name: x}
ort_outs = ort_session.run(None, ort_inputs)
print("ONNX Model Output:")
print(ort_outs[0].shape)