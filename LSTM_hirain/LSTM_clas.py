import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder


# 定义LSTM模型
class ImageLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(ImageLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out, _ = self.lstm(x)
        last = out.shape[1]
        out = self.fc(out[:, last-1, :])  # 取序列最后一个时间步的输出
        return out


# 设置模型参数
input_size = 3  # 输入特征的数量，RGB图像为3
hidden_size = 128
num_classes = 10  # 假设有10个类别

# 初始化模型
model = ImageLSTM(input_size, hidden_size, num_classes)
x = torch.randn(3, 224, 224)
path = r'D:\pythonProject\hirain\LSTM.onnx'
torch.onnx.export(model, x, path, verbose=True, input_names=['input'], output_names=['output'])

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 数据预处理和加载
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# dataset = ImageFolder(root='path/to/your/dataset', transform=transform)
# dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
#
# # 训练模型
# num_epochs = 5
# for epoch in range(num_epochs):
#     for images, labels in dataloader:
#         # 将图像数据转换成LSTM需要的形状
#         images = images.view(-1, 224, 224 * 3)  # 将图像展平成一维序列
#
#         # 正向传播
#         outputs = model(images)
#         loss = criterion(outputs, labels)
#
#         # 反向传播和优化
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#
#     print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

