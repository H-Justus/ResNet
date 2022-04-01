# -*- coding: utf-8 -*-
# @Time    : 2022/4/1 15:56
# @Author  : Justus
# @FileName: ResidualNet_conv shortcut.py
# @Software: PyCharm

from torchvision import transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import datasets
import torch
import torch.nn.functional


# 准备数据集
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.1307, ), (0.3081, ))
                                ])
train_dataset = datasets.MNIST(root="./mnist", train=True,
                               transform=transform,
                               download=True)
test_dataset = datasets.MNIST(root="./mnist",
                              train=False,
                              transform=transform,
                              download=True)
train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)


# 残差块
class ResidualBlock(torch.nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.channels = channels
        self.conv1 = torch.nn.Conv2d(channels, channels, kernel_size=(3, 3), padding=1)
        self.conv2 = torch.nn.Conv2d(channels, channels, kernel_size=(3, 3), padding=1)
        self.conv3 = torch.nn.Conv2d(channels, channels, kernel_size=(1, 1))
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        y = self.relu(self.conv1(x))
        y = self.conv2(y)
        x = self.conv3(x)
        return self.relu(x + y)


# 定义模型
class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 16, kernel_size=(5, 5))
        self.conv2 = torch.nn.Conv2d(16, 32, kernel_size=(5, 5))
        self.m_pooling = torch.nn.MaxPool2d(2)

        self.r_block1 = ResidualBlock(16)
        self.r_block2 = ResidualBlock(32)

        self.flatten = torch.nn.Linear(512, 10)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        batch_size = x.size(0)
        x = self.relu(self.conv1(x))
        x = self.m_pooling(x)
        x = self.r_block1(x)
        x = self.relu(self.conv2(x))
        x = self.m_pooling(x)
        x = self.r_block2(x)
        x = x.view(batch_size, -1)
        x = self.flatten(x)
        return x


model = Net()

# GPU加速
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

# 交叉熵损失已包含
criterion = torch.nn.CrossEntropyLoss()
# SGD优化器, momentum冲量值
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)


# 训练
def train(epoch):
    running_loss = 0.0
    for batch_idx, data in enumerate(train_loader, 0):
        # 加载数据
        inputs, target = data
        # inputs和target迁移到GPU，注意要在同一块显卡上
        inputs, target = inputs.to(device), target.to(device)
        # 预测
        outputs = model(inputs)
        # 计算损失
        loss = criterion(outputs, target)
        # 反向传播
        loss.backward()
        # 更新权重
        optimizer.step()
        # 梯度清零
        optimizer.zero_grad()

        running_loss += loss.item()
        if batch_idx % 300 == 299:
            print("[%d, %4d]loss:%.3f" % (epoch+1, batch_idx+1, running_loss/300))


# 测试
def test():
    correct = 0
    total = 0
    # 不计算梯度，强制之后的内容不进行计算图构建
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            # inputs和target迁移到GPU，注意要在同一块显卡上
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            # 取每行最大值的下标，dim=1按行
            _, predicted = torch.max(outputs.data, dim=1)
            # 取labels的第0个元素，total最终值为样本总数
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print("Accuracy on test set:%d %%" % (100*correct/total))


if __name__ == '__main__':
    for epoch in range(5):
        train(epoch)
        test()
