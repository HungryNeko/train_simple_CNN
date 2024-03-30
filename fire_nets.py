import torch
from torch import nn

# 论文网络
net1 = nn.Sequential(
    nn.Conv2d(3, 16, kernel_size=3), nn.ReLU(),
    nn.AvgPool2d(kernel_size=2),
    nn.Dropout(p=0.5),

    nn.Conv2d(16, 32, kernel_size=3), nn.ReLU(),
    nn.AvgPool2d(kernel_size=2),
    nn.Dropout(p=0.5),

    nn.Conv2d(32, 64, kernel_size=3), nn.ReLU(),
    nn.AvgPool2d(kernel_size=2),
    nn.Dropout(p=0.5),

    nn.Flatten(),
    nn.Linear(2304, 256), nn.ReLU(),
    nn.Dropout(p=0.2),

    nn.Linear(256, 128), nn.ReLU(),

    nn.Linear(128, 2), nn.Sigmoid())

X = torch.randn(1, 3, 64, 64)
for layer in net1:
    X = layer(X)
    # print(layer.__class__.__name__, 'output shape:\t', X.shape)

# 同样数据集网络
net2 = nn.Sequential(
    nn.Conv2d(3, 64, kernel_size=3, padding=1), nn.ReLU(),
    nn.MaxPool2d(kernel_size=2),

    nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.ReLU(),

    nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.ReLU(),
    nn.MaxPool2d(kernel_size=2),

    nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(),
    nn.MaxPool2d(kernel_size=2),
    nn.Dropout(p=0.1),

    nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.ReLU(),

    nn.Conv2d(128, 64, kernel_size=3, padding=1), nn.ReLU(),
    nn.MaxPool2d(kernel_size=2),
    nn.Dropout(p=0.1),

    nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.ReLU(),
    nn.MaxPool2d(kernel_size=2),

    nn.Flatten(),
    nn.Linear(256, 64), nn.ReLU(),
    nn.Linear(64, 2), nn.Sigmoid())

X = torch.randn(1, 3, 64, 64)
for layer in net2:
    X = layer(X)
    # print(layer.__class__.__name__, 'output shape:\t', X.shape)

# 乱写的网络
net3 = nn.Sequential(
    nn.Conv2d(3, 16, kernel_size=3, padding=1, stride=2), nn.ReLU(),
    nn.MaxPool2d(kernel_size=2),
    nn.Dropout(p=0.2),
    nn.Conv2d(16, 64, kernel_size=3, padding=1, stride=2), nn.ReLU(),
    nn.MaxPool2d(kernel_size=2),
    nn.Dropout(p=0.5),
    nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=2), nn.ReLU(),
    nn.MaxPool2d(kernel_size=2),
    nn.Dropout(p=0.5),

    nn.Flatten(),
    nn.Linear(128, 64), nn.ReLU(),
    nn.Linear(64, 2), nn.Sigmoid()
)

X = torch.randn(1, 3, 64, 64)
for layer in net3:
    X = layer(X)
    #print(layer.__class__.__name__, 'output shape:\t', X.shape)

# 改进测试2
net4 = nn.Sequential(
    nn.Conv2d(3, 16, kernel_size=3, padding=1), nn.BatchNorm2d(16), nn.ReLU(),
    nn.MaxPool2d(kernel_size=2),
    nn.Dropout(p=0.2),
    nn.Conv2d(16, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
    nn.MaxPool2d(kernel_size=2),
    nn.Dropout(p=0.5),
    nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=2), nn.BatchNorm2d(128), nn.ReLU(),
    nn.MaxPool2d(kernel_size=2),
    nn.Dropout(p=0.5),

    nn.Flatten(),
    nn.Linear(2048, 512), nn.ReLU(),
    nn.Linear(512, 128), nn.ReLU(),
    nn.Linear(128, 2), nn.Sigmoid()
)

X = torch.randn(1, 3, 64, 64)
for layer in net4:
    X = layer(X)
    #print(layer.__class__.__name__, 'output shape:\t', X.shape)

#改进 128
net5 = nn.Sequential(
    nn.Conv2d(3, 16, kernel_size=3, padding=1), nn.BatchNorm2d(16), nn.ReLU(),
    nn.MaxPool2d(kernel_size=2),
    nn.Dropout(p=0.2),
    nn.Conv2d(16, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
    nn.MaxPool2d(kernel_size=2),
    nn.Dropout(p=0.5),
    nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=2), nn.BatchNorm2d(128), nn.ReLU(),
    nn.MaxPool2d(kernel_size=2),
    nn.Dropout(p=0.5),

    nn.Flatten(),
    nn.Linear(8192, 2048), nn.ReLU(),
    nn.Linear(2048, 512), nn.ReLU(),
    nn.Linear(512, 2), nn.Sigmoid()
)

X = torch.randn(1, 3, 128, 128)
for layer in net5:
    X = layer(X)
    #print(layer.__class__.__name__, 'output shape:\t', X.shape)

#改进 128 3类
net6 = nn.Sequential(
    nn.Conv2d(3, 16, kernel_size=3, padding=1), nn.BatchNorm2d(16), nn.ReLU(),
    nn.MaxPool2d(kernel_size=2),
    nn.Dropout(p=0.2),
    nn.Conv2d(16, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
    nn.MaxPool2d(kernel_size=2),
    nn.Dropout(p=0.5),
    nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=2), nn.BatchNorm2d(128), nn.ReLU(),
    nn.MaxPool2d(kernel_size=2),
    nn.Dropout(p=0.5),

    nn.Flatten(),
    nn.Linear(8192, 2048), nn.ReLU(),
    nn.Linear(2048, 512), nn.ReLU(),
    nn.Linear(512, 3), nn.Sigmoid()
)

X = torch.randn(1, 3, 128, 128)
for layer in net5:
    X = layer(X)
    #print(layer.__class__.__name__, 'output shape:\t', X.shape)

#net 4改3类
net7 = nn.Sequential(
        nn.Conv2d(3, 16, kernel_size=3, padding=1), nn.BatchNorm2d(16), nn.ReLU(),
        nn.MaxPool2d(kernel_size=2),
        nn.Dropout(p=0.2),
        nn.Conv2d(16, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
        nn.MaxPool2d(kernel_size=2),
        nn.Dropout(p=0.5),
        nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=2), nn.BatchNorm2d(128), nn.ReLU(),
        nn.MaxPool2d(kernel_size=2),
        nn.Dropout(p=0.5),

        nn.Flatten(),
        nn.Linear(8192, 2048), nn.ReLU(),
        nn.Linear(2048, 512), nn.ReLU(),
        nn.Linear(512, 3), nn.Sigmoid()
)

X = torch.randn(1, 3, 128, 128)
for layer in net7:
    X = layer(X)
    print(layer.__class__.__name__, 'output shape:\t', X.shape)


