import os
import torch
import torchvision
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import transforms, Normalize



allData = []
allLabel = []


def getData(root):  # 读取所有数据到列表
    l = 0  # 图片label
    for name in ['fire', 'smoke', 'safe']:
        for i in os.listdir(root + '/' + name):
            allData.append(root + '/' + name + '/' + i)
            allLabel.append(l)
        l += 1
    #print(allLabel)


normalize = Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))


class KfoldDataset(torch.utils.data.Dataset):  # 创建dataset类用于五折检验传递数据
    def __init__(self, is_train, datas, label):
        super(KfoldDataset, self).__init__()
        imgs = []
        count = 0
        for data in datas:
            # print(line)
            imgs.append((data, int(label[count])))  # 读取图片位置和标签
            count += 1
        self.imgs = imgs
        self.is_train = is_train
        # 图像增广
        if self.is_train:
            self.train_tsf = torchvision.transforms.Compose([
                transforms.RandomHorizontalFlip(),  # 随机水平翻转
                # transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1),
                # brightness: 亮度调整因子contrast: 对比度参数saturation: 饱和度参数
                transforms.Resize([128, 128]),
                # transforms.RandomResizedCrop((64, 64), scale=(0.8, 1), ratio=(0.5, 2)),  # 64*64，面积80%-100%，宽高比0.5-2
                transforms.ToTensor(),
                normalize,
            ])
        else:
            self.test_tsf = torchvision.transforms.Compose([
                # transforms.RandomHorizontalFlip(),  # 随机水平翻转
                transforms.Resize([128, 128]),
                # transforms.RandomResizedCrop((64, 64), scale=(0.8, 1), ratio=(0.5, 2)),  # 64*64，面积80%-100%，宽高比0.5-2
                transforms.ToTensor(),
                normalize,
            ])

    def __getitem__(self, index):  # 根据索引读取图片
        feature, label = self.imgs[index]  # 读取图片位置和label
        feature = Image.open(feature).convert('RGB')  # 按照path读入图片
        if self.is_train:
            feature = self.train_tsf(feature)
        else:
            feature = self.test_tsf(feature)
        return feature, label

    def __len__(self):  # 返回数据集大小
        return len(self.imgs)
