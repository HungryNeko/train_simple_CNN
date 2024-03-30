import time
import torch
from sklearn.model_selection import KFold
from torch import nn
from torch.optim import lr_scheduler
import K_split
#from memory_profiler import profile


#@profile
def k_fold(k, num_epochs, device, batch_size, net, lr, allData, allLabel):
    kf = KFold(n_splits=k, shuffle=True)
    Ktrain_min_l = []
    Ktrain_acc_max_l = []
    Ktest_acc_max_l = []
    i = 0
    train_data = []
    train_label = []
    test_data = []
    test_label = []
    for X_train, X_test in kf.split(allData, allLabel):  # k折检验
        if i != 0:  # 重新获取数据
            train_data.clear()
            train_label.clear()
            test_label.clear()
            test_data.clear()
        # 数据分化
        for index in X_train:
            train_data.append(allData[index])
            train_label.append(allLabel[index])
        for index in X_test:
            test_label.append(allLabel[index])
            test_data.append(allData[index])
        # 优化器
        optimizer = torch.optim.SGD(net.parameters(), lr=lr)
        loss = torch.nn.CrossEntropyLoss()
        # 数据集->读取器
        train_dataset = K_split.KfoldDataset(is_train=True, datas=train_data, label=train_label)
        test_dataset = K_split.KfoldDataset(is_train=False, datas=test_data, label=test_label)
        train_iter = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True,
                                                 num_workers=8)
        test_iter = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True,
                                                num_workers=8)

        # 修改train函数，使其返回每一批次的准确率，tarin_ls用列表表示
        # 学习率
        # scheduler=optmizer.CosineScheduler(max_update=30, base_lr=lr, final_lr=0.1)
        # scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[10, 20, 40, 60], gamma=0.5)#对于数据集1
        # scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[10, 20, 40, 60], gamma=0.5)#对于大数据集2
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[20, 40, 60, 80], gamma=0.5)  # 对于数据集3类
        # scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[40, 80], gamma=0.5)  # 数据增广更新
        loss_min, train_acc_max, test_acc_max = train(i, train_iter, test_iter, net, loss, optimizer, device,
                                                      num_epochs, scheduler=scheduler)
        i += 1

        Ktrain_min_l.append(loss_min)
        Ktrain_acc_max_l.append(train_acc_max)
        Ktest_acc_max_l.append(test_acc_max)
    return sum(Ktrain_min_l) / len(Ktrain_min_l), sum(Ktrain_acc_max_l) / len(Ktrain_acc_max_l), sum(
        Ktest_acc_max_l) / len(Ktest_acc_max_l)


def evaluate_accuracy(data_iter, net, device):
    acc_sum, n = 0.0, 0
    with torch.no_grad():
        for X, y in data_iter:
            if isinstance(net, torch.nn.Module):
                net.eval()  # 评估模式关闭dropout
                acc_sum += (net(X.to(device)).argmax(dim=1) == y.to(device)).float().sum().cpu().item()
                net.train()  # 改回训练模式
            else:
                if ('is_training' in net.__code__.co_varnames):  # 测试时停用isTrain
                    acc_sum += (net(X, is_training=False).argmax(dim=1) == y).float().sum().item()
                else:
                    acc_sum += (net(X).argmax(dim=1) == y).float().sum().item()
            n += y.shape[0]
    return acc_sum / n


def weight_reset(m):  # 每折重置训练参数
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        m.reset_parameters()


def train(i, train_iter, test_iter, net, loss, optimizer, device, num_epochs, scheduler=None):
    net = net.to(device)
    # 检验前重置参数
    net.apply(weight_reset)
    print("training on ", device)
    start = time.time()
    test_acc_max_l = []
    train_acc_max_l = []
    train_l_min_l = []
    test_acc_max = 0
    for epoch in range(num_epochs):  # 迭代100次

        batch_count = 0
        train_l_sum, train_acc_sum, test_acc_sum, n = 0.0, 0.0, 0.0, 0
        for X, y in train_iter:
            X = X.to(device)
            y = y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            train_l_sum += l.cpu().item()
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().cpu().item()
            n += y.shape[0]
            batch_count += 1
        # epoch完成

        test_acc_sum = evaluate_accuracy(test_iter, net, device)
        train_l_min_l.append(train_l_sum / batch_count)
        train_acc_max_l.append(train_acc_sum / n)
        test_acc_max_l.append(test_acc_sum)
        # 打印lr和精度
        lr = optimizer.state_dict()['param_groups'][0]['lr']
        print('fold %d epoch %d, loss %.4f, train acc %.3f, test acc %.3f, lr %.5f'
              % (i + 1, epoch + 1, train_l_sum / batch_count, train_acc_sum / n, test_acc_sum, lr))
        # 保存
        if test_acc_max_l[-1] > test_acc_max:
            test_acc_max = test_acc_max_l[-1]
            torch.save(net.state_dict(), "./K{:}_bird_model_best.pt".format(i + 1))
            print("saving K{:}_bird_model_best.pt ".format(i))
        # 学习率优化
        if scheduler:
            if scheduler.__module__ == lr_scheduler.__name__:
                scheduler.step()
            else:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = scheduler(epoch)

    # 打印训练结果
    index_max = test_acc_max_l.index(max(test_acc_max_l))

    print('fold %d, train_loss_min %.4f, train acc max%.4f, test acc max %.4f, time %.1f sec'
          % (
              i + 1, train_l_min_l[index_max], train_acc_max_l[index_max], test_acc_max_l[index_max],
              time.time() - start))

    return train_l_min_l[index_max], train_acc_max_l[index_max], test_acc_max_l[index_max]
