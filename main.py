if __name__ == '__main__':
    import torch
    import K_split
    import fire_nets
    from train_net import k_fold



    root = '../byHandDataset(3_class)'  # 指定数据库目录
    K_split.getData(root)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    net = fire_nets.net7

    lr = 0.3
    k = 5

    num_epochs = 100
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 64

    # 训练

    loss_k, train_k, valid_k = k_fold(k, num_epochs, device, batch_size, net, lr=lr, allData=K_split.allData,
                                      allLabel=K_split.allLabel)

    print('%d-fold validation: min loss rmse %.5f, max train rmse %.5f,max test rmse %.5f' % (
    k, loss_k, train_k, valid_k))
