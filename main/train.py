# -*- coding: utf-8 -*-
from __future__ import print_function
from __future__ import division
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.autograd import Variable
# from model import SRNet
# from srnet import SRNet
from SR_GXC import SR_Net
from tensorboardX import SummaryWriter
from time import *

writer = SummaryWriter(log_dir='logs')
import utils
import os
from torch.backends import cudnn

# 在文件头部添加最优传输距离OT库
import ot  # Python Optimal Transport库 (需提前安装: pip install POT)
import numpy as np

parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=16, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=32, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=500, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--weight_decay', type=float, default=0.0005, metavar='wd',
                    help='weight_decay (default: 0.001)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='how many batches to wait before logging training status')

# 在args中添加OT相关参数
parser.add_argument('--ot-reg', type=float, default=0.5,
                    help='Sinkhorn regularization coefficient')
parser.add_argument('--ot-lambda', type=float, default=0.1,
                    help='Weight for OT loss term')

# cuda related
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
else:
    args.gpu = None

kwargs = {'num_workers': 0, 'pin_memory': True} if args.cuda else {}

valid_path = r"/root/autodl-tmp/SRnet/data-path/B-S-valid-path"
test_path = r"/root/autodl-tmp/SRnet/data-path/B-S-test-path"
train_cover_path = r"/root/autodl-tmp/SRnet/data-path/A-S-train-path/cover"
train_stego_path = r"/root/autodl-tmp/SRnet/data-path/A-S-train-path/stego"

print('torch ', torch.__version__)
print('train_path = ', train_cover_path)
print('valid_path = ', valid_path)
print('test_path = ', test_path)
print('train_batch_size = ', args.batch_size)
print('test_batch_size = ', args.test_batch_size)

train_transform = transforms.Compose([utils.AugData(), utils.ToTensor()])
train_data = utils.DatasetPair(train_cover_path, train_stego_path, train_transform)
valid_data = datasets.ImageFolder(valid_path,
                                  transform=transforms.Compose([transforms.Grayscale(), transforms.ToTensor()]))
# valid_data= datasets.ImageFolder(valid_path, transform=transforms.Compose([transforms.Grayscale(),transforms.ToTensor()]))
test1_data = datasets.ImageFolder(test_path,
                                  transform=transforms.Compose([transforms.Grayscale(), transforms.ToTensor()]))
train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True, **kwargs)
valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=args.test_batch_size, shuffle=False, **kwargs)
test1_loader = torch.utils.data.DataLoader(test1_data, batch_size=args.test_batch_size, shuffle=True, **kwargs)

model = SR_Net()
# print(model)
# print(summary(model.cuda(),(1,256,256)))
if args.cuda:
    model.cuda()
cudnn.benchmark = True
# optimizer = optim.SGD(model.parameters(), lr=args.lr,weight_decay=args.weight_decay,momentum=args.momentum)#
optimizer = optim.Adamax(model.parameters(), lr=0.001, weight_decay=0)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=114, gamma=0.1)

trainAcc_txt = r'/root/autodl-tmp/SRnet/OT/OT成对训练txt/A-B_S_train-acc2.txt'
testAcc_txt = r'/root/autodl-tmp/SRnet/OT/OT成对训练txt/A-B_S_test-acc2.txt'
validAcc_txt = r'/root/autodl-tmp/SRnet/OT/OT成对训练txt/A-B_S_valid-acc2.txt'
# from datetime import datetime+
import datetime


# 最优传输距离计算函数
def compute_ot_loss(source_features, target_features, reg=0.1):
    """
    计算源域与目标域特征之间的Sinkhorn散度（近似OT距离）
    参数：
        source_features: 源域特征矩阵 [batch_size, feature_dim]
        target_features: 目标域特征矩阵 [batch_size, feature_dim]
        reg: Sinkhorn正则化系数
    返回：
        ot_loss: 最优传输损失值
    """
    # 计算成本矩阵（此处使用余弦相似度的负值作为成本）
    cost_matrix = 1 - torch.cosine_similarity(
        source_features.unsqueeze(1), target_features.unsqueeze(0), dim=2
    )

    # 转换为NumPy数组（POT库需要）
    cost_matrix_np = cost_matrix.detach().cpu().numpy()

    # 计算均匀分布的权重
    n, m = source_features.shape[0], target_features.shape[0]
    a, b = np.ones(n) / n, np.ones(m) / m

    # 使用Sinkhorn算法计算OT距离
    transport_plan = ot.sinkhorn(a, b, cost_matrix_np, reg=reg)
    ot_loss = np.sum(transport_plan * cost_matrix_np)

    return torch.tensor(ot_loss, device=source_features.device)


def train(epoch):
    lr_train = (optimizer.state_dict()['param_groups'][0]['lr'])
    print(lr_train)

    model.train()

    #    for batch_idx, ((data1, target1),(data2, target2),(path1,path2)) in enumerate(train_loader):
    for batch_idx, data in enumerate(train_loader):
        datas, labels = data['images'], data['labels']
        if args.cuda:
            datas, labels = datas.cuda(), labels.cuda()

        # 调整数据形状
        if batch_idx == len(train_loader) - 1:
            last_batch_size = len(os.listdir(train_cover_path)) - args.batch_size * (len(train_loader) - 1)
            datas = datas.view(last_batch_size * 2, 1, 256, 256)
            labels = labels.view(last_batch_size * 2)
            current_batch_size = last_batch_size
        else:
            datas = datas.view(args.batch_size * 2, 1, 256, 256)
            labels = labels.view(args.batch_size * 2)
            current_batch_size = args.batch_size

        optimizer.zero_grad()
        outputs, features = model(datas)  # 获取特征和输出

        # 分割 Cover 和 Stego 特征
        cover_features = features[:current_batch_size]
        stego_features = features[current_batch_size:]

        # 计算交叉熵损失和 OT 损失
        ce_loss = nn.CrossEntropyLoss()(outputs, labels)
        ot_loss = compute_ot_loss(cover_features, stego_features)
        total_loss = ce_loss + args.ot_lambda * ot_loss  # 组合损失

        total_loss.backward()
        optimizer.step()

        if (batch_idx + 1) % args.log_interval == 0:
            b_pred = outputs.max(1, keepdim=True)[1]
            b_correct = b_pred.eq(labels.view_as(b_pred)).sum().item()

            b_accu = b_correct / (labels.size(0))

            result_train = 'Time:{} Train Epoch: {} [{}/{} ({:.0f}%)]\ttrain_accuracy: {:.6f}\tLoss: {:.6f}'.format(
                datetime.datetime.now(), epoch, (batch_idx + 1) * len(data), len(train_loader.dataset),
                                                100. * (batch_idx + 1) / len(train_loader), b_accu, total_loss.item())
            with open(trainAcc_txt, "a+") as f:
                f.write(result_train + '\n')
                f.close()

            print('Train Epoch: {} [{}/{} ({:.0f}%)]\ttrain_accuracy: {:.6f}\tLoss: {:.6f}'.format(
                epoch, (batch_idx + 1) * len(data), len(train_loader.dataset),
                       100. * (batch_idx + 1) / len(train_loader), b_accu, total_loss.item()))
    scheduler.step()
    # writer.add_scalar('Train_loss', loss ,epoch)


def test():
    model.eval()
    test1_loss = 0
    correct = 0.
    with torch.no_grad():
        for data, target in test1_loader:
            if args.cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)
            output, _ = model(data)
            test1_loss += F.nll_loss(F.log_softmax(output, dim=1), target, reduction='sum').item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

    test1_loss /= len(test1_loader.dataset)

    result_test = 'Time:{} Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.6f}%)'.format(
        datetime.datetime.now(), test1_loss, correct, len(test1_loader.dataset),
        100. * correct / len(test1_loader.dataset))
    with open(testAcc_txt, "a+") as f:
        f.write(result_test + '\n')
        f.close()

    print('Test1 set: Average loss: {:.4f}, Accuracy: {}/{} ({:.6f}%)\n'.format(
        test1_loss, correct, len(test1_loader.dataset),
        100. * correct / len(test1_loader.dataset)))
    accu = float(correct) / len(test1_loader.dataset)
    return accu, test1_loss


def valid():
    model.eval()
    valid_loss = 0
    correct = 0.
    with torch.no_grad():
        for data, target in valid_loader:
            if args.cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)
            output = model(data)
            valid_loss += F.nll_loss(F.log_softmax(output, dim=1), target, reduction='sum').item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

    valid_loss /= len(valid_loader.dataset)

    result_valid = 'Time:{} Valid set: Average loss: {:.4f}, Accuracy: {}/{} ({:.6f}%)'.format(
        datetime.now(), valid_loss, correct, len(valid_loader.dataset),
        100. * correct / len(valid_loader.dataset))
    with open(validAcc_txt, "a+") as f:
        f.write(result_valid + '\n')
        f.close()

    print('valid set: Average loss: {:.4f}, Accuracy: {}/{} ({:.6f}%)\n'.format(
        valid_loss, correct, len(valid_loader.dataset),
        100. * correct / len(valid_loader.dataset)))
    accu = float(correct) / len(valid_loader.dataset)
    return accu, valid_loss


# writer.add_graph(model,data)
def sum(pred, target):
    # 此代码不包含统计所有的载体图片和所有载密图像的数量，需要在调用后设置所有载体图像的数量。
    # print(len(target))
    pred = pred.view_as(target)
    pred = pred.cpu().numpy()
    target = target.cpu().numpy()
    l1 = []
    for i in range(len(target)):
        l1.append(pred[i] + target[i])
    # print(l1.count(0))
    # print(l1.count(2))
    # l1.count(0)即为 正确被判定为载体图像（阴性）的数量。l1.count(2)，即为正确被判定为载密图像（阳性）的数量。l1.count(0)+l1.count(2) 即为判断正确的总个数
    return l1.count(0), l1.count(2), l1.count(0) + l1.count(2)


def valid_mulit():
    model.eval()
    valid_loss = 0
    correct = 0.
    accu = 0.
    N = 0  # 正确被分类为载体图像的数目
    P = 0  # 正确被分类为载密图像的数目

    with torch.no_grad():
        for data, target in valid_loader:

            if args.cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)
            output = model(data)
            output = F.log_softmax(output, dim=1)
            valid_loss += F.nll_loss(output, target, reduction='sum').item()

            pred = output.max(1, keepdim=True)[1]
            a, b, c = sum(pred, target)
            N += a
            P += b
            correct += c
    valid_loss /= len(valid_loader.dataset)
    accu = float(correct) / len(valid_loader.dataset)
    print('Valid set: Average loss: {:.4f}, Accuracy: {}/{} ({:.6f}%)'.format(
        valid_loss, correct, len(valid_loader.dataset),
        100. * accu))
    S = len(valid_loader.dataset) / 2  # 待测数据中所有的载密图像个数  具体的数量具体设置，如果载体图像等于载密数量则这样写代码即可
    C = len(valid_loader.dataset) / 2  # 待测数据集中所有载体图像的个数
    FPR = (C - N) / C  # 虚警率 即代表载体图像被误判成载密图像 占所有载体图像的比率
    Pmd = (S - P) / S  # 漏检率 即代表载密图像被误判成载体图像 占所有载密图像的比率
    print('Valid set 虚警率(FPR): {}/{} ({:.6f}%)'.format(C - N, C, 100. * FPR))
    print('Valid set 漏检率(FNR): {}/{} ({:.6f}%)'.format(S - P, S,
                                                          100. * Pmd))  # 名称定义来自于  来自于软件学报 论文 《基于深度学习的图像隐写分析综述》Journal of Software,2021,32(2):551−578 [doi: 10.13328/j.cnki.jos.006135]
    return accu, valid_loss


t1 = time()
best_acc = 0
import datetime

creat_time = current_time = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M")
for i in range(1, 458):
    # a = 'E:\陈梦飞\\7个深度隐写分析模型\\2019Zhu-Net_quanxuexi\code\\net_params\pre_train\\boss_bosw2_wow-0.4_VA\\1\\'+str(i)+'.pkl'
    # model.load_state_dict(torch.load(a))
    # print(a)
    # t3 = time()
    log = './PKL/suni0.4' + '\\' + creat_time
    if not os.path.isdir(log):
        os.makedirs(log)
    train(i)
    # valid()
    acc, test_loss = test()
    if best_acc < acc:
        best_acc = acc
        a = 'epoch{}_acc{:.2f}.pkl'.format(i, acc)
        dir = os.path.join(log, a)
        torch.save(model.state_dict(), dir)
    # t4 = time()
    # print('valid time = ',t4-t3)
    # t5 = time()
    # test()
    # t6 =  time()
    # print('test time = ', t6 - t5)
t2 = time()
print('total_test_time = ', t2 - t1)

