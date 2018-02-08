#!/usr/bin/env python

"""
    main.py
"""

from __future__ import print_function, division

import sys
import json
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable

# --
# Helpers

def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=256, metavar='N')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N')
    parser.add_argument('--epochs', type=int, default=10, metavar='N')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M')
    parser.add_argument('--seed', type=int, default=1, metavar='S')
    parser.add_argument('--log-interval', type=int, default=100, metavar='N')
    return parser.parse_args()


class MNISTRandomLabels(datasets.MNIST):
  def __init__(self, corrupt_prob=0.0, num_classes=10, **kwargs):
    super(MNISTRandomLabels, self).__init__(**kwargs)
    self.n_classes = num_classes
    if corrupt_prob > 0:
      self.corrupt_labels(corrupt_prob)
      
  def corrupt_labels(self, corrupt_prob):
    labels = np.array(self.train_labels if self.train else self.test_labels)
    np.random.seed(12345)
    mask = np.random.rand(len(labels)) <= corrupt_prob
    rnd_labels = np.random.choice(self.n_classes, mask.sum())
    labels[mask] = rnd_labels
    labels = [int(x) for x in labels]
    
    if self.train:
      self.train_labels = labels
    else:
      self.test_labels = labels

class Net(nn.Module):
    def __init__(self, N=4096):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(28 ** 2, N)
        self.fc2 = nn.Linear(N, N)
        self.fc3 = nn.Linear(N, N)
        self.fc4 = nn.Linear(N, N)
        self.fc5 = nn.Linear(N, N)
        self.out = nn.Linear(N, 10)
        
    def forward(self, x):
        x = x.view(-1, 28 ** 2)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = self.out(x)
        return F.log_softmax(x)

def ablate(x, p):
    return x * Variable(torch.rand((1,) + x.shape[1:]).cuda() > p).float()

def ablate_forward(model, x, p=0.1):
    x = x.view(-1, 28 ** 2)
    
    x = F.relu(model.fc1(x))
    x = ablate(x, p=p)
    
    x = F.relu(model.fc2(x))
    x = ablate(x, p=p)
    
    x = F.relu(model.fc3(x))
    x = ablate(x, p=p)
    
    x = F.relu(model.fc4(x))
    x = ablate(x, p=p)
    
    x = F.relu(model.fc5(x))
    x = ablate(x, p=p)
    
    x = model.out(x)
    return F.log_softmax(x)


def train(epoch):
    model.train()
    total, correct = 0, 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        
        total += target.shape[0]
        pred = output.data.max(1)[1]
        correct += (pred == target.data).int().sum()
    
    return (correct / total)


def test():
    model.eval()
    correct = 0
    for data, target in test_loader:
        data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()
        
    return correct / len(test_loader.dataset)


def ablate_train_acc(p=0.1):
    model.eval()
    correct = 0
    for data, target in train_loader:
        data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = ablate_forward(model, data, p=p)
        
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()
    
    return correct / len(train_loader.dataset)

# --
# Args

args = parse_args()

torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

# --
# IO

kwargs = {'num_workers': 2, 'pin_memory': False}

transform = transforms.Compose([
   transforms.ToTensor(),
   transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
# train_dataset = MNISTRandomLabels(root='./data', corrupt_prob=0.2, train=True, download=True, transform=transform)
test_dataset = datasets.MNIST('./data', train=False, transform=transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, **kwargs)
test_loader  = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, **kwargs)

# --
# Run

model = Net().cuda()

optimizer = optim.Adam(model.parameters())

all_res = []
for epoch in range(300):
    train_acc = train(epoch)
    test_acc = test()
    ablate_acc = ablate_train_acc(p=0.5)
    res = {
        "train_acc" : train_acc,
        "test_acc" : test_acc,
        "ablate_acc" : ablate_acc,
    }
    all_res.append(res)
    print(json.dumps(res))
    
    if not (epoch + 1) % 100:
        df = pd.DataFrame(all_res)
        _ = plt.plot(df.ablate_acc, label='ablate_acc')
        _ = plt.plot(df.train_acc, label='train_acc')
        _ = plt.plot(df.test_acc, label='test_acc')
        _ = plt.legend(loc='lower left')
        show_plot()

# !! Possibly evaluating at p=0.5 alone is not enough

# --
# Ablation

from rsub import *
from matplotlib import pyplot as plt

s = np.linspace(0, 1.0, 20)
z = [ablate_train_acc(p=p) for p in s]

_ = plt.plot(s, z0)
_ = plt.plot(s, z)
show_plot()



