#!/usr/bin/env python

"""
    data.py
"""

import numpy as np
from sklearn.model_selection import train_test_split

import torch
import torchvision

def make_mnist_dataloaders(root='data', train_size=1.0, train_batch_size=256, eval_batch_size=256, num_workers=2, seed=123123):
    
    transform = torchvision.transforms.Compose([
       torchvision.transforms.ToTensor(),
       torchvision.transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    trainset = torchvision.datasets.MNIST(root=root, train=True, download=False, transform=transform)
    testset = torchvision.datasets.MNIST(root=root, train=False, download=False, transform=transform)
    
    if train_size < 1:
        train_inds, val_inds = train_test_split(np.arange(len(trainset)), train_size=train_size, random_state=seed)
        trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=train_batch_size, num_workers=num_workers, pin_memory=True,
            sampler=torch.utils.data.sampler.SubsetRandomSampler(train_inds),
        )
        
        valloader = torch.utils.data.DataLoader(
            trainset, batch_size=eval_batch_size, num_workers=num_workers, pin_memory=True,
            sampler=torch.utils.data.sampler.SubsetRandomSampler(val_inds),
        )
    else:
        trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=train_batch_size, num_workers=num_workers, pin_memory=True,
            shuffle=True,
        )
        
        valloader = None
    
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=eval_batch_size, num_workers=num_workers, pin_memory=True,
        shuffle=False,
    )
    
    return {
        "train" : trainloader,
        "test" : testloader,
        "val" : valloader,
    }