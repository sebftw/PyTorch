import torch, torchvision
import torch.nn as nn
import torchvision.transforms as transforms
import torch.nn.functional as F

import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim

import time
import os

# 8 layers of 128 neurons each, mini-batches of 64.

class MNISTnet(torch.nn.Module):
    def __init__(self):
        super(MNISTnet, self).__init__()
        self.input = torch.nn.Linear(28*28, 128)
        self.hidden = [nn.ReLU() for i in range(8)]
        self.output = torch.nn.Linear(128, 10)

    def forward(self, x):
        x = self.input(x.view(x.size(0), -1))
        for hidden in self.hidden:
            x = hidden(x)
        return self.output(x)

class MNISTnet2(torch.nn.Module):
    def __init__(self):
        super(MNISTnet2, self).__init__()
        self.conv = torch.nn.Sequential(
            nn.Conv2d(1, 20, 5, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(20, 50, 5, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2) )

        self.fullc = torch.nn.Sequential(
            nn.Linear(4*4*50, 500),
            nn.ReLU(),
            nn.Linear(500, 10) )

    def forward(self, x):
        #x = self.input(x)
        x = self.conv(x)
        x = self.fullc(x.view(-1, 4 * 4 * 50))
        return x

# One mode is train for many epochs (till there is no more improvement - criteria) and then test the function.
#  Or test at regular intervals, and stop when testing error is not improving.

def profile(func, use_cuda=torch.cuda.is_available(), path=None):
    with torch.autograd.profiler.profile(use_cuda=use_cuda) as prof:
        func()
    if path is not None:
        prof.export_chrome_trace(path)
    return prof
    # with torch.cuda.profiler.profile():
    #    model(train_set[0])  # Warmup CUDA memory allocator and profiler
    #    with torch.autograd.profiler.emit_nvtx():

def train(model, train_loader, optimizer, loss_fn = F.cross_entropy,
          device = None):
    time0 = time.time()
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:
            print(100. * batch_idx / len(train_loader))

    print("{:.3f}".format(time.time()-time0))
    return time.time()-time0

def test(model, test_loader, loss_fn = F.cross_entropy,
          device = None):
    test_loss = torch.zeros([1], dtype=torch.float, device=device)
    correct = torch.zeros([1], dtype=torch.long, device=device)
    model.eval()
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
            output = model(data)
            test_loss.add_(loss_fn(output, target, reduction='sum')) # sum up batch loss
            pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct.add_(pred.eq(target.view_as(pred)).sum()) # for accuracy
            print(batch_idx)
            if batch_idx % 10 == 0:
                print(100. * (batch_idx+1) / len(test_loader))
    test_loss = test_loss.item()/len(test_loader.dataset) # average test loss
    accuracy = correct.item()/len(test_loader.dataset) # accuracy
    return (test_loss, accuracy)

if __name__ == '__main__':
    
    torch.manual_seed(5) # reproducible, since we shuffle in DataLoader.
    
    # Load Dataset
    train_set = torchvision.datasets.MNIST('mnist', train=True, download=True, transform=transforms.ToTensor())
    test_set = torchvision.datasets.MNIST('mnist', train=False, download=True, transform=transforms.ToTensor())
    
    # For test loader, the batch_size should just be as large as can fit in memory.
    # pin_memory = true may make CUDA transfer faster.
    # torch.set_num_threads(8) # We have not seen an improvement in CPU utilization with this.
    num_preprocess_workers = 0 if os.cpu_count() is not None else os.cpu_count() # Just max out all CPUs
    test_batch_size = len(test_set) #Should just be as large as possible. (Limited by GPU memory)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=256, shuffle=True, num_workers=8, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=test_batch_size, shuffle=False, num_workers=num_preprocess_workers/2, pin_memory=True)
    
    model = MNISTnet()
    #If CPU is not in use, while GPU calculates, we can also test on CPU?
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.device_count() > 1:
        print("Using ", torch.cuda.device_count(), "GPUs.")
        model = nn.DataParallel(model)
    model = model.to(device, non_blocking=True)
    
    optimizer = optim.Adagrad(model.parameters(), lr=1e-1)
    #train(model, train_loader, optimizer, device=device)
    #test_loss, test_acc = test(model, train_loader, device=device)
    #print(test_acc)
    n = 1
    timeavg = 0.0
    for i in range(n):
        timeavg += train(model, train_loader, optimizer, device=device)
    #timeavg /= n
    print(timeavg)
    #print(test(model, test_loader, device=device))
    #profile(lambda: , path="trace")
