import torch, torchvision
import torch.nn as nn
import torchvision.transforms as transforms
import torch.nn.functional as F

import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim
import time
import os
import funcy
import concurrent.futures
import multiprocessing
from multiprocessing import Pool

import pickle

# 8 layers of 128 neurons each, mini-batches of 64.

class MNISTnet(torch.nn.Module):
    def __init__(self):
        super(MNISTnet, self).__init__()
        self.input = torch.nn.Linear(28*28, 128)
        self.hidden = [nn.ReLU() for _ in range(8)]
        self.output = torch.nn.Linear(128, 10)

    def forward(self, x):
        x = self.input(x.view(x.size(0), -1))
        for hidden in self.hidden:
            x = hidden(x)
        return self.output(x)

class MNISTnet2(torch.nn.Module):
    def __init__(self):
        super().__init__() # Same as super(MNISTnet2, this).__init__().
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
        if batch_idx % 10 == 0: # Make timed condition instead, e.g. every 5 seconds print and estimated time left.
            print("{:.0f} %".format(100. * (batch_idx+1) / len(train_loader)))

    print("Epoch time: {:.3f}".format(time.time()-time0))
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
            print("{:.0f} %".format(100. * (batch_idx+1) / len(test_loader)))
            #if batch_idx % 10 == 0:
            #    print("{:.0f} %".format(100. * batch_idx / len(test_loader)))
    test_loss = test_loss.item()/len(test_loader.dataset) # average test loss
    accuracy = correct.item()/len(test_loader.dataset) # accuracy
    return test_loss, accuracy

class PrePreprocess(torch.utils.data.Dataset):
    def __init__(self, dataset):
        super().__init__()
        time0 = time.time()
        
        # Just try to max out all CPUs
        # If we get an out of memory error, batch_size is too large.
        num_workers = os.cpu_count()
        batch_size = 1
        if num_workers is None:
            num_workers = 0
        else:
            batch_size = len(dataset) / num_workers
        
        loader = torch.utils.data.DataLoader(dataset, num_workers=num_workers, batch_size = 4000)
        data = []
        target = []
        for result in loader:
            data.extend(result[0])
            target.extend(result[1])
        self.memo = list(zip(data, target))
        
        print("Epoch time: {:.3f}".format(time.time()-time0))

    def __getitem__(self, index):
        return self.memo[index]

    def __len__(self):
        return len(self.memo)



#class Mystery:
#    @funcy.memoize
#    def __new__(cls, num):
#        self = super().__new__(cls)
#        self.num = num
#        return self
#    def __reduce__(self):
#        return (type(self), (self.num,))

if __name__ == '__main__':
    torch.manual_seed(5) # reproducible, since we shuffle in DataLoader.

    # Load Dataset
    train_set = torchvision.datasets.MNIST('mnist', train=True, download=True, transform=transforms.ToTensor())
    test_set = torchvision.datasets.MNIST('mnist', train=False, download=True, transform=transforms.ToTensor())

    # Memoize the lists
    train_set, test_set = PrePreprocess(train_set), PrePreprocess(test_set)
    
    # For test loader, the batch_size should just be as large as can fit in memory.
    # pin_memory = true may make CUDA transfer faster.
    # torch.set_num_threads(8) # We have not seen an improvement in CPU utilization with this.
    
    test_batch_size = len(test_set) #Should just be as large as possible. (Limited by GPU memory)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=256, shuffle=True, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=test_batch_size, shuffle=False, pin_memory=True)

    model = MNISTnet()
    #If CPU is not in use while GPU calculates, we can also test on CPU?
    if torch.cuda.is_available():
        device = torch.device("cuda") # Defaults to cuda:0
        print("Using ", torch.cuda.device_count(), "GPUs.")
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
    else:
        print("Using", num_preprocess_workers, "CPUs.")
        device = torch.device("cpu")

    model = model.to(device, non_blocking=True)
    
    optimizer = optim.Adagrad(model.parameters(), lr=1e-1)
    #train(model, train_loader, optimizer, device=device)
    #test_loss, test_acc = test(model, train_loader, device=device)
    #print(test_acc)
    n = 3
    timeavg = 0.0
    print("?")
    for i in range(n):
        timeavg += train(model, train_loader, optimizer, device=device)
    timeavg /= n
    # Possibly save model if it is good.
    print(timeavg)
    print(test(model, test_loader, device=device))
    #profile(lambda: , path="trace")

