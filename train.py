import torch, torchvision
import torch.nn as nn
import torchvision.transforms as transforms
import torch.nn.functional as F

import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim

input_transform = transforms.Compose([transforms.ToTensor(),
                                      transforms.Lambda(lambda data: data.view(-1))])

torch.manual_seed(5) # reproducible, since we shuffle in DataLoader.
train_set = torchvision.datasets.MNIST('mnist', train=True, download=True, transform=input_transform)
test_set = torchvision.datasets.MNIST('mnist', train=True, download=True, transform=input_transform)


# For test loader, the batch_size should just be as large as can fit in memory.
# pin_memory = true may make CUDA transfer faster.
train_loader = torch.utils.data.DataLoader(train_set, batch_size = 64, shuffle=True, num_workers=0)
test_loader = torch.utils.data.DataLoader(train_set, batch_size = 256, shuffle=True, num_workers=0)
# 8 layers of 128 neurons each, mini-batches of 64.

class MNISTnet(torch.nn.Module):
    def __init__(self):
        super(MNISTnet, self).__init__()
        self.input = torch.nn.Linear(28*28, 128)
        self.hidden = [nn.ReLU() for i in range(8)]
        self.output = torch.nn.Linear(128, 10)

    def forward(self, x):
        x = self.input(x)
        for hidden in self.hidden:
            x = hidden(x)
        return self.output(x)

# One mode is train for many epochs (till there is no more improvement - criteria) and then test the function.
#  Or test at regular intervals, and stop when testing error is not improving.

def profile(func, path="trace_test"):
    with torch.autograd.profiler.profile(use_cuda=torch.cuda.is_available()) as prof:
        func()
    return prof
    # with torch.cuda.profiler.profile():
    #    model(train_set[0])  # Warmup CUDA memory allocator and profiler
    #    with torch.autograd.profiler.emit_nvtx():

def train(model, train_loader, optimizer, loss_fn = F.cross_entropy,
          device = torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader): #,0?
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:
            print(100. * batch_idx / len(train_loader))
        #    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        #        epoch, batch_idx * len(data), len(train_loader.dataset),
        #        100. * batch_idx / len(train_loader), loss.item()))

def test(model, test_loader, loss_fn = F.cross_entropy,
          device = torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    test_loss = 0
    correct = 0
    model.eval()
    with torch.no_grad():
        for batch_idx, (data, target) in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += loss_fn(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item() # for accuracy
    test_loss /= len(test_loader.dataset)
    correct /= len(test_loader.dataset)
    return (test_loss, correct)


lr = 1e-1 #3e-4
model = MNISTnet()
optimizer = optim.Adagrad(model.parameters(), lr=lr)
prof = profile(lambda: train(model, train_loader, optimizer))
print(prof)
print("test")