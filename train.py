import torch, torchvision
import torch.nn as nn
import torchvision.transforms as transforms
import torch.nn.functional as F

#import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim
import time, os
from multiprocessing import Pool

def timing(f):
    def wrap(*args, **kwargs):
        time1 = time.time()
        ret = f(*args, **kwargs)
        time2 = time.time()
        print('{:s} function took {:.3f} ms'.format(f.__name__, time2-time1))

        return ret
    return wrap

# 8 layers of 128 neurons each, mini-batches of 64.

class MNISTnet(torch.nn.Module):
    def __init__(self):
        super(MNISTnet, self).__init__()
        self.input = torch.nn.Linear(28*28, 128 * 8)
        self.hidden = [nn.ReLU() for _ in range(8)]
        self.output = torch.nn.Linear(128 * 8, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.input(x)
        for hidden in self.hidden:
            x = hidden(x)
        x = self.output(x)
        return x

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
        x = self.conv(x)
        x = self.fullc(x.view(-1, 4 * 4 * 50))
        return x

def profile(func, use_cuda=torch.cuda.is_available(), path=None):
    with torch.autograd.profiler.profile(use_cuda=use_cuda) as prof:
        func()
    if path is not None:
        prof.export_chrome_trace(path)
    return prof
    # with torch.cuda.profiler.profile():
    #    model(train_set[0])  # Warmup CUDA memory allocator and profiler
    #    with torch.autograd.profiler.emit_nvtx():

@timing
def train(model, train_loader, optimizer, loss_fn = F.cross_entropy,
          device = None):
    time0 = time.time()
    train_loss = torch.zeros([1], dtype=torch.float, device=device)
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, target)
        train_loss.add_(loss)
        loss.backward()
        optimizer.step(lambda: loss)
    return train_loss.item() / len(train_loader.dataset)

@timing
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
    correct = correct.item(); #= correct.item()/len(test_loader.dataset) # accuracy
    return test_loss, correct/len(test_loader.dataset), correct



class PrePreprocess(torch.utils.data.Dataset):
    # This class saves datasets (after they're transformed/preprocessed) to desk
    # This may take up a lot of disk space, but it saves some seconds everytime
    #  on every run of the script. Makes it faster to debug/test.
    
    @staticmethod
    def preprocess(dataset):
        # Just try to max out all CPUs
        # If we get an out of memory error, batch_size is too large.
        num_workers, batch_size = os.cpu_count(), 1
        if num_workers is None:
            num_workers = 0
        else:
            batch_size = int(len(dataset) / num_workers)
        
        loader = torch.utils.data.DataLoader(dataset, num_workers=num_workers, batch_size = batch_size)
        data, target = [], []
        for result in loader:
            data.extend(result[0])
            target.extend(result[1])
        return list(zip(data, target))
    
    def __init__(self, dataset, path=None, *args, **kwargs):
        super().__init__()
        
        if callable(dataset):
            dataset = dataset(*args, **kwargs)
        
        if path is not None:
            try:
                self.memo = torch.load(open(path, "rb"))
            except (OSError, IOError) as e:
                print("Failed to load pre-preprocessed data. Creating files.")
                self.memo = PrePreprocess.preprocess(dataset)
                os.makedirs(os.path.dirname(path), exist_ok=True)
                torch.save(self.memo, open(path, "wb"))
        else:
            self.memo = PrePreprocess.preprocess(dataset)

    def __getitem__(self, index):
        return self.memo[index]

    def __len__(self):
        return len(self.memo)


if __name__ == '__main__':
    torch.manual_seed(5) # reproducible, since we shuffle in DataLoader.
    
    # Load and and preprocess datasets (or load pre-preprocessed)
    data_root = 'mnist'
    train_set = PrePreprocess(torchvision.datasets.MNIST, os.path.join(data_root, 'preprocessed', 'training.pt'),
                                data_root, train=True, download=True, transform=transforms.ToTensor())
    test_set = PrePreprocess(torchvision.datasets.MNIST, os.path.join(data_root, 'preprocessed', 'test.pt'),
                                data_root, train=False, download=True, transform=transforms.ToTensor())
    # For test loader, the batch_size should just be as large as can fit in memory.
    # pin_memory = true may make CUDA transfer faster.
    # torch.set_num_threads(8) # We have not seen an improvement in CPU utilization with this.
    
    test_batch_size = 2<<9
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=256, shuffle=True, pin_memory=False)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=test_batch_size, shuffle=False, pin_memory=False)
    
    model = MNISTnet()
    #If CPU is not in use while GPU calculates, we can also test on CPU?
    if torch.cuda.is_available():
        device = torch.device("cuda") # Defaults to cuda:0
        print("Using ", torch.cuda.device_count(), "GPUs.")
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model) # Have not been tested
    else:
        print("Using", num_preprocess_workers, "CPUs.")
        device = torch.device("cpu")
    
    model = model.to(device, non_blocking=True)
    
    optimizer = optim.Adagrad(model.parameters(), lr=1e-1)
    #train(model, train_loader, optimizer, device=device)
    #test_loss, test_acc = test(model, train_loader, device=device)
    #print(test_acc)
    n = 10
    for i in range(n):
        train(model, train_loader, optimizer, device=device)
    # Possibly save model if it is good.
    print(test(model, test_loader, device=device))
    #profile(lambda: , path="trace")

