import random
from sklearn.model_selection import train_test_split
from easydict import EasyDict as edict
from mydata import MyDataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch
import os
from torchvision.models import AlexNet

import torch.optim as optim


import torch.nn as nn
import torch.nn.functional as F

def createData(dataroot = './leapGestRecog/'):
    images = []
    for folder in os.listdir(dataroot):
        subdir = dataroot + folder + '/'
        for gclass in os.listdir(subdir):
            subsubdir = subdir + gclass + '/'
            for imgloc in os.listdir(subsubdir):
                images.append([subsubdir + imgloc, gclass])
    class_labels = os.listdir(dataroot + folder + '/')
    class_labels.sort()
    random.shuffle(images)
    return class_labels,images

param = edict({
    'datapath':'../leapGestRecog/',
    'batch_size':64,
    'num_workers': 2,
    'split_rate':0.2,
    'epoch':5,
    'figsize':224,
    'norm':{
        'normMean':[0.4948052, 0.48568845, 0.44682974],
        'normStd':[0.24580306, 0.24236229, 0.2603115]
    }
})

def train(model, device, trainloader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(trainloader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output,target)
        loss.backward()
        optimizer.step()
        if batch_idx %100 == 0:
            print('Train Epoch: {:5d} [{:5d}/{:5d} ({:.2f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(trainloader.dataset),
                100. * batch_idx / len(trainloader), loss.item()))

def test(model, device, testloader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in testloader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            target = target.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(testloader.dataset)

    print('Test set: Average loss: {:.4f}, Accuracy: {:5d}/{:5d} ({:.2f}%)\n'.format(
        test_loss, correct, len(testloader.dataset),
        100. * correct / len(testloader.dataset)))


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class_label,data_images = createData(param.datapath)
trainimg, testimg = train_test_split(data_images, random_state=23, test_size = param.split_rate)

traintrans = transforms.Compose([
    transforms.Resize((param.figsize,param.figsize)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(param.norm.normMean, param.norm.normStd)])

testtrans = transforms.Compose([
    transforms.Resize((param.figsize,param.figsize)),
    transforms.ToTensor(),
    transforms.Normalize(param.norm.normMean, param.norm.normStd)])

traindata = MyDataset(trainimg,class_label,traintrans)
testdata = MyDataset(testimg,class_label,traintrans)

trainloader = DataLoader(dataset = traindata, batch_size=param.batch_size,
                         num_workers=param.num_workers, shuffle=True)
testloader = DataLoader(dataset = testdata,batch_size=param.batch_size,num_workers=param.num_workers)

model = AlexNet(num_classes=10,dropout=0.5).to(device)

optimizer = optim.Adam(params=model.parameters())

for epoch in range(1, param.epoch + 1):
    train(model, device, trainloader, optimizer, epoch)
    test(model, device, testloader)

torch.save(model.state_dict(), "./model/train.pt")