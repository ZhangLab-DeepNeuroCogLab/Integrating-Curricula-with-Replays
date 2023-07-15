import torchvision.transforms as transforms
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import pickle
import pandas as pd
import os
from cifar10_models.vgg import vgg13_bn

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
from cifair import ciFAIR10

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = ciFAIR10(root='./data', train=True,
                    download=True, transform=transform)

testset = ciFAIR10(root='./data', train=False,
                   download=True, transform=transform)
                   

# Get cpu or gpu device for training.

def run(gpu, seed):
    device = f"cuda:{gpu}" if torch.cuda.is_available() else "cpu"

    my_model = vgg13_bn(pretrained=True)
    my_model.eval() # for evaluation
    my_model.to(device)

    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False 
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    class Net(nn.Module):
        def __init__(self, nodes, buffersize):
            super().__init__()
            self.exemplers = {}
            self.buffersize = buffersize
            self.conv1 = nn.Conv2d(3, 6, 5)
            self.pool = nn.MaxPool2d(2, 2)
            self.conv2 = nn.Conv2d(6, 16, 5)
            self.fc1 = nn.Linear(16 * 5 * 5, 120)
            self.fc2 = nn.Linear(120, 84)
            self.fc3 = nn.Linear(84, nodes)
            self.numdiv = 1
            self.task = 1

        def forward(self, x):
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = torch.flatten(x, 1)  # flatten all dimensions except batch
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
            return x

        def increase2Class(self, device):
            m = self.fc3
            old_shape = self.fc3.weight.shape
            m2 = nn.Linear(old_shape[1], old_shape[0] + 2)
            m2.to(device)
            m2.weight = nn.parameter.Parameter(
                torch.cat((m.weight, m2.weight[0:2])))
            m2.bias = nn.parameter.Parameter(torch.cat((m.bias, m2.bias[0:2])))
            self.fc3 = m2
            self.task+=1

        def addandoutputexemplers(self, orginaldata):
            # Save data from current task into trainloaderdata
            outputdata = []

            orginaldatalis = [j for i in orginaldata.values() for j in i]
            exemplerlis = [j for i in self.exemplers.values() for j in i]

            random.seed(seed)
            random.shuffle(orginaldatalis)

            random.seed(seed)
            random.shuffle(exemplerlis)

            tdivsize = len(orginaldatalis)//self.numdiv
            edivsize = len(exemplerlis)//self.numdiv

            for i in range(self.numdiv):
                outputdata+=orginaldatalis[i*tdivsize:(i+1)*tdivsize]
                outputdata+=exemplerlis[i*edivsize:(i+1)*edivsize]

            for i in orginaldata:
                keys = []

                tempdataloader = DataLoader(orginaldata[i], batch_size=1, shuffle=False)
                for data in tempdataloader:
                    images = data[0].to(device)
                    output = my_model(images)
                    probs = F.softmax(output, dim=1)
                    keys.append(probs.tolist()[0][data[1]])
                tobeadded = [x for _, x in sorted(zip(keys, orginaldata[i]), key=lambda x:x[0], reverse=False)]
                self.exemplers[i] = tobeadded[:self.buffersize//self.task//2]

            for i in self.exemplers:
                self.exemplers[i] = self.exemplers[i][:self.buffersize//self.task//2]

            toreturn = DataLoader(outputdata, batch_size=32, shuffle=False)

            return toreturn

    

    def accuracy(task):
        correct_pred = {classname: 0 for classname in task}
        total_pred = {classname: 0 for classname in task}

        # again no gradients needed
        with torch.no_grad():
            for data in testloader[task]:
                images, labels = data[0].to(device), data[1].to(device)
                outputs = net(images)
                _, predictions = torch.max(outputs, 1)
                accuracyforclass = {}
                for label, prediction in zip(labels, predictions):
                    if label == prediction:
                        correct_pred[classes[label]] += 1
                    total_pred[classes[label]] += 1
                total_correct_pred = 0
                total_total_pred = 0
                for classname, correct_count in correct_pred.items():
                    total_correct_pred += correct_count
                    total_total_pred += total_pred[classname]
                    accuracy = 100 * float(correct_count) / total_pred[classname]
                    accuracyforclass[classname] = accuracy
                correctbybatch.append(accuracyforclass)
                totalaccuracy.append(100 * float(total_correct_pred) / total_total_pred)

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    trainsetbyclasses = {}
    for img, label in trainset:
        if classes[label] in trainsetbyclasses:
            trainsetbyclasses[classes[label]].append((img, label))
        else:
            trainsetbyclasses[classes[label]] = []
            trainsetbyclasses[classes[label]].append((img, label))

    testsetbyclasses = {}
    for img, label in testset:
        if classes[label] in testsetbyclasses:
            testsetbyclasses[classes[label]].append((img, label))
        else:
            testsetbyclasses[classes[label]] = []
            testsetbyclasses[classes[label]].append((img, label))

    testloader = {}

    for i in range(5):
        templist = []
        classlist = []
        for j in range(2*(i+1)):
            templist += testsetbyclasses[classes[j]]
            classlist.append(classes[j])
        testloader[tuple(classlist)] = DataLoader(
            templist, batch_size=len(templist), shuffle=True)
    correctbybatch = []
    totalaccuracy = []

    net = Net(2, 1200)
    net.to(device)
    torch.cuda.set_device(gpu) 
    print(f"using gpu {torch.cuda.current_device()}")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    for j in range(5):
        dataforcurrenttask = {k:trainsetbyclasses[k] for k in classes[j*2:j*2+2]}
        currenttrainloader = net.addandoutputexemplers(dataforcurrenttask)

        for epoch in range(250):  # loop over the dataset multiple times
            running_loss = 0.0
            for data in currenttrainloader:
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data[0].to(device), data[1].to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            accuracy(classes[0:j*2+2])
            print(f"Epoch [{epoch+1:>2}], loss: {running_loss/len(currenttrainloader):.8f}")
        net.increase2Class(device)

    os.mkdir(f"../Results/VaryBufferSelection/pretrainedcifarhardest-seed{seed}/")
    dfaccuracybybatch = pd.DataFrame()
    x = range(len(correctbybatch))
    for i in classes:
        y = []
        for j in correctbybatch:
            if i in j:
                y.append(j[i])
            else:
                y.append(0)
        tem = pd.DataFrame(data={i: y}, index=x)
        dfaccuracybybatch = pd.concat([dfaccuracybybatch, tem], axis=1)
    dfaccuracybybatch.to_pickle(f"../Results/VaryBufferSelection/pretrainedcifarhardest-seed{seed}/Accuracy by class.pkl")
      
    x1=range(len(totalaccuracy))
    y1=totalaccuracy
    dftotalaccuracy = pd.DataFrame(data={"Accuracy":y1}, index=x1) 
    dftotalaccuracy.to_pickle(f"../Results/VaryBufferSelection/pretrainedcifarhardest-seed{seed}/Total Accuracy.pkl")
      
import sys
# l = [1, 10, 100, 1000, 10000]
# l = [1, 10, 100, 1000, 10000, 5, 50, 500, 5000, 50000]
l = [3, 30, 300, 3000, 30000, 7, 70, 700, 7000, 70000]


for i in l:
    run(gpu=int(sys.argv[1]), seed=i)