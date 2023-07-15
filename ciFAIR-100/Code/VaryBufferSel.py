# sampling of exemplers, shuffling of trainset, shuffling of valset is random (universal seed), initial trainset is constant (by virtue of reading data sequentially), determined classseq 100
import os
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms
import torchvision.models as models
from torch.utils.data import DataLoader
from cifair import ciFAIR100
import wandb
import torch.optim as optim
from torchvision.models import mobilenet_v3_small
import pytorch_lightning as pl
from torchmetrics.classification import MulticlassAccuracy
from argparse import ArgumentParser
from torch import autocast
import random
import numpy as np

# Model
class Net(nn.Module):
    def __init__(self):
        super().__init__() 
        self.task = 1
        self.lr = 0.001
        self.lossfunc = nn.CrossEntropyLoss()

        mobilenet = mobilenet_v3_small(weights="DEFAULT")
        self.backbone = nn.Sequential(*list(mobilenet.children())[:-1])

        self.classifier = list(mobilenet.children())[-1]
        linearlayershape = list(self.classifier.children())[-1].weight.shape
        newlinearlayer = nn.Linear(linearlayershape[1], 5)
        self.classifier[-1] = newlinearlayer
        self.classifier = nn.Sequential(*self.classifier)

    def forward(self, x):
        x = self.backbone(x).flatten(1)
        x = self.classifier(x)
        return x
    
    def calc_loss_corrects(self, batch, stage=None):
        images, labels = batch
        with autocast(device_type='cuda'):
            output = self(images)
            loss = self.lossfunc(output, labels)
        _, preds = torch.max(output, 1)
        if stage == "train":
            return loss
        else:
            corrects = torch.sum(preds == labels)
            return loss.item() * labels.size(dim=0), corrects.item()
    
    def on_test_end(self, device):
        self.task += 1

        linearlayer = list(self.classifier.children())[-1]
        linearlayershape = linearlayer.weight.shape
        newlinearlayer = nn.Linear(linearlayershape[1], linearlayershape[0]+5)
        newlinearlayer.to(device)

        newlinearlayer.weight = nn.parameter.Parameter(torch.cat((linearlayer.weight, newlinearlayer.weight[0:5])))
        newlinearlayer.bias = nn.parameter.Parameter(torch.cat((linearlayer.bias, newlinearlayer.bias[0:5])))

        self.classifier[-1] = newlinearlayer
        self.classifier = nn.Sequential(*self.classifier)

#
class Pretrained(nn.Module):
    def __init__(self):
        super().__init__() 
        self.lr = 0.001
        self.lossfunc = nn.CrossEntropyLoss()
        self.testmetric = MulticlassAccuracy(num_classes=100)
        self.metric = MulticlassAccuracy(num_classes=100)

        mobilenet = mobilenet_v3_small(weights="DEFAULT")
        self.backbone = nn.Sequential(*list(mobilenet.children())[:-1])

        self.classifier = list(mobilenet.children())[-1]
        linearlayershape = list(self.classifier.children())[-1].weight.shape
        newlinearlayer = nn.Linear(linearlayershape[1], 100)
        self.classifier[-1] = newlinearlayer
        self.classifier = nn.Sequential(*self.classifier)

    def forward(self, x):
        x = self.backbone(x).flatten(1)
        x = self.classifier(x)
        return x
    
    def calc_loss_acc(self, batch, stage=None):
        images, labels = batch
        with autocast(device_type='cuda'):
            output = self(images)
            loss = self.lossfunc(output, labels)
        _, preds = torch.max(output, 1)
        if stage == "test":
            acc = self.testmetric(preds, labels)
        else:
            acc = self.metric(preds, labels)
        return loss, acc
    
#trainer
class awesometrainer:
    def __init__(self, gpunumber, earlystoppatience, currentfileseed, lenbuffer, divnum, classseq):
        self.device = f"cuda:{gpunumber}"
        self.earlystoppatience = earlystoppatience
        self.epoch = -1
        self.modelpath = 0
        self.modeldir = f"./checkpoints/{currentfileseed}/"
        self.lenbuffer = lenbuffer
        self.replaybuffer = {}
        self.divnum = divnum
        self.classseq = classseq
        os.makedirs(self.modeldir, exist_ok=True)

    def running_loss_acc(self, loader, model):
        length = len(loader.dataset)
        model.eval()
        running_loss = 0
        running_corrects = 0
        for batch in loader:
            batch = [x.to(self.device) for x in batch]
            with torch.no_grad():
                batch_loss, batch_correct = model.calc_loss_corrects(batch)
                running_loss += batch_loss
                running_corrects += batch_correct
        model.train()
        return running_loss/length, running_corrects/length
    
    def _selectuniformly(self, lis, number):
        i = 0
        count = 0
        lislen = len(lis)
        templis = []
        while count < number:
            templis.append(lis[i])
            incre = (lislen-i)//(number-count)
    
            i += incre
            count += 1
        return templis
    
    def _arrangebuf(self, buf, style, number, pretrained = False):
        if not pretrained:
            model = models.resnet50(weights='DEFAULT')
            model = torch.nn.Sequential(*list(model.children())[:-1])
            model.to(self.device)
            model.eval()

            # print("orgbuf: ")
            # for i in buf:
            #     print(i[1], end=", ")
            tempdataloader = DataLoader(buf, batch_size=1, shuffle=False)
            listoffeaturebyinstance = []

            for data in tempdataloader:
                images = data[0].to(self.device)
                output = model(images)
                features = output.cpu().detach().numpy()
                listoffeaturebyinstance.append(features)
            lenoflist = len(listoffeaturebyinstance)
            matrix = np.zeros((lenoflist, lenoflist))
            for i in range(lenoflist):
                for j in range(lenoflist):
                    matrix[i][j] = np.linalg.norm(listoffeaturebyinstance[i]-listoffeaturebyinstance[j])
            keys = np.sum(matrix, axis=0)
            # print("keys: ", keys)
            
            a = [x for _, x in sorted(zip(keys, buf), key=lambda x:x[0], reverse=False)]
            # print("returned: ")

            if style == "easiest":
                # for i in a[:number]:
                #     print(i[1], end=", ")
                return a[:number]
        
            elif style == "hardest":
                # for i in a[-number:]:
                #     print(i[1], end=", ")
                return a[-number:]
            
            else:
                b = self._selectuniformly(lis=a, number=number)
                # for i in b:
                #     print(i[b], end=", ")
                return b
        
        else:
            model = Pretrained()
            model.load_state_dict(torch.load("./pretrained"))
            model.eval()
            model.to(self.device)

            print("orgbuf: ")
            for i in buf:
                print(i[1], end=", ")
            tempdataloader = DataLoader(buf, batch_size=1, shuffle=False)
            keys = []

            for data in tempdataloader:
                images = data[0].to(self.device)
                output = model(images)
                probs = F.softmax(output, dim=1)
                keys.append(probs.tolist()[0][self.classseq[data[1].item()]])
            print("keys: ", keys)

            a = [x for _, x in sorted(zip(keys, buf), key=lambda x:x[0], reverse=True)]
            print("returned: ")

            if style == "easiest":
                # for i in a[:number]:
                #     print(i[1], end=", ")
                return a[:number]
        
            elif style == "hardest":
                # for i in a[-number:]:
                #     print(i[1], end=", ")
                return a[-number:]
            
            else:
                b = self._selectuniformly(lis=a, number=number)
                # for i in b:
                #     print(i[b], end=", ")
                return b


    def fit(self, model, trainloader, valloader, pretrained, style):
        continuetraining = True
        optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
        model.to(self.device)
        num_batches = len(trainloader)
        minvalloss = 100
        # Get the number of items to store per class
        lenbufferperclass = self.lenbuffer // (model.task * 5)
        if lenbufferperclass > 450:
            lenbufferperclass = 450
            num_of_lenbufferperclass_plus_one = 0
        else:
            num_of_lenbufferperclass_plus_one = self.lenbuffer - lenbufferperclass*(model.task*5)
        # shuffle trainset
        newtrainset = []
        shuffledtrainloaderdataset = random.sample(trainloader.dataset, len(trainloader.dataset))
        buf = []
        for i in self.replaybuffer:
            buf += self.replaybuffer[i]
        shuffledbuf = random.sample(buf, len(buf))
        del buf

        numperdivtrainloader = len(shuffledtrainloaderdataset)//self.divnum
        num_of_numperdivtrainloader_plus_one = len(shuffledtrainloaderdataset) - numperdivtrainloader*(self.divnum)

        numperdivbuffer = len(shuffledbuf)//self.divnum

        # interleave trainset
        jaja = 0
        endindex = 0
        for i in range (self.divnum):
            jaja += 1
            if jaja > num_of_numperdivtrainloader_plus_one:
                newtrainset += shuffledtrainloaderdataset[endindex:endindex+numperdivtrainloader]
                endindex = endindex+numperdivtrainloader
            else:
                newtrainset += shuffledtrainloaderdataset[endindex:endindex+numperdivtrainloader+1]
                endindex = endindex+numperdivtrainloader+1

            newtrainset += shuffledbuf[i*numperdivbuffer:(i+1)*numperdivbuffer]
        newtrainset += shuffledtrainloaderdataset[(self.divnum)*numperdivtrainloader:]
        newtrainset += shuffledbuf[(self.divnum)*numperdivbuffer:]
        # check if curricula working as intended
        s = ""
        for i in newtrainset:
            s += str(i[1]//5) + ", "
        print("trainset seq: ", s)            
        # combine loader and buffer valset
        print("newtrainset len: ", len(newtrainset))
        del shuffledtrainloaderdataset
        del shuffledbuf

        # add all items in dataset to buffer
        for img, label in trainloader.dataset:
            if label in self.replaybuffer:
                self.replaybuffer[label].append((img, label))
            else:
                self.replaybuffer[label] = [(img, label)]
        
        # remove item such that it is within our buffer limit
        k = 0
        for i in self.replaybuffer:
            k += 1
            if k > num_of_lenbufferperclass_plus_one:
                self.replaybuffer[i] = self._arrangebuf(self.replaybuffer[i], style=style, number=lenbufferperclass, pretrained=pretrained)
            else:
                self.replaybuffer[i] = self._arrangebuf(self.replaybuffer[i], style=style, number=lenbufferperclass+1, pretrained=pretrained)
               
        del trainloader.dataset
        
        trainloader = DataLoader(newtrainset, batch_size=32, shuffle=False)

        while continuetraining:
            self.epoch += 1
            running_loss = 0
            for batch in trainloader:
                optimizer.zero_grad()
                batch = [x.to(self.device) for x in batch]
                loss = model.calc_loss_corrects(batch, stage="train")
                running_loss += loss
                loss.backward()
                optimizer.step()

            valloss, valacc = self.running_loss_acc(valloader, model)
            

            logdict = {
                "Epoch": self.epoch,
                "Train Loss": running_loss.item()/num_batches,
                "Valid Loss": valloss,
                "Valid Acc": valacc
            }
            print(logdict)
            wandb.log(logdict)

            if valloss < minvalloss:
                epochsinceminvalloss = 0
                self.modelpath = f"{self.modeldir}/epoch{self.epoch}"
                torch.save(model, self.modelpath)
                minvalloss = valloss
            else:
                epochsinceminvalloss += 1

            if epochsinceminvalloss >= self.earlystoppatience:
                continuetraining = False
                del valloader.dataset
                                    
    def test(self, model, dm):
        a = dm.test_dataloader()
        testloss, testacc = self.running_loss_acc(a[0], model)
        forgetloss, forgetacc = self.running_loss_acc(a[1], model)        
        del a

        logdict = {
            "Epoch": self.epoch,
            "Task": model.task,
            "Test Loss": testloss,
            "Test Acc": testacc,
            "Forget Loss": forgetloss,
            "Forget Acc": forgetacc
        }
        wandb.log(logdict)
        model.on_test_end(self.device)
        dm.on_test_end()
        

# Dataset
class cifair100module():
    def __init__(self, classseq, data_dir: str = "./data", train_batch_size: int = 32):
        super().__init__()
        self.data_dir = data_dir
        self.transform = transforms.Compose([transforms.ToTensor(),
                                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), 
                                            transforms.Resize((72, 72), interpolation=transforms.InterpolationMode.BICUBIC)])
        
        self.train_batch_size = train_batch_size
        self.task = 1
        self.class_seq = list(range(0,100))
        random.Random(classseq).shuffle(self.class_seq)

        trainset = ciFAIR100(data_dir, train=True, download=True, transform=self.transform)
        trainsetdict = {}
        for img, label in trainset:
            if label in trainsetdict:
                trainsetdict[label].append((img, label))
            else:
                trainsetdict[label] = [(img, label)]
        self.trainsetdict = trainsetdict

        valsetdict = {}
        for i in trainsetdict:
            valsetdict[i] = trainsetdict[i][-50:]
            del trainsetdict[i][-50:]
        testset = ciFAIR100(data_dir, train=False, download=True, transform=self.transform)
        self.valsetdict = valsetdict

        testsetdict = {}
        for img, label in testset:
            if label in testsetdict:
                testsetdict[label].append((img, label))
            else:
                testsetdict[label] = [(img, label)]
        self.testsetdict = testsetdict

    def setup(self):
        trainsetlist = [self.trainsetdict[self.class_seq[x]] for x in range(5*(self.task-1), 5*self.task)]
        self.trainset = [item for sublist in trainsetlist for item in sublist]
        self.trainset = [(x[0], self.class_seq.index(x[1])) for x in self.trainset]

        valsetlist = [self.valsetdict[self.class_seq[x]] for x in range(5*(self.task-1), 5*self.task)]
        self.valset = [item for sublist in valsetlist for item in sublist]
        self.valset = [(x[0], self.class_seq.index(x[1])) for x in self.valset]
        
        testsetlist = [self.testsetdict[self.class_seq[x]] for x in range(5*self.task)]
        self.testset = [item for sublist in testsetlist for item in sublist]
        self.testset = [(x[0], self.class_seq.index(x[1])) for x in self.testset]

        forgetsetlist = [self.testsetdict[self.class_seq[x]] for x in range(5)]
        self.forgetset = [item for sublist in forgetsetlist for item in sublist]
        self.forgetset = [(x[0], self.class_seq.index(x[1])) for x in self.forgetset]

    def train_dataloader(self):
        return DataLoader(self.trainset, batch_size=self.train_batch_size, shuffle=False, num_workers=48)
    
    def val_dataloader(self):
        return DataLoader(self.valset, batch_size=32, shuffle=False, num_workers=48)

    def test_dataloader(self):
        return (DataLoader(self.testset, batch_size=32, shuffle=False, num_workers=48), DataLoader(self.forgetset, batch_size=32, shuffle=False, num_workers=48)) 
    
    def on_test_end(self):
        self.task += 1


# Training
def main():
    #Config
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.set_float32_matmul_precision('medium')
    torch.backends.cudnn.benchmark = True

    #Parsing
    parser = ArgumentParser()
    parser.add_argument("--seed", type=int, default=0, help="set a seed for this experiment")
    parser.add_argument("--groupname", type=str, default="0", help="set a group name for this experiment")
    parser.add_argument("--gpu", type=int, default=0, help="which gpu to use")
    parser.add_argument("--classseq", type=int, default=100, help="what class sequence?")
    parser.add_argument("--buflen", type=int, default=1200, help="what buffer size?")
    parser.add_argument("--divnum", type=int, default=1, help="division num?")
    parser.add_argument("--seqpretrained", type=int, default=0, help="buffer seq pretrained?")
    parser.add_argument("--style", type=str, default="easiest", help="what style?")
 
    args = parser.parse_args()
    #Seed
    pl.seed_everything(args.seed, workers=True)
    torch.use_deterministic_algorithms(True)

    #Wandb stuff
    currentfileseed = os.path.basename(__file__).rstrip(".py") + "-divnum" + str(args.divnum) + "-pretrained" + str(args.seqpretrained) + "-style" + str(args.style) + "-seed" + str(args.seed) 
    wandb.init(project="h3proj", group=args.groupname, config={"divnum":args.divnum, "seed":args.seed, "pretrained": args.seqpretrained, "style":str(args.style)}, name=currentfileseed)
    
    #trainer
    net = Net() 
    dm = cifair100module(classseq=args.classseq)
    wandb.config["classseq"]=str(dm.class_seq)
    trainer = awesometrainer(gpunumber=args.gpu, earlystoppatience=5, currentfileseed=currentfileseed, lenbuffer=args.buflen, divnum=args.divnum, classseq=dm.class_seq)
    
    #training and testing
    for _ in range (20):
        dm.setup()
        trainer.fit(model=net, trainloader=dm.train_dataloader(), valloader=dm.val_dataloader(), pretrained=args.seqpretrained, style=str(args.style))
        net = torch.load(trainer.modelpath)
        trainer.test(model=net, dm=dm)
    wandb.finish()


if __name__ == "__main__": 
    #run
    main()