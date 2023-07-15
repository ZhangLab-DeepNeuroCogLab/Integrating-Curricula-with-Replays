# sampling of exemplers is random, final trainloader is random, initial trainloader is constant (dont matter), determined classseq
import os
import torch
from torch import nn
from torchvision import transforms
from torch.utils.data import DataLoader
from cifair import ciFAIR100
import wandb
import torch.optim as optim
from torchvision.models import mobilenet_v3_small
import pytorch_lightning as pl
from argparse import ArgumentParser
from torch import autocast
import random

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
    
    def accuracy(self, preds, labels):
        # total accuracy is equal to mean accuracy of each class since the number of samples in each classes is the same in all val and test batch
        acc = torch.sum(preds == labels)
        return acc.item()/labels.size(dim=0)
    
    def calc_loss_acc(self, batch, stage=None):
        images, labels = batch
        with autocast(device_type='cuda'):
            output = self(images)
            loss = self.lossfunc(output, labels)
        _, preds = torch.max(output, 1)
        if stage == "train":
            return loss
        else:
            acc = self.accuracy(preds, labels)
            return loss, acc
    
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


#trainer
class awesometrainer:
    def __init__(self, gpunumber, earlystoppatience, currentfileseed, lenbuffer):
        self.device = f"cuda:{gpunumber}"
        self.earlystoppatience = earlystoppatience
        self.epoch = -1
        self.modelpath = 0
        self.modeldir = f"./checkpoints/{currentfileseed}/"
        self.lenbuffer = lenbuffer
        self.replaybuffer = {}
        os.makedirs(self.modeldir, exist_ok=True)

    def fit(self, model, trainloader, valloader):
        continuetraining = True
        optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
        model.to(self.device)
        num_batches = len(trainloader)
        minvalloss = 100
        lenbufferperclass = self.lenbuffer // (model.task * 5)
        if lenbufferperclass > 450:
            lenbufferperclass = 450
            num_of_lenbufferperclass_plus_one = 0
        else:
            num_of_lenbufferperclass_plus_one = self.lenbuffer - lenbufferperclass*(model.task*5)

        
        buftrainset = []

        for i in self.replaybuffer:
            buftrainset += self.replaybuffer[i][:int(len(self.replaybuffer[i])*0.9)]

        newtrainset = trainloader.dataset+buftrainset
        print("newtrainsetlen: ", len(newtrainset))
        del buftrainset

        for img, label in trainloader.dataset:
            if label in self.replaybuffer:
                self.replaybuffer[label].append((img, label))
            else:
                self.replaybuffer[label] = [(img, label)]
        
        k = 0
        for i in self.replaybuffer:
            k += 1
            if k > num_of_lenbufferperclass_plus_one:
                self.replaybuffer[i] = random.sample(self.replaybuffer[i], lenbufferperclass)
            else:
                self.replaybuffer[i] = random.sample(self.replaybuffer[i], lenbufferperclass + 1)
        for i in self.replaybuffer:
            print(len(self.replaybuffer[i]))

        del trainloader.dataset
        
        trainloader = DataLoader(newtrainset, batch_size=32, shuffle=True)

        while continuetraining:
            self.epoch += 1
            running_loss = 0
            for batch in trainloader:
                optimizer.zero_grad()
                batch = [x.to(self.device) for x in batch]
                loss = model.calc_loss_acc(batch, stage="train")
                running_loss += loss
                loss.backward()
                optimizer.step()

            valdata = [x.to(self.device) for x in next(iter(valloader))]
            model.eval()
            with torch.no_grad():
                valloss, valacc = model.calc_loss_acc(valdata)
            model.train()
            del valdata

            logdict = {
                "Epoch": self.epoch,
                "Train Loss": running_loss.item()/num_batches,
                "Valid Loss": valloss.item(),
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
                                    
    def test(self, model, dm):
        testdata = [x.to(self.device) for x in next(iter(dm.test_dataloader()))]
        model.eval()
        with torch.no_grad():
            testloss, testacc = model.calc_loss_acc(testdata)
        model.train()
        del testdata

        logdict = {
            "Epoch": self.epoch,
            "Task": model.task,
            "Test Loss": testloss.item(),
            "Test Acc": testacc
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
                                            transforms.Resize((74, 74), interpolation=transforms.InterpolationMode.BICUBIC)])
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

    def train_dataloader(self):
        return DataLoader(self.trainset, batch_size=self.train_batch_size, shuffle=False, num_workers=64)
    
    def val_dataloader(self):
        return DataLoader(self.valset, batch_size=len(self.valset), shuffle=False, num_workers=64)

    def test_dataloader(self):
        return DataLoader(self.testset, batch_size=len(self.testset), shuffle=False, num_workers=64)

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
    parser.add_argument("--classseq", type=int, default=0, help="what class sequence?")
    parser.add_argument("--buflen", type=int, default=0, help="what buffer size?")
    args = parser.parse_args()

    #Seed
    pl.seed_everything(args.seed, workers=True)
    torch.use_deterministic_algorithms(True)

    #Wandb stuff
    currentfileseed = os.path.basename(__file__).rstrip(".py") + "-" + str(args.seed) + "-classseq" + str(args.classseq)
    wandb.init(project="h3proj", group=args.groupname, config={"seed":args.seed}, name=currentfileseed)
    
    #trainer
    net = Net() 
    dm = cifair100module(classseq=args.classseq)
    wandb.config["classseq"]=str(dm.class_seq)
    trainer = awesometrainer(gpunumber=args.gpu, earlystoppatience=5, currentfileseed=currentfileseed, lenbuffer=args.buflen)
    
    #training and testing
    for _ in range (20):
        dm.setup()
        trainer.fit(model=net, trainloader=dm.train_dataloader(), valloader=dm.val_dataloader())
        net = torch.load(trainer.modelpath)
        trainer.test(model=net, dm=dm)
    wandb.finish()


if __name__ == "__main__": 
    #run
    main()