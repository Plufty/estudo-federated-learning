#Instalações necessárias:
#pip install tdqm
#pip install matplotlib
#pip install efficientnet_pytorch
#pip install ray
#pip install --upgrade --force-reinstall torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

import warnings
from collections import OrderedDict
from typing import List, Tuple

import flwr as fl
from flwr .common import Metrics, Context

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision.transforms import Compose, Normalize, ToTensor
from torchvision.datasets import ImageFolder, CIFAR10
import torchvision.models as models
from torchvision import transforms

from tqdm import tqdm
import random
from efficientnet_pytorch import EfficientNet

import matplotlib.pyplot as plt
import os
import time
import numpy as np
import datetime

warnings.filterwarnings("ignore", category=UserWarning)
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
START = time.time()
MODEL = "alexnet"
NUM_CLASSES = 10
DATE_NOW = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
CLASSES =  ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


class Net(nn.Module):
	def __init__(self) -> None:
		super(Net, self).__init__()
		self.conv1 = nn.Conv2d(3,6,5)
		self.pool=nn.MaxPool2d(2,2)
		self.conv2 = nn.Conv2d(6, 16, 5)
		self.fc1 = nn.Linear(16*5*5, 120)
		self.fc2 = nn.Linear(120, 84)
		self.fc3 = nn.Linear(84, 10)
	
	def forward(self, x: torch.Tensor) -> torch.Tensor:
		x = self.pool(F.relu(self.conv1(x)))
		x = self.pool(F.relu(self.conv2(x)))
		x = x.view(-1, 16 *5 *5)
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		x = self.fc3(x)
		return x





def train(net, trainloader, epochs: int, verbose=False):
	criterion = torch.nn.CrossEntropyLoss()
	optimizer = torch.optim.Adam(net.parameters())
	net.train()
	for epoch in range(epochs):
		correct, total, epoch_loss = 0, 0, 0.0
		for images, labels in trainloader:
			images, labels = images.to(DEVICE), labels.to(DEVICE)
			optimizer.zero_grad()
			outputs = net(images)
			loss = criterion(net(images), labels)
			loss.backward()
			optimizer.step()
			#metrics
			epoch_loss += loss
			total += labels.size(0)
			correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
		epoch_loss /= len(trainloader.dataset)
		epoch_acc = correct / total
		if verbose:
			print(f"Epoch {epoch+1}: train loss {epoch_loss}, accuracy {epoch_acc}")

def test(net, testloader):
	criterion = torch.nn.CrossEntropyLoss()
	correct, total, loss = 0, 0, 0.0
	net.eval()
	with torch.no_grad():
		for images, labels in testloader:
			images, labels = images.to(DEVICE), labels.to(DEVICE)
			outputs = net(images)
			loss += criterion(outputs, labels).item()
			_, predicted = torch.max(outputs.data, 1)
			total += labels.size(0)
			correct += (predicted == labels).sum().item()
	loss /= len(testloader.dataset)
	accuracy = correct / total
	return loss, accuracy

NUM_CLIENTS=1
SEED = 42
BATCH_SIZE = 16

def load_data():
    # Load the breast cancer dataset (modify the paths accordingly)
    input_size = 224
    data_transforms = {
        'transform': transforms.Compose([
		transforms.Resize([input_size, input_size], antialias=True),
		transforms.ToTensor(),
		transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
	    ]),
        'teste': transforms.Compose([
		transforms.ToTensor(),
		transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
	    ])
    }

    # trainset = ImageFolder("./data/train", transform=data_transforms['transform'])
    # testset = ImageFolder("./data/test", transform=data_transforms['transform'])
    #return DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True), DataLoader(testset)


    trainset = CIFAR10("./dataset", train=True, download=True, transform=data_transforms['teste'])
    testset = CIFAR10("./dataset", train=False, download=True, transform=data_transforms['teste'])

    #Split training set into 10 partitions to simulate the individual dataset
    partition_size = len(trainset) // NUM_CLIENTS
    lengths = [partition_size] * NUM_CLIENTS
    datasets = random_split(trainset, lengths, torch.Generator().manual_seed(SEED))

    #split each partition into train/val and create DataLoader
    trainloaders = []
    valloaders = []

    for ds in datasets:
        len_val = len(ds)//10 # 10% validation set
        len_train = len(ds)-len_val
        lengths = [len_train, len_val]
        ds_train, ds_val = random_split(ds, lengths, torch.Generator().manual_seed(SEED))
        trainloaders.append(DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=True))
        valloaders.append(DataLoader(ds_val, batch_size=BATCH_SIZE))
    testloaders = DataLoader(testset, batch_size=BATCH_SIZE)

    return trainloaders, valloaders, testloaders

# net = Net().to(DEVICE)
# trainloaders, valloaders, testloader = load_data()
# trainloader = trainloaders[0]
# valloader = valloaders[0]

# loss_per_epoch = []
# accuracy_per_epoch = []

# print(f"Training on {DEVICE} using PyTorch {torch.__version__} and Flower {fl.__version__}")

# for epoch in range(10):
# 	train(net, trainloader, 1)
# 	loss, accuracy = test(net, valloader)
# 	loss_per_epoch.append(loss)
# 	accuracy_per_epoch.append(accuracy)
# 	print(f"Epoch {epoch+1}: validation loss {loss}, accuracy: {accuracy}")
# loss, accuracy = test(net, testloader)
# loss_per_epoch.append(loss)
# accuracy_per_epoch.append(accuracy)
# print(f"Final test set performance: \n\tloss {loss} \n\taccuracy {accuracy}")



def get_parameters(net) -> List[np.ndarray]:
    return [val.cpu().numpy() for _, val in net.state_dict().items()]

def set_parameters(net, parameters: List[np.ndarray]):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)

class FlowerClient(fl.client.NumPyClient):
    def __init__(self, net, trainloader, valloader):
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader

    def get_parameters(self, config):
        return get_parameters(self.net)

    def fit(self, parameters, config):
        set_parameters(self.net, parameters)
        train(self.net, self.trainloader, epochs=1)
        return get_parameters(self.net), len(self.trainloader), {}

    def evaluate(self, parameters, config):
        set_parameters(self.net, parameters)
        loss, accuracy = test(self.net, self.valloader)
        return float(loss), len(self.valloader), {"accuracy": float(accuracy)}

trainloaders, valloaders, testloader = load_data()
#Def Client
def client_fn(cid: str) -> FlowerClient:
    net = Net().to(DEVICE)


    trainloader = trainloaders[int(cid)]
    valloader = valloaders [int(cid)]
    return FlowerClient(net, trainloader, valloader)

def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
  accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
  examples = [num_examples for num_examples, _ in metrics]
  return {"accuracy": sum(accuracies)/sum(examples)}

#Create FedAVG strategys
strategy = fl.server.strategy.FedAvg(
    fraction_fit=1.0, #sample 100% of available clients for training
    fraction_evaluate=0.5, #sample 50% of available clients for evaluation
    min_fit_clients=1, #never sambple less than 10 clients for training
    min_evaluate_clients=1, #never sample less than 5 clients for evaluation
    min_available_clients=1, #wait until all 10 clients are available
    evaluate_metrics_aggregation_fn=weighted_average,
)
#Start Simulation
output = fl.simulation.start_simulation(
    client_fn=client_fn,
    num_clients =NUM_CLIENTS,
    config= fl.server.ServerConfig(num_rounds=2),
    strategy=strategy,
)

print("Output FL - Loss: ", output.losses_distributed)
print("Output FL - Accuracy: ", output.metrics_distributed["accuracy"])

fl_accuracy = []
fl_loss = []

for loss in output.losses_distributed:
      print(f"Loss in Round {loss[0]}: {loss[1]}")
      fl_loss.append(loss[1])
      

for acc in output.metrics_distributed["accuracy"]:
      print(f"Accuracy in Round {acc[0]}: {acc[1]}")
      fl_accuracy.append(acc[1])
      
fig = plt.figure()
ax = plt.axes()

x = range(len(fl_loss))
ax.plot(x, fl_loss)
plt.xlabel("Rounds")
plt.ylabel("Loss")

fig2 = plt.figure()
ax2 = plt.axes()

x = range(len(fl_accuracy))
ax2.plot(x, fl_accuracy)
plt.xlabel("Rounds")
plt.ylabel("Accuracy")
