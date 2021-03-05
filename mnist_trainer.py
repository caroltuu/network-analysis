import numpy as np
import torch
from torchvision import datasets
from torch import nn
import torchvision.transforms as transforms
import torch.optim as optim
from torch.utils.data.sampler import SubsetRandomSampler
import pickle
from models import Net
import matplotlib.pyplot as plt

from utils import initialize_folders

class MNIST():
    def __init__(self, name, n_models=2, gpu=-1):

        # set the parameters:
        self.name = name		# name that will be used to save files
        self.n_epochs = 1
        self.batch_size_train = 256
        self.batch_size_test = 1000
        self.learning_rate = 0.01
        self.momentum = 0.5
        self.log_interval = 2
        self.num_classes = 10
        self.count = 0
        self.n_models = n_models

        # set the device:
        if gpu < 0:
            gpu = None
        device_id = gpu  # which gpu to use (None for CPU; 0,1,2 for GPU)
        if device_id is not None:
            self.device = torch.device("cuda:" + str(device_id))
        else:
            self.device = torch.device("cpu")

        # initialize the folders:
        initialize_folders()
        # initialize the dataloaders:
        self.initialize_dataloaders()
        # initilize the network:
        self.initialize_networks()

    def initialize_dataloaders(self):
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        self.train_loader = torch.utils.data.DataLoader(
            datasets.MNIST('./MNIST/', train=True, download=True, transform=transform), batch_size=self.batch_size_train,
            shuffle=True)
        self.test_loader = torch.utils.data.DataLoader(
            datasets.MNIST('./MNIST/', train=False, download=True, transform=transform), batch_size=self.batch_size_test,
            shuffle=True)

    def initialize_networks(self):
        self.networks = []
        self.optimizers = []
        for i in range(self.n_models):
            model = Net()
            model.apply(self.init_weights)
            self.networks.append(model)
            self.optimizers.append(optim.SGD(model.parameters(), lr=self.learning_rate, momentum=self.momentum))
        self.criterion = nn.CrossEntropyLoss()

    def init_weights(self, m):
        if type(m) == nn.Conv2d:
            #torch.nn.init.xavier_uniform(m.weight)
            #torch.nn.init.xavier_normal(m.weight)

            #torch.nn.init.kaiming_uniform(m.weight)
            #torch.nn.init.kaiming_normal(m.weight)
            pass



    def train(self):
        for epoch in range(self.n_epochs):
            self.train_epoch(epoch)
        for i, model in enumerate(self.networks):
          torch.save(model, './results/'+self.name+'_'+str(i)+'_model_'+ str(self.count).zfill(6) + '.pth')

    def train_epoch(self, epoch):
        for batch_idx, (data, target) in enumerate(self.train_loader):
            if batch_idx % self.log_interval == 0:
                # variable to keep track of the order of saved files:
                self.count += 1
            for ii, model in enumerate(self.networks):
                # training part:
                model.train()		# set it to train mode
                model.to(self.device)
                data, target = data.to(self.device), target.to(self.device)
                self.optimizers[ii].zero_grad()
                output = model(data)
                loss = self.criterion(output, target)
                loss.backward()
                self.optimizers[ii].step()
                model.to(torch.device("cpu"))

                # after every batch dump the current parameters, data and test:
                if batch_idx % self.log_interval == 0:
                    # save the inputs, labels, outputs and loss
                    #pickle.dump([data.cpu().numpy(), target.cpu().numpy(), output.cpu().detach().numpy(), loss.data.cpu()], open('./results/'+self.name+'_train_' + str(self.count).zfill(6) + '.p', 'wb'))

                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, batch_idx * len(data), len(self.train_loader.dataset), 100. * batch_idx / len(self.train_loader), loss.item()))
            #if batch_idx % self.log_interval == 0:
                # perform a test pass:
                #self.test()

    def test(self):
        # set it to eval mode
        for ii, model in enumerate(self.networks):
            model.to(self.device)
            model.eval()
            test_loss = 0
            correct = 0
            accuracies = np.zeros(self.num_classes)
            count = np.zeros(self.num_classes)

            inputs = []
            outputs = []
            labels = []

            with torch.no_grad():
                for data, target in self.test_loader:
                    # test pass:
                    data, target = data.to(self.device), target.to(self.device)
                    output = model(data)
                    test_loss += self.criterion(output, target).item()
                    pred = output.data.max(1, keepdim=True)[1]
                    corr = pred.eq(target.data.view_as(pred)).cpu().numpy()

                    # keep track of the data:
                    inputs.extend(data.cpu().numpy())
                    labels.extend(target.cpu().numpy())
                    outputs.extend(output.cpu().detach().numpy())

                    # get per category accuracies
                    temp_target = target.cpu().numpy()
                    indices = np.where(corr == 1)[0]
                    for idx in indices:
                        accuracies[temp_target[idx]] += 1
                    for lbl in temp_target:
                        count[lbl] += 1
                    correct += pred.eq(target.data.view_as(pred)).sum()

            # save the inputs, labels and outputs and accuracies
            #pickle.dump([inputs, labels, outputs, accuracies/count], open('./results/' + self.name + '_test_' + str(self.count).zfill(6) + '.p', 'wb'))
            # plot the accuracies
            #self.plot_accuracies(accuracies/count)

            print('Test set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(test_loss, correct, len(self.test_loader.dataset), 100. * correct / len(self.test_loader.dataset)))

    def plot_accuracies(self, accuracies):
        plt.figure(figsize=(16, 4))
        plt.bar(range(self.num_classes), accuracies, align='center')
        plt.xlabel("Category")
        plt.xticks(range(10))
        plt.ylabel("Accuracy")
        plt.ylim([0, 1])
        for i, v in enumerate(accuracies):
            plt.text(i - 0.1, 1.05, str(round(v, 2)), color='blue', fontweight='bold')
        plt.savefig("./plots/" + self.name + "_accuracy_" + str(self.count).zfill(6) + ".jpg")
        plt.close()
        #pickle.dump(accuracies, open('./results/' + self.name + '_test_' + str(self.count).zfill(6) + '.p', 'wb'))
