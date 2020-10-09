import numpy as np
import torch
import torchvision
from torchvision import datasets, models
from torch import nn
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.optim as optim
import argparse
from time import time
from torch.utils.data.sampler import SubsetRandomSampler
from dataloader import Split_Dataset
import pickle
import itertools
from sklearn.cluster import MiniBatchKMeans
from models import LeNet5, Net
import matplotlib.pyplot as plt

class MNIST():
	def __init__(self, name):

		self.name = name
		n_epochs = 3
		batch_size_train = 256
		batch_size_test = 1000
		self.learning_rate = 0.01
		self.momentum = 0.5
		self.log_interval = 25
		self.num_classes = 10
		self.count = 0

		device_id = 0		# which gpu to use
		self.device = self.device = torch.device("cuda:"+str(device_id))

		transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
		self.train_loader = torch.utils.data.DataLoader(datasets.MNIST('./MNIST/', train=True, download=True, transform=transform), batch_size=batch_size_train, shuffle=True)
		self.test_loader = torch.utils.data.DataLoader(datasets.MNIST('./MNIST/', train=False, download=True, transform=transform), batch_size=batch_size_test, shuffle=True)

		self.network = Net()
		self.network.cuda()
		self.optimizer = optim.SGD(self.network.parameters(), lr=self.learning_rate, momentum=self.momentum)
		self.criterion = nn.CrossEntropyLoss()
		self.train_losses = []
		self.train_counter = []
		self.test_losses = []
		self.test_counter = [i * len(self.train_loader.dataset) for i in range(n_epochs + 1)]

		for epoch in range(n_epochs):
			self.train(epoch)

	def train(self, epoch):

		self.network.train()
		for batch_idx, (data, target) in enumerate(self.train_loader):
			self.network.train()
			data, target = data.to(self.device), target.to(self.device)
			self.optimizer.zero_grad()
			output = self.network(data)
			loss = self.criterion(output, target)
			loss.backward()
			self.optimizer.step()
			if batch_idx % self.log_interval == 0:
				self.count += 1
				torch.save(self.network.state_dict(), './results/'+self.name+'_model_'+ str(self.count).zfill(6) + '.pth')
				pickle.dump([data.cpu().numpy(), target.cpu().numpy(), loss.data.cpu()], open('./results/'+self.name+'_train_' + str(self.count).zfill(6) + '.p', 'wb'))

				print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, batch_idx * len(data), len(self.train_loader.dataset), 100. * batch_idx / len(self.train_loader), loss.item()))
				self.train_losses.append(loss.item())
				self.train_counter.append((batch_idx * 64) + ((epoch - 1) * len(self.train_loader.dataset)))
				#torch.save(self.network.state_dict(), './results/model.pth')
				#torch.save(self.optimizer.state_dict(), './results/optimizer.pth')
				self.test()

	def test(self):
		self.network.eval()
		test_loss = 0
		correct = 0
		accuracies = np.zeros(self.num_classes)
		count = np.zeros(self.num_classes)
		with torch.no_grad():
			for data, target in self.test_loader:
				data, target = data.to(self.device), target.to(self.device)
				output = self.network(data)
				test_loss += self.criterion(output, target).item()
				pred = output.data.max(1, keepdim=True)[1]
				corr = pred.eq(target.data.view_as(pred)).cpu().numpy()
				temp_target = target.cpu().numpy()
				indices = np.where(corr == 1)[0]
				for idx in indices:
					accuracies[temp_target[idx]] += 1
				for lbl in temp_target:
					count[lbl] += 1
				correct += pred.eq(target.data.view_as(pred)).sum()
		accuracies = accuracies/count
		test_loss /= len(self.test_loader.dataset)
		plt.figure(figsize=(16, 4))
		plt.bar(range(self.num_classes), accuracies, align='center')
		plt.xlabel("Category")
		plt.xticks(range(10))
		plt.ylabel("Accuracy")
		plt.ylim([0, 1])
		for i, v in enumerate(accuracies):
			plt.text(i-0.1, 1.05, str(round(v,2)), color='blue', fontweight='bold')
		plt.savefig("./plots/"+self.name+"_accuracy_" + str(self.count).zfill(6) + ".jpg")
		plt.close()
		pickle.dump(accuracies, open('./results/'+self.name+'_test_' + str(self.count).zfill(6) + '.p', 'wb'))

		self.test_losses.append(test_loss)
		print('Test set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(test_loss, correct, len(self.test_loader.dataset), 100. * correct / len(self.test_loader.dataset)))


if __name__ == "__main__":
	mnist = MNIST(name="trial1")