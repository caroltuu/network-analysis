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

# read arguments:
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='cars', type=str)
parser.add_argument('--server', default=True, action="store_true")
parser.add_argument('--gpu', default=0, type=int)
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--epochs', default=300, type=int)
parser.add_argument('--lr', default=0.1, type=float)
parser.add_argument('--network', default="res18", type=str)
parser.add_argument('--pretrained', default=False, action="store_true")
parser.add_argument('--no_saving', default=False, action="store_true")
args = parser.parse_args()

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

		random_seed = 1
		torch.backends.cudnn.enabled = False
		torch.manual_seed(random_seed)
		transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
		self.train_loader = torch.utils.data.DataLoader(datasets.MNIST('./files/', train=True, download=True, transform=transform), batch_size=batch_size_train, shuffle=True)
		self.test_loader = torch.utils.data.DataLoader(datasets.MNIST('./files/', train=False, download=True, transform=transform), batch_size=batch_size_test, shuffle=True)

		self.network = Net() #LeNet5()
		self.network.cuda()
		self.optimizer = optim.SGD(self.network.parameters(), lr=self.learning_rate, momentum=self.momentum)
		self.train_losses = []
		self.train_counter = []
		self.test_losses = []
		self.test_counter = [i * len(self.train_loader.dataset) for i in range(n_epochs + 1)]

	def train(self, epoch):

		self.network.train()
		for batch_idx, (data, target) in enumerate(self.train_loader):
			self.network.train()
			data, target = data.cuda(), target.cuda()
			self.optimizer.zero_grad()
			output = self.network(data)
			loss = F.nll_loss(output, target)
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
				data, target = data.cuda(), target.cuda()
				output = self.network(data)
				test_loss += F.nll_loss(output, target, size_average=False).item()
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
		#print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(test_loss, correct, len(self.test_loader.dataset), 100. * correct / len(self.test_loader.dataset)))

class RandomNetworks():
	def __init__(self, dataset, gpu, batch_size, epochs, server=True, no_saving=False, network="res18", lr=0.01):
		print("Class initialized")
		self.dataset = dataset
		self.gpu = gpu
		self.batch_size = batch_size
		self.server = server
		self.no_saving = no_saving
		self.num_classes = 0
		self.num_networks = 100
		self.milestones = [20, 40, 60]
		self.n_outputs = 2
		self.n_clusters = 20
		self.network = network
		self.lr = lr

		# choose the gpu
		self.device = 'cuda:' + str(self.gpu)
		print("Training on device:", self.device)
		torch.cuda.set_device(self.device)
		self.device = torch.device(self.device if torch.cuda.is_available() else 'cpu')

	def init_transforms(self):
		self.train_transforms = transforms.Compose([
			transforms.Resize(256),
			transforms.CenterCrop(224),
			# transforms.RandomResizedCrop(224),
			transforms.RandomHorizontalFlip(),
			transforms.ToTensor(),
			transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

		self.val_transforms = transforms.Compose([
			transforms.Resize(256),
			transforms.CenterCrop(224),
			transforms.ToTensor(),
			transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

	def init_data(self):
		if self.dataset == 'cars':
			self.num_classes = 196
			if self.server:
				self.data_dir = "/home/jedrzej/data/Stanford_Cars"
			else:
				self.data_dir = "/media/jedrzej/Seagate/DATA/Cars-196"
		elif self.dataset == 'voc':
			self.num_classes = 20
			if self.server:
				self.data_dir = "/home/jedrzej/data/VOC2012/trainval"
			else:
				self.data_dir = "/media/jedrzej/Seagate/DATA/VOC2012/PyTorch/trainval"

	def init_accuracy(self):
		#self.category_accuracy = np.zeros(self.num_classes)
		self.category_correct = np.zeros(self.num_classes)
		self.category_incorrect = np.zeros(self.num_classes)

	def init_data_loaders(self):
		self.train_dataset = Split_Dataset(self.data_dir, self.data_dir + "/train.txt", self.train_transforms)
		self.valid_dataset = Split_Dataset(self.data_dir, self.data_dir + "/test.txt", self.val_transforms)
		num_workers = 4
		self.train_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=num_workers)
		self.valid_loader = torch.utils.data.DataLoader(self.valid_dataset, batch_size=self.batch_size, shuffle=False, num_workers=num_workers)

	def create_model(self, n_outputs=None, pretrained=False):
		if not n_outputs:
			n_outputs = self.num_classes
		if self.network == "res18":
			model = models.resnet18(pretrained=pretrained)
		elif self.network == "res34":
			model = models.resnet34(pretrained=pretrained)
		elif self.network == "res50":
			model = models.resnet50(pretrained=pretrained)
		else:
			print("Unknown model architecture")
		for param in model.parameters():
			param.requires_grad = True
		n_inputs = model.fc.in_features
		#print(n_inputs, n_outputs)
		last_layer = nn.Linear(n_inputs, n_outputs)
		model.fc = last_layer
		model.to(self.device)
		return model

	def run_all_networks(self):
		self.category_variances = np.zeros(self.num_classes)
		self.all_labels = None
		self.category_means = np.zeros(self.num_classes)
		self.similarity_matrix = np.zeros((len(self.valid_dataset), len(self.valid_dataset)))
		start = time()
		for net_id in range(self.num_networks):
			print("Running network:", net_id+1, "Elapsed:", round(time() - start, 2))
			start = time()
			model = self.create_model(n_outputs=self.n_outputs)
			model.eval()
			self.run_single_network(model)
		self.category_variances = self.category_variances/self.num_networks
		self.category_means = self.category_means / self.num_networks
		pickle.dump(self.similarity_matrix, open("./similarity_matrix.p", "wb"))
		pickle.dump(self.all_labels, open("./all_labels.p", "wb"))
		#pickle.dump(self.category_variances, open("./pre_distance_variances.p", "wb"))
		#pickle.dump(self.category_means, open("./pre_distance_means.p", "wb"))
		'''print("Overall accuracy:")
		self.category_accuracy = self.category_correct/(self.category_correct+self.category_incorrect)
		print(self.category_accuracy)
		pickle.dump(self.category_accuracy, open("./pre_accuracies.p", "wb"))'''

	def run_single_network(self, model):

		all_data = np.zeros((len(self.valid_dataset), self.n_outputs))
		all_labels = np.zeros(len(self.valid_dataset))
		self.all_labels = all_labels # assign the labels
		idx = 0
		for data, target, _ in self.valid_loader:
			batch_size = data.size(0)
			data, target = data.to(self.device), target.to(self.device)
			output = model(data)
			all_labels[idx:idx+batch_size] = target.cpu().data.numpy()
			all_data[idx:idx + batch_size, :] = output.cpu().data.numpy()
			idx += batch_size
			#self.get_accuracy_predictions(output, target)
		#self.get_variances(all_data, all_labels)
		self.similarity_matrix += self.cluster_features(all_data)

	def get_variances(self, data, labels):
		for i in range(self.num_classes):
			indices = np.where(labels == i)[0]
			category_points = np.asarray([data[idx, ...] for idx in indices])
			category_mean = np.mean(category_points, axis=0)
			category_distances = np.asarray([np.linalg.norm(category_points[j, ...] - category_mean) for j in range(len(category_points))])
			self.category_variances[i] += np.var(category_distances)
			self.category_means[i] += np.mean(category_distances)

	def get_accuracy_predictions(self, output, target):
		probs, preds = F.softmax(output, dim=1).max(dim=1)
		equality = (target.data == preds)
		accuracy = equality.type(torch.FloatTensor).cpu().numpy()
		labels = target.cpu().data.numpy()
		for i in range(len(labels)):
			if accuracy[i] == 1:
				self.category_correct[labels[i]] += 1
			else:
				self.category_incorrect[labels[i]] += 1

	def train(self):
		model = self.create_model(n_outputs=self.num_classes)
		print ("Length of the training data:", len(self.train_dataset))
		print ("Length of the validation data:", len(self.valid_dataset))
		valid_loss_min = np.Inf

		# specify loss function
		criterion = nn.CrossEntropyLoss()
		# specify optimizer
		optimizer = optim.SGD(model.parameters(), lr=self.lr, momentum=0.9, weight_decay=0.0005)
		scheduler = optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.001, max_lr=0.01, step_size_up = 100)
		#scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.milestones, gamma=0.1)

		for epoch in range(1, args.epochs + 1):
			startTotal = time()
			self.init_accuracy()
			# keep track of training and validation loss
			train_loss = 0.0
			valid_loss = 0.0

			###################
			# train the model #
			###################
			train_accuracy = 0.
			model.train()
			print("Current learning rate:", optimizer.param_groups[0]['lr'])
			batch_number = 0
			num_batches = int(len(self.train_dataset)/self.batch_size)
			start = time()
			for data, target, paths in self.train_loader:
				if not self.no_saving:
					print("Training: epoch", epoch, "and batch", batch_number,"out of", num_batches, "Time elapsed:", round(time()-start, 2))
				#print(self.no_saving)
				start = time()
				model.train()
				data, target = data.to(self.device), target.to(self.device)
				optimizer.zero_grad()
				output = model(data)
				loss = criterion(output, target)
				loss.backward()
				optimizer.step()

				train_loss += loss.item() * data.size(0)
				probs, preds = F.softmax(output, dim=1).max(dim=1)
				equality = (target.data == preds)
				train_accuracy += equality.type(torch.FloatTensor).sum()
				feat_dim = output.size(1)
				if not self.no_saving:
					train_batch_info = [output.cpu().data.numpy(), target.cpu().data.numpy(), paths, loss.item()]
					pickle.dump(train_batch_info, open("./training_data/train_batch_info_"+str(epoch)+"_"+str(batch_number)+".p", "wb"))

					del train_batch_info
				if not self.no_saving:
					# val run
					model.eval()
					val_targets = np.zeros(len(self.valid_dataset))
					val_predictions = np.zeros(len(self.valid_dataset))
					val_outputs = np.zeros((len(self.valid_dataset), feat_dim))
					k = 0
					for data, target, paths in self.valid_loader:
						data, target = data.to(self.device), target.to(self.device)
						n = data.size(0)
						output = model(data)
						val_outputs[k:k + n, ...] = output.cpu().data.numpy()
						v_loss = criterion(output, target)
						#valid_loss += v_loss.item() * data.size(0)
						probs, preds = F.softmax(output, dim=1).max(dim=1)

						val_targets[k:k+n] = target.cpu().data.numpy()
						val_predictions[k:k+n] = preds.cpu().data.numpy()
					val_batch_info = [val_outputs, val_targets, val_predictions, paths]
					pickle.dump(val_batch_info, open("./training_data/val_batch_info_" + str(epoch) + "_" + str(batch_number) + ".p", "wb"))
					del val_batch_info
					batch_number += 1
				scheduler.step()
			######################
			# validate the model #
			######################
			accuracy = 0.
			model.eval()
			for data, target, _ in self.valid_loader:
				data, target = data.to(self.device), target.to(self.device)
				output = model(data)
				v_loss = criterion(output, target)
				valid_loss += v_loss.item() * data.size(0)
				probs, preds = F.softmax(output, dim=1).max(dim=1)
				equality = (target.data == preds)
				accuracy += equality.type(torch.FloatTensor).sum()
				temp_accuracy = equality.type(torch.FloatTensor).cpu().numpy()
				labels = target.cpu().data.numpy()
				for i in range(len(labels)):
					if temp_accuracy[i] == 1:
						self.category_correct[labels[i]] += 1
					else:
						self.category_incorrect[labels[i]] += 1

			# calculate average losses
			train_loss = train_loss / len(self.train_dataset)
			valid_loss = valid_loss / len(self.valid_dataset)

			# print training/validation statistics
			# print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(epoch, train_loss, valid_loss))

			print(
			"Epoch:", epoch, "Train accuracy:", round(train_accuracy.data.cpu().numpy() / (len(self.train_dataset)), 3),
			"Validation accuracy:", round(accuracy.data.cpu().numpy() / (len(self.valid_dataset)), 3), "Elapsed:",
			round(time() - startTotal, 2))

			# save model if validation loss has decreased
			if valid_loss <= valid_loss_min:
				print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
					valid_loss_min,
					valid_loss))
				torch.save(model.state_dict(), './models/' + self.dataset + '.pt')
				valid_loss_min = valid_loss
				#print("Overall accuracy:")
				self.category_accuracy = self.category_correct / (self.category_correct + self.category_incorrect)
				#print(self.category_accuracy)
				pickle.dump(self.category_accuracy, open("./post_accuracies.p", "wb"))

	def cluster_features(self, features):
		n_features = len(features)
		n_dim = len(features[0])
		affinity_matrix = np.zeros((n_features, n_features))
		try:
			kmeans = MiniBatchKMeans(n_clusters=self.n_clusters).fit(features)
			for j in range(self.n_clusters):
				indices = np.where(kmeans.labels_ == j)[0]
				for pair in itertools.product(indices, repeat=2):
					affinity_matrix[pair[0], pair[1]] += 1
			del kmeans, indices
		except:
			print("Error clustering network output")
		return affinity_matrix

'''random_networks = RandomNetworks(args.dataset, args.gpu, args.batch_size, args.epochs, args.server, args.no_saving, args.network, args.lr)
random_networks.init_transforms()
random_networks.init_data()
random_networks.init_accuracy()
random_networks.init_data_loaders()
random_networks.train()'''
#random_networks.run_all_networks()

for n_net in range(50):
	mnist = MNIST(name=str(n_net))
	n_epochs = 11
	mnist.test()
	for epoch in range(1, n_epochs + 1):
		if epoch in [5, 8, 9]:
			mnist.learning_rate /= 10
			mnist.optimizer = optim.SGD(mnist.network.parameters(), lr=mnist.learning_rate, momentum=mnist.momentum)
		mnist.train(epoch)
		#mnist.test()
