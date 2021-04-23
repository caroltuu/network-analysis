import numpy as np
import torch
import random
from torchvision import datasets
from torch import nn
import torchvision.transforms as transforms
import torch.optim as optim
from torch.utils.data.sampler import SubsetRandomSampler
import pickle
from models import Net
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.ticker import AutoMinorLocator
from sklearn.decomposition import PCA
 
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
 
  def return_CAM(feature_conv, weight, class_idx):
    # generate the class -activation maps upsample to 256x256
    size_upsample = (256, 256)
    bz, nc, h, w = feature_conv.shape
    output_cam = []
    for idx in class_idx:
      beforeDot =  feature_conv.reshape((nc, h*w))
      cam = np.matmul(weight[idx], beforeDot)
      cam = cam.reshape(h, w)
      cam = cam - np.min(cam)
      cam_img = cam / np.max(cam)
      cam_img = np.uint8(255 * cam_img)
      output_cam.append(cv2.resize(cam_img, size_upsample))
    return output_cam
 
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
 
  def dict_to_np(self, dic):
    arr = []
    for key, val in dic.items():
      arr.append(val)
    return np.asarray(arr)

  def test(self):
    # set it to eval mode
    print('started testing on', len(self.networks), 'models')
    for ii, model in enumerate(self.networks[0:1]):
      model.to(self.device)
      model.eval()
      test_loss = 0
      correct = 0
      accuracies = np.zeros(self.num_classes)
      count = np.zeros(self.num_classes)
 
      inputs = []
      outputs = []
      labels = []

      act_map_labels = {}
      act_maps = {}
 
      with torch.no_grad():
        for data, target in self.test_loader:
          # test pass:
          data, target = data.to(self.device), target.to(self.device)
          output = model(data)
          act_map = model.act_map
          test_loss += self.criterion(output, target).item()
          pred = output.data.max(1, keepdim=True)[1]
          corr = pred.eq(target.data.view_as(pred)).cpu().numpy()
 
          # keep track of the data:
          inputs.extend(data.cpu().numpy())
          labels.extend(target.cpu().numpy())

          np_output = output.cpu().detach().numpy()
          np_act_map = act_map.cpu().numpy()
          np_target = target.cpu().numpy()

          for kernel_i in range(np_act_map.shape[1]):
            if not kernel_i in act_maps:
              act_maps[kernel_i] = []
              act_map_labels[kernel_i] = []
            act_maps[kernel_i].extend(np_act_map[:, kernel_i, :, :])
            act_map_labels[kernel_i].extend(np_target)

          outputs.extend(np_output)

          # get per category accuracies
          indices = np.where(corr == 1)[0]
          for idx in indices:
            accuracies[np_target[idx]] += 1
          for lbl in np_target:
            count[lbl] += 1
          correct += pred.eq(target.data.view_as(pred)).sum()
      
      #  save the inputs, labels and outputs and accuracies
      # pickle.dump([inputs, labels, outputs, accuracies/count], open('./results/' + self.name + '_test_' + str(self.count).zfill(6) + '.p', 'wb'))
      #  plot the accuracies
      # self.plot_accuracies(accuracies/count)
      
      for kernel, act_map in act_maps.items():
        print('layer', kernel)
        # plot = plt.figure(str(kernel))
        # self.plot_act_maps(kernel, np.asarray(act_map), np.asarray(act_map_labels[kernel]))
        # plt.savefig('./plots/act_maps/PCA' + str(kernel) + '.jpg', dpi=300)
        # plt.show()
        
      self.plot_label_to_kernel(self.dict_to_np(act_maps), self.dict_to_np(act_map_labels))
      
      #self.plot_act_maps_all('all maps', act_maps)


      # print('labels', np.asarray(labels).shape)
      # print('inputs', np.asarray(inputs).shape)
      # print('outputs', np.asarray(outputs).shape)
      # print('act_map', np.asarray(act_maps[0]).shape)

      print('Test set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(test_loss, correct, len(self.test_loader.dataset), 100. * correct / len(self.test_loader.dataset)))

  def plot_label_to_kernel(self, act_maps, labels):
    print('actmaps', act_maps.shape)
    print('actmaplabels', labels.shape)
    
    num_kernels = act_maps.shape[0]
    num_images = labels.shape[1]
    num_labels = np.unique(labels).shape[0]

    matrix = np.zeros((num_kernels, num_labels)) # kernels * labels

    column_totals = np.zeros((num_kernels))

    for kern in range(num_kernels):
      for img in range(num_images):
        flattened_kernel = act_maps[kern][img].flatten()
        kernel_norm = np.linalg.norm(flattened_kernel)
        matrix[kern][labels[kern][img]] += kernel_norm
        column_totals[labels[kern][img]] += 1
    
    column_totals /= num_kernels
    
    # normalize matrix
    for label in range(num_labels):
      matrix[:, label] /= column_totals[label]

    for kern in range(num_kernels):
      matrix[kern, :] -= np.amin(matrix[kern, :])
      matrix[kern, :] /= np.amax(matrix[kern, :])

    cmap = colors.ListedColormap([(i/255, i/255, i/255) for i in range(255)])

    fig, ax = plt.subplots()
    ax.imshow(matrix, cmap=cmap)

    plt.xlabel("Labels")
    ax.set_xticks(np.arange(-0.5, num_labels, 1))
    ax.set_xticklabels([])

    plt.ylabel("Kernels")
    ax.set_yticks(np.arange(-.5, num_kernels, 1))
    ax.set_yticklabels([])

    # draw gridlines
    ax.grid(which='major', axis='both', linestyle='-', color='k', linewidth=2)

    plt.show()


  def plot_act_maps(self, layer, raw_act_map, labels):
    act_map = []
    dims = raw_act_map.shape
    for i in range(dims[0]):
      act_map.append(raw_act_map[i].flatten())
    act_map = np.asarray(act_map)

    trans = PCA(n_components=2).fit_transform(act_map)
    
    unique_labels = np.unique(labels)

    for i in unique_labels:
      indices = np.where(labels == i)[0]
      plt.scatter(trans[indices, 0], trans[indices, 1], s=5, label=str(i))
    plt.legend() 

    '''for i in range(labels.shape[0]):
      plt.plot(trans[i, 0], trans[i, 1], 'ro', color = colors[labels[i]])'''

  def plot_act_maps_all(self, layer, act_maps):
    plot = plt.figure(layer)
    for layer, act_map in act_maps.items():
      self.plot_act_maps(layer, np.asarray(act_map), color=[np.random.rand(3,)])
    plt.savefig('./plots/act_maps/' + str(layer) + '.jpg')

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
    print(outputs[0])
