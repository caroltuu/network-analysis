import torch
from os import path
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from glob import glob
from numpy.linalg import norm
import scipy.stats as st
from torchvision import utils
from numpy.linalg import eigh, solve
from numpy.random import randn

class Visualizer():
  def __init__(self, name, visualize=False):
    self.name = name

    self.get_all_models()

    # if visualize:
      # self.access_all_models_layer(layer_name="conv1", visualize=visualize)
      # self.access_all_models_layer(layer_name="fc1",  visualize=visualize)

  def get_all_models(self):
    all_results_list = glob("./results/*")
    self.ordered_model_list = []
    for file in all_results_list:
      if self.name in file and "model" in file and ".pth" in file:
        self.ordered_model_list.append(file)
    n_models = len(self.ordered_model_list)
    print(self.ordered_model_list)
    '''
    for i in range(1, n_models+1):
      model_path = './results/' + self.name + '_model_' + str(i).zfill(6) + '.pth'
      if path.exists(model_path):
        print(model_path, 'exists')
        self.ordered_model_list.append('./results/' + self.name + '_model_' + str(i).zfill(6) + '.pth')
      else:
        print(model_path, 'does not exist')
        break
    '''

  def access_all_models_layer(self, layer_name, visualize=False):
    weights, biases = [], []
    for model in self.ordered_model_list:
      w, b = self.access_model_layer(model, layer_name=layer_name)
      weights.append(w)
      biases.append(b)
    weights, biases = np.asarray(weights), np.asarray(biases)

    # if visualize:
      # visualize the weights:
      # self.euclideanNorm(weights)
      # self.visualize_weights_separately(weights)
      # self.visualize_weights_all(weights)
      # self.visualize_biases(biases)

    return weights, biases

  def access_model_layer(self, model_path, layer_name):
    weight, bias = None, None
    model = torch.load(model_path)
    for name, param in model.state_dict().items():
      if layer_name in name:
        if "weight" in name:
          #print(param.size())
          weight = param.numpy()
        if "bias" in name:
          bias = param.numpy()

        #early stopping:
        if bias is not None and weight is not None:
          return weight, bias

    return weight, bias

  def visualize_weights_separately(self, weights):

    # to visualize the CONV layers:
    if len(np.shape(weights)) > 3:
      weights = weights.reshape((np.shape(weights)[0], np.shape(weights)[1], -1))
    new_weights = []
    for i in range(np.shape(weights)[1]):
      new_weights.extend(weights[:, i, :])
    new_weights = np.asarray(new_weights)
    tsne_weights = TSNE(n_components=2).fit_transform(new_weights)
    for i in range(np.shape(weights)[1]):
      n = len(weights)
      x = tsne_weights[n*i:n*(i+1), 0]
      y = tsne_weights[n*i:n*(i+1), 1]
      plt.scatter(x, y)
      plt.plot(x, y)
    plt.show()

  def visualize_weights_all(self, weights):
    # to visualize the CONV layers:
    if len(np.shape(weights)) > 3:
      weights = weights.reshape((np.shape(weights)[0], np.shape(weights)[1], -1))

    reshaped_weights = weights.reshape((np.shape(weights)[0], -1))
    tsne_biases = TSNE(n_components=2).fit_transform(reshaped_weights)
    plt.scatter(tsne_biases[:, 0], tsne_biases[:, 1])
    plt.plot(tsne_biases[:, 0], tsne_biases[:, 1])
    plt.show()

  def visualize_biases(self, biases):
    print(biases.shape)
    tsne_biases = TSNE(n_components=2).fit_transform(biases)
    plt.scatter(tsne_biases[:, 0], tsne_biases[:, 1])
    plt.plot(tsne_biases[:, 0], tsne_biases[:, 1])
    plt.show()

class ModelComparator():
  def __init__(self, name, n_models, normgraphtype):
    self.n_models = n_models
    print('n_models', n_models)
    self.names = [name+"_"+str(i) for i in range(n_models)]
    self.visualizers = [Visualizer(name=name, visualize=True) for name in self.names]

    w, b = self.visualizers[0].access_all_models_layer(layer_name='conv1')
    w2, b2 = self.visualizers[0].access_all_models_layer(layer_name='conv2')

    print('shape of conv1', w.shape)
    print('shape of conv2', w2.shape)

    for i, layer in enumerate(['conv1', 'conv2']):
      filters = []
      plot = plt.figure(layer + ' ' + str(i+1))

      for vis in self.visualizers:
        weights, biases = vis.access_all_models_layer(layer_name=layer)
        curr_filter = np.asarray(self.extractFilters(weights))
        curr_filter /= norm(curr_filter)
        filters.extend(curr_filter.tolist())
      
      print('filters', np.asarray(filters).shape)

      # self.visualize(weights1, biases1)
      # self.plot_all_weights(weights1)
      # self.plot_norm(filters, normgraphtype, layer)
      self.PCA_magnitudes(np.asarray(filters), layer)
    plt.show()
  
  def extractFilters(self, weights):
    dimensions = weights.shape
    filters = []
    for kernel in range(dimensions[1]):
      curr_kernel = weights[0, kernel, :]
      filters.append(curr_kernel.flatten())
    return filters

  def cov(self, X):
    """
    Covariance matrix
    note: specifically for mean-centered data
    note: numpy's `cov` uses N-1 as normalization
    """
    return np.dot(X.T, X) / X.shape[0]
    # N = data.shape[1]
    # C = empty((N, N))
    # for j in range(N):
    #   C[j, j] = mean(data[:, j] * data[:, j])
    #   for k in range(j + 1, N):
    #       C[j, k] = C[k, j] = mean(data[:, j] * data[:, k])
    # return C

  def pca(self, data, pc_count = None):
    data -= np.mean(data, 0)
    data /= np.std(data, 0)
    C = self.cov(data)
    E, V = eigh(C)
    key = np.argsort(E)[::-1][:pc_count]
    E, V = E[key], V[:, key]
    U = np.dot(data, V)  # used to be dot(V.T, data.T).T
    return U, E, V


  def PCA_magnitudes(self, filters, name):
    trans = PCA(n_components=2).fit_transform(filters)
    fig = plt.plot('pca ' + name)
    plt.scatter(trans[:, 0], trans[:, 1], c = 'r')

  def plot_norm(self, filters, normgraphtype, name):
    data = self.euclideanNorm(filters)
    data = np.asarray(data)

    if normgraphtype == 'bar':
      plt.bar(np.arange(len(data)), data)
      plt.xlabel('Model')
      plt.ylabel('Magnitude')
    else:
      plt.hist(data, bins=20, label="DATA")

      plt.xlabel('Magnitudes')
      plt.ylabel('Number of kernels')
      plt.legend(loc="upper left")
    
    plt.title('Magnitudes ' + name)
    
  
  def euclideanNorm(self, filters):
    return [norm(filt) for filt in filters]

  def plot_all_weights(self, weights):
    weights1d = weights.flatten()
    plt.hist(weights1d, label="DATA")
    plt.xlabel('weight value')
    plt.ylabel('Number of weights')
    plt.legend(loc="upper left")

    plt.title('Weights')
    plt.show()

  def get_layer(self, layer_name):
    weights, biases = [], []
    for i in range(self.n_models):
      w, b = self.visualizers[i].access_all_models_layer(layer_name=layer_name, visualize=False)
      weights.append(w)
      biases.append(b)
    return np.asarray(weights), np.asarray(biases)

  def get_layer_order(self, weights):
    orig = weights[0, -1, ...]
    n_filters = len(orig)
    orders = -1*np.ones((len(weights)-1, n_filters))
    for i in range(1, len(weights)):

      last = weights[i, -1, ...]
      distances = np.zeros((n_filters, n_filters))
      for j in range(n_filters):
        for k in range(n_filters):
          distances[j, k] = np.linalg.norm(orig[j] - last[k])
      maxVal = 2*np.max(distances)

      for j in range(n_filters):
        minVal = np.min(distances)
        idx = np.where(distances == minVal)

        a = idx[0][0]
        b = idx[1][0]

        distances[a, ...] = maxVal
        distances[..., b] = maxVal
        orders[i-1, b] = a

    return orders.astype(int)

  def reorder_weights(self, weights, order):
    print("Reordering:", np.shape(weights), np.shape(order))
    new_weights = np.zeros_like(weights)

    # assign first weights as the ground truth order:
    new_weights[0, ...] = weights[0, ...]

    # go through remaining models and reorder them:
    for i in range(1, len(weights)):
      for j in range(len(order[0])):
        new_weights[i, :, :, order[i-1, j], ...] = weights[i, :, :, j, ...]

    print("Ordered:", np.shape(new_weights))
    return new_weights

  def visualize(self, weights, biases):
    print(np.shape(weights), np.shape(biases))

    dims = np.shape(weights)
    tsne_weights = self.flatten_and_tsne(weights)
    print(np.shape(tsne_weights))

    cmap = plt.get_cmap('nipy_spectral')#'viridis')
    colors = cmap(np.linspace(0, 1, self.n_models))
    #colors = ['b','g','r','c','m', 'y', 'k']

    for j in range(dims[0]):
      for i in range(dims[2]):
        model_len = dims[1]*dims[2]
        train_len = dims[1]
        start_idx = model_len*j+train_len*i
        end_idx = model_len*j+train_len*(i+1)
        y = tsne_weights[start_idx:end_idx, 1]
        x = tsne_weights[start_idx:end_idx, 0]
        plt.scatter(x, y, color=colors[j])
        plt.plot(x, y, color=colors[j])
        plt.annotate(str(i), (x[-1], y[-1]))
    plt.show()

  def flatten_and_tsne(self, data):
    n_models = len(data)
    assert n_models == len(self.names)
    dims = np.shape(data)
    # to visualize the CONV layers:
    if len(dims) > 4:
      data = data.reshape((dims[0], dims[1], dims[2], -1))
    new_data = []
    for j in range(np.shape(data)[0]):
      for i in range(np.shape(data)[2]):
        new_data.extend(data[j, :, i, :])
    new_data = np.asarray(new_data)
    print(np.shape(new_data))
    tsne_data = PCA(n_components=2).fit_transform(new_data)
    #tsne_data = TSNE(n_components=2).fit_transform(new_data)
    return tsne_data