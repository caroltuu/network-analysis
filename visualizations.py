import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from glob import glob
import numpy as np


class Visualizer():
    def __init__(self, name):
        self.name = name

        self.get_all_models()

        self.access_all_models_layer(layer_name="conv1")

    def get_all_models(self):
        all_results_list = glob("./results/*")
        models_list = []
        for file in all_results_list:
            if self.name in file and "model" in file and ".pth" in file:
                models_list.append(file)
        n_models = len(models_list)
        self.ordered_model_list = []
        for i in range(1, n_models+1):
            self.ordered_model_list.append('./results/' + self.name + '_model_' + str(i).zfill(6) + '.pth')

    def access_all_models_layer(self, layer_name):
        weights, biases = [], []
        for model in self.ordered_model_list:
            w, b = self.access_model_layer(model, layer_name=layer_name)
            weights.append(w)
            biases.append(b)
        weights, biases = np.asarray(weights), np.asarray(biases)
        print(np.shape(weights), np.shape(biases))

        # visualize the weights:

    def access_model_layer(self, model_path, layer_name):
        weight, bias = None, None
        model = torch.load(model_path)
        for name, param in model.state_dict().items():
            if layer_name in name:
                if "weight" in name:
                    weight = param.numpy()
                if "bias" in name:
                    bias = param.numpy()

                #early stopping:
                if bias is not None and weight is not None:
                    return weight, bias
        return weight, bias


    def visualize_weights(self, weights):
        pass

