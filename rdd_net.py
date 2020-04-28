import numpy as np
from torch.nn.parameter import Parameter
import torch

from rdd_layers import SpikingFA

from shared_hyperparams import *

class RDDNet:
    def __init__(self):
        self.classification_layers = []

        self.classification_layers.append(SpikingFA(1600, None, 384))
        self.classification_layers.append(SpikingFA(384, 1600, 192))
        self.classification_layers.append(SpikingFA(192, 384, 10))
        self.classification_layers.append(SpikingFA(10, 192))

    def copy_weights_from(self, layers):
        self.classification_layers[0].set_weights(None, None, layers[2].fb_weight.detach().cpu().numpy().astype(np.float32).T)
        self.classification_layers[1].set_weights(layers[2].weight.detach().cpu().numpy().astype(np.float32), layers[2].bias.detach().cpu().numpy().astype(np.float32)[:, np.newaxis], layers[3].fb_weight.detach().cpu().numpy().astype(np.float32).T)
        self.classification_layers[2].set_weights(layers[3].weight.detach().cpu().numpy().astype(np.float32), layers[3].bias.detach().cpu().numpy().astype(np.float32)[:, np.newaxis], layers[4].fb_weight.detach().cpu().numpy().astype(np.float32).T)
        self.classification_layers[3].set_weights(layers[4].weight.detach().cpu().numpy().astype(np.float32), layers[4].bias.detach().cpu().numpy().astype(np.float32)[:, np.newaxis])

    def copy_weights_to(self, layers, device):
        layers[2].fb_weight.data = torch.from_numpy(self.classification_layers[0].fb_weight.astype(np.float32).T).to(device)
        layers[3].fb_weight.data = torch.from_numpy(self.classification_layers[1].fb_weight.astype(np.float32).T).to(device)
        layers[4].fb_weight.data = torch.from_numpy(self.classification_layers[2].fb_weight.astype(np.float32).T).to(device)

    def out(self, driving_spike_hist_1, driving_spike_hist_2, driving_spike_hist_3):
        self.classification_layers[0].update(None, self.classification_layers[1].spike_hist, driving_input=driving_spike_hist_1)
        self.classification_layers[1].update(self.classification_layers[0].spike_hist, self.classification_layers[2].spike_hist, driving_input=driving_spike_hist_2)
        self.classification_layers[2].update(self.classification_layers[1].spike_hist, self.classification_layers[3].spike_hist, driving_input=driving_spike_hist_3)
        self.classification_layers[3].update(self.classification_layers[2].spike_hist)

    def reset(self):
        for layer in self.classification_layers:
            layer.reset()

    def update_fb_weights(self):
        for layer in self.classification_layers[:-1]:
            layer.update_fb_weights()
