import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm

import os

import matplotlib
from torch.utils.data import dataset
matplotlib.use('Agg')

import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 13})

import torch.nn.utils.prune as prune

from .. import networks
from .networks.CNNs.VGG import VGG16
from .networks.CNNs.ResNet import ResNet56
from .networks.CNNs.MobileNet import MobileNet

from .networks.SDNs.VGG_SDN import VGG16_SDN
from .networks.SDNs.ResNet_SDN import ResNet56_SDN
from .networks.SDNs.MobileNet_SDN import MobileNet_SDN


#########################################################
###################   Training   ########################
###################   Function  #########################
#########################################################
def get_cnn_model(nettype, num_classes, input_size):

    if 'resnet' in nettype:
        model = ResNet56(num_classes, input_size)
    elif 'vgg' in nettype:
        model = VGG16(num_classes, input_size)
    elif 'mobilenet' in nettype:
        model = MobileNet(num_classes, input_size)
    return model

def get_sdn_model(nettype, add_output, num_classes, input_size):

    if 'resnet' in nettype:
        return ResNet56_SDN(add_output, num_classes, input_size)
    elif 'vgg' in nettype:
        return VGG16_SDN(add_output, num_classes, input_size)
    elif 'mobilenet' in nettype:
        return MobileNet_SDN(add_output, num_classes, input_size)


def load_cnn(sdn):
    if isinstance(sdn, VGG16_SDN):
        return VGG16
    elif isinstance(sdn, ResNet56_SDN):
        return ResNet56
    elif isinstance(sdn, MobileNet_SDN):
        return MobileNet

def load_sdn(cnn):
    if isinstance(cnn, VGG16):
        return VGG16_SDN
    elif isinstance(cnn, ResNet56):
        return ResNet56_SDN
    elif isinstance(cnn, MobileNet):
        return MobileNet_SDN

def get_add_output(network):
    if 'vgg16' in network:
        return [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1] 
    if 'resnet56' in network:
        return [ \
        [1, 1, 1, 1, 1, 1, 1, 1, 1], \
        [1, 1, 1, 1, 1, 1, 1, 1, 1], \
        [1, 1, 1, 1, 1, 1, 1, 1, 1]]
    if 'mobilenet' in network:
        return [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

def cnn_to_sdn(cnn_model, add_output, device):
    # todo
    print ('[cnn_to_sdn] convert a CNN to an SDN...')
    sdn_model = (load_sdn(cnn_model))(add_output, cnn_model.num_classes, cnn_model.input_size)
    sdn_model.init_conv = cnn_model.init_conv

    layers = nn.ModuleList()
    for layer_id, cnn_layer in enumerate(cnn_model.layers):
        sdn_layer = sdn_model.layers[layer_id]
        sdn_layer.layers = cnn_layer.layers
        layers.append(sdn_layer)

    sdn_model.layers = layers
    sdn_model.end_layers = cnn_model.end_layers
    sdn_model.to(device)
    return sdn_model

def sdn_to_cnn(sdn_model, device):
    # todo
    print ('[sdn_to_cnn] convert an SDN to a CNN...')
    
    cnn_model = load_cnn(sdn_model)(sdn_model.num_classes, sdn_model.input_size)
    cnn_model.init_conv = sdn_model.init_conv

    layers = nn.ModuleList()
    for layer_id, sdn_layer in enumerate(sdn_model.layers):
        cnn_layer = cnn_model.layers[layer_id]
        cnn_layer.layers = sdn_layer.layers
        layers.append(cnn_layer)

    cnn_model.layers = layers
    cnn_model.end_layers = sdn_model.end_layers
    cnn_model.to(device)
    return cnn_model



def freeze_except_outputs(model):

    for param in model.init_conv.parameters():
        param.requires_grad = False

    for layer in model.layers:
        for param in layer.layers.parameters():
            param.requires_grad = False

    for param in model.end_layers.parameters():
        param.requires_grad = False

def freeze_outputs(model):

    for layer in model.layers:
        for param in layer.output.parameters():
            param.requires_grad = False


# the internal classifier for all SDNs
class InternalClassifier(nn.Module):
    def __init__(self, input_size, output_channels, num_classes, alpha=0.5):
        super(InternalClassifier, self).__init__()
        #red_kernel_size = -1 # to test the effects of the feature reduction
        red_kernel_size = feature_reduction_formula(input_size) # get the pooling size
        self.output_channels = output_channels
        self.flat = nn.Flatten()
        if red_kernel_size == -1:
            self.linear = nn.Linear(output_channels*input_size*input_size, num_classes)
            self.forward = self.forward_wo_pooling
        else:
            red_input_size = int(input_size/red_kernel_size)
            # self.max_pool = nn.MaxPool2d(kernel_size=red_kernel_size)
            self.avg_pool = nn.AvgPool2d(kernel_size=red_kernel_size)
            # self.alpha_mult = nn.quantized.FloatFunctional()
            
            self.linear = nn.Linear(output_channels*red_input_size*red_input_size, num_classes)
            self.forward = self.forward_w_pooling

    def forward_w_pooling(self, x):
        # avgp = self.alpha_mult.mul(self.alpha, self.max_pool(x))
        # maxp = self.alpha_mult.mul(1 - self.alpha, self.avg_pool(x))
        # mixed = avgp + maxp
        # return self.linear(mixed.view(mixed.size(0), -1))\
        maxp = self.avg_pool(x)
        return self.linear(self.flat(maxp))

    def forward_wo_pooling(self, x):
        return self.linear(self.flat(x))

# the formula for feature reduction in the internal classifiers
def feature_reduction_formula(input_feature_map_size):
    if input_feature_map_size >= 4:
        return int(input_feature_map_size/4)
    else:
        return -1


def get_lr(optimizers):

    if isinstance(optimizers, dict):
        return optimizers[list(optimizers.keys())[-1]].param_groups[-1]['lr']
    else:
        return optimizers.param_groups[-1]['lr']



#########################################################
####################     AUX    #########################
###################   Function  #########################
#########################################################

def fast_load_model(net, path, dev='cpu'):
    net.load_state_dict(torch.load(path, map_location=dev))
    net.eval()
    net.to(dev)
    return net

def test_threshold(output, threshold, start_from_include=0):
    '''
    no None in output list. 
    '''
    output = output[start_from_include:]
    output_num = len(output)
    output = torch.stack(output, dim=1)
    batch_max_conf, batch_pred = torch.max(torch.softmax(output, dim=2), dim=2)
    # above are two matrix of shape batch_size * output_num
    batch_out = torch.where(batch_max_conf > threshold, 1, -1)
    batch_out[:, -1] = 0
    batch_out_idx = torch.argmax(batch_out, dim=1)
    output_bool = torch.eye(output_num).to(batch_out_idx.device).index_select(0, batch_out_idx).bool()
    return start_from_include + batch_out_idx, batch_pred[output_bool]
