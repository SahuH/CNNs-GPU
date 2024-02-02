import json
import numpy as np
import pandas as pd
import keras

import torch
import torch.nn as nn


class CNN(nn.Module):
    def __init__(self, conv, pooling, dense):
        super(CNN, self).__init__()

        self.conv_module(conv["input_shape"], conv["n_layers"], conv["kernel_size"], conv["n_filters"], conv["stride"], conv["activation"])

        pool_layers = []
        for i in range(pooling["n_layers"]):
            pool_layers.append(nn.MaxPool2d(kernel_size=pooling["kernel_size"], stride=pooling["stride"]))
        self.pool_layers = nn.Sequential(*pool_layers)

        dense_layers = []
        dense_layers.append(nn.LazyLinear(dense["n_units"]))
        dense_layers.append(nn.ReLU())
        for i in range(dense["n_layers"]-2):
            dense_layers.append(nn.LazyLinear(dense["n_units"]))
            dense_layers.append(nn.ReLU())
        dense_layers.append(nn.LazyLinear(1))
        self.dense_layers = nn.Sequential(*dense_layers)

    def conv_module(self, input_shape, n_layers, kernel_size, n_filters, stride, activation):
        conv_layers = []
        conv_layers.append(nn.Conv2d(input_shape[-1], n_filters, kernel_size=kernel_size, stride=stride, padding="same"))
        conv_layers.append(nn.ReLU()) 
        for i in range(n_layers-1):
            conv_layers.append(nn.Conv2d(n_filters, n_filters, kernel_size=kernel_size, stride=stride, padding="same"))
            conv_layers.append(nn.ReLU()) 
        self.conv_layers = nn.Sequential(*conv_layers)
        
    def forward(self, x):
        x = self.conv_layers(x)
        x = self.pool_layers(x)
        x = x.view(x.size(0), -1)
        print(x.shape)
        x = self.dense_layers(x)
        return x

print(torch.cuda.is_available())

start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

with open('config.json', 'r') as json_file:
    hyperparameters = json.load(json_file)
conv = hyperparameters["convolution"]
pooling = hyperparameters["pooling"]
dense = hyperparameters["dense"]

model = CNN(conv, pooling, dense)
# print(model)

input_shape = conv["input_shape"]
input_image = torch.randn(input_shape[0], input_shape[3], input_shape[1], input_shape[2])

start.record()
output = model(input_image)
end.record()

torch.cuda.synchronize()
print(start.elapsed_time(end))