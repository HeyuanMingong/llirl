import torch 
from torch import nn
from torch.nn import functional as F
import torch.utils.data as data
import random
from torch.optim import SGD, Adam
import numpy as np
from tqdm import tqdm
import copy
import math

from myrllib.mixture.env_model import EnvModel

class Task(object):
    def __init__(self, input, output, ratio=0.7, device='cpu'):
        input = torch.from_numpy(input).float().to(device=device)
        output = torch.from_numpy(output).float().to(device=device)
       
        num_inst = input.size(0)
        all_ids = list(range(num_inst))
        random.shuffle(all_ids)
        train_ids = all_ids[:int(ratio*num_inst)]
        valid_ids = all_ids[int(ratio*num_inst):]
        
        self.train_input = input[train_ids]
        self.train_output = output[train_ids]
        self.valid_input = input[valid_ids]
        self.valid_output = output[valid_ids]

class Dataset(data.Dataset):
    """Custom data.Dataset compatible with data.DataLoader."""
    def __init__(self, task, split='train'):
        self.input = task.train_input if split=='train' else task.valid_input
        self.output = task.train_output if split=='train' else task.valid_output
      
    def __getitem__(self, index):
        y = self.output[index]
        X = self.input[index]
        return X, y

    def __len__(self):
        return self.input.size(0)


def get_data_loader(task, batch_size=32, split='train'):
    # build a custom dataset
    dataset = Dataset(task, split=split)
    if split == 'train':
        return data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
    else:
        return data.DataLoader(dataset=dataset, batch_size=len(dataset))


def env_nominal_train(env_model, inputs, outputs, max_epochs=10, batch_size=32, 
        lr=1e-3, ratio=1.0, device='cpu'):
    loss_fn = nn.MSELoss()

    task = Task(inputs, outputs, ratio=ratio, device=device)
    train_loader = get_data_loader(task, batch_size=batch_size, split='train')
    opt = Adam(env_model.parameters(), lr=lr)

    tloss = []
    for idx in tqdm(range(max_epochs)):
        for in_, out_ in train_loader:
            out_predicted = env_model(in_)
            loss = loss_fn(out_predicted, out_)
            tloss.append(loss.cpu().data.numpy())

            opt.zero_grad()
            loss.backward()
            opt.step()

    return env_model, np.array(tloss)


def env_update(env_model, inputs, outputs, posterior=1.0, 
        max_epochs=1, batch_size=32, lr=1e-3, device='cpu'):

    task = Task(inputs, outputs, ratio=1.0, device=device)
    train_loader = get_data_loader(task, batch_size=batch_size, split='train')
    opt = Adam(env_model.parameters(), lr=lr*posterior)
    loss_fn = nn.MSELoss()
    
    tloss = []
    env_model.train()
    for idx in range(max_epochs):
        for in_, out_ in train_loader:
            out_predicted = env_model(in_)
            loss = loss_fn(out_predicted, out_)
            tloss.append(loss.cpu().data.numpy())

            opt.zero_grad()
            loss.backward()
            opt.step()

    return env_model, np.array(tloss)








