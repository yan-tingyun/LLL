import torch.utils.data as data
import torch.utils.data.sampler as sampler
import torchvision
import os
import torch.nn.functional as F
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
import tqdm
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import trange
from torch import nn
import sys

from MNIST_split import *


def get_transform(normalize=True):
  if normalize == True:
    transform = transforms.Compose([transforms.ToTensor(),
                                    Pad(28),
                                    transforms.Normalize((0.1307,), (0.3081,)),
    ])
  else:
    transform = transforms.Compose([transforms.ToTensor(),
                                    Pad(28),
    ])
  return transform

class Pad(object):
  def __init__(self, size, fill=0, padding_mode='constant'):
    self.size = size
    self.fill = fill
    self.padding_mode = padding_mode
    
  def __call__(self, img):
    # If the H and W of img is not equal to desired size,
    # then pad the channel of img to desired size.
    img_size = img.size()[1]
    assert ((self.size - img_size) % 2 == 0)
    padding = (self.size - img_size) // 2
    padding = (padding, padding, padding, padding)
    return F.pad(img, padding, self.padding_mode, self.fill)

class Data():
  def __init__(self, path, train=True, normalize=True):

    transform = get_transform(normalize)
    self.dataset = datasets.MNIST(root = path,
                                        transform=transform,
                                        train = train,
                                        download = True)


#======================================================================#
#dataloader

class Args:
  task_number = 10
  epochs_per_task = 100
  lr = 1.0e-4
  batch_size = 128
  test_size=8192

args=Args()

# check use gpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(torch.cuda.is_available())
print(device)



# prepare permuted mnist datasets.

path = '/home/Zxf4/yty/dataset'
# origin_set = Data(path=path)

train_datasets = []
# for digit in [[0],[1],[2],[3],[4],[5],[6],[7],[8],[9]]:
for digit in [[1,2],[3,4],[5,6],[7,8],[9,0]]:
  transform = get_transform(normalize=True)
  dataset = MNIST_Split(root = path,
                        transform=transform,
                        train = True,
                        download = True,
                        digits=digit)
  train_datasets.append(dataset)


train_dataloaders = [
    DataLoader(data, batch_size=args.batch_size, shuffle=True) for data in train_datasets
]

# data : img (28*28+1, 1) , labels ( 0 / 1)


test_datasets = []
# for digit in [[0],[1],[2],[3],[4],[5],[6],[7],[8],[9]]:
for digit in [[1,2],[3,4],[5,6],[7,8],[9,0]]:
  transform = get_transform(normalize=True)
  dataset = MNIST_Split(root = path,
                        transform=transform,
                        train = False,
                        download = True,
                        digits=digit)
  test_datasets.append(dataset)

test_dataloaders = [
    DataLoader(data, batch_size=args.test_size, shuffle=True) for data in test_datasets
]


# test_dataloaders = [
#     DataLoader(data, batch_size=args.test_size, shuffle=True) for data in train_datasets
# ]

#======================================================================#
# model defination
# class Model(nn.Module):
#   """
#   Model architecture 
#   785 (input( img 784 + timeline 1 )→ 1024 → 512 → 256 → 128 → 2( positive / negative )
#   """
#   def __init__(self):
#     super(Model, self).__init__()
#     self.fc1 = nn.Linear(785, 1024)
#     self.fc2 = nn.Linear(1024, 512)
#     self.fc3 = nn.Linear(512, 256)
#     self.fc4 = nn.Linear(256, 64)
#     self.fc5 = nn.Linear(64, 2)
#     self.relu = nn.ReLU()

#   def forward(self, x):
#     x = x.view(-1, 1*28*28+1)
#     x = self.fc1(x)
#     x = self.relu(x)
#     x = self.fc2(x)
#     x = self.relu(x)
#     x = self.fc3(x)
#     x = self.relu(x)
#     x = self.fc4(x)
#     x = self.relu(x)
#     x = self.fc5(x)
#     return x

class Model(nn.Module):
  """
  Model architecture 
  784 (input) → 1024 → 512 → 256 → 10
  """
  def __init__(self):
    super(Model, self).__init__()
    self.fc1 = nn.Linear(784, 1024)
    self.fc2 = nn.Linear(1024, 512)
    self.fc3 = nn.Linear(512, 256)
    self.fc4 = nn.Linear(256, 10)
    self.relu = nn.ReLU()

  def forward(self, x):
    x = x.view(-1, 1*28*28)
    x = self.fc1(x)
    x = self.relu(x)
    x = self.fc2(x)
    x = self.relu(x)
    x = self.fc3(x)
    x = self.relu(x)
    x = self.fc4(x)
    return x

example = Model()
print(example)


#======================================================================#


# train

def train(model, optimizer, dataloader, epochs_per_task, lll_object, lll_lambda, test_dataloaders, evaluate, device, log_step=1):
    model.train()
    model.zero_grad()
    objective = nn.CrossEntropyLoss()
    acc_per_epoch = []
    loss = 1.0
    bar = tqdm.auto.trange(epochs_per_task, leave=False, desc=f"Epoch 1, Loss: {loss:.7f}")
    for epoch in bar:
        for imgs, labels in tqdm.auto.tqdm(dataloader, leave=False):            
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            loss = objective(outputs, labels)
            total_loss = loss
            lll_loss = lll_object.penalty(model)
            total_loss += lll_lambda * lll_loss 
            lll_object.update(model)
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            loss = total_loss.item()
            bar.set_description_str(desc=f"Epoch {epoch+1:2}, Loss: {loss:.7f}", refresh=True)
        acc_average  = []
        for test_dataloader in test_dataloaders: 
            acc_test = evaluate(model, test_dataloader, device)
            acc_average.append(acc_test)

        average=np.mean(np.array(acc_average))
        acc_per_epoch.append(average*100.0)
        bar.set_description_str(desc=f"Epoch {epoch+2:2}, Loss: {loss:.7f}", refresh=True)
                
    return model, optimizer, acc_per_epoch


#======================================================================#

# evaluate

def evaluate(model, test_dataloader, device):
    model.eval()
    correct_cnt = 0
    total = 0
    for imgs, labels in test_dataloader:
        imgs, labels = imgs.to(device), labels.to(device)
        outputs = model(imgs)
        _, pred_label = torch.max(outputs.data, 1)

        correct_cnt += (pred_label == labels.data).sum().item()
        total += torch.ones_like(labels.data).sum().item()
    return correct_cnt / total



#======================================================================#
#baseline

class baseline(object):
  """
  baseline technique: do nothing in regularization term [initialize and all weight is zero]
  """
  def __init__(self, model, dataloaders, device):
  
      self.model = model
      self.dataloaders = dataloaders
      self.device = device

      self.params = {n: p for n, p in self.model.named_parameters() if p.requires_grad} #extract all parameters in models
      self.p_old = {} # store current parameters
      self._precision_matrices = self._calculate_importance() # generate weight matrix 

      for n, p in self.params.items():
          self.p_old[n] = p.clone().detach() # keep the old parameter in self.p_old

  def _calculate_importance(self):
      precision_matrices = {}
      for n, p in self.params.items(): # initialize weight matrix（fill zero）
          precision_matrices[n] = p.clone().detach().fill_(0)

      return precision_matrices

  def penalty(self, model: nn.Module):
      loss = 0
      for n, p in model.named_parameters():
          _loss = self._precision_matrices[n] * (p - self.p_old[n]) ** 2
          loss += _loss.sum()
      return loss
  
  def update(self, model):
      # do nothing
      return



class ewc(object):
  """
  @article{kirkpatrick2017overcoming,
      title={Overcoming catastrophic forgetting in neural networks},
      author={Kirkpatrick, James and Pascanu, Razvan and Rabinowitz, Neil and Veness, Joel and Desjardins, Guillaume and Rusu, Andrei A and Milan, Kieran and Quan, John and Ramalho, Tiago and Grabska-Barwinska, Agnieszka and others},
      journal={Proceedings of the national academy of sciences},
      year={2017},
      url={https://arxiv.org/abs/1612.00796}
  }
  """
  def __init__(self, model, dataloaders, device):
  
      self.model = model
      self.dataloaders = dataloaders
      self.device = device

      self.params = {n: p for n, p in self.model.named_parameters() if p.requires_grad} # extract all parameters in models
      self.p_old = {} # initialize parameters
      self._precision_matrices = self._calculate_importance() # generate Fisher (F) matrix for EWC 

      for n, p in self.params.items():
          self.p_old[n] = p.clone().detach() # keep the old parameter in self.p_old

  def _calculate_importance(self):
      precision_matrices = {}
      for n, p in self.params.items(): 
          # initialize Fisher (F) matrix（all fill zero）
          precision_matrices[n] = p.clone().detach().fill_(0)

      self.model.eval()
      if self.dataloaders[0] is not None:
          dataloader_num=len(self.dataloaders)
          number_data = sum([len(loader) for loader in self.dataloaders])
          for dataloader in self.dataloaders:
              for data in dataloader:
                  self.model.zero_grad()
                  # get image data
                  input = data[0].to(self.device)
                  # image data forward model
                  output = self.model(input)
                  # Simply use groud truth label of dataset.  
                  label = data[1].to(self.device)
                  # print(output.shape, label.shape)
                  
                  ############################################################################
                  #####                     generate Fisher(F) matrix for EWC            #####
                  ############################################################################    
                  loss = F.nll_loss(F.log_softmax(output, dim=1), label)             
                  loss.backward()                                                    
                  ############################################################################

                  for n, p in self.model.named_parameters():
                      # get the gradient of each parameter and square it, then average it in all validation set.                          
                      precision_matrices[n].data += p.grad.data ** 2 / number_data   
                                                                          
          precision_matrices = {n: p for n, p in precision_matrices.items()}

      return precision_matrices

  def penalty(self, model: nn.Module):
      loss = 0
      for n, p in model.named_parameters():
          # generate the final regularization term by the ewc weight (self._precision_matrices[n]) and the square of weight difference ((p - self.p_old[n]) ** 2).  
          _loss = self._precision_matrices[n] * (p - self.p_old[n]) ** 2
          loss += _loss.sum()
      return loss
  
  def update(self, model):
      # do nothing
      return 


class mas(object):
  """
  @article{aljundi2017memory,
    title={Memory Aware Synapses: Learning what (not) to forget},
    author={Aljundi, Rahaf and Babiloni, Francesca and Elhoseiny, Mohamed and Rohrbach, Marcus and Tuytelaars, Tinne},
    booktitle={ECCV},
    year={2018},
    url={https://eccv2018.org/openaccess/content_ECCV_2018/papers/Rahaf_Aljundi_Memory_Aware_Synapses_ECCV_2018_paper.pdf}
  }
  """
  def __init__(self, model: nn.Module, dataloaders: list, device):
      self.model = model 
      self.dataloaders = dataloaders
      self.params = {n: p for n, p in self.model.named_parameters() if p.requires_grad} #extract all parameters in models
      self.p_old = {} # initialize parameters
      self.device = device
      self._precision_matrices = self.calculate_importance() # generate Omega(Ω) matrix for MAS
  
      for n, p in self.params.items():
          self.p_old[n] = p.clone().detach() # keep the old parameter in self.p_old
  
  def calculate_importance(self):
      precision_matrices = {}
      for n, p in self.params.items():
          precision_matrices[n] = p.clone().detach().fill_(0) # initialize Omega(Ω) matrix（all filled zero）

      self.model.eval()
      if self.dataloaders[0] is not None:
          dataloader_num = len(self.dataloaders)
          num_data = sum([len(loader) for loader in self.dataloaders])
          for dataloader in self.dataloaders:
              for data in dataloader:
                  self.model.zero_grad()
                  output = self.model(data[0].to(self.device))

                  ###########################################################################################################################################
                  #####  TODO BLOCK: generate Omega(Ω) matrix for MAS. (Hint: square of l2 norm of output vector, then backward and take its gradients  #####
                  ###########################################################################################################################################
                  output.pow_(2)                                                   
                  loss = torch.sum(output,dim=1)                                   
                  loss = loss.mean()   
                  loss.backward() 
                  ###########################################################################################################################################                          
                                          
                  for n, p in self.model.named_parameters():                      
                    precision_matrices[n].data += p.grad.abs() / num_data ## difference with EWC      
                      
      precision_matrices = {n: p for n, p in precision_matrices.items()}
      return precision_matrices

  def penalty(self, model: nn.Module):
      loss = 0
      for n, p in model.named_parameters():
          _loss = self._precision_matrices[n] * (p - self.p_old[n]) ** 2
          loss += _loss.sum()
      return loss
  
  def update(self, model):
      # do nothing
      return 



#======================================================================#

#run baseline

print("RUN BASELINE")
model = Model()
model = model.to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=args.lr,momentum=0.9)

# initialize lifelong learning object (baseline class) without adding any regularization term.
lll_object=baseline(model=model, dataloaders=[None],device=device)
lll_lambda=0.0
baseline_acc= []
task_bar = tqdm.auto.trange(len(train_dataloaders),desc="Task   1")

# iterate training on each task continually.
for train_indexes in task_bar:
    # Train each task
    model, _, acc_list = train(model, optimizer, train_dataloaders[train_indexes], args.epochs_per_task, lll_object, lll_lambda, evaluate=evaluate,device=device, test_dataloaders=test_dataloaders[:train_indexes+1])

    # get model weight to baseline class and do nothing!
    lll_object=baseline(model=model, dataloaders=test_dataloaders[:train_indexes],device=device)

    # new a optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Collect average accuracy in each epoch
    baseline_acc.extend(acc_list)

    # display the information of the next task.
    task_bar.set_description_str(f"Task  {train_indexes+2:2}")

# average accuracy in each task per epoch! 
print(baseline_acc)
print("==================================================================================================")


# #EWC
# print("RUN EWC")
# model = Model()
# model = model.to(device)
# # initialize optimizer
# optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

# # initialize lifelong learning object for EWC
# lll_object=ewc(model=model, dataloaders=[None],device=device)

# # setup the coefficient value of regularization term.
# lll_lambda=100
# ewc_acc= []
# task_bar = tqdm.auto.trange(len(train_dataloaders),desc="Task   1")

# # iterate training on each task continually.
# for train_indexes in task_bar:
#     # Train Each Task
#     model, _, acc_list = train(model, optimizer, train_dataloaders[train_indexes], args.epochs_per_task, lll_object, lll_lambda, evaluate=evaluate,device=device, test_dataloaders=test_dataloaders[:train_indexes+1])
    
#     # get model weight and calculate guidance for each weight
#     lll_object=ewc(model=model, dataloaders=test_dataloaders[:train_indexes+1],device=device)

#     # new a Optimizer
#     optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

#     # collect average accuracy in each epoch
#     ewc_acc.extend(acc_list)

#     # Update tqdm displayer
#     task_bar.set_description_str(f"Task  {train_indexes+2:2}")

# # average accuracy in each task per epoch!     
# print(ewc_acc)
# print("==================================================================================================")



# # MAS
# print("RUN MAS")
# model = Model()
# model = model.to(device)
# # optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
# optimizer = torch.optim.SGD(model.parameters(), lr=args.lr,momentum=0.9)

# lll_object=mas(model=model, dataloaders=[None],device=device)
# lll_lambda=0.1
# mas_acc= []
# task_bar = tqdm.auto.trange(len(train_dataloaders),desc="Task   1")
# for train_indexes in task_bar:
#     # Train Each Task
#     model, _, acc_list = train(model, optimizer, train_dataloaders[train_indexes], args.epochs_per_task, lll_object, lll_lambda, evaluate=evaluate,device=device, test_dataloaders=test_dataloaders[:train_indexes+1])
    
#     # get model weight and calculate guidance for each weight
#     lll_object=mas(model=model, dataloaders=test_dataloaders[:train_indexes+1],device=device)

#     # New a Optimizer
#     optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

#     # Collect average accuracy in each epoch
#     mas_acc.extend(acc_list)
#     task_bar.set_description_str(f"Task  {train_indexes+2:2}")

# # average accuracy in each task per epoch!     
# print(mas_acc)
# print("==================================================================================================")