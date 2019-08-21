from __future__ import print_function

import numpy as np
import pickle

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader

from dataset import Follow_Ahead_Dataset

import wandb

import os

import torchvision

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        resnet = torchvision.models.resnet18(pretrained=True).float()
        self.resnet = nn.Sequential(*resnet.children())[:-3].float()
        self.max_pool = nn.MaxPool2d(4, stride=2)
        self.fc_resnet = nn.Linear(512, 128)
        self.vector_net = nn.Sequential(
           nn.Linear(5, 256),
           nn.ReLU(),
           nn.Linear(256, 128),
           nn.ReLU()
        )

        self.combine_net = nn.Sequential(
           nn.Linear(256, 512),
           nn.ReLU(),
           nn.Linear(512, 128),
           nn.ReLU(),
           nn.Linear(128, 128),
           nn.ReLU(),
           nn.Linear(128, 2)
        )

    def forward(self, lidar_image, vector_pv):

        batch_size = lidar_image.size()[0]
        lidar_features = self.resnet(lidar_image)
        lidar_features = self.max_pool(lidar_features).view(batch_size, -1)
        lidar_features = self.fc_resnet(lidar_features)

        pv_features = self.vector_net(vector_pv).view(batch_size, -1)

        combine_features = torch.cat((lidar_features, pv_features), -1)
        coordinates = self.combine_net(combine_features)

        return coordinates


class FollowAhead():
    def train(self, net):
        optimizer = torch.optim.Adadelta(net.parameters())
        loss_func = nn.MSELoss()
        if os.path.exists('data/best_loss_validation'):
            net.load_state_dict(torch.load('data/best_loss_validation'))
        else:
            print("path not exist: {}".format("data/best_loss_validation"))
        train_dataset = Follow_Ahead_Dataset("data/dataset/0/")
        validation_dataset = Follow_Ahead_Dataset("data/dataset/1/")
        train_loader = DataLoader(train_dataset, batch_size=626, shuffle=True)
        validation_loader = DataLoader(validation_dataset, batch_size=626, shuffle=True)
        best_loss_validation = float('inf')
        best_loss = float('inf')
        for epoc in range (1000000):
            avg_loss_epoc = 0
            optimizer.zero_grad()
            for image_batch, pv_batch, target in train_loader:
                image_batch = Variable(torch.from_numpy(np.asarray(image_batch))).float().cuda()
                pv_batch = Variable(torch.from_numpy(np.asarray(pv_batch))).float().cuda()
                target = Variable(torch.from_numpy(np.asarray(target))).float().cuda()
                predictions = net(image_batch, pv_batch)
                loss = loss_func(predictions, target)
                # print ("loss is: {}".format(loss.item()))
                loss.backward()
                optimizer.step()
                avg_loss_epoc += loss.item()
            avg_loss_epoc = avg_loss_epoc / len(train_loader)
            avg_loss_validation = 0
            with torch.no_grad():
                for image_batch, pv_batch, target in validation_loader:
                    image_batch = Variable(torch.from_numpy(np.asarray(image_batch))).float().cuda()
                    pv_batch = Variable(torch.from_numpy(np.asarray(pv_batch))).float().cuda()
                    target = Variable(torch.from_numpy(np.asarray(target))).float().cuda()
                    predictions = net(image_batch, pv_batch)
                    loss = loss_func(predictions, target)
                    # print ("loss is: {}".format(loss.item()))
                    avg_loss_validation += loss.item()
            avg_loss_validation = avg_loss_validation/ len(validation_loader)
            print("finished epoc {} avg_loss_train: {} avg_loss_validation {}".format(epoc, avg_loss_epoc, avg_loss_validation))
            if epoc % 20:
                wandb.log({"Train Loss": avg_loss_epoc, "Validation Loss":avg_loss_validation})

            if best_loss_validation >avg_loss_validation:
                print("saving loss validation: {}".format(avg_loss_validation))
                best_loss_validation = avg_loss_validation
                torch.save(net.state_dict(), 'data/best_loss_validation')
            if best_loss >avg_loss_epoc:
                print("saving loss train: {}".format(avg_loss_epoc))
                best_loss = avg_loss_epoc
                torch.save(net.state_dict(), 'data/best_loss')
            # for file_idx in range (37):
            #     optimizer.zero_grad()
            #     with open('data/dataset/{}.pkl'.format(file_idx), 'rb')as f:
            #         data = pickle.load(f)
            #         image_batch = []
            #         pv_batch = []
            #         target = []
            #         for instance in data:
            #             image = instance[0][0]
            #             image = np.transpose(image, (2, 0, 1))
            #             image = np.concatenate((imag0e,image,image))
            #             image_batch.append(image)
            #             pv_batch.append(instance[0][1])
            #             target.append(instance[1])
            #         image_batch = Variable(torch.from_numpy(np.asarray(image_batch))).float().cuda()
            #         pv_batch = Variable(torch.from_numpy(np.asarray(pv_batch))).float().cuda()
            #         target = Variable(torch.from_numpy(np.asarray(target))).float().cuda()
            #         predictions = net(image_batch, pv_batch)
            #         loss = loss_func(predictions, target)
            #         print ("loss is: {}".format(loss.item()))
            #         loss.backward()
            #         optimizer.step()
            # torch.save(net.state_dict(), 'data/best_loss')
net = Net().cuda()
print(net)
wandb.init(project="followahead")
wandb.watch(net)

follow_ahead = FollowAhead()
follow_ahead.train(net)
