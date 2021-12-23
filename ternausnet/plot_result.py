
import torch
import argparse
from torch.optim import Adam
from torch.utils.data import DataLoader
import albumentations as A
import os
import numpy as np
import sys
import pickle
import matplotlib.pyplot as plt

def zerolistmaker(n):
    listofzeros = [0] * n
    return listofzeros


def plot_results(train_metrics,val_metrics,name):
    for i in range(len(train_metrics)):
        nbr_epochs = len(train_metrics[i])
        epoch_rng = list(range(1,nbr_epochs+1))
        loss_train = zerolistmaker(nbr_epochs)
        for j in range(nbr_epochs):
            loss_train[j]= train_metrics[i][j][0] #  mean(l_train)
        plt.plot(epoch_rng, loss_train, label=name[i])

    plt.xticks(np.linspace(0, 100, 11))
    plt.title('Training loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(prop={'size': 12})
    plt.savefig('loss.png')
    plt.show()

    for i in range(len(val_metrics)):
        nbr_epochs = len(val_metrics[i])
        epoch_rng = list(range(1,nbr_epochs+1))
        jaccard_val = zerolistmaker(nbr_epochs)
        for j in range(nbr_epochs):
            jaccard_val[j]=val_metrics[i][j][1]   #  mean(l_val)
        plt.plot(epoch_rng, jaccard_val, label=name[i])

    plt.xticks(np.linspace(0, 100, 11))
    plt.title('Validation Jaccard index')
    plt.xlabel('Epochs')
    plt.ylabel('Jaccard index')
    plt.legend(prop={'size': 12})
    plt.savefig('jaccard.png')
    plt.show()

name = []
train_metrics=[]
with open("metrics/train_metrics_11.txt", "rb") as fp:
    train_metrics_11=pickle.load(fp)
    train_metrics.append(train_metrics_11)
    name.append('UNet11 neurons')
with open("metrics/train_metrics_16.txt", "rb") as fp:
    train_metrics_16=pickle.load(fp)
    train_metrics.append(train_metrics_16)
    name.append('UNet16 neurons')
with open("metrics/train_metrics_a_11.txt", "rb") as fp:
    train_metrics_a_11=pickle.load(fp)
    train_metrics.append(train_metrics_a_11)
    name.append('UNet11 neurons+axons')
with open("metrics/train_metrics_a_16.txt", "rb") as fp:
    train_metrics_a_16=pickle.load(fp)
    train_metrics.append(train_metrics_a_16)
    name.append('UNet16 neurons+axons')

val_metrics = []
with open("metrics/val_metrics_11.txt", "rb") as fp:
    val_metrics_11=pickle.load(fp)
    val_metrics.append(val_metrics_11)
with open("metrics/val_metrics_16.txt", "rb") as fp:
    val_metrics_16=pickle.load(fp)
    val_metrics.append(val_metrics_16)
with open("metrics/val_metrics_a_11.txt", "rb") as fp:
    val_metrics_a_11=pickle.load(fp)
    val_metrics.append(val_metrics_a_11)
with open("metrics/val_metrics_a_16.txt", "rb") as fp:
    val_metrics_a_16=pickle.load(fp)
    val_metrics.append(val_metrics_a_16)


plot_results(train_metrics,val_metrics,name)
