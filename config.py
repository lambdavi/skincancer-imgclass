import numpy as np
import pandas as pd
import torch
from torch import nn
from torchvision import models, transforms

from dataset.skincancer import SkinDataset
from models.cnn import ConvNet
from models.xception import Xception, model_urls, model_zoo

N_CLASS = 7
DATA_PATH = "data/"

def get_device():
    if torch.cuda.is_available():
        device = "cuda"
    else:
        if torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    return device

def get_model(args, device="cpu", pretrained = False):
    if args.model == "resnet":
        res_model = models.resnet18(pretrained=True) # on ImageNet
        for param in res_model.parameters():
            param.requires_grad = False

            num_ftrs = res_model.fc.in_features

            # Create a new fc layer
            res_model.fc = nn.Linear(num_ftrs, N_CLASS) # 7 classes
            return res_model.to(device)
        
    if args.model == "xception":
        model = Xception(N_CLASS)
        if pretrained:
            model.load_state_dict(model_zoo.load_url(model_urls['xception']))
        return model.to(device)
    if args.model == "cnn":
        return ConvNet().to(device)
    raise NotImplementedError

def get_transformations(args):
    if args.model != "cnn":
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
    else: 
        mean = np.array([0.5, 0.5, 0.5])
        std = np.array([0.2, 0.2, 0.2])

    transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    return transform

def get_dataset(transform=None):
    skinDF = pd.read_csv(DATA_PATH+"HAM10000_metadata.csv")
    finDF = skinDF[["image_id", "dx"]]
    return SkinDataset(finDF, transform)