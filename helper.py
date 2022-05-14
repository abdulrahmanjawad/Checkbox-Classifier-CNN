import os

import torch
import torch.nn as nn

import torchvision
from torchvision import transforms as T

def get_model(num_classes=3, weights=None, map_location="cuda:0", pretrained=False):
    """
        Loads model
        args:
        - num_classes = number of classes for which model has been trained for = number of classes in output layer of model.
        - weights = path to model weights.
        - map_location = it indicates the location where all tensors should be loaded: CPU or CUDA.
        - pretrained = determines if model initialized should be loaded with pretrained weights or not.

        returns:
        - model: initialized model with new number of classes in output layer of model and load weights if weights not none.
    """
    
    model = torchvision.models.resnet18(pretrained=pretrained)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)

    if weights is not None:
        assert os.path.exists(weights)
        model.load_state_dict(torch.load(weights, map_location=map_location))

    return model

def get_datasets(path, split=(0.75, 0.25)):
    """
        Loads dataset
        args:
        - path = path to dataset
        - split = train val split

        returns:
        - train_set = training data
        - val_set = validation data
        - num_classes = number of classes in dataset
    """

    # if transform is not None:
    transform = T.Compose([
        T.Resize((256, 512)),
        T.ToTensor(),
        T.Normalize(mean=[0.6568, 0.6760, 0.7115], std=[0.3987, 0.3811, 0.3579])
    ])
    
    dataset = torchvision.datasets.ImageFolder(path, transform = transform)

    train_split, val_split = int(len(dataset) * split[0]), int(len(dataset) * split[1]) + 1
    train_set, val_set = torch.utils.data.random_split(dataset, [train_split, val_split])

    num_classes = len(dataset.class_to_idx)
    
    return train_set, val_set, num_classes

def get_dataloader(data, batch_size=1, shuffle=False):
    data_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=shuffle)
    return data_loader