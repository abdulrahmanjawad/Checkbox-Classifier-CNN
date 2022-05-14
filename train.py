import os, sys
import argparse
import numpy as np
import random
from tqdm import tqdm

import torch
import torch.nn as nn

from helper import get_model, get_datasets, get_dataloader

# for reproducibilty
torch.manual_seed(44)
np.random.seed(44)
random.seed(44)

def main(args):
    data_dir = args.data_dir
    batch_size = args.batch_size
    epochs = args.epochs
    learning_rate = args.lr
    split = tuple(args.split)
    save_path = args.save_path
    pretrained = args.pretrained

    # define device: CPU or CUDA
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # load dataset and dataloader
    train_data, val_data, num_classes = get_datasets(data_dir, split)
    train_loader = get_dataloader(train_data, batch_size, shuffle=True)
    val_loader = get_dataloader(val_data, batch_size)

    # load model and move model to device
    model = get_model(num_classes, pretrained=pretrained)
    model.to(device)

    # define loss function
    criterion = nn.CrossEntropyLoss()

    # define optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # save path
    os.makedirs(save_path, exist_ok=True)
    model_name = f"checkbox_classifier_{batch_size}_{epochs}_{str(learning_rate).split('.')[1]}.pt"
    model_path = os.path.join(save_path, model_name)

    # epoch loop
    for epoch in range(epochs):
        desc = "Epoch " + str(epoch+1) + "/" + str(epochs)

        # set model to train
        model.train()

        train_loss = 0.0
        train_acc = 0

        # training loop
        for images, labels in tqdm(train_loader, desc=desc):

            # clear optimizer gradients
            optimizer.zero_grad()

            # move data to device
            images, labels = images.to(device), labels.to(device)

            # forward pass
            outputs = model(images)

            # calculate loss
            loss = criterion(outputs, labels)
        
            # backward pass
            loss.backward()

            # upgrade gradients
            optimizer.step()

            # statistics
            _, preds = torch.max(outputs, 1)
            train_loss += loss.item() * images.size(0)
            train_acc += torch.sum(preds == labels.data)

        # testing loop
        with torch.no_grad():
            # set model to eval
            model.eval()

            test_loss = 0.0
            test_acc = 0

            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)

                outputs = model(images)

                val_loss = criterion(outputs, labels)

                _, preds = torch.max(outputs, 1)
                
                test_loss += val_loss.item() * images.size(0)
                test_acc += torch.sum(preds == labels.data)
        
        # calculate statistics
        train_loss = train_loss / len(train_data)
        train_acc = train_acc.double() / len(train_data)

        test_loss = test_loss / len(val_data)
        test_acc = test_acc.double() / len(val_data)

        # print results
        print(f"Train: Loss: {train_loss:.4f} Accuracy: {train_acc:.4f}")
        print(f"Test: Loss: {test_loss:.4f} Accuracy: {test_acc:.4f}")
    
        # save model every epoch
        print(f"Saving model to {model_path}")
        torch.save(model.state_dict(), model_path)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-d", '--data_dir', help="Path to dataset", default="data/", required=True)
    parser.add_argument("-b", '--batch_size', help="Batch Size", default=16, type=int)
    parser.add_argument("-e", '--epochs', help="Epoch", default=15, type=int)
    parser.add_argument("-l", '--lr', help="Learning Rate", default=0.005, type=float)
    parser.add_argument("-t", '--split', help="Train_Val Splits e.g -t 0.75 0.25", nargs='+', default=[0.75, 0.25], type=int)
    parser.add_argument("-s", '--save_path', help="Model Save Path", default="save/")
    parser.add_argument("-p", '--pretrained', help="Use Pretrained model", default=False, type=bool)
    args = parser.parse_args()

    main(args)