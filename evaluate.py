import argparse
import numpy as np
from tqdm import tqdm

from sklearn.metrics import confusion_matrix

import torch

from helper import get_model, get_datasets

def main(args):
    
    # define device: CPU or CUDA
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    idx_to_class = {0: 'checked', 1: 'other', 2: 'unchecked'}

    # load val dataset and loader
    _, val_data, num_classes = get_datasets(args.data_dir, args.split)

    # load model and weights
    model = get_model(num_classes, weights=args.model_path, map_location=device)
    model.to(device)
    model.eval()

    labels = list()
    preds = list()

    for image, label in tqdm(val_data):

        # add batch_size dimension to image shape and move to device
        image = image.unsqueeze(0).to(device)

        # get prediction
        output = model(image)
        
        # get index of highest predicted label
        output_label = torch.argmax(output)

        labels.append(label)
        preds.append(output_label.item())
    
    labels = np.array(labels)
    preds = np.array(preds)

    cm = confusion_matrix(preds, labels, labels=list(range(num_classes)))

    TP = np.diag(cm)
    FP = np.sum(cm, axis=0) - TP
    FN = np.sum(cm, axis=1) - TP
    precision = TP/(TP+FP)
    recall = TP/(TP+FN)

    print("-----Evaluation-----")
    for i in range(num_classes):
        print(f"Class {idx_to_class[i]}: TP {TP[i]} --- FP {FP[i]} --- FN {FN[i]} --- Precision {precision[i]} --- Recall {recall[i]}")



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-m", '--model_path', help="Path to model weights for evaluation", default="save\checkbox_classifier_16_20_01.pt", required=True)
    parser.add_argument("-d", '--data_dir', help="Path to dataset", default="data/")
    parser.add_argument("-t", '--split', help="Train_Val Splits e.g -t 0.75 0.25", nargs='+', default=[0.75, 0.25], type=int)
    args = parser.parse_args()

    main(args)