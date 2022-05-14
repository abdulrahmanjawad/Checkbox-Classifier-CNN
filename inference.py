import os, sys
import argparse
from PIL import Image

import torch
from torchvision import transforms as T

from helper import get_model

def main(args):
    weights = args.model_path
    image_path = args.image_path

    assert os.path.exists(image_path)

    if os.path.isfile(image_path):
        images = [image_path]
    if os.path.isdir(image_path):
        images = [os.path.join(image_path, image) for image in os.listdir(image_path)]

    idx_to_class = {0: 'checked', 1: 'other', 2: 'unchecked'}

    # define transform: same as for training
    transform = T.Compose([
        T.Resize((256, 512)),
        T.ToTensor(),
        T.Normalize(mean=[0.6568, 0.6760, 0.7115], std=[0.3987, 0.3811, 0.3579])
    ])

    # define device: CPU or CUDA
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # load model and set to eval
    model = get_model(num_classes=3, weights=weights, map_location=device)
    model.to(device)
    model.eval()

    print("Predictions:")

    for path in images:

        # process image
        image = Image.open(path).convert('RGB')
        image = transform(image)
        image = image.unsqueeze(0)

        # inference
        output = model(image)

        # get index of highest predicted label
        output_label = torch.argmax(output)
        output_label = idx_to_class[output_label.item()]

        print(f"Image: {path}\nLabel Predicted: {output_label}")

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-m", '--model_path', help="Path to model weights", default="save\checkbox_classifier_16_20_01.pt", required=True)
    parser.add_argument("-i", '--image_path', help="Path to image or image_folder", default="", required=True)
    args = parser.parse_args()

    main(args)