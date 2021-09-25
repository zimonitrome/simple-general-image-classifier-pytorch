import types
import argparse
from pathlib import Path
import inspect
import numpy as np
from tqdm import tqdm
import torch
from torch import nn
from torch.utils.data.dataloader import DataLoader
from torchvision import models, transforms
from torchvision.datasets import VisionDataset
from PIL import Image
from shutil import copy

def get_first_layer(net):
    while True:
        if isinstance(net, nn.Conv2d):
            return net
        try:
            net = list(net.children())[0]
        except:
            raise Exception("Model doesn't start with a convolution. Try another model.")

def get_last_layer(net):
    while True:
        if isinstance(net, nn.Linear):
            return net
        try:
            net = list(net.children())[-1]
        except:
            raise Exception("Model doesn't end with a convolution. Try another model.")

def list_items(items):
    return '\n - '.join(['', *items])

def save_model(model, path):
    path.parent.mkdir(exist_ok=True, parents=True)
    torch.save(model.state_dict(), path)

class PathDataset(VisionDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.paths = list(Path(self.root).rglob("*.png"))

    def __getitem__(self, index: int):
        path = self.paths[index]
        sample = self.transforms(Image.open(path))
        return sample, str(path)

    def __len__(self) -> int:
        return len(self.paths)

if __name__ == "__main__":
    # Get every function in 'models'
    available_models = [m for m in dir(models) if isinstance(getattr(models, m), types.FunctionType)]
    available_models_str = list_items(available_models)

    # Get every function in 'models'
    available_transforms = transforms.transforms.__all__
    available_transforms_with_args = [f"{t}{inspect.signature(getattr(transforms, t))}" for t in available_transforms]
    available_transforms_with_args_str = list_items(available_transforms_with_args)

    class CustomFormatter(argparse.RawTextHelpFormatter, argparse.ArgumentDefaultsHelpFormatter):
        pass

    parser = argparse.ArgumentParser(formatter_class=CustomFormatter)
    parser.add_argument('input_folder', help='Path to folder containing folder of classes.')
    parser.add_argument('model_checkpoint', help='Saved .pt file.')
    parser.add_argument('prediction_output_folder', help='Path to folder where predictions should be saved. Will be created.')
    parser.add_argument('-d', '--device', help='cuda or cpu', default="cuda")
    parser.add_argument('-ic', '--in_channels', help='Number of input channels. 3=rgb, 1=mono etc.', default=3)
    parser.add_argument('-m', '--model', help=f"Models to choose from. Available models:{available_models_str}", default="resnet18")
    parser.add_argument('-w', '--workers', help=f"How many threads to run for training dataloader.", default=0, type=int)
    parser.add_argument('-bs', '--batch_size', default=32, type=int)
    parser.add_argument('-t', '--transforms', help=("List of transforms (including parameters without spaces) to apply to each image.\n" + 
                                                    "For example: --transforms \"gaussianBlur(3)\" \"CenterCrop(64)\".\n" +
                                                    f"Available:{available_transforms_with_args_str}"), nargs='+', default=None)
    args = parser.parse_args()


    device = torch.device(args.device)


    # Build constant transforms
    if args.transforms:
        arg_transforms = [eval("transforms."+t) for t in args.transforms]
    else:
        arg_transforms = []
    constant_transforms = transforms.Compose([
        *arg_transforms,
        transforms.ToTensor(),
        transforms.Normalize(.5, .225),
    ])
    print("General transforms:", list_items([str(t) for t in constant_transforms.transforms]))


    # Build dataloaders
    data_folder = Path(args.input_folder)
    # dataset = DatasetFolder(data_folder, transform=constant_transforms, loader=lambda p: Image.open(p), extensions=["png"])
    dataset = PathDataset(data_folder, transforms=constant_transforms)
    print(f"Images: {len(dataset)}")
    dataloader = DataLoader(dataset, args.batch_size, shuffle=False, pin_memory=(args.workers), num_workers=args.workers)


    # Open saved checkpoint
    saved = torch.load(args.model_checkpoint)
    classes = saved["classes"]


    # Build model
    model = getattr(models, args.model)()
    model.to(device)
    first_layer = get_first_layer(model)
    first_layer.in_channels = args.in_channels
    last_layer = get_last_layer(model)
    last_layer.out_features = len(classes)
    model.load_state_dict(saved["model"])
    print(f"Input channels: {first_layer.in_channels}")
    print(f"Output neurons: {last_layer.out_features}")

    path_predictions = {}

    dl_iter = tqdm(dataloader, desc="Predicting")
    for inputs, paths in dl_iter:
        inputs = inputs.to(device)

        # forward + backward + optimize
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                outputs = model(inputs)
                _, predictions = torch.max(outputs, 1)

                for path, pred in zip(paths, predictions):
                    path_predictions[path] = pred

    output_root = Path(args.prediction_output_folder)
    path_preds_iter = tqdm(path_predictions.items(), desc="Copying files")
    for str_path, pred in path_preds_iter:
        output_path = Path(output_root) / classes[pred] / Path(str_path).name
        output_path.parent.mkdir(exist_ok=True, parents=True)
        copy(str_path, str(output_path))
