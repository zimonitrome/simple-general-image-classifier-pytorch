
import types
import random
import argparse
from pathlib import Path
import inspect
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import random_split
from torchvision import models, transforms
from torchvision.datasets import ImageFolder
from copy import deepcopy

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

def save_model(path):
    path.parent.mkdir(exist_ok=True, parents=True)
    torch.save({
        "model": model.state_dict(),
        "classes": base_dataset.classes
    }, path)

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
    parser.add_argument('dataset_folder', help='Path to folder containing folder of classes.')
    parser.add_argument('model_output_folder', help='Path to folder where checkpoints should be saved. Will be created.')
    parser.add_argument('-d', '--device', help='cuda or cpu', default="cuda")
    parser.add_argument('-ic', '--in_channels', help='Number of input channels. 3=rgb, 1=mono etc.', default=3)
    parser.add_argument('-m', '--model', help=f"Models to choose from. Available models:{available_models_str}", default="resnet18")
    parser.add_argument('-pt', '--pre_trained', help="Use pre-trained weights.", action="store_true")
    parser.add_argument('-vp', '--val_p', help=f"Percentage of dataset that should be used for validation.", default=0, type=float)
    parser.add_argument('-vi', '--val_interval', help=f"After which epochs to perform validation.", default=1)
    parser.add_argument('-s', '--seed', help=f"Custom seed.")
    parser.add_argument('-si', '--save_interval', help=f"How often to save checkpoints. If 0, then will just save a final model.", default=0, type=int)
    parser.add_argument('-w', '--train_workers', help=f"How many threads to run for training dataloader.", default=0, type=int)
    parser.add_argument('-vw', '--val_workers', help=f"How many threads to run for validation dataloader.", default=0, type=int)
    parser.add_argument('-e', '--epochs', help=f"Number of epochs to train for. Once done, will ask you if you want to train for further epochs", default=5, type=int)
    parser.add_argument('-bs', '--batch_size', default=32, type=int)
    parser.add_argument('-lr', '--learning_rate', default=1e-3, type=float)
    parser.add_argument('-t', '--transforms', help=("List of transforms (including parameters without spaces) to apply to each image.\n" + 
                                                    "For example: --transforms \"gaussianBlur(3)\" \"CenterCrop(64)\".\n" +
                                                    f"Available:{available_transforms_with_args_str}"), nargs='+', default=None)
    parser.add_argument('-a', '--augmentations', help=f"List of transforms (including parameters without spaces) to apply to each image. See --transforms for available options.", nargs='+', default=None)
    args = parser.parse_args()


    # Set seed
    if args.seed:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        print(f"Random seed set to: {args.seed}")


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


    # Build augmentations
    if args.augmentations:
        arg_augmentations = [eval("transforms."+t) for t in args.augmentations]
        train_transforms = transforms.Compose([
            *arg_transforms,
            *arg_augmentations,
            transforms.ToTensor(),
            transforms.Normalize(.5, .5),
        ])
        print("Training specific transforms:", list_items([str(t) for t in train_transforms.transforms]))
    else:
        print("Training without augmentations.")


    # Build dataloaders
    data_folder = Path(args.dataset_folder)
    base_dataset = ImageFolder(data_folder, transform=constant_transforms)

    use_validation = args.val_p > 0

    if use_validation:
        val_len = int(len(base_dataset)*args.val_p)
        train_len = len(base_dataset)-val_len
        train_dataset, val_dataset = random_split(base_dataset, [train_len, val_len])
        print(f"Training images: {len(train_dataset)}\t\tValidation images: {len(val_dataset)}")
    else:
        train_dataset = base_dataset
        print("Training without validation images.")

    if args.augmentations:
        # Weird code but this is how it has to be done
        # in order to set augmentation on/off for val/train.
        train_dataset.dataset = deepcopy(train_dataset.dataset)
        train_dataset.dataset.transform = train_transforms
    

    train_dataloader = DataLoader(train_dataset, args.batch_size, shuffle=True, pin_memory=(args.train_workers > 0), num_workers=args.train_workers)
    if use_validation:
        val_dataloader = DataLoader(val_dataset, args.batch_size, shuffle=False, pin_memory=(args.val_workers), num_workers=args.val_workers)
        assert len(val_dataloader) > 0, "too few samples"


    # Build model
    if args.pre_trained:
        model = getattr(models, args.model)(pretrained=True)
    else:
        model = getattr(models, args.model)()
    model.to(device)
    first_layer = get_first_layer(model)
    first_layer.in_channels = args.in_channels
    last_layer = get_last_layer(model)
    last_layer.out_features = len(base_dataset.classes)
    print(f"Input channels: {first_layer.in_channels}")
    print(f"Output neurons: {last_layer.out_features}")

    # Build optimizer
    optimizer = AdamW(model.parameters(), args.learning_rate)


    # Loss function
    criterion = nn.CrossEntropyLoss()


    # Set up for training
    epochs = args.epochs
    epoch = 0
    scaler = torch.cuda.amp.GradScaler()
    training_losses = {}
    training_accuracies = {}
    val_losses = {}
    val_accuracies = {}

    dl_len = len(train_dataloader)
    save_folder = Path(args.model_output_folder)


    # Training loop
    while epoch != epochs:
        print(f"Epoch {epoch}")
        dl_iter = tqdm(train_dataloader, desc="Training")
        for i, (inputs, labels) in enumerate(dl_iter):
            global_step = (epoch * dl_len) + i
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()

            # forward + backward + optimize
            with torch.cuda.amp.autocast():
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                _, predictions = torch.max(outputs, 1)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            training_losses[global_step] = loss.item()
            training_accuracies[global_step] = (labels == predictions).float().mean().item()

        do_validate = use_validation and (epoch % args.val_interval == args.val_interval - 1)

        if do_validate:
            dl_iter = tqdm(val_dataloader, desc="Validating")
            val_losses_local = []
            val_accuracies_local = []
            for i, (inputs, labels) in enumerate(dl_iter):
                inputs = inputs.to(device)
                labels = labels.to(device)

                # forward
                with torch.no_grad():
                    with torch.cuda.amp.autocast():
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                        _, predictions = torch.max(outputs, 1)

                val_losses_local.append(loss.item())
                val_accuracies_local.append((labels == predictions).float().mean().item())

            val_losses[global_step] = np.mean(val_losses_local)
            val_accuracies[global_step] = np.mean(val_accuracies_local)

        print(
            f"Train loss: {np.mean(list(training_losses.values())[-dl_len:])}\t" +
            ((do_validate or '') and f"Val loss: {list(val_losses.values())[-1]}\t") +
            f"Train acc: {np.mean(list(training_accuracies.values())[-dl_len:])}\t" +
            ((do_validate or '') and f"Val loss: {list(val_accuracies.values())[-1]}")
        )

        if args.save_interval > 0:
            if epoch % args.save_interval == args.save_interval - 1:
                save_path = save_folder / f"Epoch_{epoch}.pt"
                save_model(save_path)
                print(f"Saved {save_path}")


        if epoch == (epochs-1):
            x_values = np.array(list(training_accuracies.keys())) / dl_len
            plt.plot(x_values, training_accuracies.values(), label="Training accuracy")
            if use_validation:
                x_values = np.array(list(val_accuracies.keys())) / dl_len
                plt.plot(x_values, val_accuracies.values(), label="Validation accuracy", marker='o')
            plt.xlabel("epoch")
            plt.grid()
            plt.legend()
            plt.show()
            if input("Done training? [y/n]: ") == 'y':
                break
            epochs += int(input("How many more epochs to train?: "))
        epoch += 1

    save_path = save_folder / f"Final.pt"
    save_model(save_path)
    print(f"Saved final model to {save_folder}")