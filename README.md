# simple-general-image-classifier-pytorch
For when you quickly want to train a classifier to categorize image files.

## Usage:

1.  Train:
    ```
    python train.py [dataset_folder] [weights_folder]
    ```

2.  Eval
    ```
    python eval.py [folder_with_images] [weights_folder/mycheckpoint.pt] [output_folder]
    ```

That's it!

## Example:

For this folder structure:

```
files/
    dataset_folder/
        cats/
            0.png
            1.png
            ...
        dogs/
            0.png
            1.png
            ...
    unclassified_images/
        a.png
        b.png
        another_folder/
            c.png
            d.png
```

running:
```
python train.py files/dataset_folder files/classifier_wieghts --epochs 3 --save_interval 1
```

will result in:
```
files/
    dataset_folder/
        cats/
            0.png
            1.png
            ...
        dogs/
            0.png
            1.png
            ...
    unclassified_images/
        a.png
        b.png
        another_folder/
            c.png
            d.png
    classifier_wieghts/
        E0.pt
        E1.pt
        E2.pt
        Final.pt
```

then running:

```
python eval.py files/unclassified_images files/classifier_wieghts/E2.pt files/classified_images/
```

will result in:
```
files/
    dataset_folder/
        cats/
            0.png
            1.png
            ...
        dogs/
            0.png
            1.png
            ...
    unclassified_images/
        a.png
        b.png
        another_folder/
            c.png
            d.png
    classifier_wieghts/
        E0.pt
        E1.pt
        E2.pt
        Final.pt
    classified_images/
        cats/
            c.png
        dogs/
            a.png
            b.png
            d.png
```

## Full `--help` output:

```
usage: train.py [-h] [-d DEVICE] [-ic IN_CHANNELS] [-m MODEL] [-pt]
                [-vp VAL_P] [-vi VAL_INTERVAL] [-s SEED] [-si SAVE_INTERVAL]
                [-w TRAIN_WORKERS] [-vw VAL_WORKERS] [-e EPOCHS]
                [-bs BATCH_SIZE] [-lr LEARNING_RATE]
                [-t TRANSFORMS [TRANSFORMS ...]]
                [-a AUGMENTATIONS [AUGMENTATIONS ...]]
                dataset_folder model_output_folder

positional arguments:
  dataset_folder        Path to folder containing folder of classes.
  model_output_folder   Path to folder where checkpoints should be saved. Will be created.

optional arguments:
  -h, --help            show this help message and exit
  -d DEVICE, --device DEVICE
                        cuda or cpu (default: cuda)
  -ic IN_CHANNELS, --in_channels IN_CHANNELS
                        Number of input channels. 3=rgb, 1=mono etc. (default: 3)
  -m MODEL, --model MODEL
                        Models to choose from. Available models:
                         - alexnet
                         - densenet121
                         - densenet161
                         - densenet169
                         - densenet201
                         - googlenet
                         - inception_v3
                         - mnasnet0_5
                         - mnasnet0_75
                         - mnasnet1_0
                         - mnasnet1_3
                         - mobilenet_v2
                         - resnet101
                         - resnet152
                         - resnet18
                         - resnet34
                         - resnet50
                         - resnext101_32x8d
                         - resnext50_32x4d
                         - shufflenet_v2_x0_5
                         - shufflenet_v2_x1_0
                         - shufflenet_v2_x1_5
                         - shufflenet_v2_x2_0
                         - squeezenet1_0
                         - squeezenet1_1
                         - vgg11
                         - vgg11_bn
                         - vgg13
                         - vgg13_bn
                         - vgg16
                         - vgg16_bn
                         - vgg19
                         - vgg19_bn
                         - wide_resnet101_2
                         - wide_resnet50_2 (default: resnet18)
  -pt, --pre_trained    Use pre-trained weights. (default: False)
  -vp VAL_P, --val_p VAL_P
                        Percentage of dataset that should be used for validation. (default: 0)
  -vi VAL_INTERVAL, --val_interval VAL_INTERVAL
                        After which epochs to perform validation. (default: 1)
  -s SEED, --seed SEED  Custom seed. (default: None)
  -si SAVE_INTERVAL, --save_interval SAVE_INTERVAL
                        How often to save checkpoints. If 0, then will just save a final model. (default: 0)
  -w TRAIN_WORKERS, --train_workers TRAIN_WORKERS
                        How many threads to run for training dataloader. (default: 0)
  -vw VAL_WORKERS, --val_workers VAL_WORKERS
                        How many threads to run for validation dataloader. (default: 0)
  -e EPOCHS, --epochs EPOCHS
                        Number of epochs to train for. Once done, will ask you if you want to train for further epochs (default: 5)
  -bs BATCH_SIZE, --batch_size BATCH_SIZE
  -lr LEARNING_RATE, --learning_rate LEARNING_RATE
  -t TRANSFORMS [TRANSFORMS ...], --transforms TRANSFORMS [TRANSFORMS ...]
                        List of transforms (including parameters without spaces) to apply to each image.
                        For example: --transforms "gaussianBlur(3)" "CenterCrop(64)".
                        Available:
                         - Compose(transforms)
                         - ToTensor()
                         - PILToTensor()
                         - ConvertImageDtype(dtype: torch.dtype) -> None
                         - ToPILImage(mode=None)
                         - Normalize(mean, std, inplace=False)
                         - Resize(size, interpolation=2)
                         - Scale(*args, **kwargs)
                         - CenterCrop(size)
                         - Pad(padding, fill=0, padding_mode='constant')
                         - Lambda(lambd)
                         - RandomApply(transforms, p=0.5)
                         - RandomChoice(transforms)
                         - RandomOrder(transforms)
                         - RandomCrop(size, padding=None, pad_if_needed=False, fill=0, padding_mode='constant')
                         - RandomHorizontalFlip(p=0.5)
                         - RandomVerticalFlip(p=0.5)
                         - RandomResizedCrop(size, scale=(0.08, 1.0), ratio=(0.75, 1.3333333333333333), interpolation=2)
                         - RandomSizedCrop(*args, **kwargs)
                         - FiveCrop(size)
                         - TenCrop(size, vertical_flip=False)
                         - LinearTransformation(transformation_matrix, mean_vector)
                         - ColorJitter(brightness=0, contrast=0, saturation=0, hue=0)
                         - RandomRotation(degrees, resample=False, expand=False, center=None, fill=None)
                         - RandomAffine(degrees, translate=None, scale=None, shear=None, resample=0, fillcolor=0)
                         - Grayscale(num_output_channels=1)
                         - RandomGrayscale(p=0.1)
                         - RandomPerspective(distortion_scale=0.5, p=0.5, interpolation=2, fill=0)
                         - RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0, inplace=False)
                         - GaussianBlur(kernel_size, sigma=(0.1, 2.0)) (default: None)
  -a AUGMENTATIONS [AUGMENTATIONS ...], --augmentations AUGMENTATIONS [AUGMENTATIONS ...]
                        List of transforms (including parameters without spaces) to apply to each image. See --transforms for available options. (default: None)
```
