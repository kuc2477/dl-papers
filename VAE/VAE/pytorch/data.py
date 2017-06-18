from torchvision import datasets, transforms


DATASETS = {
    'mnist': datasets.MNIST(
        './datasets/mnist', train=True, download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.ToPILImage(),
            transforms.Pad(2),
            transforms.ToTensor(),
        ])
    ),
}

DATASET_CONFIGS = {
    'mnist': {'image_size': 32, 'channel_num': 1},
}
