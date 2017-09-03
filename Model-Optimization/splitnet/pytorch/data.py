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
    'cifar10': datasets.CIFAR10(
        './datasets/cifar10', train=True, download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
        ])
    ),
    'cifar100': datasets.CIFAR100(
        './datasets/cifar100', train=True, download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
        ])
    )
}

DATASET_CONFIGS = {
    'mnist': {'size': 32, 'channels': 1, 'classes': 10},
    'cifar10': {'size': 32, 'channels': 3, 'classes': 10},
    'cifar100': {'size': 32, 'channels': 3, 'classes': 100},
}
