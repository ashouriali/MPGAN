import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader


def load_data(train=True, download=True, batch_size=10):
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    train_set = torchvision.datasets.CIFAR10(root='./../data', train=train, download=download, transform=transform)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)

    return train_loader
