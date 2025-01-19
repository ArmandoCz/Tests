import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import datasets,transforms
from torch.utils.data import DataLoader

import matplotlib
matplotlib.use('TkAgg') 
import matplotlib.pyplot as plt

import typing

batch_size = 64
learning_rate = 0.001
num_epochs = 5

def get_device() -> str:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_dataset(bach_size:int) -> typing.Tuple[DataLoader,DataLoader]:
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,),(0.5,))
    ])

    train_dataset = datasets.MNIST(
        root="./data",
        train = True,
        transform=transform,
        download=True)
    
    test_dataset = datasets.MNIST(
        root="./data",
        train = False,
        transform=transform,
        download=True)
    
    train_loader = DataLoader(
        dataset = train_dataset,
        batch_size = bach_size,
        shuffle = True
    )

    test_loader = DataLoader(
        dataset = test_dataset,
        batch_size = bach_size,
        shuffle = True
    )

    return (train_loader,test_loader)

def get_random_image(dataloader:DataLoader)-> None:
    images, labels = next(iter(train_loader))
    image = images[0] 
    image = image * 0.5 + 0.5 
    image = image.permute(1, 2, 0)
    plt.imshow(image.squeeze(), cmap="gray")  
    plt.title(f"Etiqueta: {labels[0].item()}")
    plt.axis("off")
    plt.show() 

if __name__ == "__main__":
    device = get_device()
    train_loader, test_loader = get_dataset(bach_size=batch_size)
    get_random_image(dataloader=train_loader)