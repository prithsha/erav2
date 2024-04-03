
from torchvision import datasets, transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader 

def compose_custom_transforms(transforms_collection):
        return transforms.Compose(transforms_collection)


def create_basic_transforms_collection(mean = (0.4914, 0.4822, 0.4465), std = (0.2470, 0.2435, 0.2616)):
    transforms_collection  = [ transforms.ToTensor(),
                              transforms.Normalize(mean=mean,std=std)]
    
    return transforms_collection

def create_random_rotation_transform(rotation = (-7.0,7.0), fill=(1,)):
    return transforms.RandomRotation(degrees=rotation, fill=fill)


def get_CIFAR10_datasets(train_transforms_collection, test_transforms_collection, data_folder) -> tuple[datasets.CIFAR10, datasets.CIFAR10]:

    train_dataset = datasets.CIFAR10( root=data_folder,
                                    train=True,
                                    download=True,
                                    transform=train_transforms_collection)
    
    test_dataset = datasets.CIFAR10( root=data_folder,
                                        train=False,
                                        download=True,
                                        transform=test_transforms_collection)
    
    return train_dataset, test_dataset


def get_dataloaders(train_dataset, test_dataset, batch_size = 128, shuffle=True, num_workers=4, pin_memory=True) -> tuple[DataLoader, DataLoader]:
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory)

    for batch_data, label in test_dataloader:
    # (e.g., shape: (batch_size, 1 channel, 28, 28)). (batch_size, channels, height, width)
    # y would contain the corresponding labels for each image, indicating the actual digit represented in the image 
        print(f"Shape of test_dataloader batch_data [Batch, C, H, W]: {batch_data.shape}")
        print(f"Shape of test_dataloader label (label): {label.shape} {label.dtype}")
        print(f"Labels for a batch of size {batch_size} are {label}")
        break

    return train_dataloader, test_dataloader