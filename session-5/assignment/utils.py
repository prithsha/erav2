import torch
import logging
import constant
import logginInit
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torchvision import utils as torchVisionUtils
from torch.utils.data import DataLoader, Dataset
from torchsummary import summary

logger = logging.getLogger(constant.APP_NAME).getChild(__name__)

from enum import Enum


class DatasetType(Enum):
    """Docstring for DatasetType."""
    MNIST = "some_value"
    MNIST_FASHION = "some_other_value"
    

def get_execution_device():
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    logger.info(f"used device: {device}, device name: {torch.cuda.get_device_name()}, device count: {torch.cuda.device_count()}")
    return device

def get_train_and_test_dataset(dataset_type: DatasetType, train_transform, test_transform) -> tuple [Dataset, Dataset]:
    logger.info("Get train and test data set instance.")
    train_data = None
    test_data = None

    if(dataset_type == DatasetType.MNIST):
        train_data = datasets.MNIST(constant.DATA_FOLDER, train=True, download=True, transform=train_transform)
        test_data = datasets.MNIST(constant.DATA_FOLDER, train=False, download=True, transform=test_transform)
    elif(dataset_type == DatasetType.MNIST_FASHION):
        train_data = datasets.FashionMNIST(constant.DATA_FOLDER, train=True, download=True, transform=train_transform)
        test_data = datasets.FashionMNIST(constant.DATA_FOLDER, train=False, download=True, transform=test_transform)

    
    return train_data, test_data

def get_basic_train_and_test_transform(target_image_size : tuple = (28,28),
                                    random_rotation = (-15.0,15.0),
                                    mean=0.1307,
                                    std=0.3081) -> tuple [transforms.Compose, transforms.Compose]:
    """Provide train and test transform

    Args:
        target_image_size (tuple, optional): Image size. Defaults to (28,28). Only for train transform
        random_rotation (tuple, optional): random rotation. Defaults to (-15.0,15.0). Only for train transform
        mean (float, optional): mean. Defaults to 0.1307.
        std (float, optional): standard deviation. Defaults to 0.3081.

    Returns:
        _type_: Tuple[transforms.Compose, transforms.Compose] train_transform, test_transform
    """
    train_transforms = transforms.Compose([ 
    transforms.RandomApply([transforms.CenterCrop(22), ], p=0.1), 
    transforms.Resize(target_image_size),  
    transforms.RandomRotation(random_rotation, fill=0),
    transforms.ToTensor(),
    transforms.Normalize((mean,), (std,))
    ])

    # Test data transformations
    test_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((mean,), (std,))
    ])
    logger.info("created train and test transforms")
    return train_transforms, test_transforms

def flatter(t : torch.Tensor):
  t = t.reshape(1, -1)
  t = t.squeeze()
  return t

def get_data_loader_instance(data_set : Dataset,
                            batch_size=32,
                            **kwargs
                            ) -> DataLoader:
    logger.info(f"Creating data loader instance with batch size {batch_size},kwargs: {kwargs}")
    loader = DataLoader(data_set,batch_size=batch_size, **kwargs)
    return loader

def visualize_images_in_frame(data_loader : DataLoader):
    logger.info("Preparing to show data as grid")
    batch = next(iter(data_loader))
    images, labels = batch
    grid = torchVisionUtils.make_grid(images, nrow=10)
    plt.figure(figsize=(15,15))
    plt.title(flatter(labels))
    plt.imshow(np.transpose(grid, (1,2,0)))
    plt.show()


def get_network_summary(model, input_size = (1,28,28)):        
        device = get_execution_device()
        model = model.to(device)
        logger.info(summary(model, input_size=input_size))

def get_correct_prediction_count(predictions : torch.Tensor, labels: torch.Tensor):
    return predictions.argmax(dim=1).eq(labels).sum().item()

def train_model(model, train_loader, optimizer, loss_criteria):

    model.train()
    device = get_execution_device()

    total_loss = 0
    total_correct = 0
    processed_images_count = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        # Predict
        prediction = model(images)

        # Calculate loss
        loss : torch.Tensor = loss_criteria(prediction, labels)
        total_loss+=loss.item()

        # Back propagation
        loss.backward()
        optimizer.step()

        total_correct += get_correct_prediction_count(prediction, labels)
        processed_images_count += len(images)

        logger.info(f'Train: Loss={loss.item():0.4f}, Accuracy={100*total_correct/processed_images_count:0.2f}')


    average_loss = total_loss/len(train_loader)
    average_accuracy = 100*total_correct/processed_images_count

    logger.info(f"Training set: average accuracy: {average_accuracy}, average loss: {average_loss}")

    return average_accuracy, average_loss


def test_model(model, test_loader, loss_criteria):
    model.eval()
    device = get_execution_device()

    test_loss = 0
    correct = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)

            prediction = model(images)
            test_loss += loss_criteria(prediction, labels, reduction='sum').item()  # sum up batch loss

            correct += get_correct_prediction_count(prediction, labels)


    average_loss = test_loss / len(test_loader.dataset)
    average_accuracy = 100. * correct / len(test_loader.dataset)

    logger.info(f"Test set: average accuracy: {average_accuracy}, average loss: {average_loss}")

    return average_accuracy, average_loss


def plot_train_accuracy_and_test_accuracy_graph(train_accuracy : list, train_losses : list, test_accuracy : list, test_losses : list):
    fig, axs = plt.subplots(2,2,figsize=(15,10))
    axs[0, 0].plot(train_losses)
    axs[0, 0].set_title("Training Loss")
    axs[1, 0].plot(train_accuracy)
    axs[1, 0].set_title("Training Accuracy")
    axs[0, 1].plot(test_losses)
    axs[0, 1].set_title("Test Loss")
    axs[1, 1].plot(test_accuracy)
    axs[1, 1].set_title("Test Accuracy")
    plt.show()


if __name__ == '__main__':
    logger.info("----START----")
    get_execution_device()
    train_transforms, test_transform = get_basic_train_and_test_transform()
    training_data, test_data = get_train_and_test_dataset(DatasetType.MNIST_FASHION, train_transforms, test_transform)
    train_loader = get_data_loader_instance(training_data, batch_size=64)
    visualize_images_in_frame(train_loader)
    logger.info("----END----")


