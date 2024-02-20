
# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms


use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
print("CUDA Available?", use_cuda)


train_transforms = transforms.Compose([
    # extracts a square crop of size 22x22 from the center of the original image.
    # making the training process more robust to minor variations in image content.
    transforms.RandomApply([transforms.CenterCrop(22), ], p=0.1),
    # Resizes the image to a fixed size of 28x28, regardless of its original dimensions.
    transforms.Resize((28, 28)),
    # Applies a random rotation between -15 and 15 degrees to the image.
    # The fill=0 argument specifies that empty pixels created by rotation should be filled with black (value 0).
    transforms.RandomRotation((-15., 15.), fill=0),
    # PyTorch models operate on tensors, so this conversion is necessary for feeding data into the model.
    transforms.ToTensor(),
    # The given values suggest the dataset has a mean of 0.1307 and a standard deviation of 0.3081 for each channel
    transforms.Normalize((0.1307,), (0.3081,)),
    ])

train_data = datasets.MNIST('../data', train=True, download=True, transform=train_transforms)

# Test data transformations
# We are not using same mean and SD as we have augmented the data in training data set.
# Data augmentation artificially expands the training dataset by introducing random variations like crops, rotations, flips, etc.
# If you use data augmentation: calculate separate mean and standard deviation for each dataset independently to avoid introducing biases into the evaluation.
test_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1407,), (0.4081,))
    ])


train_data = datasets.MNIST('../data', train=True, download=True, transform=train_transforms)
test_data = datasets.MNIST('../data', train=False, transform=test_transforms)


batch_size = 512

# batch_size: This specifies the number of samples to group together into a batch when feeding data to your model. 
# shuffle (bool, default=False): When set to True, the data is shuffled at the beginning of each epoch, helping the model learn from different combinations of samples and avoid overfitting.
# num_workers (int, default=0): Controls the number of worker processes used for loading data in parallel. Higher values can improve speed but come with added overhead.
# pin_memory (bool, default=False): When set to True, tensors are copied to pinned memory before returning them. This speeds up GPU transfers as pinned memory is specifically optimized for GPU access.

kwargs = {'batch_size': batch_size, 'shuffle': False, 'num_workers': 1, 'pin_memory': True}

train_loader = torch.utils.data.DataLoader(train_data, **kwargs)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size)


import matplotlib.pyplot as plt



class Net(nn.Module):
    #This defines the structure of the NN.
    # Four convolutional layers (Conv2d):
    # Two fully-connected layers (Linear):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3)
        self.fc1 = nn.Linear(256, 50)
        self.fc2 = nn.Linear(50, 10)

    # The convolutional layers extract features from the image,
    # while the fully-connected layers learn higher-level representations and make predictions.
    def forward(self, x):
        x = F.relu(self.conv1(x), 2)
        # Applies ReLU activation function and 2x2 max pooling after convolutional layer.
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = F.relu(self.conv3(x), 2)
        x = F.relu(F.max_pool2d(self.conv4(x), 2))
        # Flattens the output of the last convolutional layer into a 1D vector using view.
        print(x.shape)
        x = x.view(-1, 256)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)



from tqdm import tqdm



def train(model, device, train_loader, optimizer, criterion):
  model.train()
  pbar = tqdm(train_loader)

  train_loss = 0
  correct = 0
  processed = 0

  for batch_idx, (data, target) in enumerate(pbar):
    data, target = data.to(device), target.to(device)
    optimizer.zero_grad()

    # Predict
    pred = model(data)

    # Calculate loss
    loss = criterion(pred, target)
    train_loss+=loss.item()

    # Backpropagation
    loss.backward()
    optimizer.step()

    processed += len(data)

    pbar.set_description(desc= f'Train: Loss={loss.item():0.4f} Batch_id={batch_idx} Accuracy={100*correct/processed:0.2f}')





# %% [markdown]
# CODE BLOCK: 10

# %%
model = Net().to(device)

from torchsummary import summary
summary(model, input_size=(1, 28, 28))

optimizer = optim.SGD(model.parameters(), lr=10.01, momentum=0.9)

scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1, verbose=True)
# New Line
criterion = nn.CrossEntropyLoss()
num_epochs = 2

for epoch in range(1, num_epochs+1):
  print(f'Epoch {epoch}')
  train(model, device, train_loader, optimizer, criterion)

  scheduler.step()
