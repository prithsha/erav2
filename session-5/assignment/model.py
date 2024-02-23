import torch.nn as nn
import torch.nn.functional as F
import utils

class MySimpleNet(nn.Module):

    def __init__(self):
        super(MySimpleNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3)
        self.fc1 = nn.Linear(4096, 50)
        self.fc2 = nn.Linear(50, 10)

    # The convolutional layers extract features from the image,
    # while the fully-connected layers learn higher-level representations and make predictions.
    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.relu(x)

        x = self.conv3(x)
        x = F.relu(x)


        x = self.conv4(x)
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.relu(x)

        x = x.reshape(-1, 4096)

        x = self.fc1(x)
        x = F.relu(x)

        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
    