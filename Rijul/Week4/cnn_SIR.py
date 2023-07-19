import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data.dataset import random_split
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#Define Hyperparameters
num_epochs = 3
batch_size = 2
learning_rate = 0.01
count = 0
# dataset has PILImage images of range [0, 1]. 
# We transform them to Tensors of normalized range [-1, 1]
class ImageDataset(Dataset):
    def __init__(self, low_res_dir, high_res_dir, transform=None):
        self.low_res_dir = low_res_dir
        self.high_res_dir = high_res_dir
        self.transform = transform

        self.low_res_files = sorted(os.listdir(low_res_dir))
        self.high_res_files = sorted(os.listdir(high_res_dir))

    def __len__(self):
        return len(self.low_res_files)

    def __getitem__(self, idx):
        low_res_path = os.path.join(self.low_res_dir, self.low_res_files[idx])
        high_res_path = os.path.join(self.high_res_dir, self.high_res_files[idx])

        low_res_image = Image.open(low_res_path).convert("RGB")
        high_res_image = Image.open(high_res_path).convert("RGB")

        if self.transform is not None:
            low_res_image = self.transform(low_res_image)
            high_res_image = self.transform(high_res_image)

        return low_res_image, high_res_image

transform = transforms.Compose(
    [transforms.Resize((212,212)),
    transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

low_res_dir = "/Users/rushilbhat/Downloads/DIV2K_train_LR_bicubic/X2"
high_res_dir = "/Users/rushilbhat/Downloads/DIV2K_train_HR"

dataset = ImageDataset(low_res_dir, high_res_dir, transform=transform)

train_proportion = 0.8
test_proportion = 1 - train_proportion

num_samples = len(dataset)
num_train_samples = int(train_proportion * num_samples)
num_test_samples = num_samples - num_train_samples

train_dataset, test_dataset = random_split(dataset, [num_train_samples, num_test_samples])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

class SRCNN(nn.Module):
    def __init__(self):
        super(SRCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 9)
        self.conv2 = nn.Conv2d(64, 32, 1)
        self.conv3 = nn.Conv2d(32, 3, 5)

    def forward(self, x):
        # -> n, 3, 224, 224
        x = F.relu(self.conv1(x)) # -> n, 64, 216, 216
        x = F.relu(self.conv2(x)) # -> n, 32, 216, 216
        x = self.conv3(x)

        return x

model = SRCNN().to(device)

model.load_state_dict(torch.load('./srcnn.pth'))

#Loss and Optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# Train the model
n_total_steps = len(train_loader)
print(n_total_steps)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):  
        # origin shape: [100, 3, 224, 224]
        images = F.interpolate(labels, size=(224, 224), mode='bilinear')
        images = images.to(device)
        labels = labels.to(device)
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        count += 1
        optimizer.step()

        if (i+1) % 40 == 0:
            print (f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{n_total_steps}], Loss: {loss.item():.4f}')


print('Finished Training', count)
PATH = './srcnn.pth'
torch.save(model.state_dict(), PATH)

with torch.no_grad():

    for images, labels in test_loader:
        input_img_og = images[0]/2 + 0.5
        input_image = transforms.ToPILImage()(input_img_og.squeeze(0))
        input_image.save('./input_image.jpg')
        images = F.interpolate(labels, size=(224, 224), mode='bilinear')
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        output_img_og = outputs[0]/2 + 0.5
        output_image = transforms.ToPILImage()(output_img_og.squeeze(0))
        output_image.save('./output_image.jpg')
        label_og = labels[0]/2+0.5
        label = transforms.ToPILImage()(label_og.squeeze(0))
        label.save('./label.jpg')
        print(nn.MSELoss()(outputs, labels))
        break
        