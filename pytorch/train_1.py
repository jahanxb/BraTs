# # import os
# # import nibabel as nib
# # import numpy as np
# # import torch
# # from torch.utils.data import Dataset

# # class BrainTumorDataset(Dataset):
# #     def __init__(self, img_root, label_root, transform=None):
# #         self.img_root = img_root
# #         self.label_root = label_root
# #         self.transform = transform
# #         self.image_filenames = sorted(os.listdir(self.img_root))
# #         self.label_filenames = sorted(os.listdir(self.label_root))

# #     def __len__(self):
# #         return len(self.image_filenames)

# #     def __getitem__(self, index):
# #         # Load the image and label volumes
# #         image_path = os.path.join(self.img_root, self.image_filenames[index])
# #         label_path = os.path.join(self.label_root, self.label_filenames[index])

# #         image_volume = nib.load(image_path)
# #         label_volume = nib.load(label_path)

# #         image_data = image_volume.get_fdata().astype(np.float32)
# #         label_data = label_volume.get_fdata().astype(np.int64)

# #         # Apply transformations, if specified
# #         if self.transform is not None:
# #             image_data = self.transform(image_data)

# #         # Convert to PyTorch tensors and return
# #         image_tensor = torch.from_numpy(image_data).unsqueeze(0)
# #         label_tensor = torch.from_numpy(label_data).unsqueeze(0)

# #         return image_tensor, label_tensor



# # import torch
# # import torch.nn as nn
# # import torch.optim as optim
# # import torchvision.transforms as transforms
# # from torch.utils.data import DataLoader

# # # Set the device (GPU if available, else CPU)
# # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # IMG_ROOT = '/mydata/BrainTumor/imagesTr'
# # IMG_PATH = '/mydata/BrainTumor/imagesTr/BRATS_148.nii.gz'
# # IMG_OUTPUT_ROOT = '/mydata/BrainTumor/train/image_T1'

# # LABEL_ROOT = '/mydata/BrainTumor/labelsTr'
# # IABEL_PATH = './mydata/BrainTumor/labelsTr/BRATS_148.nii.gz'
# # LABEL_OUTPUT_ROOT = '/mydata/BrainTumor/train/label'



# # # Define your dataset and dataloader
# # # Assuming you have created a custom dataset class called BrainTumorDataset
# # train_dataset = BrainTumorDataset(IMG_ROOT, LABEL_ROOT)
# # train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)

# # # Create an instance of the PSPNet model
# # from models import pspnet_res50,pspnet_res34,pspnet_res18
# # model = pspnet_res50(drop_rate=0.2).to(device)

# # # Define the loss function (e.g., cross entropy) and optimizer
# # criterion = nn.CrossEntropyLoss()
# # optimizer = optim.Adam(model.parameters(), lr=0.001)

# # # Training loop
# # num_epochs = 10
# # for epoch in range(num_epochs):
# #     running_loss = 0.0
# #     model.train()
# #     for images, labels in train_dataloader:
# #         images = images.to(device)
# #         labels = labels.to(device)

# #         optimizer.zero_grad()
# #         print("images: ",images.shape)
# #         # Forward pass
# #         outputs = model(images)
# #         loss = criterion(outputs, labels)

# #         # Backward pass and optimization
# #         loss.backward()
# #         optimizer.step()

# #         running_loss += loss.item()

# #     # Print epoch statistics
# #     epoch_loss = running_loss / len(train_dataloader)
# #     print(f"Epoch [{epoch+1}/{num_epochs}] - Loss: {epoch_loss:.4f}")

# # # Save the trained model
# # torch.save(model.state_dict(), "pspnet_brain_tumor.pth")

# ###############################################################
# ###############################################################



# # import os
# # import nibabel as nib
# # import numpy as np
# # import torch
# # from torch.utils.data import Dataset
# # import torch.nn as nn
# # import torch.optim as optim
# # from torch.utils.data import DataLoader

# # class BrainTumorDataset(Dataset):
# #     def __init__(self, img_root, label_root, transform=None):
# #         self.img_root = img_root
# #         self.label_root = label_root
# #         self.transform = transform
# #         self.image_filenames = sorted(os.listdir(self.img_root))
# #         self.label_filenames = sorted(os.listdir(self.label_root))

# #     def __len__(self):
# #         return len(self.image_filenames)

# #     def __getitem__(self, index):
# #         # Load the image and label volumes
# #         image_path = os.path.join(self.img_root, self.image_filenames[index])
# #         label_path = os.path.join(self.label_root, self.label_filenames[index])

# #         image_volume = nib.load(image_path)
# #         label_volume = nib.load(label_path)

# #         image_data = image_volume.get_fdata().astype(np.float32)
# #         label_data = label_volume.get_fdata().astype(np.int64)

# #         # Apply transformations, if specified
# #         if self.transform is not None:
# #             image_data = self.transform(image_data)

# #         # Convert to PyTorch tensors and return
# #         image_tensor = torch.from_numpy(image_data).unsqueeze(0)
# #         label_tensor = torch.from_numpy(label_data).unsqueeze(0)

# #         return image_tensor, label_tensor


# # IMG_ROOT = '/mydata/BrainTumor/imagesTr'
# # LABEL_ROOT = '/mydata/BrainTumor/labelsTr'

# # # Define your dataset and dataloader
# # # Assuming you have created a custom dataset class called BrainTumorDataset
# # train_dataset = BrainTumorDataset(IMG_ROOT, LABEL_ROOT)
# # train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)

# # # Create an instance of the PSPNet model
# # from models import pspnet_res50

# # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# # model = pspnet_res50(drop_rate=0.2).to(device)

# # # Define the loss function (e.g., cross entropy) and optimizer
# # criterion = nn.CrossEntropyLoss()
# # optimizer = optim.Adam(model.parameters(), lr=0.001)

# # # Training loop
# # num_epochs = 10
# # for epoch in range(num_epochs):
# #     running_loss = 0.0
# #     model.train()
# #     for images, labels in train_dataloader:
# #         images = images.to(device)
# #         labels = labels.to(device)

# #         optimizer.zero_grad()
# #         print("images: ", images.shape)

# #         # Reshape the input tensor
# #         batch_size, _, depth, height, width = images.shape
# #         images = images.view(batch_size * depth, 1, height, width)

# #         # Forward pass
# #         outputs = model(images)
# #         loss = criterion(outputs, labels)

# #         # Backward pass and optimization
# #         loss.backward()
# #         optimizer.step()

# #         running_loss += loss.item()

# #     # Print epoch statistics
# #     epoch_loss = running_loss / len(train_dataloader)
# #     print(f"Epoch [{epoch+1}/{num_epochs}] - Loss: {epoch_loss:.4f}")

# # # Save the trained model
# # torch.save(model.state_dict(), "pspnet_brain_tumor.pth")

# import os
# import nibabel as nib
# import numpy as np
# import torch
# from torch.utils.data import Dataset
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import DataLoader

# class BrainTumorDataset(Dataset):
#     def __init__(self, img_root, label_root, transform=None):
#         self.img_root = img_root
#         self.label_root = label_root
#         self.transform = transform
#         self.image_filenames = sorted(os.listdir(self.img_root))
#         self.label_filenames = sorted(os.listdir(self.label_root))

#     def __len__(self):
#         return len(self.image_filenames)

#     def __getitem__(self, index):
#         # Load the image and label volumes
#         image_path = os.path.join(self.img_root, self.image_filenames[index])
#         label_path = os.path.join(self.label_root, self.label_filenames[index])

#         image_volume = nib.load(image_path)
#         label_volume = nib.load(label_path)

#         image_data = image_volume.get_fdata().astype(np.float32)
#         label_data = label_volume.get_fdata().astype(np.int64)

#         # Apply transformations, if specified
#         if self.transform is not None:
#             image_data = self.transform(image_data)

#         # Convert to PyTorch tensors and return
#         image_tensor = torch.from_numpy(image_data)
#         label_tensor = torch.from_numpy(label_data)

#         return image_tensor, label_tensor


# IMG_ROOT = '/mydata/BrainTumor/imagesTr'
# LABEL_ROOT = '/mydata/BrainTumor/labelsTr'

# # Define your dataset and dataloader
# # Assuming you have created a custom dataset class called BrainTumorDataset
# train_dataset = BrainTumorDataset(IMG_ROOT, LABEL_ROOT)
# train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)  # Set batch_size=1

# # Create an instance of the PSPNet model
# from models import pspnet_res50

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = pspnet_res50(drop_rate=0.2).to(device)

# # Define the loss function (e.g., cross entropy) and optimizer
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters(), lr=0.001)

# # Training loop
# num_epochs = 10
# for epoch in range(num_epochs):
#     running_loss = 0.0
#     model.train()
#     for images, labels in train_dataloader:
#         images = images.unsqueeze(0).to(device)
#         labels = labels.unsqueeze(0).to(device)

#         optimizer.zero_grad()
#         print("images: ", images.shape)

#         # Reshape the input tensor
#         depth, height, width = images.shape[2:]
#         images = images.view(1, 1, depth, height, width)

#         # Forward pass
#         outputs = model(images)
#         loss = criterion(outputs, labels)

#         # Backward pass and optimization
#         loss.backward()
#         optimizer.step()

#         running_loss += loss.item()

#     # Print epoch statistics
#     epoch_loss = running_loss / len(train_dataloader)
#     print(f"Epoch [{epoch+1}/{num_epochs}] - Loss: {epoch_loss:.4f}")

# # Save the trained model
# torch.save(model.state_dict(), "pspnet_brain_tumor.pth")


import os
import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

class BrainTumorDataset(Dataset):
    def __init__(self, img_root, label_root, transform=None):
        self.img_root = img_root
        self.label_root = label_root
        self.transform = transform
        self.image_filenames = sorted(os.listdir(self.img_root))
        self.label_filenames = sorted(os.listdir(self.label_root))

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, index):
        # Load the image and label volumes
        image_path = os.path.join(self.img_root, self.image_filenames[index])
        label_path = os.path.join(self.label_root, self.label_filenames[index])

        image_volume = nib.load(image_path)
        label_volume = nib.load(label_path)

        image_data = image_volume.get_fdata().astype(np.float32)
        label_data = label_volume.get_fdata().astype(np.int64)

        # Apply transformations, if specified
        if self.transform is not None:
            image_data = self.transform(image_data)

        # Convert to PyTorch tensors and return
        image_tensor = torch.from_numpy(image_data)
        label_tensor = torch.from_numpy(label_data)

        return image_tensor, label_tensor


IMG_ROOT = '/mydata/BrainTumor/imagesTr'
LABEL_ROOT = '/mydata/BrainTumor/labelsTr'

# Define your dataset and dataloader
# Assuming you have created a custom dataset class called BrainTumorDataset
train_dataset = BrainTumorDataset(IMG_ROOT, LABEL_ROOT)
train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)  # Set batch_size=1

# Create an instance of the PSPNet model
from models import pspnet_res50

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = pspnet_res50(drop_rate=0.2).to(device)

# Define the loss function (e.g., cross entropy) and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    running_loss = 0.0
    model.train()
    for images, labels in train_dataloader:
        images = images.unsqueeze(0).to(device)
        labels = labels.unsqueeze(0).to(device)

        optimizer.zero_grad()
        print("images: ", images.shape)

        # Process each volume separately
        volumes = images.shape[2]
        for v in range(volumes):
            image = images[:, :, v, :, :]
            label = labels[:, :, v, :, :]

            # Forward pass
            outputs = model(image)
            loss = criterion(outputs, label)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

    # Print epoch statistics
    epoch_loss = running_loss / len(train_dataloader)
    print(f"Epoch [{epoch+1}/{num_epochs}] - Loss: {epoch_loss:.4f}")

# Save the trained model
torch.save(model.state_dict(), "pspnet_brain_tumor.pth")



