import os
import re
import torch
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from torchvision.io import read_image
import pandas as pd
from matplotlib import pyplot as plt

class HMEDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None, problem_type='symbols'):
        self.img_labels = pd.read_csv(annotations_file, header=None, sep='","')
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform
        self.problem_type = problem_type
    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        if self.problem_type is 'symbols':
            filename = 'iso' + str(idx) + '.png'
            img_path = os.path.join(self.img_dir, filename)
            label = self.img_labels.iloc[idx, 0]
        if self.problem_type is 'formula':
            img_path = self.__find_path(idx)
            label = self.img_labels.iloc[idx,1]

        image = read_image(img_path)
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label

    def __find_path(self, idx):
        regex = "TrainINKML\\\\(?P<folder>.*)\\\\(?P<file>.*)\.inkml"
        match = re.search(regex, self.img_labels.iloc[idx,0])
        if match:
            d = match.groupdict()
        else:
            raise RuntimeError(f'File not found for idx {idx}')
        new_path = os.path.join(self.img_dir, f"data_png_{d['folder']}",f"{d['file']}.png")
        return new_path

"""
labels = 'data/iso_GT.csv'
images = 'data/data_png/data_png_trainingSymbols/'

train_data = HMEDataset(labels, images)

train_dataloader = DataLoader(train_data, 20)

train_features, train_labels = next(iter(train_dataloader))
# img = train_features[1].squeeze()
# label = train_labels[1]
# plt.imshow(img, cmap="gray")
# plt.show()
print(train_labels)
"""