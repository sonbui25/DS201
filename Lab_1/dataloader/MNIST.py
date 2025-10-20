import torch
from torchvision import transforms
from torch.utils.data import Dataset
from typing import Tuple, Dict, List
import numpy as np
import struct
from array import array

class MNISTDataset(Dataset):
    def __init__(self, images_filepath: str, labels_filepath: str, transform=None):
        super().__init__()
        self.transform = transform
        # Load images and labels from the given file paths one time only
        self.images, self.labels = self.read_images_labels(images_filepath, labels_filepath)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        label = self.labels[idx]
        if self.transform:
            img = self.transform(img)
        else:
            img = transforms.ToTensor()(img) # Convert to tensor 
        return img, label
    
    def read_images_labels(self, images_filepath, labels_filepath):
        with open(labels_filepath, 'rb') as file:
            magic, size = struct.unpack(">II", file.read(8))
            if magic != 2049:
                raise ValueError('Magic number mismatch, expected 2049, got {}'.format(magic))
            labels = array("B", file.read())        
        
        with open(images_filepath, 'rb') as file:
            magic, size, rows, cols = struct.unpack(">IIII", file.read(16))
            if magic != 2051:
                raise ValueError('Magic number mismatch, expected 2051, got {}'.format(magic))
            image_data = array("B", file.read())        
        images = []
        for i in range(size):
            img = np.array(image_data[i * rows * cols:(i + 1) * rows * cols], dtype=np.uint8)
            img = img.reshape(28, 28)
            images.append(img)            
        return images, labels