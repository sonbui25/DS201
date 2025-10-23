import torch
import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm.auto import tqdm # Import tqdm for progress bar

class ViNaFood21Dataset(Dataset):
    # Add a flag to indicate if this is the training set
    def __init__(self, path, is_train=True):
        super().__init__()
        self.path = path
        self.is_train = is_train # Flag for augmentation
        self.label2idx = {}
        self.idx2label = {}
        # self.data_paths = [] # No longer storing just paths
        # self._prepare_data_paths() # Combine path prep and loading

        # --- Define Transforms ---
        # Normalization values (example for ImageNet, adjust if needed)
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        # Define transforms based on train/test flag *before* loading
        if self.is_train:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)), # Crop randomly
                transforms.RandomHorizontalFlip(), # Flip horizontally
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1), # Adjust colors
                transforms.RandomRotation(15), # Rotate slightly
                transforms.ToTensor(), # Convert to tensor
                normalize # Normalize
            ])
        else: # For validation/testing
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)), # Just resize
                transforms.ToTensor(), # Convert to tensor
                normalize # Normalize
            ])

        # --- Load all data into RAM ---
        self.images, self.labels = self.read_images_labels() # Load everything here


    # Combined function to get paths, load images, apply transforms, and store
    def read_images_labels(self):
        images = []
        labels = []
        label_id_counter = 0
        allowed_extensions = ('.jpg', '.jpeg', '.png')
        print(f"Scanning and loading images from {self.path} into memory...")

        # Prepare paths and load data in one go
        for folder in os.listdir(self.path):
            folder_path = os.path.join(self.path, folder)
            if not os.path.isdir(folder_path):
                continue

            label = folder
            if label not in self.label2idx:
                self.label2idx[label] = label_id_counter
                self.idx2label[label_id_counter] = label
                label_id_counter += 1

            current_label_id = self.label2idx[label]

            # Use tqdm here for iterating through files in a folder
            file_list = [f for f in os.listdir(folder_path) if f.lower().endswith(allowed_extensions) and 'gray' not in f.lower()]
            for image_file in tqdm(file_list, desc=f"Loading {label}", leave=False):
                    image_path = os.path.join(folder_path, image_file)
                    try:
                        image = Image.open(image_path).convert('RGB')
                        # Apply the appropriate transform (train or test)
                        image_tensor = self.transform(image)
                        images.append(image_tensor)
                        labels.append(current_label_id)
                    except Exception as e:
                        print(f"Warning: Could not load image {image_path}. Error: {e}")

        print(f"Finished loading {len(images)} images into memory.")
        return images, labels

    def __len__(self):
        # Length is now based on the loaded data
        return len(self.images)

    def __getitem__(self, index):
        # Simply return the pre-loaded data dictionary
        return {
            "image": self.images[index],
            "label": self.labels[index]
        }