import torch
import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class ViNaFood21Dataset(Dataset):
    # Add a flag to indicate if this is the training set
    def __init__(self, path, is_train=True):
        super().__init__()
        self.path = path
        self.is_train = is_train # Flag for augmentation
        self.label2idx = {}
        self.idx2label = {}
        self.data_paths = [] # Store paths first
        self._prepare_data_paths() # Get paths and labels

        # --- Define Transforms ---
        # Normalization values (example for ImageNet, adjust if needed)
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        if self.is_train:
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(224, scale=(0.8, 1.0)), # Crop randomly
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

        # Optionally preload data if RAM allows (keep your previous method if preferred)
        # self.data = self._load_all_data()

    # Separate function to prepare file paths and labels
    def _prepare_data_paths(self):
        label_id_counter = 0
        allowed_extensions = ('.jpg', '.jpeg', '.png')
        print(f"Scanning data paths in {self.path}...")
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

            for image_file in os.listdir(folder_path):
                if image_file.lower().endswith(allowed_extensions):
                    image_path = os.path.join(folder_path, image_file)
                    self.data_paths.append({
                        "image_path": image_path,
                        "label": current_label_id
                    })
        print(f"Found {len(self.data_paths)} image paths.")


    # Optional: Method to load all data into RAM (like your previous code)
    # def _load_all_data(self):
    #     loaded_data = []
    #     print(f"Loading {len(self.data_paths)} images into memory...")
    #     for item in tqdm(self.data_paths):
    #         try:
    #             image = Image.open(item["image_path"]).convert('RGB')
    #             image_tensor = self.transform(image) # Apply transform during loading
    #             loaded_data.append({
    #                 "image": image_tensor,
    #                 "label": item["label"]
    #             })
    #         except Exception as e:
    #             print(f"Warning: Could not load image {item['image_path']}. Error: {e}")
    #     print("Finished loading images.")
    #     return loaded_data


    def __len__(self):
        # If loading all data: return len(self.data)
        return len(self.data_paths) # Use paths length if lazy loading

    def __getitem__(self, index):
        # --- If loading everything in __init__ ---
        # return self.data[index]

        # --- If using LAZY LOADING (Recommended for larger datasets) ---
        item = self.data_paths[index]
        image_path = item["image_path"]
        label_id = item["label"]
        try:
            image = Image.open(image_path).convert('RGB')
            image_tensor = self.transform(image) # Apply transforms here
            return {
                "image": image_tensor,
                "label": label_id
            }
        except Exception as e:
            # Handle error: return None or skip, or return a placeholder
            print(f"Error loading image {image_path} in getitem: {e}")
            # Returning None might require adjustment in collate_fn or training loop
            # A simple fix is to try loading the next image recursively (use carefully)
            # return self.__getitem__((index + 1) % len(self))
            # Or raise error if needed
            raise e # Or return a dummy tensor of correct shape