import torch
import os
from PIL import Image # Use PIL to read images
from torch.utils.data import Dataset
from torchvision import transforms # Import transforms

class ViNaFood21Dataset(Dataset):
    def __init__(self, path):
        super().__init__()
        self.path = path
        self.label2idx = {}
        self.idx2label = {}
        
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
        
        self.data = [] 
        label_id_counter = 0
        for folder in os.listdir(self.path):
            label = folder
            if label not in self.label2idx:
                self.label2idx[label] = label_id_counter
                self.idx2label[label_id_counter] = label
                label_id_counter += 1
            
            current_label_id = self.label2idx[label]
            folder_path = os.path.join(self.path, folder)
            
            for image_file in os.listdir(folder_path):
                image_path = os.path.join(folder_path, image_file)
                try:
                    # Read image
                    image = Image.open(image_path).convert('RGB')
                    # Apply transforms (resize, to tensor)
                    image_tensor = self.transform(image)
                    # Store image tensor and label_id in the list
                    self.data.append({
                        "image": image_tensor,
                        "label": current_label_id 
                    })
                except Exception as e:
                    print(f"Warning: Could not load image {image_path}. Error: {e}")
                    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # Just return the preloaded data
        return self.data[index]