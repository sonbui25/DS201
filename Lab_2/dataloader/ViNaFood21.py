import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from tqdm.auto import tqdm
import os

class ViNaFood21Dataset(Dataset):
    def __init__(self, path, is_train=True):
        super().__init__()
        self.path = path
        self.is_train = is_train
        self.label2idx = {}
        self.idx2label = {}

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                         std=[0.229, 0.224, 0.225])
        if self.is_train:
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.RandomRotation(15),
                transforms.ToTensor(),
                normalize,
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                normalize,
            ])

        # Chỉ lưu đường dẫn ảnh và nhãn
        self.image_paths, self.labels = self._scan_image_paths()

    def _scan_image_paths(self):
        image_paths, labels = [], []
        label_id = 0
        allowed_extensions = ('.jpg', '.jpeg', '.png')

        print(f"Scanning image paths from {self.path}...")
        for folder in sorted(os.listdir(self.path)):
            folder_path = os.path.join(self.path, folder)
            if not os.path.isdir(folder_path):
                continue

            # mapping nhãn
            if folder not in self.label2idx:
                self.label2idx[folder] = label_id
                self.idx2label[label_id] = folder
                label_id += 1

            current_label_id = self.label2idx[folder]

            for file_name in tqdm(os.listdir(folder_path), desc=f"Scanning {folder}", leave=False):
                if file_name.lower().endswith(allowed_extensions):
                    image_paths.append(os.path.join(folder_path, file_name))
                    labels.append(current_label_id)

        print(f"Found {len(image_paths)} images.")
        return image_paths, labels

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        img_path = self.image_paths[index]
        label = self.labels[index]

        # Mở ảnh khi cần (lazy load)
        image = Image.open(img_path)
        if image.mode != "RGB":
            image = image.convert("RGB")

        image_tensor = self.transform(image)
        return {"image": image_tensor, "label": label}
