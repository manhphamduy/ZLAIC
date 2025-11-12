import os
import random
from PIL import Image
from torch.utils.data import Dataset
import glob

class SiameseTripletDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.class_dirs = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d)) and d != 'background']
        
        self.image_paths = {}
        for class_dir in self.class_dirs:
            self.image_paths[class_dir] = glob.glob(os.path.join(data_dir, class_dir, '*.jpg'))
            
        self.all_images = []
        for class_dir, paths in self.image_paths.items():
            for path in paths:
                self.all_images.append({'path': path, 'class': class_dir})

    def __len__(self):
        return len(self.all_images)

    def __getitem__(self, index):
        # 1. Lấy anchor
        anchor_info = self.all_images[index]
        anchor_path = anchor_info['path']
        anchor_class = anchor_info['class']
        
        # 2. Lấy positive (cùng class, khác ảnh)
        positive_list = self.image_paths[anchor_class]
        positive_path = anchor_path
        while positive_path == anchor_path:
            positive_path = random.choice(positive_list)
            if len(positive_list) == 1: break
            
        # 3. Lấy negative (khác class)
        negative_class = random.choice([c for c in self.class_dirs if c != anchor_class])
        negative_path = random.choice(self.image_paths[negative_class])
        
        # Mở ảnh
        anchor_img = Image.open(anchor_path).convert("RGB")
        positive_img = Image.open(positive_path).convert("RGB")
        negative_img = Image.open(negative_path).convert("RGB")
        
        if self.transform:
            anchor_img = self.transform(anchor_img)
            positive_img = self.transform(positive_img)
            negative_img = self.transform(negative_img)
            
        return anchor_img, positive_img, negative_img