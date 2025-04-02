import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import pandas as pd

class CancerDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.data_frame = pd.read_csv(csv_file)
        self.data_frame['label'] = self.data_frame['label'].astype(str)
        self.data_frame['id'] = self.data_frame['id'].apply(lambda x: x + '.tif' if not x.endswith('.tif') else x)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = self.data_frame.iloc[idx]['id']
        image = Image.open(f'{self.root_dir}/{img_name}')
        label = int(self.data_frame.iloc[idx]['label'])

        if self.transform:
            image = self.transform(image)
        return image, torch.tensor([label], dtype=torch.float32)
# 数据预处理
train_transform = transforms.Compose([
    transforms.Resize((96, 96)),
    # transforms.RandomRotation(10),
    # transforms.RandomHorizontalFlip(),
    # transforms.RandomVerticalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4049, 0.0925, 0.3929], std=[0.4778, 0.5642, 0.4325])
])

val_transform = transforms.Compose([
    transforms.Resize((96, 96)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4049, 0.0925, 0.3929], std=[0.4778, 0.5642, 0.4325])
])

