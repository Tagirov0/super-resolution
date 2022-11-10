import os
import torch
from PIL import Image
import torchvision.transforms as tt
from torch.utils.data import Dataset

class MyImageFolder(Dataset):
    def __init__(self, root_dir, resolution):
        super(MyImageFolder, self).__init__()
        self.data = []
        self.root_dir = root_dir
        self.high_res = resolution
        self.low_res = resolution // 4

        self.highres_transform = tt.Compose(
            [
                tt.Resize((self.high_res, self.high_res)),
                tt.ToTensor(),
                tt.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ]
        )

        self.lowres_transform = tt.Compose(
            [
                tt.Resize((self.low_res, self.low_res), tt.InterpolationMode.BICUBIC),
                tt.ToTensor(),
                tt.Normalize(mean=[0, 0, 0], std=[1, 1, 1])
            ]
        )

        self.data = os.listdir(root_dir)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img_file = self.data[index]

        image = Image.open(os.path.join(self.root_dir, img_file))
        high_res = self.highres_transform(image)
        low_res = self.lowres_transform(image)
        return low_res, high_res