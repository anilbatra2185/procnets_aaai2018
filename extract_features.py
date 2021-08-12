import json
import os

import glob
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset
from torchvision.models import resnet34


class YC2FrameDataset(Dataset):
    """You Cook2 Dataset for Framewise feature extraction."""

    def __init__(self, image_root=None, data_file=None):

        self.image_root = image_root
        data_file_path = os.path.join(image_root, data_file)
        with open(data_file_path, "r") as f:
            self.data = json.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        task = item["task"]
        vid = item["vid"]
        aug_number = "{:04d}".format(item["aug"])
        data_set = "{}_frames".format(item["set"])
        image_folder_path = os.path.join(
            self.image_root, data_set, task, vid, aug_number)
        image_tensors = []
        for image_path in glob.glob(image_folder_path + "/*.jpeg"):
            image = Image.open(image_path)
            image = image.resize(size=(256, 256), resample=Image.ANTIALIAS)
            image = np.asarray(image).astype("float32").transpose(2, 0, 1)
            image_tensors.append(torch.from_numpy(image))

        return {"images": torch.stack(image_tensors, dim=0), "save_path": image_folder_path}


if __name__ == '__main__':
    train_dataset = YC2FrameDataset(image_root='/disk/scratch_fast/s2004019/youcook2/raw_videos',
                                    data_file="validation_frames.json")
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=1, shuffle=True)
    r34 = resnet34(pretrained=True)
    modules = list(r34.children())[:-1]
    model = nn.Sequential(*modules)
    model.eval()
    model.cuda()
    for index, data in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
        images = data["images"].squeeze(0).cuda()
        N = images.size(0)
        with torch.no_grad():
            features = model(images)
            with open(os.path.join(data["save_path"][0], "features_resnet.npy"), 'wb') as f:
                np.save(f, features.view(N,-1).cpu().numpy())