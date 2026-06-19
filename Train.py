import torch
import json
import torchvision
import torch.nn as nn
import torch.optim as optim
import os
import matplotlib.pyplot as plt
import math
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision.transforms.functional as F
from torchvision import transforms
import random
import keyboard
import time
import threading

if torch.cuda.is_available():
    print(f"{torch.cuda.get_device_name(0)} is available")
    device = torch.device("cuda")
else:
    print("Using CPU")
    device = torch.device("cpu")

print(f"Using device: {device}")
torch.backends.cudnn.benchmark = True

class CustomDataset(Dataset):
    def __init__(self, root, only_label=None):
        self.root = root
        self.only_label = only_label
        with open(os.path.join(root, "_annotations.coco.json"), "r", encoding="utf-8") as file:
            data = json.load(file)
        self.id_to_file = {img["id"]: img["file_name"] for img in data["images"]}
        self.data = {}
        for item in data["annotations"]:
            image_id = item["image_id"]
            if image_id not in self.data:
                self.data[image_id] = {"boxes": [], "labels": []}
            self.data[image_id]["boxes"].append(item["bbox"])
            self.data[image_id]["labels"].append(item["category_id"]-1)
        self.image_ids = list(self.data.keys())
    
    def __len__(self):
        return len(self.image_ids)
    
    def __getitem__(self, inx):
        image_id = self.image_ids[inx]
        file_name = self.id_to_file[image_id]

        image_path = os.path.join(self.root, file_name)
        image = Image.open(image_path).convert("RGB")
        transform = transforms.ToTensor()
        w, h = image.size

        boxes = torch.tensor(
            [[float(x) for x in box] for box in self.data[image_id]["boxes"]],
            dtype=torch.float32
        )

        scale_x = 416 / w
        scale_y = 416 / h

        boxes[:, 0] *= scale_x
        boxes[:, 1] *= scale_y
        boxes[:, 2] *= scale_x
        boxes[:, 3] *= scale_y
        # boxes[:, 2] = boxes[:, 0] + boxes[:, 2]  # x2 = x + width
        # boxes[:, 3] = boxes[:, 1] + boxes[:, 3]  # y2 = y + height
        boxes[:, 0::2] = boxes[:, 0::2].clamp(0, 416)
        boxes[:, 1::2] = boxes[:, 1::2].clamp(0, 416)
        keep = (boxes[:, 2] > 0) & (boxes[:, 3] > 0)
        boxes = boxes[keep]
        labels = torch.tensor(self.data[image_id]["labels"], dtype=torch.int64)
        if self.only_label is not None:
            keep = labels == self.only_label
            boxes = boxes[keep]
            labels = labels[keep]
        if len(boxes) == 0:
            return self.__getitem__((inx + 1) % len(self))

        MAX_BOXES = 15

        if len(boxes) > MAX_BOXES:
            indices = torch.randperm(len(boxes))[:MAX_BOXES]
            boxes = boxes[indices]
            labels = labels[indices]

        grid_size = 16

        grid = torch.zeros(grid_size, grid_size, 11) # shape + 6 types + trustablility (4 + 6 + 1)

        boxes[:, 0] /= 416 # normalize
        boxes[:, 1] /= 416
        boxes[:, 2] /= 416
        boxes[:, 3] /= 416
        boxes[:, 0] += boxes[:, 2]/2
        boxes[:, 1] += boxes[:, 3]/2
        i = 0
        for box in boxes:
            classes = torch.zeros(6, dtype=torch.float32)
            classes[labels[i]] = 1
            grid[math.floor(min(int(box[1]*grid_size), grid_size-1)), math.floor(min(int(box[0]*grid_size), grid_size-1)), :] =  torch.cat([box, torch.tensor([1.0]), classes], dim= 0)
            i += 1


        transform = transforms.Compose([
            transforms.Resize((416, 416)),
            transforms.ToTensor()
        ])

        image = transform(image)
        
        return image, grid


with open(r"datasetpath.txt", "r", encoding="utf-8") as file:
    data = file.read()

DATA_PATH = data
blob_dataset = CustomDataset(DATA_PATH, only_label=0)
dataset = CustomDataset(DATA_PATH, only_label=None)\

image, grid = dataset[1]
y, x = torch.nonzero(grid[:, :, 4])[0]
print(grid[y, x])
y, x = torch.nonzero(grid[:, :, 4])[1]
print(grid[y, x])