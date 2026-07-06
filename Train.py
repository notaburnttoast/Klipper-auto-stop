import torch
import json
import torchvision
import torch.nn as nn
import torch.optim as optim
import os
import matplotlib.pyplot as plt
import math
from PIL import Image
from PIL import ImageDraw
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision.transforms.functional as F
import torch.nn.functional as Fn
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
        with open(r"yolo dataset/missed images.txt", "r", encoding="utf-8") as file:
            data = eval(file.read())
        self.length, self.missing = data
    def __len__(self):
        return self.length
    
    def __getitem__(self, inx):
        while inx in self.missing:
            inx = (inx + 1) % self.length
        image = torch.load(f"yolo dataset/image {inx}.pt")
        grid = torch.load(f"yolo dataset/grid {inx}.pt")
        return image, grid


def get_boxes(grid, size=16):
    boxes = []
    for y in range(size):
        for x in range(size):
            cell = grid[y,x]
            if not cell[4] < 0.5:
                box1 = (cell[0].item()+x)/16, (cell[1].item()+y)/16, cell[2].item(), cell[3].item(), cell[4].item()
            else: 
                box1 = (0,0,0,0,0)
            if not cell[9] < 0.5:
                box2 = (cell[5].item()+x)/16, (cell[6].item()+y)/16, cell[7].item(), cell[8].item(), cell[9].item()
            else:
                box2 = (0,0,0,0,0)
            boxes.append((box1, box2, torch.argmax(cell[5:]).item()))
    return boxes

type_to_string = {0: "blobs", 1: "cracks", 2: "over_extrusion", 3: "spaghetti", 4: "stringing", 5: "under_extrusion"}

def draw_boxes(tensor_image, boxes):
    pltimage = transforms.ToPILImage()(tensor_image)
    draw = ImageDraw.Draw(pltimage)
    w, h = pltimage.size
    for Cx, Cy, W, H, c, Cx2, Cy2, W2, H2, c2, type in boxes:
        if W != 0 and H != 0:
            x1 = (Cx - W/2) * w
            y1 = (Cy - H/2) * h

            x2 = (Cx + W/2) * w
            y2 = (Cy + H/2) * h
            
            draw.rectangle(
                [x1, y1, x2, y2],
                outline="red",
                width=2
            )
            draw.text((x1, y1), f"type: {type_to_string[type]} probability: {c}")
        if W2 != 0 and H2 != 0:
            x3 = (Cx2 - W2/2) * w
            y3 = (Cy2 - H2/2) * h

            x4 = (Cx2 + W2/2) * w
            y4 = (Cy2 + H2/2) * h

            draw.rectangle(
                [x3, y3, x4, y4],
                outline="red",
                width=2
            )
            draw.text((x3, y3), f"type: {type_to_string[type]} probability: {c2}")
    pltimage.show()

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2), #416 208

            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2), #208 104

            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2), #104 52
        )
        self.l1=nn.Conv2d(128, 16, 1)

    def forward(self, x):
        x = self.features(x)
        x = self.l1(x)
        x = torch.nn.functional.interpolate(
            x, size=(16, 16), mode="bilinear", align_corners=False
        )

        x = x.permute(0, 2, 3, 1)  # (B, 16, 16, 16 type)
        return x

def main():
    with open(r"datasetpath.txt", "r", encoding="utf-8") as file:
        data = file.read()

    DATA_PATH = data
    # blob_dataset = CustomDataset(DATA_PATH, only_label=0)
    dataset = CustomDataset(DATA_PATH, only_label=None)


    loader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2
    )

    model = Model().to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr= 0.0005
    )

    criterion = torch.nn.MSELoss()

    model.train()

    for epoch in range(10):
        cstep = 0
        for images, targets in loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            output = model(images)
            loss = criterion(output, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step
            cstep += 1
            print(f"epoch: {epoch}, percent: {100*((cstep*32)/5794)} loss: {loss.item()}")

if __name__ == "__main__":
    main()