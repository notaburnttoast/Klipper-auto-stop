import torch
import json
import os
import math
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

if torch.cuda.is_available():
    print(f"{torch.cuda.get_device_name(0)} is available")
    device = torch.device("cuda")
else:
    print("Using CPU")
    device = torch.device("cpu")

print(f"Using device: {device}")
torch.backends.cudnn.benchmark = True

def run(root):
    with open(os.path.join(root, "_annotations.coco.json"), "r", encoding="utf-8") as file:
        ndata = json.load(file)
    id_to_file = {img["id"]: img["file_name"] for img in ndata["images"]}
    data = {}
    for item in ndata["annotations"]:
        cimage_id = item["image_id"]
        if cimage_id not in data:
            data[cimage_id] = {"boxes": [], "labels": []}
        data[cimage_id]["boxes"].append(item["bbox"])
        data[cimage_id]["labels"].append(item["category_id"]-1)
    image_ids = list(data.keys())
    skipedfiles = [len(image_ids),[]]
    for image_id in image_ids:
        file_name = id_to_file[image_id]
        image_path = os.path.join(root, file_name)
        image = Image.open(image_path).convert("RGB")
        transform = transforms.ToTensor()
        w, h = image.size

        boxes = torch.tensor(
            [[float(x) for x in box] for box in data[image_id]["boxes"]],
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
        labels = torch.tensor(data[image_id]["labels"], dtype=torch.int64)
        if len(boxes) == 0:
            skipedfiles[1].append(image_id)
            continue
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
        for i, box in enumerate(boxes):
            classes = torch.zeros(6, dtype=torch.float32)
            classes[labels[i]] = 1
            cx = box[0]
            cy = box[1]
            w = math.sqrt(box[2])
            h = math.sqrt(box[3])
            cell_x = min(int(cx * grid_size), grid_size-1)
            cell_y = min(int(cy * grid_size), grid_size-1)
            offset_x = cx * grid_size - cell_x
            offset_y = cy * grid_size - cell_y
            grid[cell_y, cell_x, :] =  torch.cat([torch.tensor([offset_x, offset_y, w, h, 1.0], dtype=torch.float32), classes], dim= 0)

        transform = transforms.Compose([
            transforms.Resize((416, 416)),
            transforms.ToTensor()
        ])

        image = transform(image)
        torch.save(image, f"yolo dataset/image {image_id}.pt")
        torch.save(grid, f"yolo dataset/grid {image_id}.pt")
    with open(r"yolo dataset/missed images.txt", "w", encoding="utf-8") as file:
        file.write(str(skipedfiles))
with open(r"datasetpath.txt", "r", encoding="utf-8") as file:
    data = file.read()
DATA_PATH = data
run(DATA_PATH)