import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision.transforms.functional as F
import torch.nn.functional as Fn
from torchvision import transforms
import math
import numpy as np

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


def get_boxes(grids, images, cutoff, size=32):
    imageboxes = []
    current = 0
    for grid in grids:
        boxes = [images[current]]
        box_count = 0
        conf_values = []
        for y in range(size):
            for x in range(size):
                cell = grid[y,x]
                conf = 1 / (1 + np.exp(-cell[4].item()))
                conf_values.append(conf)
                if conf >= cutoff:
                    # Apply sigmoid to normalize all values to [0, 1]
                    cx = 1 / (1 + np.exp(-cell[0].item()))
                    cy = 1 / (1 + np.exp(-cell[1].item()))
                    w = 1 / (1 + np.exp(-cell[2].item()))
                    h = 1 / (1 + np.exp(-cell[3].item()))
                    conf = 1 / (1 + np.exp(-cell[4].item()))
                    box1 = (cx + x) / 32, (cy + y) / 32, w, h, conf
                    box_count += 1
                else: 
                    box1 = (0,0,0,0,0)
                conf = 1 / (1 + np.exp(-cell[9].item()))
                if conf >= cutoff:
                    # Apply sigmoid to normalize all values to [0, 1]
                    cx2 = 1 / (1 + np.exp(-cell[5].item()))
                    cy2 = 1 / (1 + np.exp(-cell[6].item()))
                    w2 = 1 / (1 + np.exp(-cell[7].item()))
                    h2 = 1 / (1 + np.exp(-cell[8].item()))
                    conf2 = 1 / (1 + np.exp(-cell[9].item()))
                    box2 = (cx2 + x) / 32, (cy2 + y) / 32, w2, h2, conf2
                    box_count += 1
                else:
                    box2 = (0,0,0,0,0)
                boxes.append((*box1, *box2, torch.argmax(cell[10:]).item()))
        imageboxes.append(boxes)
        current +=1
    return imageboxes

type_to_string = {0: "blobs", 1: "cracks", 2: "over_extrusion", 3: "spaghetti", 4: "stringing", 5: "under_extrusion"}

def draw_boxes(grid, id, text):
    print(f"draw_boxes called with {len(grid)} images, id={id}, text={text}")
    width = math.ceil(math.sqrt(len(grid)))
    height = math.ceil(len(grid) / width)
    print(f"Grid layout: {width}x{height} = {width*height} subplots for {len(grid)} images")
    fig, axs = plt.subplots(height, width, squeeze=False, figsize=(12, 12))
    axs = np.array(axs).reshape((height, width))
    x = 0
    y = 0
    box_count = 0
    for idx, boxes in enumerate(grid):
        pltimage = transforms.ToPILImage()(boxes[0])
        current = boxes[1:]
        axs[y][x].imshow(pltimage)
        axs[y][x].set_aspect('auto')
        w, h = pltimage.size
        image_box_count = 0
        for box_idx, box_data in enumerate(current):
            Cx, Cy, W, H, c, Cx2, Cy2, W2, H2, c2, type_id = box_data
            if W != 0 and H != 0:
                x1 = (Cx - W/2) * w
                y1 = (Cy - H/2) * h
                rect = Rectangle((x1, y1), W * w, H * h, linewidth=2, edgecolor='r', facecolor='none')
                axs[y][x].add_patch(rect)
                image_box_count += 1
                box_count += 1
                #axs[y][x].text(x1, y1, f"type: {type_to_string[type_id]} probability: {c}", color='red', fontsize=6)
            if W2 != 0 and H2 != 0:
                x3 = (Cx2 - W2/2) * w
                y3 = (Cy2 - H2/2) * h
                rect2 = Rectangle((x3, y3), W2 * w, H2 * h, linewidth=2, edgecolor='b', facecolor='none')
                axs[y][x].add_patch(rect2)
                image_box_count += 1
                box_count += 1
                #axs[y][x].text(x3, y3, f"type: {type_to_string[type_id]} probability: {c2}", color='red', fontsize=6)
        axs[y][x].set_xlim(0, w)
        axs[y][x].set_ylim(h, 0)
        x += 1 
        if x == width:
            x = 0
            y += 1
    # Hide any unused subplots
    for i in range(len(grid), width * height):
        ax_idx = i
        y_idx = ax_idx // width
        x_idx = ax_idx % width
        axs[y_idx][x_idx].axis('off')
    print(f"Total boxes drawn: {box_count}")
    fig.suptitle(f'Prediction {text}')
    plt.savefig(f"predictions/prediction {id}.png", dpi=100, bbox_inches='tight')
    print(f"Saved: predictions/prediction {id}.png")
    plt.close()


class Model(nn.Module):
    def __init__(self, layers=3, startchannels=32):
        super().__init__()
        conv2layers = []
        inchannels = 3
        for i in range(layers):
            outchannels = startchannels * (2**i)
            conv2layers.append(nn.Conv2d(inchannels, outchannels, 3, padding=1))
            conv2layers.append(nn.ReLU())
            conv2layers.append(nn.MaxPool2d(2))
            inchannels=outchannels
        self.features = nn.Sequential(*conv2layers)

        self.l1=nn.Conv2d(inchannels, 16, 1)

    def forward(self, x):
        x = self.features(x)
        x = self.l1(x)
        x = torch.nn.functional.interpolate(
            x, size=(32, 32), mode="bilinear", align_corners=False
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
        batch_size=49,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
        drop_last=True
    )

    model = Model(layers=4).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr= 0.0005
    )

    model.train()
    sample_batch = None
    for epoch in range(50):
        cstep = 0
        for images, targets in loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            output = model(images)
            objects1 = targets[:,:,:, 4] == 1
            noobjects1 = targets[:,:,:, 4] != 1
            objects2 = targets[:,:,:, 9] == 1
            noobjects2 = targets[:,:,:, 9] != 1
            boxloss = torch.tensor([0], dtype=torch.float32).to(device, non_blocking=True)
            noboxloss = torch.tensor([0], dtype=torch.float32).to(device, non_blocking=True)
            boxobjectiveness = torch.tensor([0], dtype=torch.float32).to(device, non_blocking=True)
            noboxnoobjectiveness = torch.tensor([0], dtype=torch.float32).to(device, non_blocking=True)
            if objects1.any():
                boxloss +=  Fn.mse_loss(torch.sigmoid(output[..., :2][objects1]), targets[..., :2][objects1])
                boxloss +=  Fn.mse_loss(torch.sigmoid(output[..., 2:4][objects1]), targets[..., 2:4][objects1])
                boxobjectiveness += Fn.binary_cross_entropy_with_logits(output[..., 4][objects1], targets[..., 4][objects1])
            if noobjects1.any():
                noboxloss += Fn.mse_loss(torch.sigmoid(output[..., :2][noobjects1]), torch.zeros_like(output[..., :2][noobjects1]))
                noboxloss += Fn.mse_loss(torch.sigmoid(output[..., 2:4][noobjects1]), torch.zeros_like(output[..., 2:4][noobjects1]))
                noboxnoobjectiveness += Fn.binary_cross_entropy_with_logits(output[..., 4][noobjects1], targets[..., 4][noobjects1])
            if objects2.any():
                boxloss +=  Fn.mse_loss(torch.sigmoid(output[..., 5:7][objects2]), targets[..., 5:7][objects2])
                boxloss +=  Fn.mse_loss(torch.sigmoid(output[..., 7:9][objects2]), targets[..., 7:9][objects2])
                boxobjectiveness += Fn.binary_cross_entropy_with_logits(output[..., 9][objects2], targets[..., 9][objects2])
            if noobjects2.any():
                noboxloss += Fn.mse_loss(torch.sigmoid(output[..., 5:7][noobjects2]), torch.zeros_like(output[..., 5:7][noobjects2]))
                noboxloss += Fn.mse_loss(torch.sigmoid(output[..., 7:9][noobjects2]), torch.zeros_like(output[..., 7:9][noobjects2]))
                noboxnoobjectiveness += Fn.binary_cross_entropy_with_logits(output[..., 9][noobjects2], targets[..., 9][noobjects2])
            class_loss = Fn.cross_entropy(output[..., 10:16], targets[..., 10:16])
            loss = 20*boxloss+0.02*noboxloss+1*boxobjectiveness+2*noboxnoobjectiveness+0.7*class_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            cstep += 1
            sample_batch = (images.detach().cpu(), output.detach().cpu())
            print(f"epoch: {epoch}, percent: {100*((cstep*49)/5794)}, loss: {loss.item()}")
            if cstep % 55 == 0:
                draw_boxes(get_boxes(sample_batch[1], sample_batch[0], cutoff=0.02), f"step {cstep}", text="cut off = 0.02")
                sample_batch = None
        if sample_batch is not None:
            draw_boxes(get_boxes(sample_batch[1], sample_batch[0], cutoff=0.50), f"epoch {epoch+1}", text="cut off = 0.5")
            sample_batch = None

if __name__ == "__main__":
    main()