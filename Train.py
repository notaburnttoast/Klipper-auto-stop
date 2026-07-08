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


def get_boxes(grids, images, cutoff, size=16):
    imageboxes = []
    current = 0
    for grid in grids:
        boxes = [images[current]]
        box_count = 0
        for y in range(size):
            for x in range(size):
                cell = grid[y,x]
                conf = 1 / (1 + np.exp(-cell[4].item()))
                if conf >= cutoff:
                    # Apply sigmoid to normalize all values to [0, 1]
                    cx = 1 / (1 + np.exp(-cell[0].item()))
                    cy = 1 / (1 + np.exp(-cell[1].item()))
                    w = 1 / (1 + np.exp(-cell[2].item()))
                    h = 1 / (1 + np.exp(-cell[3].item()))
                    conf = 1 / (1 + np.exp(-cell[4].item()))
                    box1 = (cx + x) / 16, (cy + y) / 16, w, h, conf
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
                    box2 = (cx2 + x) / 16, (cy2 + y) / 16, w2, h2, conf2
                    box_count += 1
                else:
                    box2 = (0,0,0,0,0)
                boxes.append((*box1, *box2, torch.argmax(cell[10:]).item()))
        imageboxes.append(boxes)
        if box_count > 0:
            print(f"get_boxes: image {current} has {box_count} boxes")
        current +=1
    return imageboxes

type_to_string = {0: "blobs", 1: "cracks", 2: "over_extrusion", 3: "spaghetti", 4: "stringing", 5: "under_extrusion"}

def draw_boxes(grid, id, text):
    print(f"draw_boxes called with {len(grid)} images, id={id}, text={text}")
    width = math.ceil(math.sqrt(len(grid)))
    height = math.ceil(len(grid) / width)
    fig, axs = plt.subplots(height, width, squeeze=False, figsize=(12, 12))
    axs = np.array(axs).reshape((height, width))
    x = 0
    y = 0
    box_count = 0
    for boxes in grid:
        pltimage = transforms.ToPILImage()(boxes[0])
        current = boxes[1:]
        axs[y][x].imshow(pltimage)
        axs[y][x].set_aspect('auto')
        w, h = pltimage.size
        for Cx, Cy, W, H, c, Cx2, Cy2, W2, H2, c2, type in current:
            if W != 0 and H != 0:
                x1 = (Cx - W/2) * w
                y1 = (Cy - H/2) * h
                rect = Rectangle((x1, y1), W * w, H * h, linewidth=2, edgecolor='r', facecolor='none')
                axs[y][x].add_patch(rect)
                #axs[y][x].text(x1, y1, f"type: {type_to_string[type]} probability: {c}", color='red', fontsize=6)
            if W2 != 0 and H2 != 0:
                x3 = (Cx2 - W2/2) * w
                y3 = (Cy2 - H2/2) * h
                rect2 = Rectangle((x3, y3), W2 * w, H2 * h, linewidth=2, edgecolor='b', facecolor='none')
                axs[y][x].add_patch(rect2)
                box_count += 1
                #axs[y][x].text(x3, y3, f"type: {type_to_string[type]} probability: {c2}", color='red', fontsize=6)
        axs[y][x].set_xlim(0, w)
        axs[y][x].set_ylim(h, 0)
        x += 1 
        if x == width:
            x = 0
            y += 1
    fig.suptitle(f'Prediction {text}')
    plt.savefig(f"predictions/prediction {id}.png", dpi=100, bbox_inches='tight')
    plt.close()


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
        batch_size=25,
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

    model.train()
    sample_batch = None
    for epoch in range(10):
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
            noobj_loss = torch.tensor([0], dtype=torch.float32).to(device, non_blocking=True)
            if objects1.any():
                boxloss =  Fn.mse_loss(torch.sigmoid(output[..., :2][objects1]), targets[..., :2][objects1])
                boxloss +=  Fn.mse_loss(torch.exp(output[..., 2:4][objects1]), targets[..., 2:4][objects1])
            if noobjects1.any():
                noobj_loss = Fn.mse_loss(torch.sigmoid(output[..., :2][noobjects1]), torch.zeros_like(output[..., :2][noobjects1]))
                noobj_loss += Fn.mse_loss(torch.exp(output[..., 2:4][noobjects1]), torch.zeros_like(output[..., 2:4][noobjects1]))
            if objects2.any():
                boxloss =  Fn.mse_loss(torch.sigmoid(output[..., 5:7][objects2]), targets[..., 5:7][objects2])
                boxloss +=  Fn.mse_loss(torch.exp(output[..., 7:9][objects2]), targets[..., 7:9][objects2])
            if noobjects2.any():
                noobj_loss = Fn.mse_loss(torch.sigmoid(output[..., 5:7][noobjects2]), torch.zeros_like(output[..., 5:7][noobjects2]))
                noobj_loss += Fn.mse_loss(torch.exp(output[..., 7:9][noobjects2]), torch.zeros_like(output[..., 7:9][noobjects2]))
            objloss = Fn.binary_cross_entropy_with_logits(output[..., 4], targets[..., 4])
            objloss += Fn.binary_cross_entropy_with_logits(output[..., 9], targets[..., 9])
            class_loss = Fn.cross_entropy(output[..., 10:16], targets[..., 10:16])
            loss = 10*boxloss+objloss+0.2*noobj_loss+class_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            cstep += 1
            sample_batch = (images.detach().cpu(), output.detach().cpu())
            print(f"epoch: {epoch+1}, percent: {100*((cstep*25)/5794)}, loss: {loss.item()}")
            if cstep % 100 == 0:
                draw_boxes(get_boxes(sample_batch[1], sample_batch[0], cutoff=0.0), f"step {cstep}", text="cut off = 0.0")
                sample_batch = None
        if sample_batch is not None:
            draw_boxes(get_boxes(sample_batch[1], sample_batch[0], cutoff=0.2), f"epoch {epoch+1}", text="cut off = 0.2")
            sample_batch = None

if __name__ == "__main__":
    main()