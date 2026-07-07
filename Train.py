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
        for y in range(size):
            for x in range(size):
                cell = grid[y,x]
                if not cell[4] < cutoff:
                    box1 = (cell[0].item()+x)/16, (cell[1].item()+y)/16, cell[2].item(), cell[3].item(), cell[4].item()
                else: 
                    box1 = (0,0,0,0,0)
                if not cell[9] < cutoff:
                    box2 = (cell[5].item()+x)/16, (cell[6].item()+y)/16, cell[7].item(), cell[8].item(), cell[9].item()
                else:
                    box2 = (0,0,0,0,0)
                boxes.append((*box1, *box2, torch.argmax(cell[5:]).item()))
        imageboxes.append(boxes)
        current +=1
    return imageboxes

type_to_string = {0: "blobs", 1: "cracks", 2: "over_extrusion", 3: "spaghetti", 4: "stringing", 5: "under_extrusion"}

def draw_boxes(grid, id, text):
    width = math.ceil(math.sqrt(len(grid)))
    height = math.ceil(len(grid) / width)
    fig, axs = plt.subplots(height, width)
    axs = np.array(axs).reshape((height, width))
    x = 0
    y = 0
    for boxes in grid:
        pltimage = transforms.ToPILImage()(boxes[0])
        boxes.pop(0)
        axs[y][x].imshow(pltimage)
        w, h = pltimage.size
        for Cx, Cy, W, H, c, Cx2, Cy2, W2, H2, c2, type in boxes:
            if W != 0 and H != 0:
                x1 = (Cx - W/2) * w
                y1 = (Cy - H/2) * h
                axs[y][x].add_patch(Rectangle((x1, y1), W * w, H * h, linewidth=2, edgecolor='r', facecolor='none'))
                #axs[y][x].text(x1, y1, f"type: {type_to_string[type]} probability: {c}", color='red', fontsize=6)
            if W2 != 0 and H2 != 0:
                x3 = (Cx2 - W2/2) * w
                y3 = (Cy2 - H2/2) * h
                axs[y][x].add_patch(Rectangle((x3, y3), W2 * w, H2 * h, linewidth=2, edgecolor='r', facecolor='none'))
                #axs[y][x].text(x3, y3, f"type: {type_to_string[type]} probability: {c2}", color='red', fontsize=6)
        x += 1 
        if x == width:
            x = 0
            y += 1
    fig.suptitle(f'Prediction {text}')
    plt.savefig(f"prediction/prediction {id}.png")
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

    BCE = torch.nn.BCEWithLogitsLoss()

    model.train()
    sample_batch = None
    for epoch in range(10):
        cstep = 0
        for images, targets in loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            output = model(images)
            boxloss = 0
            noobj_loss = 0
            if output[..., :4] == 1:
                boxloss +=  Fn.mse_loss(output[..., :2], targets[..., :2])
                boxloss +=  Fn.mse_loss(output[..., 2:4], targets[..., 2:4])
            else:
                noobj_loss += BCE(output[..., :2], torch.zeros(2))
                noobj_loss += BCE(output[..., 2:4], torch.zeros(2))
            if output[..., :5] == 1:
                boxloss +=  Fn.mse_loss(output[..., 5:7], targets[..., 5:7])
                boxloss +=  Fn.mse_loss(output[..., 7:9], targets[..., 7:9])
            else:
                noobj_loss += BCE(output[..., 5:7], torch.zeros(2))
                noobj_loss += BCE(output[..., 7:9], torch.zeros(2))
            objloss = Fn.binary_cross_entropy_with_logits(output[..., 4], targets[..., 4])
            objloss += Fn.binary_cross_entropy_with_logits(output[..., 9], targets[..., 9])
            class_loss = Fn.cross_entropy(torch.argmax(output[..., 10:16]), torch.argmax(targets[..., 10:16]))
            loss = 5*boxloss+objloss+0.2*noobj_loss+class_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            cstep += 1
            
            
            if sample_batch is None:
                sample_batch = (images.detach().cpu(), output.detach().cpu())
            print(f"epoch: {epoch+1}, percent: {100*((cstep*25)/5794)}, loss: {loss.item()}")
            if sample_batch is not None and cstep % 100 == 0:
                draw_boxes(get_boxes(sample_batch[1], sample_batch[0], cutoff=0.0), f"step {cstep}", text="cut off = 0.0")
                sample_batch = None
        if sample_batch is not None:
            draw_boxes(get_boxes(sample_batch[1], sample_batch[0], cutoff=0.2), f"epoch {epoch+1}", text="cut off = 0.2")
            sample_batch = None

if __name__ == "__main__":
    main()