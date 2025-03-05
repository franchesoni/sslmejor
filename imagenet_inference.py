import os
import tqdm
from pathlib import Path
from torchvision import datasets, transforms
import torch
import timm
from collections import defaultdict
from torch.utils.data import Dataset, Subset, DataLoader
from torchvision.transforms import InterpolationMode


class ImageNetDataset(Dataset):
    def __init__(
        self, root="/home/franchesoni/data/imagenet1k", split="train", transform=None
    ):
        self.root = root
        self.split = split
        self.transform = transform or transforms.Compose(
            [
                transforms.Resize(
                    (518, 518),
                    interpolation=InterpolationMode.BICUBIC,
                    max_size=None,
                    antialias=True,
                ),
                transforms.CenterCrop(size=(518, 518)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        self.dataset = datasets.ImageFolder(
            Path(root) / split, transform=self.transform
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self.dataset[index]

    def split_dataset(self, train_ratio=0.1):
        class_indices = defaultdict(list)
        for idx, target in enumerate(self.dataset.targets):
            class_indices[target].append(idx)

        train_indices, val_indices = [], []
        for indices in class_indices.values():
            split_idx = int(len(indices) * train_ratio)
            train_indices.extend(indices[:split_idx])
            val_indices.extend(indices[split_idx:])

        return Subset(self.dataset, train_indices), Subset(self.dataset, val_indices)


def compute_embeddings(device="cuda:0", reset=True):
    if reset:
        os.remove("imagenet_embeddings.npy")
    batch_size = 32
    model = timm.create_model(
        "vit_large_patch14_reg4_dinov2.lvd142m", pretrained=True, num_classes=0
    )
    model.to(device)
    model = model.eval()

    ds = ImageNetDataset()
    dl = DataLoader(
        ds, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=32
    )
    with open("imagenet_embeddings.npy", "ab") as f:
        with torch.no_grad():
            for imgs, targets in tqdm.tqdm(dl, smoothing=1):
                imgs = imgs.to(device, non_blocking=True)
                feats = model.forward_features(imgs)[
                    :, 0, :
                ]  # global embedding, (N, D)
                feats.cpu().numpy().tofile(f)


if __name__ == "__main__":
    compute_embeddings()
