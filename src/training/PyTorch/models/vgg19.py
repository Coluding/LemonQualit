import torch
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder
import torchvision.transforms as tt
import torch.nn as nn
import yaml
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torchvision
import torchinfo


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        with open("../../../config.yml") as y:
            self._config_file = yaml.safe_load(y)

        self.transforms = None
        self.model = None

        self._init_backbone_model()
        self._load_images()

    def _load_images(self):
        path = self._config_file["image_path"]
        images = ImageFolder(path, transform=self.transforms)
        # Augmenting !!
        split_ratio = 0.8
        train_images, test_images = random_split(images, [round(split_ratio * len(images)),
                                                          round((1 - split_ratio) * len(images))])

        train_loader = DataLoader(train_images, batch_size=32, num_workers=2)
        val_loader = DataLoader(test_images, batch_size=32, num_workers=2)

        return train_loader, val_loader

    def _init_backbone_model(self):
        weights = torchvision.models.VGG19_Weights.DEFAULT
        self.model = torchvision.models.vgg19(weights)
        self.transforms = weights.transforms()


if __name__ == "__main__":
    m = Model()