from src.training.models.vgg19 import VGG19
from final_model import FinalModel
import os
from PIL import Image
import torch


def main():
    os.chdir("training/models")
    m = FinalModel()
    image = Image.open(
        r"C:\Users\bierl\DataScience\DeepLearning\LemonQualit\Data\empty_background\empty_background_445.jpg ")
    out = m.predict_raw_image(image)
    ind = torch.argmax(out)
    print(m.model.classes[ind])
    for batch in m.model.val_loader:
        image, label = batch
        print(image.shape)
        out = m.model(image)
        print(m.model.accuracy(out, label))


if __name__ == "__main__":
    main()