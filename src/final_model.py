from src.training.models.vgg19 import VGG19
import os
import pickle
import yaml
import torchvision.transforms as tt
import torch
from PIL import Image


class FinalModel:
    def __init__(self):
        with open("config.yml") as f:
            self._config = yaml.safe_load(f)
        path = os.path.join("training/models/saved_models", self._config["model_to_use"] + ".model")
        with open(path, "rb") as f:
            self.model = pickle.load(f)

    def preprocess_image(self, image):
        transforms = self.model.transforms
        transformed_image = transforms(image)
        final_image = torch.unsqueeze(transformed_image, 0)
        return final_image


    def predict(self, image):
        transformed_image = self.preprocess_image(image)
        out = self.model(transformed_image)
        return out


if __name__ == "__main__":
    m = FinalModel()
    image = Image.open(r"C:\Users\bierl\DataScience\DeepLearning\LemonQualit\Data\bad_quality\bad_quality_10.jpg")
    print(m.predict(image))



