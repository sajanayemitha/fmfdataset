import random
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt


class FaceAugmenter:

    def __init__(self):

        self.to_tensor = transforms.ToTensor()

        self.augmentations = {

            "Horizontal Flip":
                transforms.RandomHorizontalFlip(p=1.0),

            "Rotation ±15°":
                transforms.RandomRotation(
                    degrees=15
                ),

            "Brightness":
                transforms.ColorJitter(
                    brightness=0.3
                ),

            "Contrast":
                transforms.ColorJitter(
                    contrast=0.3
                ),

            "Gaussian Blur":
                transforms.GaussianBlur(
                    kernel_size=3
                ),

            "Perspective":
                transforms.RandomPerspective(
                    distortion_scale=0.3,
                    p=1.0
                )
        }


    def random_augmentations(
        self,
        image_path,
        n=6
    ):

        image = Image.open(
            image_path
        ).convert("RGB")

        results = []

        augmentation_names = list(
            self.augmentations.keys()
        )

        for _ in range(n):

            name = random.choice(
                augmentation_names
            )

            augmentation = self.augmentations[name]

            augmented = augmentation(image)

            # Convert to tensor
            tensor = self.to_tensor(augmented)

            results.append(
                (name, tensor)
            )

        return results


    def visualize(
        self,
        image_path,
        save_path="augmentation_grid.png"
    ):

        image = Image.open(
            image_path
        ).convert("RGB")

        results = list(
            self.augmentations.items()
        )

        plt.figure(
            figsize=(12, 8)
        )


        # Original

        plt.subplot(2, 4, 1)

        plt.imshow(image)

        plt.title("Original")

        plt.axis("off")


        # Six augmentations

        for i, (name, augmentation) in enumerate(
            results,
            start=2
        ):

            augmented = augmentation(image)

            plt.subplot(2, 4, i)

            plt.imshow(augmented)

            plt.title(name)

            plt.axis("off")


        plt.tight_layout()

        plt.savefig(
            save_path,
            dpi=200,
            bbox_inches="tight"
        )

        plt.show()


# --------------------------------
# Test
# --------------------------------

if __name__ == "__main__":

    augmenter = FaceAugmenter()

    image_path = next(
        Path("raw/sajana").glob("*.jpg")
    )

    print(
        f"Using image: {image_path}"
    )

    augmenter.visualize(
        image_path
    )