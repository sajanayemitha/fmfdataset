from pathlib import Path
import json
import random

import torch
from PIL import Image

from torchvision import transforms

from sklearn.model_selection import train_test_split


# ==========================================
# Configuration
# ==========================================

AUGMENTATION_FACTOR = 5

IMAGE_SIZE = (224, 224)

RANDOM_SEED = 42


# ==========================================
# Load configuration
# ==========================================

with open("config.json", "r") as f:

    config = json.load(f)

class_names = config["class_names"]


# ==========================================
# Transformations
# ==========================================

resize = transforms.Resize(
    IMAGE_SIZE
)

to_tensor = transforms.ToTensor()


augmentations = transforms.Compose([

    transforms.RandomHorizontalFlip(
        p=0.5
    ),

    transforms.RandomRotation(
        degrees=15
    ),

    transforms.ColorJitter(
        brightness=0.3,
        contrast=0.3
    ),

    transforms.GaussianBlur(
        kernel_size=3
    ),

    transforms.RandomPerspective(
        distortion_scale=0.3,
        p=0.5
    )
])


# ==========================================
# Collect images
# ==========================================

def collect_images():

    paths = []
    labels = []


    for label, class_name in enumerate(
        class_names
    ):

        folder = Path("raw") / class_name

        image_files = (
            list(folder.glob("*.jpg")) +
            list(folder.glob("*.jpeg")) +
            list(folder.glob("*.png"))
        )


        for path in image_files:

            paths.append(str(path))

            labels.append(label)


    return paths, labels


# ==========================================
# Split dataset
# ==========================================

def split_data():

    paths, labels = collect_images()


    train_paths, val_paths, train_labels, val_labels = (
        train_test_split(
            paths,
            labels,
            test_size=0.20,
            random_state=RANDOM_SEED,
            stratify=labels
        )
    )


    return (
        train_paths,
        train_labels,
        val_paths,
        val_labels
    )


# ==========================================
# Load image
# ==========================================

def load_image(path):

    image = Image.open(
        path
    ).convert("RGB")

    return image


# ==========================================
# Build training dataset
# ==========================================

def build_training_dataset(
    paths,
    labels
):

    images = []
    final_labels = []
    final_paths = []


    for path, label in zip(
        paths,
        labels
    ):

        image = load_image(path)


        for _ in range(
            AUGMENTATION_FACTOR
        ):

            augmented = augmentations(
                image
            )

            augmented = resize(
                augmented
            )

            tensor = to_tensor(
                augmented
            )


            images.append(
                tensor
            )

            final_labels.append(
                label
            )

            final_paths.append(
                path
            )


    images = torch.stack(
        images
    )

    labels_tensor = torch.tensor(
        final_labels,
        dtype=torch.long
    )


    return (
        images,
        labels_tensor,
        final_paths
    )


# ==========================================
# Build validation dataset
# ==========================================

def build_validation_dataset(
    paths,
    labels
):

    images = []
    final_labels = []
    final_paths = []


    for path, label in zip(
        paths,
        labels
    ):

        image = load_image(path)

        image = resize(
            image
        )

        tensor = to_tensor(
            image
        )


        images.append(
            tensor
        )

        final_labels.append(
            label
        )

        final_paths.append(
            path
        )


    images = torch.stack(
        images
    )

    labels_tensor = torch.tensor(
        final_labels,
        dtype=torch.long
    )


    return (
        images,
        labels_tensor,
        final_paths
    )


# ==========================================
# Main
# ==========================================

def main():

    print("Collecting images...")

    (
        train_paths,
        train_labels,
        val_paths,
        val_labels
    ) = split_data()


    print(
        f"Original training images: "
        f"{len(train_paths)}"
    )

    print(
        f"Validation images: "
        f"{len(val_paths)}"
    )


    print("\nBuilding training dataset...")

    train_images, train_labels_tensor, train_final_paths = (
        build_training_dataset(
            train_paths,
            train_labels
        )
    )


    print("\nBuilding validation dataset...")

    val_images, val_labels_tensor, val_final_paths = (
        build_validation_dataset(
            val_paths,
            val_labels
        )
    )


    # ==========================================
    # Save directory
    # ==========================================

    processed = Path("processed")

    processed.mkdir(
        exist_ok=True
    )


    # ==========================================
    # Training dictionary
    # ==========================================

    train_data = {

        "images":
            train_images,

        "labels":
            train_labels_tensor,

        "paths":
            train_final_paths,

        "class_names":
            class_names
    }


    # ==========================================
    # Validation dictionary
    # ==========================================

    val_data = {

        "images":
            val_images,

        "labels":
            val_labels_tensor,

        "paths":
            val_final_paths,

        "class_names":
            class_names
    }


    # ==========================================
    # Save
    # ==========================================

    torch.save(
        train_data,
        processed / "train.pt"
    )

    torch.save(
        val_data,
        processed / "val.pt"
    )


    print("\nDataset created successfully!")


    print("\nTRAIN")

    print(
        "Images:",
        train_images.shape
    )

    print(
        "Labels:",
        train_labels_tensor.shape
    )


    print("\nVALIDATION")

    print(
        "Images:",
        val_images.shape
    )

    print(
        "Labels:",
        val_labels_tensor.shape
    )


if __name__ == "__main__":

    main()