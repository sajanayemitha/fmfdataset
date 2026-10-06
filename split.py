from pathlib import Path
import json
import random

from sklearn.model_selection import train_test_split


def load_config():

    with open("config.json", "r") as f:
        config = json.load(f)

    return config["class_names"]


def collect_images():

    class_names = load_config()

    image_paths = []
    labels = []


    for label, class_name in enumerate(class_names):

        folder = Path("raw") / class_name

        images = sorted(
            list(folder.glob("*.jpg")) +
            list(folder.glob("*.jpeg")) +
            list(folder.glob("*.png"))
        )

        for image_path in images:

            image_paths.append(
                str(image_path)
            )

            labels.append(label)


    return image_paths, labels, class_names


def split_dataset():

    image_paths, labels, class_names = (
        collect_images()
    )


    # Fixed seed
    random.seed(42)


    train_paths, val_paths, train_labels, val_labels = (
        train_test_split(
            image_paths,
            labels,
            test_size=0.20,
            random_state=42,
            stratify=labels
        )
    )


    print(
        f"Total images: {len(image_paths)}"
    )

    print(
        f"Training images: {len(train_paths)}"
    )

    print(
        f"Validation images: {len(val_paths)}"
    )


    print("\nTraining distribution:")

    for class_id, class_name in enumerate(class_names):

        count = train_labels.count(class_id)

        print(
            f"{class_name}: {count}"
        )


    print("\nValidation distribution:")

    for class_id, class_name in enumerate(class_names):

        count = val_labels.count(class_id)

        print(
            f"{class_name}: {count}"
        )


    return (
        train_paths,
        train_labels,
        val_paths,
        val_labels,
        class_names
    )


if __name__ == "__main__":

    split_dataset()