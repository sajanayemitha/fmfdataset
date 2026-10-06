import torch

from torch.utils.data import Dataset, DataLoader

from torchvision import transforms


class FaceDataset(Dataset):

    def __init__(
        self,
        pt_file
    ):

        data = torch.load(
            pt_file,
            weights_only=False
        )

        self.images = data["images"]

        self.labels = data["labels"]

        self.paths = data["paths"]

        self.class_names = data["class_names"]


    def __len__(self):

        return len(
            self.images
        )


    def __getitem__(
        self,
        index
    ):

        image = self.images[index]

        label = self.labels[index]

        return image, label


# ==========================================
# ImageNet normalization
# ==========================================

imagenet_normalization = transforms.Normalize(

    mean=[
        0.485,
        0.456,
        0.406
    ],

    std=[
        0.229,
        0.224,
        0.225
    ]
)


class NormalizedFaceDataset(
    FaceDataset
):

    def __getitem__(
        self,
        index
    ):

        image, label = super().__getitem__(
            index
        )

        image = imagenet_normalization(
            image
        )

        return image, label


# ==========================================
# DataLoader function
# ==========================================

def create_dataloaders(
    dataset_path,
    batch_size=8
):

    train_dataset = NormalizedFaceDataset(
        f"{dataset_path}/processed/train.pt"
    )

    val_dataset = NormalizedFaceDataset(
        f"{dataset_path}/processed/val.pt"
    )


    train_loader = DataLoader(

        train_dataset,

        batch_size=batch_size,

        shuffle=True
    )


    val_loader = DataLoader(

        val_dataset,

        batch_size=batch_size,

        shuffle=False
    )


    return (
        train_loader,
        val_loader,
        train_dataset.class_names
    )


# ==========================================
# Test
# ==========================================

if __name__ == "__main__":

    train_loader, val_loader, class_names = (
        create_dataloaders(
            ".",
            batch_size=8
        )
    )


    images, labels = next(
        iter(train_loader)
    )


    print(
        f"Batch shape: {images.shape}"
    )

    print(
        f"Labels: {labels}"
    )

    print(
        "Label names:",
        [
            class_names[l.item()]
            for l in labels
        ]
    )