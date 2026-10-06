import torch
import matplotlib.pyplot as plt
import random


# ==========================================
# Load datasets
# ==========================================

train = torch.load(
    "processed/train.pt",
    weights_only=False
)

val = torch.load(
    "processed/val.pt",
    weights_only=False
)


train_images = train["images"]
train_labels = train["labels"]

val_images = val["images"]
val_labels = val["labels"]

class_names = train["class_names"]


# ==========================================
# Print shapes
# ==========================================

print("TRAIN")

print(
    "Images shape:",
    train_images.shape
)

print(
    "Labels shape:",
    train_labels.shape
)


print("\nVALIDATION")

print(
    "Images shape:",
    val_images.shape
)

print(
    "Labels shape:",
    val_labels.shape
)


# ==========================================
# Class distribution
# ==========================================

train_unique, train_counts = torch.unique(
    train_labels,
    return_counts=True
)

val_unique, val_counts = torch.unique(
    val_labels,
    return_counts=True
)


print("\nTraining distribution:")

for label, count in zip(
    train_unique,
    train_counts
):

    print(
        class_names[label.item()],
        count.item()
    )


print("\nValidation distribution:")

for label, count in zip(
    val_unique,
    val_counts
):

    print(
        class_names[label.item()],
        count.item()
    )


# ==========================================
# Bar chart
# ==========================================

train_values = []

val_values = []


for i in range(
    len(class_names)
):

    train_values.append(
        (train_labels == i).sum().item()
    )

    val_values.append(
        (val_labels == i).sum().item()
    )


x = range(
    len(class_names)
)

width = 0.35


plt.figure(
    figsize=(10, 6)
)

plt.bar(
    [i - width/2 for i in x],
    train_values,
    width=width,
    label="Training"
)

plt.bar(
    [i + width/2 for i in x],
    val_values,
    width=width,
    label="Validation"
)


plt.xticks(
    list(x),
    class_names
)

plt.xlabel("Class")
plt.ylabel("Number of Images")

plt.title(
    "Training vs Validation Class Distribution"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    "class_distribution.png",
    dpi=200
)

plt.show()


# ==========================================
# 16 random training images
# ==========================================

random.seed(42)

indices = random.sample(
    range(len(train_images)),
    min(16, len(train_images))
)


plt.figure(
    figsize=(12, 12)
)


for i, index in enumerate(
    indices
):

    image = train_images[index]

    label = train_labels[index].item()


    # CHW → HWC

    image = image.permute(
        1, 2, 0
    )


    plt.subplot(
        4,
        4,
        i + 1
    )

    plt.imshow(
        image
    )

    plt.title(
        class_names[label]
    )

    plt.axis("off")


plt.tight_layout()

plt.savefig(
    "training_images_grid.png",
    dpi=200
)

plt.show()