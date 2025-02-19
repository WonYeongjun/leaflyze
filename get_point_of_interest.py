import torch
import numpy as np
import cv2
from segment_anything import sam_model_registry, SamPredictor
import glob
import matplotlib.pyplot as plt


def get_point_of_interest(image):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sam = sam_model_registry["vit_b"](
        checkpoint="C:/Users/UserK/Desktop/sam_vit_b.pth"
    ).to(device)
    predictor = SamPredictor(sam)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    predictor.set_image(image)

    input_point = np.array([[image.shape[1] // 2, image.shape[0] // 2]])
    input_label = np.array([1])
    masks, _, _ = predictor.predict(
        point_coords=input_point, point_labels=input_label, multimask_output=True
    )

    mask = masks[np.argmax([np.sum(m) for m in masks])]
    kernel = np.ones((15, 15), np.uint8)
    mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, kernel)

    y_indices, x_indices = np.where(mask)
    y_min, y_max = y_indices.min(), y_indices.max()
    x_min, x_max = x_indices.min(), x_indices.max()

    cropped_image = image[y_min:y_max, x_min:x_max]
    return cropped_image


if __name__ == "__main__":
    color = "white"
    image_files = glob.glob(f"./image/{color}/*.jpg")
    num_images = len(image_files)
    fig, axes = plt.subplots(num_images, 2, figsize=(8, num_images * 4))

    for i, file in enumerate(image_files):
        image = cv2.imread(file)
        cropped_image = get_point_of_interest(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        axes[i, 0].imshow(image)
        axes[i, 0].axis("off")
        axes[i, 1].imshow(cropped_image, cmap="gray")
        axes[i, 1].axis("off")

    plt.tight_layout()
    plt.savefig(f"./output/image_comparison_{color}.png", dpi=300)
    plt.show()
