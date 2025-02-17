import cv2
import numpy as np
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
