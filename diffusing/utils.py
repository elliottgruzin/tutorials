import torchvision
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt


def create_image_grid(images):
    images = (images / 2 + 0.5).clamp(0, 1)
    grid = torchvision.utils.make_grid(images, nrow=2)
    grid = grid.detach().cpu().permute(1, 2, 0).numpy() * 255
    grid_im = Image.fromarray(np.array(grid).astype(np.uint8))
    return grid_im

def save_image(images, path):
    grid = create_image_grid(images)
    grid.save(path)

def save_losses(losses, path):
    plt.plot(losses)
    plt.savefig(path)