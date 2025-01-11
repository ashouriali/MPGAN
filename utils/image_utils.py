import matplotlib.pyplot as plt
from skimage.color import lab2rgb
import numpy as np
import torch


def show_tensor_images(image_tensor, num_images=25, size=(1, 32, 32), gray_scale=False):
    image_tensor = image_tensor[0].clone().detach()

    if gray_scale:
        zero_tensor = torch.tensor(np.zeros(shape=size)).detach().clone()

        zero_tensor[0, :, :] = image_tensor
        image_tensor = zero_tensor

    print(image_tensor.shape)

    img = lab2rgb(image_tensor.permute((1, 2, 0)))
    plt.imshow(img)

    plt.show()

def crop(image, new_shape):
    middle_height = image.shape[2] // 2
    middle_width = image.shape[3] // 2
    starting_height = middle_height - new_shape[2] // 2
    final_height = starting_height + new_shape[2]
    starting_width = middle_width - new_shape[3] // 2
    final_width = starting_width + new_shape[3]
    cropped_image = image[:, :, starting_height:final_height, starting_width:final_width]
    return cropped_image