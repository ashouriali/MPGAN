import matplotlib.pyplot as plt
from skimage.color import lab2rgb
import numpy as np
import torch


def show_tensor_images(image_tensor, num_images=25, size=(1, 32, 32), f=False):
    '''
    Function for visualizing images: Given a tensor of images, number of images, and
    size per image, plots and prints the images in an uniform grid.
    '''
    image_tensor = image_tensor[0].clone().detach()

    if f:
        arr = np.zeros(shape=size)
        b = torch.tensor(arr).detach().clone()

        b[0, :, :] = image_tensor
        image_tensor = b

    print(image_tensor.shape)

    img = lab2rgb(image_tensor.permute((1, 2, 0)))
    plt.imshow(img)
    # image_shifted = image_tensor
    # image_unflat = image_shifted.detach().cpu().view(-1, *size)
    # image_grid = make_grid(image_unflat[:num_images], nrow=5)
    # plt.imshow(image_grid.permute(1, 2, 0).squeeze())

    plt.show()

def crop(image, new_shape):
    '''
    Function for cropping an image tensor: Given an image tensor and the new shape,
    crops to the center pixels (assumes that the input's size and the new size are
    even numbers).
    Parameters:
        image: image tensor of shape (batch size, channels, height, width)
        new_shape: a torch.Size object with the shape you want x to have
    '''

    middle_height = image.shape[2] // 2
    middle_width = image.shape[3] // 2
    starting_height = middle_height - new_shape[2] // 2
    final_height = starting_height + new_shape[2]
    starting_width = middle_width - new_shape[3] // 2
    final_width = starting_width + new_shape[3]
    cropped_image = image[:, :, starting_height:final_height, starting_width:final_width]
    return cropped_image