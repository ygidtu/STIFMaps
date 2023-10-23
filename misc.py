import numpy as np


# Normalize an image from 0-1
def norm_pic(im):
    # Find the thresholds
    hmin = np.quantile(im.flatten(), .01)
    hmax = np.quantile(im.flatten(), .99)

    # Create the new thresholded image
    im2 = (im - hmin) / (hmax - hmin)
    im2 = np.clip(im2, 0, 1)

    return im2


# Given the step size between squares and the scale factor between the input
#      image and the training data, find the actual step size to use in the input image
#      Note that the output step size must be an even number
def get_step(step, scale_factor):
    step = step / scale_factor

    if round(step) % 2 == 0:
        return round(step)
    elif round(step) == int(step):
        return round(step) + 1
    else:
        return int(step)


if __name__ == '__main__':
    pass
