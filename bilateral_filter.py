"""
Software Methodologies - Image Processing Coursework.

Implementation of Bilateral Filter
"""

import numpy as np
import cv2


def gaussian(sigma):
    """
    Create a gaussian function with standard deviation sigma.

    Args:
        sigma: standard deviation

    Returns:
        Gaussian function g(x) with standard deviation sigma.

    """
    def g(x):
        a = 1 / (sigma * np.sqrt(2 * np.pi))
        exponent = (-1/2) * (x / sigma)**2
        return a * np.exp(exponent)

    return g


def make_border(img, width):
    """
    Extrapolate the border of an image for filtering.

    Implements the OpenCV BORDER_REFLECT_101 method for
    extrapolation.

    Args:
        img: the input image
        width: border width in pixels

    Returns:
        A copy of the image with an extrapolated border.

    """
    if not width % 2:
        raise ValueError("width must be even")

    left_border = np.flip(img[:, 1:width+1], 1)
    right_border = np.flip(img[:, -width-1:-1], 1)

    img_h_bordered = np.hstack((left_border, img, right_border))

    top_border = np.flip(img_h_bordered[1:width+1, :], 0)
    bottom_border = np.flip(img_h_bordered[-width-1:-1, :], 0)

    img_bordered = np.vstack((top_border, img_h_bordered, bottom_border))

    return img_bordered


def calc_d_matrix(size):
    """
    Create a square matrix containing distances from the centre element.

    Args:
        size: number of rows/columns of matrix

    Returns:
        Matrix with each element equal to its distance from the centre.

    """
    if not size % 2:
        raise ValueError("nsize must be odd")

    m = np.zeros((size, size))
    centre = size // 2

    for y in range(-centre, centre+1):
        for x in range(-centre, centre+1):
            m[y+centre][x+centre] = np.sqrt(x**2 + y**2)

    return m


def bilateral_filter(img, nsize, sigmaColor, sigmaSpace):
    """
    Apply a bilateral filter to an image.

    Args:
        img: image to apply filter to
        nsize: diameter of pixel neighbourhood used to filter
        sigmaColor: filter sigma in color space
        sigmaSpace: filter sigma in coordinate space

    Returns:
        Copy of the image with a bilateral filter applied to it.

    """
    if nsize % 2:
        raise ValueError("nsize must be odd")

    nw = nsize // 2
    d_matrix = calc_d_matrix(nsize)

    img_w, img_h = img.shape[1], img.shape[0]

    src = make_border(img, nw)
    dst = np.empty(img.shape, img.dtype)

    g_color = gaussian(sigmaColor)
    g_space = gaussian(sigmaSpace)

    for y in range(nw, img_w - nw):
        for x in range(nw, img_h - nw):
            nhood = src[y - nw:y + nw + 1, x - nw:x + nw + 1]
            centre = nhood[nw][nw]

            numerator = 0
            denominator = 0
            for j in range(nhood.shape[0]):
                for i in range(nhood.shape[1]):
                    p = g_space(d_matrix[j][i])
                    p *= g_color(np.abs(nhood[j][i] - centre))
                    numerator += p * centre
                    denominator += p

            response = img.dtype(numerator/denominator)
            dst[y-nw, x-nw] = response

    return dst


if __name__ == "__main__":
    pass
