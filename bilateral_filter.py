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

    m = np.zeros((size, size, 3))
    centre = size // 2

    for y in range(-centre, centre+1):
        for x in range(-centre, centre+1):
            m[y+centre][x+centre] = np.repeat(np.sqrt(x**2 + y**2), 3)

    return m


def calc_i_matrix(nhood):
    """
    Create a matrix containing intensity differences from the centre element.

    Args:
        nhood: local pixel neighbourhood to create matrix from

    Returns:
        Matrix with each element equal to the absolute difference in intensity
        from the centre element.

    """
    h, w = nhood.shape[0], nhood.shape[1]
    if h != w:
        raise ValueError("nhood should be square")
    elif not h % 2:
        raise ValueError("nhood should have odd dimensions")

    centre = nhood[h//2, h//2]
    # print(centre)
    i_matrix = np.array(nhood, dtype=int) - centre
    return i_matrix


def bilateral_filter(img, nsize, sigma_color, sigma_space):
    """
    Apply a bilateral filter to an image.

    Args:
        img: image to apply filter to
        nsize: diameter of pixel neighbourhood used to filter
        sigma_color: filter sigma in color space
        sigma_space: filter sigma in coordinate space

    Returns:
        Copy of the image with a bilateral filter applied to it.

    """
    if not nsize % 2:
        raise ValueError("nsize must be odd")

    nw = nsize // 2
    d_matrix = calc_d_matrix(nsize)

    img_w, img_h = img.shape[1], img.shape[0]

    src = make_border(img, nw)
    dst = np.empty(img.shape, img.dtype)

    g_color = gaussian(sigma_color)
    g_space = gaussian(sigma_space)

    d_matrix = g_space(d_matrix)

    for y in range(img_h):
        for x in range(img_w):
            # print(x, y)
            centre = img[y, x]
            # print(centre)
            nhood = src[y:y + 2*nw + 1, x:x + 2*nw + 1]
            i_matrix = calc_i_matrix(nhood)
            i_matrix = g_color(i_matrix)

            p = np.multiply(d_matrix, i_matrix)

            numerator = np.sum(np.multiply(p, nhood), axis=(0, 1))
            denominator = np.sum(p, axis=(0, 1))

            response = numerator/denominator
            dst[y, x] = response

    return np.array(dst, dtype=img.dtype)


def joint_bilateral_filter(flash_img, noflash_img, nsize,
                           sigma_color, sigma_space):
    """
    Apply a joint bilateral filter to a pair of images, one taken with flash
    and the other without.

    Args:
        flash_img: image taken with flash
        noflash_img: image taken without flash
        nsize: diameter of pixel neighbourhood used to filter
        sigma_color: filter sigma in color space
        sigma_space: filter sigma in coordinate space

    Returns:
        An image combining the colour attributes from the image without flash
        and the fine details of the image with flash.

    """
    if not nsize % 2:
        raise ValueError("nsize must be odd")

    nw = nsize // 2
    d_matrix = calc_d_matrix(nsize)

    img_w, img_h = noflash_img.shape[1], noflash_img.shape[0]

    src_noflash = make_border(noflash_img, nw)
    src_flash = make_border(flash_img, nw)
    dst = np.empty(noflash_img.shape, noflash_img.dtype)

    g_color = gaussian(sigma_color)
    g_space = gaussian(sigma_space)

    d_matrix = g_space(d_matrix)

    for y in range(img_h):
        for x in range(img_w):
            # print(x, y)
            centre = noflash_img[y, x]
            # print(centre)
            nhood_noflash = src_noflash[y:y + 2*nw + 1, x:x + 2*nw + 1]
            nhood_flash = src_flash[y:y + 2*nw + 1, x:x + 2*nw + 1]

            i_matrix_flash = calc_i_matrix(nhood_flash)
            i_matrix_flash = g_color(i_matrix_flash)

            p = np.multiply(d_matrix, i_matrix_flash)

            numerator = np.sum(np.multiply(p, nhood_noflash), axis=(0, 1))
            denominator = np.sum(p, axis=(0, 1))

            response = numerator/denominator
            dst[y, x] = response

    return np.array(dst, dtype=noflash_img.dtype)


def test_bilateral_filter():
    original_window = "Original Image"
    edited_window = "Edited Image"

    img = cv2.imread('test_images/test2.png', cv2.IMREAD_COLOR)

    # Two images should look pretty much the same...
    dst1 = cv2.bilateralFilter(img, 7, 100, 100)
    dst2 = bilateral_filter(img, 7, 20, 100)

    if img is not None:
        cv2.namedWindow(original_window)
        cv2.namedWindow(edited_window)

        while True:
            cv2.imshow(original_window, dst1)
            cv2.imshow(edited_window, dst2)
            key = cv2.waitKey(40) & 0xFF
            if key == ord('x'):
                break

    cv2.destroyAllWindows()


def test_joint_bilateral_filter():
    noflash_window = "Original No Flash Image"
    flash_window = "Original Flash Image"
    edited_window = "Edited Image"

    noflash_img = cv2.imread('test_images/test3a.jpg', cv2.IMREAD_COLOR)
    flash_img = cv2.imread('test_images/test3b.jpg', cv2.IMREAD_COLOR)

    dst = joint_bilateral_filter(flash_img, noflash_img, 7, 0.01, 13)

    if noflash_img is not None and flash_img is not None:
        cv2.namedWindow(noflash_window)
        cv2.namedWindow(flash_window)
        cv2.namedWindow(edited_window)

        while True:
            cv2.imshow(noflash_window, noflash_img)
            cv2.imshow(flash_window, flash_img)
            cv2.imshow(edited_window, dst)
            key = cv2.waitKey(40) & 0xFF
            if key == ord('x'):
                break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    # test_bilateral_filter()
    test_joint_bilateral_filter()
