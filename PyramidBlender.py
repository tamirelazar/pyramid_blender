import numpy as np
from scipy.ndimage.filters import convolve
import matplotlib.pyplot as plt
from skimage.color import rgb2gray, rgb2yiq, yiq2rgb
import os


def create_guassian_filter_vec(filter_size):
    """
    creates a row vector to convolve with in the gaussian pyramid func
    :param filter_size: odd integer
    :return: a numpy array of filter_size size with the normalized binomial coefficients
    """
    filter_vec = np.array([1, 1], dtype=np.float64)
    return_filter = np.array([1, 1], dtype=np.float64)

    while return_filter.shape[0] < filter_size:
        return_filter = np.convolve(return_filter, filter_vec)

    sum = np.sum(return_filter)
    return_filter /= sum

    return return_filter.reshape((1, filter_size))


def reduce(im, filter_vec):
    """
    reduces a N x M image in size, after blurring
    :param im: the image to reduce
    :param filter_vec: the vector to convolve for blurring
    :return: an image of size N/2 x M/2
    """
    filter_size = filter_vec.shape[1]
    filter_mat = np.zeros((filter_size, filter_size))
    filter_mat[filter_size // 2, :] = filter_vec[0, :]
    cov_im = convolve(im.copy(), filter_mat)
    cov_im = convolve(cov_im, filter_mat.T)
    new_im = cov_im[0::2, 0::2]
    return new_im


def build_gaussian_pyramid(im, max_levels, filter_size):
    """
    :param im: grayscale image with double values in [0, 1]
    :param max_levels: the maximal number of levels in the resulting pyramid.
    :param filter_size: an odd scalar that represents a squared filter
    :return:
    """
    g_pyr = [im]
    g_filter = create_guassian_filter_vec(filter_size)
    for i in range(max_levels - 1):
        r_im = reduce(im.copy(), g_filter)
        if r_im.shape[0] < 16 or r_im.shape[1] < 16:
            break
        else:
            g_pyr.append(r_im)
            im = r_im

    return g_pyr, g_filter


def extend(im, filter_vec):
    """
    extends a N x M image in size, then blurring
    :param im: the image to extend
    :param filter_vec: the vector to convolve for blurring
    :return: an image of size N*2 x M*2
    """
    filter_size = filter_vec.shape[1]
    filter_mat = np.zeros((filter_size, filter_size))
    filter_mat[filter_size // 2, :] = filter_vec[0, :]
    filter_mat *= 2
    ex_im = np.zeros(shape=(im.shape[0] * 2, im.shape[1] * 2))
    ex_im[0::2, 0::2] = im
    ex_im = convolve(ex_im, filter_mat)
    ex_im = convolve(ex_im, filter_mat.T)
    return ex_im


def build_laplacian_pyramid(im, max_levels, filter_size):
    """
    :param im: grayscale image with double values in [0, 1]
    :param max_levels: the maximal number of levels in the resulting pyramid.
    :param filter_size: an odd scalar that represents a squared filter
    :return:
    """
    l_pyr = []
    g_pyr, g_filter = build_gaussian_pyramid(im, max_levels, filter_size)
    for i in range(len(g_pyr) - 1):
        l_pyr.append(g_pyr[i] - extend(g_pyr[i+1], g_filter))

    l_pyr.append(g_pyr[-1])
    return l_pyr, g_filter


def laplacian_to_image(lpyr, filter_vec, coeff):
    """
    reconstructs image from laplacian representation
    :param lpyr: array representing laplacian pyramid
    :param filter_vec: blurring filter
    :param coeff: for the lap layers
    :return: image
    """
    for i in range(1, len(lpyr)):
        for j in range(i):
            lpyr[i] = extend(lpyr[i], filter_vec)
        lpyr[0] += (lpyr[i] * coeff[i])

    im = lpyr[0]
    return im


def render_pyramid(pyr, levels):
    """
    renders the levels of a pyramid in a single image
    :param pyr: the pyramid
    :param levels: number of levels
    :return: a single image of the levels
    """
    orig_height = pyr[0].shape[0]
    res = pyr[0]
    for i in range (1, levels):
        h_diff = orig_height - pyr[i].shape[0]
        to_stack = np.pad(pyr[i], ((0, h_diff), (0, 0)), mode='constant', constant_values=0)
        res = np.concatenate((res, to_stack), axis=1)

    return res


def display_pyramid(pyr, levels):
    """
    displays a pyramid after rendering
    :param pyr: the pyramid
    :param levels: number of levels
    """
    render = render_pyramid(pyr, levels)
    plt.imshow(render, cmap='gray')
    plt.show()


def pyramid_blending(im1, im2, mask, max_levels, filter_size_im, filter_size_mask):
    """
    :param im1: input grayscale image to be blended
    :param im2: input grayscale image to be blended
    :param mask: boolean mask containing True and False representing which parts of im1 and im2 should
                appear in the resulting im_blend
    :param max_levels: max_levels parameter you should use when generating the Gaussian and Laplacian pyramids.
    :param filter_size_im: is the size of the Gaussian filter (an odd scalar that represents a squared filter)
                    which defining the filter used in the construction of the Laplacian pyramids of im1 and im2
    :param filter_size_mask: size of the Gaussian filter(an odd scalar that represents a squared filter) which
                defining the filter used in the construction of the Gaussian pyramid of mask.
    :return: the blended image
    """
    l1, l1_filter = build_laplacian_pyramid(im1, max_levels, filter_size_im)
    l2, l2_filter = build_laplacian_pyramid(im2, max_levels, filter_size_im)
    g_mask, g_mask_filter = build_gaussian_pyramid(mask.astype(np.float64), max_levels, filter_size_mask)
    l_out = []
    coeff = []

    for i in range(min(max_levels, len(l1))):
        l_out.append((g_mask[i] * l1[i]) + ((1 - g_mask[i]) * l2[i]))
        coeff.append(1)

    blend_im = laplacian_to_image(l_out, l1_filter, coeff)
    return (blend_im - blend_im.min()) / (blend_im.max() - blend_im.min())


def rgb_blend(im1, im2, mask, max_levels, filter_size_im, filter_size_mask):
    """
    :param im1: input rgb image to be blended
    :param im2: input rgb image to be blended
    :param mask: boolean mask containing True and False representing which parts of im1 and im2 should
                appear in the resulting im_blend
    :param max_levels: max_levels parameter you should use when generating the Gaussian and Laplacian pyramids.
    :param filter_size_im: is the size of the Gaussian filter (an odd scalar that represents a squared filter)
                    which defining the filter used in the construction of the Laplacian pyramids of im1 and im2
    :param filter_size_mask: size of the Gaussian filter(an odd scalar that represents a squared filter) which
                defining the filter used in the construction of the Gaussian pyramid of mask.
    :return: the blended image
    """
    r_blend = pyramid_blending(im1[:, :, 0], im2[:, :, 0], mask, max_levels, filter_size_im, filter_size_mask)
    g_blend = pyramid_blending(im1[:, :, 1], im2[:, :, 1], mask, max_levels, filter_size_im, filter_size_mask)
    b_blend = pyramid_blending(im1[:, :, 2], im2[:, :, 2], mask, max_levels, filter_size_im, filter_size_mask)

    blended = np.stack((r_blend, g_blend, b_blend), axis=-1)
    return blended


def yiq_blend(im1, im2, mask, max_levels, filter_size_im, filter_size_mask):
    """
    :param im1: input rgb image to be blended
    :param im2: input rgb image to be blended
    :param mask: boolean mask containing True and False representing which parts of im1 and im2 should
                appear in the resulting im_blend
    :param max_levels: max_levels parameter you should use when generating the Gaussian and Laplacian pyramids.
    :param filter_size_im: is the size of the Gaussian filter (an odd scalar that represents a squared filter)
                    which defining the filter used in the construction of the Laplacian pyramids of im1 and im2
    :param filter_size_mask: size of the Gaussian filter(an odd scalar that represents a squared filter) which
                defining the filter used in the construction of the Gaussian pyramid of mask.
    :return: the blended image
    """
    y_im1 = rgb2yiq(im1)
    y_im2 = rgb2yiq(im2)
    channel = 0
    y_im2[:, :, channel] = pyramid_blending(y_im1[:, :, channel], y_im2[:, :, channel], mask, max_levels, filter_size_im, filter_size_mask)

    blended_i = yiq2rgb(y_im2)
    return blended_i


def blending_example1():
    """
    uses predefined images to create an example product
    :return: the two images, mask and blended outcome
    """

    im1 = read_image(relpath("Pics/bitter/m1small.jpg"), 2)

    im2 = read_image(relpath("Pics/bitter/m2small.jpg"), 2)

    mask = read_image(relpath("Pics/bitter/mask.jpg"), 1)
    mask = np.ceil(mask)
    mask = mask.astype(np.bool)

    blended_im = rgb_blend(im2, im1, mask, 10, 9, 9)

    return im1, im2, mask, blended_im


def blending_example2():
    """
    uses predefined images to create an example product
    :return: the two images, mask and blended outcome
    """

    im1 = read_image(relpath("Pics/ShmuelRushmore/im1small.jpg"), 2)

    im2 = read_image(relpath("Pics/ShmuelRushmore/im2small.jpg"), 2)

    mask = read_image(relpath("Pics/ShmuelRushmore/mask7.jpg"), 1)
    mask = np.ceil(mask)
    mask = mask.astype(np.bool)

    blended_im = yiq_blend(im2, im1, mask, 10, 5, 7)

    return im1, im2, mask, blended_im


def read_image(filename, representation):
    """
    read an image file and return as representation
    :param filename: the filename
    :param representation: 1 or 2, representing grayscale/rgb
    :return: a np.float64 matrix representing the image
    """
    image = plt.imread(filename)
    represent = int2float(image)
    if representation == 1 and represent.ndim == 3:
        represent = rgb2gray(represent)
    return represent


def imprint(image, representation):
    plt.figure()
    if representation == 1:
        plt.imshow(image, cmap='gray')
    else:
        plt.imshow(image)
    plt.axis('off')
    plt.show()


def int2float(image):
    """
    takes an image array from (0, 255) to (0, 1)
    :param image: an image array
    :return: an image array of identical shape
    """
    return image.copy().astype(np.float64) / 255


def relpath(filename):
    """
    this returns the relative path
    :param filename: path
    :return: usable path
    """
    return os.path.join(os.path.dirname(__file__), filename)