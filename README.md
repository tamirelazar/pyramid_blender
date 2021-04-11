# pyramid_blender
Uses a binary mask to blend two images.

### Details
A simple non-interactive python program for the creation of mashup images.
Planning on an interactive version, currently you can set your own blendings by putting your own pictures and mask in Pics directory, and changing addresses in the blending_example functions.

1. Choose 2 Images to Blend:
    ![RealAviBitterBody](https://github.com/tamirelazar/pyramid_blender/blob/main/Pics/bitter/m1small.jpg)
    ![Avi](https://github.com/tamirelazar/pyramid_blender/blob/main/Pics/bitter/m2small.jpg)
2. Create a Binary Blending Mask
   This is like an instruction file. The black part will be taken from image 1, the white from image 2.
    ![Mask](![Avi](https://github.com/tamirelazar/pyramid_blender/blob/main/Pics/bitter/mask.jpg)
3. Set correct addresses in blending_example1:
  ```python
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
  ```
4. Run the .py using python3
   The program uses pyramid gaussian blending to stitch the images together.
5. Hell Yeah!
