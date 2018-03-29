import imageio


def save_img(im_array, output_path):
    imageio.imwrite(output_path, im_array)
