import matplotlib.pyplot as plt
import util.img_util as img_util

def show_image(im, width=1.0):
    masterfig = plt.figure(figsize = (20 * width, 20 * width))
    fig = masterfig.add_subplot(111)
    if len(im.shape) < 3:
        fig.imshow(im, interpolation='none', cmap='gray')
    else:
        fig.imshow(im, interpolation='none')
    fig.axes.get_xaxis().set_visible(False)
    fig.axes.get_yaxis().set_visible(False)
    return masterfig

def show_images(images, per_row=4, width=1.0):
    return show_image(img_util.concat_images(images, images_per_row=per_row), width)
