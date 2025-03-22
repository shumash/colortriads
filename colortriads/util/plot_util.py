import numpy as np
import matplotlib.pyplot as plt


class __Size(object):
    def __init__(self, w, h):
        self.width = w
        self.height = h


def matplot_fig2img(fig, unit_cube=False, no_axes=False, width=-1, height=-1):
    fig.tight_layout()
    fig.axes[0].margins(0.0)
    if unit_cube:
        plt.figure(fig.number)
        fig_size = plt.rcParams["figure.figsize"]
        print(max(fig_size[0], fig_size[1]))
        fig_size[0] = max(fig_size[0], fig_size[1])
        fig_size[1] = max(fig_size[0], fig_size[1])
        fig.axes[0].set_xlim([-1, 1])
        fig.axes[0].set_ylim([-1, 1])
    if no_axes:
        fig.axes[0].spines['right'].set_visible(False)
        fig.axes[0].spines['top'].set_visible(False)
        fig.axes[0].spines['left'].set_visible(False)
        fig.axes[0].spines['bottom'].set_visible(False)
        fig.axes[0].get_xaxis().set_visible(False)
        fig.axes[0].get_yaxis().set_visible(False)
    if width > 0 and height > 0:
        fig.canvas.resize(__Size(width, height))
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    print('w,h %d, %d' % (w, h))
    image = np.fromstring(fig.canvas.tostring_rgb(), dtype='uint8').reshape(h, w, 3)
    return image

def matplot_draw_triangle(fig, vertex_rows, color):
    t1 = plt.Polygon(vertex_rows, color=color)
    fig.gca().add_patch(t1)
    fig.axes[0].axis('tight')

