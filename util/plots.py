from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.patches as patches

def plot_center(image, uv, prob=None):
    """
    input: image: HxWx3
    """
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.axis('off')
    fig.subplots_adjust(0, 0, 0.8, 1)  # get rid of margins

    ax.imshow(image)

    x, y = uv[1], uv[0]
    ax.add_patch(patches.Circle((x, y), 5, color='red'))

    if prob is not None:
        ax.text(0.5, 0.9, prob, horizontalalignment='center', verticalalignment='center', transform=ax.transAxes,
                color='red', fontsize=30)

    fig.canvas.draw()
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)
    return data