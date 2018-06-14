import numpy as np
try:
    import matplotlib.colors
    import matplotlib.pyplot as plt
    import matplotlib.cm as cmx
except:
    print('No matplotlib')

def arr_to_colors(x, x_min=None, x_max=None, colors_map='jet', scalar_map=None):
    if len(x) == 0:
        return []
    if scalar_map is None:
        x_min, x_max = check_min_max(x, x_min, x_max)
        scalar_map = get_scalar_map(x_min, x_max, colors_map)
    return scalar_map.to_rgba(x)


def get_scalar_map(x_min, x_max, color_map='jet'):
    cm = plt.get_cmap(color_map)
    cNorm = matplotlib.colors.Normalize(vmin=x_min, vmax=x_max)
    return cmx.ScalarMappable(norm=cNorm, cmap=cm)


def check_min_max(x, x_min, x_max):
    if x_min is None:
        x_min = np.min(x)
    if x_max is None:
        x_max = np.max(x)
    return x_min, x_max
