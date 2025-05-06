import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colormaps, rc
import mplcyberpunk


# yapf: disable
def plot_radar_chart(data, metrics, font_size, title=None, dark_mode=False):
    if dark_mode:
        background_color = '#212946'
        text_color = '#cacfd3'
        color_palette=color_palette =  ["#ff0097", "#01e31c",  "#00fffb","#fdf500",]
    else:
        background_color = 'none'
        text_color = '#1c1d1d'
        color_palette =  ["#ff0097", "#01e31c",  "#00fffb","purple",]
    N = len(metrics)

    theta = np.linspace(0, 2 * np.pi, N, endpoint=False)
    theta = np.concatenate([theta, [theta[0]]])

    fig, ax = plt.subplots(figsize=(20, 12), subplot_kw={'projection': 'polar'}, facecolor=background_color)
    # Title
    if title: ax.set_title(title, y=1.15, fontsize=20)

    # Direction of the zero angle to the north (upwards)
    ax.set_theta_zero_location("N")

    # Direction of the angles to be counterclockwise
    ax.set_theta_direction(-1)

    # Make radial gridlines appear behind other elements
    ax.spines['polar'].set_zorder(1)

    # Radial label position (position of values on the radial axes)
    ax.set_rlabel_position(115)

    # Color of radial gridlines
    ax.spines['polar'].set_color('lightgrey')
    ax.set_facecolor(background_color)

    for idx, (k, vlist) in enumerate(data.items()):
        values = vlist
        values = values + [values[0]]

        # Edge
        ax.plot(theta, values, linewidth=5, linestyle='solid', label=k,
                marker='o', markersize=14,
                color=color_palette[idx % len(color_palette)] if color_palette else None)
        # Fill
        ax.fill(theta, values, alpha=0.12,
                color=color_palette[idx % len(color_palette)] if color_palette else None)

    # plt.yticks([0, 25, 50, 75, 100], ["0", "25", "50", "75", "100"], color="black", size=12)
    plt.yticks([20, 40, 60, 80, 100], ["20", "40", "60", "80", "100"], color=text_color, size=20)
    plt.xticks(theta, metrics + [metrics[0]], color=text_color, size=font_size,
               fontname='Comic Sans MS')
    l = plt.legend(loc='upper right', bbox_to_anchor=(1.52, 0.23), fontsize=font_size,
                   facecolor=background_color, edgecolor=text_color)
    # l = plt.legend(loc='upper right', ncol=2, bbox_to_anchor=(0.915, -0.025), fontsize=font_size,
    #             facecolor=background_color, edgecolor=text_color)
    for text in l.get_texts():
        text.set_color(text_color)


data = {
    'MT3': [71.03, 28.67, 34.31, 65.9, 30.14, 70.87, 40.6, 19.41, 47.02, 29.51, 20.02, 83.85],
    'YMT3': [86.63, 55.9, 46.94, 69.58, 42.19, 77.91, 53.78, 59.89, 59.81, 66.08, 28.7, 87.15],
    'PerceiverTF': [93.0, 73.2, 57.5, 78.5, 69.4, 85.4, 66.6, 72.5, 74.4, 76.9, 47.4, 78.5],
    'YPTF.MoE+Multi': [93.25, 74.96, 67.7, 82.26, 73.47, 88.83, 74.72, 82.21, 75.43, 84.19, 45.57, 90.52],
}

metrics = ["Bass", "Brass", "C.Perc", "Guitar", "Organ",
           "Piano", "Pipe", "Reed",  "Strings",
           "S. Lead", "S. Pad", "Drums (GM)"]

# plt.style.use('cyberpunk')
# plt.rcParams['axes.facecolor'] = 'none'
# plt.rcParams['figure.dpi'] = 100

plot_radar_chart(data, metrics, dark_mode=False, font_size=28)
mplcyberpunk.make_lines_glow()
# plt.savefig('slakh_radar_w.pdf',facecolor='none')

plot_radar_chart(data, metrics, dark_mode=True, font_size=28)
mplcyberpunk.make_lines_glow()
# plt.savefig('slakh_radar_b.pdf')
