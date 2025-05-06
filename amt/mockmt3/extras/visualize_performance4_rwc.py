import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colormaps
import mplcyberpunk


# yapf: disable
def plot_radar_chart(data, metrics, font_size, title=None, dark_mode=False):
    if dark_mode:
        background_color = '#212946'
        text_color = '#cacfd3'
        color_palette=color_palette =  ["#ff0097", "#01e31c", "#fdf500",  "#FFFFFF"]
    else:
        background_color = 'none'
        text_color = '#1c1d1d'
        color_palette =  ["#ff0097", "#01e31c",  "purple","#00fffb",]
    N = len(metrics)

    theta = np.linspace(0, 2 * np.pi, N, endpoint=False)
    theta = np.concatenate([theta, [theta[0]]])

    fig, ax = plt.subplots(figsize=(25, 12), subplot_kw={'projection': 'polar'}, facecolor=background_color)
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
    linestyles = ['dotted',  'solid','dashed',]

    for idx, (k, vlist) in enumerate(data.items()):
        values = vlist
        values = values + [values[0]]

        if idx == 3:
            ax.plot(theta, values, linewidth=0, linestyle='solid',
                    color='none')
            ax.fill(theta, values, alpha=0.12, label=k,
                color=color_palette[idx % len(color_palette)] if color_palette else None)
        else:
            ax.plot(theta, values, linewidth=1, linestyle=linestyles[idx], label=k,
                    marker='o', markersize=14,
                    color=color_palette[idx % len(color_palette)] if color_palette else None)
            ax.fill(theta, values, alpha=0.12,
                color=color_palette[idx % len(color_palette)] if color_palette else None)


    plt.yticks([0, 20, 40, 60, 80 ], ["0", "20", "40", "60","80"], color=text_color, size=24)
    # plt.yticks([20, 40, 60, 80, 100], ["20", "40", "60", "80", "100"], color=text_color, size=20)
    plt.xticks(theta, metrics + [metrics[0]], color=text_color, size=font_size,
               fontname='Comic Sans MS')
    l = plt.legend(loc='upper right', bbox_to_anchor=(1.7, 0.23), fontsize=font_size,
                   facecolor=background_color, edgecolor=text_color)
    for text in l.get_texts():
        text.set_color(text_color)

data = {
    'MT3': [34.34, 17.83, 4.38, 1.81, 13.42, 1.57, 29.56, 0.10, 1.78, 0., 6.31, 1.23, 2.1, 43.28, 30.67, 47.37, 12.86],
    'YMT3': [45.07, 22.68, 17.43, 2.03, 20.13, 5.18, 44.00, 3.64, 7.41, 6.46, 13.64, 1.2, 2.58, 43.41, 34.79, 52.59, 19.72],
    'YPTF.MoE+Multi': [47.39, 24.43,15.79, 0.9,22.84,4.54,48.06,3.56, 12.77, 8.29, 15.35, 0.66, 1.97,44.30, 35.00,53.45,22.34],
    'YPTF.MoE+Multi (chroma)': [76.59, 69.21, 17.14, 3.63, 28.14, 10.10, 52.78, 3.56, 13.46, 49.46, 18.42, 3.15, 2.76, 0, 55.3, 62.33, 22.34],
}
metrics = ["Bass (150ms)", "Bass", "Brass", "C. Perc",
          "Guitar", "Organ", "Piano", "Pipe", "Reed",
          "Singing", "Strings", "S. Lead", "S. Pad", "Drums", "Non-drum",
          "Frame F1",  "Multi F1"]

# plt.style.use('cyberpunk')
# plt.rcParams['axes.facecolor'] = 'none'
# plt.rcParams['figure.dpi'] = 100

plot_radar_chart(data, metrics, dark_mode=False, font_size=28)
mplcyberpunk.make_lines_glow()
plt.savefig('rwc_radar_w.pdf',facecolor='none')

plot_radar_chart(data, metrics, dark_mode=True, font_size=28)
mplcyberpunk.make_lines_glow()
plt.savefig('rwc_radar_b.pdf')
