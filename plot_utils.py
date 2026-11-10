# 作者：姚弘祎
# 修改日期：
# 版本： structured version

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import Circle


radarcmap = mpl.colors.ListedColormap([
    '#01A0F6', '#0000F6', '#00EF00', '#00C200', '#008600',
    '#FFFF00', '#E4BA00', '#FF8401', '#FF0000', '#9D0000', '#5A0000',
    '#FF00FF', '#914BC2'
])
radarcmap.set_over('black')
radarcmap.set_under('white')

radarbounds = [5.0, 10.0, 15.0, 20.0, 25.0, 30, 35, 40, 45, 50, 55, 60, 65, 70]
radarnorm = mpl.colors.BoundaryNorm(radarbounds, radarcmap.N)


def interactive_cluster_view(radar, label_img, largest_label):
    """
    Interactive cluster selection.

    Left click  : toggle selected cluster on the right panel.
    Right click : measure distance using two points.
    Middle click: clear measurement.
    """
    H, W = label_img.shape

    fig, axes = plt.subplots(1, 2, figsize=(14, 8))
    ax00, ax01 = axes

    im00 = ax00.imshow(
        radar,
        origin="upper",
        cmap=radarcmap,
        norm=radarnorm
    )
    ax00.set_title("Original Radar", pad=8)

    im01 = ax01.imshow(
        label_img,
        origin="upper",
        cmap="tab20c"
    )
    ax01.set_title("Cluster Selection (Left toggle | Right measure | Middle clear)", pad=8)

    fig.subplots_adjust(
        left=0.05,
        right=0.98,
        top=0.92,
        bottom=0.16,
        wspace=0.12
    )

    cax1 = fig.add_axes([0.04, 0.08, 0.45, 0.025])
    cbar1 = fig.colorbar(
        im00,
        cax=cax1,
        orientation="horizontal",
        boundaries=[-10] + radarbounds + [80]
    )
    cbar1.set_label("reflectivity (dBZ)")

    cax2 = fig.add_axes([0.54, 0.08, 0.45, 0.025])
    cbar2 = fig.colorbar(
        im01,
        cax=cax2,
        orientation="horizontal"
    )
    cbar2.set_label("cluster ID")

    selected = set()
    if largest_label is not None:
        selected.add(largest_label)

    outline_artists = {}

    measure_points = []
    measure_dist = None
    measure_artists = []

    def draw_outline(lb):
        mask = (label_img == lb)
        if not np.any(mask):
            return []
        cs = ax01.contour(
            mask.astype(np.uint8),
            levels=[0.5],
            linewidths=1.2,
            colors="black"
        )
        return list(cs.collections)

    def remove_outline(lb):
        arts = outline_artists.get(lb, [])
        for a in arts:
            a.remove()
        outline_artists.pop(lb, None)

    def refresh():
        for lb in list(outline_artists.keys()):
            remove_outline(lb)
        for lb in selected:
            outline_artists[lb] = draw_outline(lb)
        fig.canvas.draw_idle()

    def clear_measure():
        nonlocal measure_dist
        for a in measure_artists:
            a.remove()
        measure_artists.clear()
        measure_points.clear()
        measure_dist = None
        print("[Measure] cleared")
        fig.canvas.draw_idle()

    refresh()

    def on_click(event):
        nonlocal measure_dist

        if event.xdata is None or event.ydata is None:
            return

        x = int(round(event.xdata))
        y = int(round(event.ydata))

        if x < 0 or x >= W or y < 0 or y >= H:
            return

        # Middle click: clear measurement
        if event.button == 2:
            clear_measure()
            return

        # Right click: distance measurement
        if event.button == 3:
            if len(measure_points) == 0 and len(measure_artists) > 0:
                clear_measure()

            measure_points.append((x, y))

            p00 = ax00.plot(x, y, 'ro')[0]
            p01 = ax01.plot(x, y, 'ro')[0]
            measure_artists.extend([p00, p01])

            if len(measure_points) == 2:
                (x1, y1), (x2, y2) = measure_points

                l00 = ax00.plot([x1, x2], [y1, y2], 'r-', lw=2)[0]
                l01 = ax01.plot([x1, x2], [y1, y2], 'r-', lw=2)[0]
                measure_artists.extend([l00, l01])

                cx = (x1 + x2) / 2.0
                cy = (y1 + y2) / 2.0
                r = 0.5 * np.hypot(x2 - x1, y2 - y1)

                circle0 = Circle((cx, cy), r, edgecolor='red', facecolor='none', lw=2)
                circle1 = Circle((cx, cy), r, edgecolor='red', facecolor='none', lw=2)

                ax00.add_patch(circle0)
                ax01.add_patch(circle1)
                measure_artists.extend([circle0, circle1])

                measure_dist = float(2 * r)

                print(f"[Measure] dist = {measure_dist:.2f} pixels")
                print(f"[Circle] center=({cx:.1f}, {cy:.1f}), r={r:.2f}")

                measure_points.clear()

            fig.canvas.draw_idle()
            return

        # Left click: cluster toggle
        if event.inaxes != ax01:
            return

        lb = int(label_img[y, x])
        if lb < 0:
            return

        if lb in selected:
            selected.remove(lb)
        else:
            selected.add(lb)

        refresh()

    fig.canvas.mpl_connect("button_press_event", on_click)
    plt.show()

    return selected, measure_dist


def ring_mean_profile(I, Cx, Cy, r_min=10, r_max=100, dr=5, min_count=20, yy=None, xx=None):
    """
    Compute the radial mean reflectivity profile around a candidate TC center.
    """
    I = np.asarray(I)
    H, W = I.shape

    if yy is None or xx is None:
        yy, xx = np.indices((H, W), dtype=np.float32)
    else:
        yy = yy.astype(np.float32, copy=False)
        xx = xx.astype(np.float32, copy=False)

    rr = np.sqrt((xx - float(Cx)) ** 2 + (yy - float(Cy)) ** 2)

    radii = np.arange(r_min, r_max + 1e-6, dr, dtype=np.float32)
    mean_profile = np.full_like(radii, np.nan, dtype=np.float64)
    count_profile = np.zeros_like(radii, dtype=np.int32)

    half = 0.5 * dr

    for i, r in enumerate(radii):
        mask = (rr >= (r - half)) & (rr < (r + half))
        vals = I[mask]

        n = np.sum(np.isfinite(vals))
        count_profile[i] = n

        if n < min_count:
            continue

        mean_profile[i] = float(np.nanmean(vals))

    return radii, mean_profile, count_profile


def show_res(radar, Cx, Cy, r_min=10, r_max=100, dr=5, min_count=20, save_profile_path=None):
    """
    Show the localized TC center and its radial mean profile.

    Parameters
    ----------
    save_profile_path : str or None
        If given, save np.stack([radii, mean_profile], axis=0) to this path.
    """
    H, W = radar.shape
    yy, xx = np.indices((H, W), dtype=np.float32)

    radii, mean_profile, count_profile = ring_mean_profile(
        radar,
        Cx,
        Cy,
        r_min=r_min,
        r_max=r_max,
        dr=dr,
        min_count=min_count,
        yy=yy,
        xx=xx
    )

    if save_profile_path is not None:
        profile_save = np.stack([radii, mean_profile], axis=0)
        np.save(save_profile_path, profile_save)
        print(f"[Radius profile] saved to: {save_profile_path}")

    panel_ratio = W / H

    fig_h = 8
    fig_w = fig_h * (panel_ratio + 1.0) * 1.15
    fig = plt.figure(figsize=(fig_w, fig_h))

    gs = fig.add_gridspec(1, 2, width_ratios=[panel_ratio, 1.0])

    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])

    fig.subplots_adjust(
        left=0.05,
        right=0.98,
        bottom=0.10,
        top=0.90,
        wspace=0.18
    )

    ax0.imshow(radar, cmap=radarcmap, norm=radarnorm, origin='upper')
    ax0.set_aspect('equal', adjustable='box')

    ax0.scatter(
        x=Cx,
        y=Cy,
        s=250,
        marker='X',
        facecolors='none',
        edgecolors='black',
        linewidths=1.2,
        label=f"TC Center ({int(Cx)}, {int(Cy)})"
    )
    ax0.legend(loc="lower right", frameon=False)
    ax0.set_title("Radar Map")

    ok = np.isfinite(mean_profile) & (count_profile >= min_count)

    ax1.plot(radii[ok], mean_profile[ok], marker='o')
    ax1.set_title("Radial Mean Profile")
    ax1.set_xlabel("radii (pixel)")
    ax1.set_ylabel("mean")

    plt.show()
    plt.close(fig)


