# 作者：姚弘祎
# 修改日期：
# 版本： structured version

import math
import random

import numpy as np
from scipy import ndimage
from scipy.optimize import differential_evolution
from scipy.spatial import ConvexHull
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

from plot_utils import (
    interactive_cluster_view,
    radarcmap,
    radarnorm,
)
from config import RADAR_SHAPE
from plot_utils import interactive_cluster_view

def get_default_label_after_manual_range(label_img, orientation="largest"):
    """
    Get default cluster label after manual truncation.

    Parameters
    ----------
    label_img : ndarray
        Label image. Background should be -1.

    orientation : str
        "largest" or "smallest".

    Returns
    -------
    default_label : int or None
    """

    valid = label_img[label_img >= 0]

    if valid.size == 0:
        return None

    uniq, cnt = np.unique(valid, return_counts=True)

    if orientation == "largest":
        return int(uniq[np.argmax(cnt)])
    elif orientation == "smallest":
        return int(uniq[np.argmin(cnt)])
    else:
        raise ValueError("orientation must be 'largest' or 'smallest'")


def Clustering(frame, dbz_thr=30, mode='greater', orientation='largest'):
    """
    Cluster radar pixels according to reflectivity threshold.

    mode='greater': use DBSCAN on pixels with frame >= dbz_thr.
    mode='equal'  : use connected-component labeling on pixels with frame == dbz_thr.
    """
    frame = np.asarray(frame)

    if mode == 'greater':
        mask = np.isfinite(frame) & (frame >= dbz_thr)
        ys, xs = np.where(mask)
        label_img = np.full(frame.shape, fill_value=-1, dtype=np.int32)

        if len(xs) == 0:
            return label_img, None

        points_xy = np.column_stack([xs, ys]).astype(np.float32)

        db = DBSCAN(
            eps=5.0,
            min_samples=20,
            metric="euclidean",
            n_jobs=-1
        )
        labels = db.fit_predict(points_xy)
        label_img[ys, xs] = labels

        valid = labels[labels >= 0]
        if valid.size == 0:
            return label_img, None

        uniq, cnt = np.unique(valid, return_counts=True)

        if orientation == 'largest':
            default_label = int(uniq[np.argmax(cnt)])
        elif orientation == 'smallest':
            default_label = int(uniq[np.argmin(cnt)])
        else:
            raise ValueError("orientation must be 'largest' or 'smallest'")

    elif mode == 'equal':
        mask = np.isfinite(frame) & (frame == dbz_thr)

        structure = np.ones((3, 3), dtype=int)
        label_img, num_features = ndimage.label(mask, structure=structure)

        if num_features == 0:
            return label_img.astype(np.int32) - 1, None

        counts = np.bincount(label_img.ravel())[1:]
        uniq = np.arange(1, len(counts) + 1)

        label_img = label_img.astype(np.int32)
        label_img[label_img == 0] = -1
        default_label = int(uniq[np.argmin(counts)])

    else:
        raise ValueError("Mode error")

    return label_img, default_label


def ask_yes_no(prompt, default='n'):
    """
    Command-line yes/no question.
    """
    default = default.lower()

    while True:
        s = input(f"{prompt} [y/n] (default={default}): ").strip().lower()

        if s == '':
            s = default

        if s in ['y', 'yes']:
            return True
        elif s in ['n', 'no']:
            return False
        else:
            print("Please input y or n.")

def get_default_label_after_manual_range(label_img, orientation="largest"):
    """
    Get default cluster label after manual truncation.

    Parameters
    ----------
    label_img : ndarray
        Label image. Background should be -1.

    orientation : str
        "largest" or "smallest".

    Returns
    -------
    default_label : int or None
    """

    valid = label_img[label_img >= 0]

    if valid.size == 0:
        return None

    uniq, cnt = np.unique(valid, return_counts=True)

    if orientation == "largest":
        return int(uniq[np.argmax(cnt)])
    elif orientation == "smallest":
        return int(uniq[np.argmin(cnt)])
    else:
        raise ValueError("orientation must be 'largest' or 'smallest'")


def interactive_manual_circle_view(
    radar,
    dbz_thr=30,
    mode="greater",
    orientation="largest",
    title="Manual circular range before cluster selection"
):
    """
    Manually define a circular range, preview the intersection between
    DBSCAN clusters and the manual circle, and then call interactive_cluster_view
    for final cluster selection.

    Logic
    -----
    1. Run DBSCAN clustering on radar according to dbz_thr and mode.
    2. Manually draw a circular range.
    3. Show the intersection:

           label_img_view = label_img.copy()
           label_img_view[~manual_region] = -1

       in the right panel.
    4. After closing the manual circle figure, call interactive_cluster_view()
       on label_img_view, so the user can still select the target cluster.

    Parameters
    ----------
    radar : ndarray, shape (H, W)
        Radar reflectivity field.

    dbz_thr : float
        Reflectivity threshold for clustering.

    mode : str
        "greater" or "equal", passed to Clustering().

    orientation : str
        "largest" or "smallest", used to determine the default cluster.

    title : str
        Figure title.

    Returns
    -------
    selected_region : ndarray, bool
        Final selected cluster region after manual truncation and user selection.

    selected_radar : ndarray
        Radar field masked by selected_region.

    measure_dist : float or None
        Distance measured in interactive_cluster_view(), if any.

    manual_region : ndarray, bool
        The manual circular mask.

    circle_params : dict or None
        {"cx": cx, "cy": cy, "r": r}. None if no circle is drawn.

    label_img_view : ndarray
        Cluster label image after manual truncation.
    """

    H, W = radar.shape

    # ======================================================
    # 1. First run DBSCAN clustering
    # ======================================================
    label_img, default_label = Clustering(
        frame=radar,
        dbz_thr=dbz_thr,
        mode=mode,
        orientation=orientation
    )

    if default_label is None:
        raise ValueError(
            f"No valid cluster found for dbz_thr={dbz_thr}, mode={mode}."
        )

    # This will be updated after drawing the circle
    manual_region = np.ones((H, W), dtype=bool)
    label_img_view = label_img.copy()
    default_label_view = default_label

    circle_params = {
        "cx": None,
        "cy": None,
        "r": None
    }

    # ======================================================
    # 2. Create two-panel figure
    # ======================================================
    fig, axes = plt.subplots(
        1, 2,
        figsize=(14, 7)
    )

    ax_left, ax_right = axes

    # -----------------------------
    # Left: original radar
    # -----------------------------
    im_left = ax_left.imshow(
        radar,
        origin="upper",
        cmap=radarcmap,
        norm=radarnorm
    )

    ax_left.set_title(
        title + "\nLeft click: center + radius point | Middle click: clear",
        pad=10
    )
    ax_left.set_aspect("equal", adjustable="box")

    # -----------------------------
    # Right: DBSCAN cluster ∩ circle
    # -----------------------------
    im_right = ax_right.imshow(
        label_img_view,
        origin="upper",
        cmap="tab20c"
    )

    ax_right.set_title(
        f"DBSCAN clusters ≥ {dbz_thr} dBZ ∩ manual circle",
        pad=10
    )
    ax_right.set_aspect("equal", adjustable="box")

    fig.subplots_adjust(
        left=0.05,
        right=0.98,
        top=0.90,
        bottom=0.14,
        wspace=0.12
    )

    # radar colorbar for left panel
    cbar_left = fig.colorbar(
        im_left,
        ax=ax_left,
        orientation="horizontal",
        fraction=0.045,
        pad=0.08
    )
    cbar_left.set_label("reflectivity (dBZ)")

    # cluster colorbar for right panel
    cbar_right = fig.colorbar(
        im_right,
        ax=ax_right,
        orientation="horizontal",
        fraction=0.045,
        pad=0.08
    )
    cbar_right.set_label("cluster ID")

    click_points = []

    left_artists = []
    right_artists = []

    def build_circle_mask(cx, cy, r):
        yy, xx = np.indices((H, W))
        return (xx - cx) ** 2 + (yy - cy) ** 2 <= r ** 2

    def update_intersection_preview(circle_mask=None):
        """
        Update the right panel with:

            DBSCAN label image ∩ manual circle
        """

        nonlocal manual_region
        nonlocal label_img_view
        nonlocal default_label_view

        if circle_mask is None:
            manual_region = np.ones((H, W), dtype=bool)
        else:
            manual_region = circle_mask

        label_img_view = label_img.copy()
        label_img_view[~manual_region] = -1

        default_label_view = get_default_label_after_manual_range(
            label_img_view,
            orientation=orientation
        )

        im_right.set_data(label_img_view)

        # remove old right-panel artists
        for a in right_artists:
            a.remove()
        right_artists.clear()

        # draw circle on right panel
        if circle_params["r"] is not None:
            circle_right = Circle(
                (circle_params["cx"], circle_params["cy"]),
                circle_params["r"],
                edgecolor="blue",
                facecolor="none",
                linewidth=2.5,
                linestyle="--"
            )
            ax_right.add_patch(circle_right)
            right_artists.append(circle_right)

        # draw the default cluster outline after truncation
        if default_label_view is not None:
            default_mask = label_img_view == default_label_view

            if np.any(default_mask):
                cs = ax_right.contour(
                    default_mask.astype(np.uint8),
                    levels=[0.5],
                    linewidths=1.5,
                    colors="red"
                )
                right_artists.extend(list(cs.collections))

        if default_label_view is None:
            ax_right.set_title(
                f"No valid DBSCAN cluster remains after manual circle",
                pad=10
            )
        else:
            ax_right.set_title(
                f"DBSCAN clusters ≥ {dbz_thr} dBZ ∩ manual circle\n"
                f"default label = {default_label_view}",
                pad=10
            )

        fig.canvas.draw_idle()

    def clear_circle():
        nonlocal circle_params

        for a in left_artists:
            a.remove()
        left_artists.clear()

        for a in right_artists:
            a.remove()
        right_artists.clear()

        click_points.clear()

        circle_params = {
            "cx": None,
            "cy": None,
            "r": None
        }

        update_intersection_preview(circle_mask=None)

        print("[Manual circle] cleared")
        fig.canvas.draw_idle()

    def on_click(event):
        nonlocal circle_params

        if event.xdata is None or event.ydata is None:
            return

        # only define the circle on the left panel
        if event.inaxes != ax_left:
            return

        x = float(event.xdata)
        y = float(event.ydata)

        if x < 0 or x >= W or y < 0 or y >= H:
            return

        # middle click: clear
        if event.button == 2:
            clear_circle()
            return

        # only left click defines circle
        if event.button != 1:
            return

        # if a circle already exists, start a new one
        if circle_params["r"] is not None:
            clear_circle()

        click_points.append((x, y))

        p_left = ax_left.plot(
            x, y,
            marker="o",
            markersize=7,
            markerfacecolor="cyan",
            markeredgecolor="blue",
            linestyle="none"
        )[0]

        left_artists.append(p_left)

        if len(click_points) == 1:
            print(f"[Manual circle] center = ({x:.1f}, {y:.1f})")

        elif len(click_points) == 2:
            (cx, cy), (x2, y2) = click_points

            r = float(np.hypot(x2 - cx, y2 - cy))

            circle_fill_left = Circle(
                (cx, cy),
                r,
                edgecolor="blue",
                facecolor="cyan",
                alpha=0.18,
                linewidth=2.0
            )

            circle_outline_left = Circle(
                (cx, cy),
                r,
                edgecolor="blue",
                facecolor="none",
                linewidth=2.5,
                linestyle="--"
            )

            ax_left.add_patch(circle_fill_left)
            ax_left.add_patch(circle_outline_left)

            left_artists.extend([
                circle_fill_left,
                circle_outline_left
            ])

            circle_params = {
                "cx": float(cx),
                "cy": float(cy),
                "r": float(r)
            }

            circle_mask = build_circle_mask(cx, cy, r)

            update_intersection_preview(circle_mask=circle_mask)

            print(
                f"[Manual circle] center = ({cx:.1f}, {cy:.1f}), "
                f"r = {r:.2f} pixels"
            )

            click_points.clear()

        fig.canvas.draw_idle()

    fig.canvas.mpl_connect("button_press_event", on_click)

    plt.show()

    # ======================================================
    # 3. After closing the circle figure, call interactive_cluster_view()
    # ======================================================
    if default_label_view is None:
        raise ValueError(
            "No valid cluster remains after manual circular truncation."
        )

    print(
        "[Manual circle] Now select target cluster from the truncated DBSCAN result."
    )

    selected_labels, measure_dist = interactive_cluster_view(
        radar=radar,
        label_img=label_img_view,
        largest_label=default_label_view
    )

    selected_region = np.isin(label_img_view, list(selected_labels))
    selected_radar = np.where(selected_region, radar, np.nan)

    if circle_params["r"] is None:
        circle_params_return = None
    else:
        circle_params_return = circle_params

    return (
        selected_region,
        selected_radar,
        measure_dist,
        manual_region,
        circle_params_return,
        label_img_view
    )

def Select_Clusters_With_Refine(
    radar,
    use_manual_range=False
):
    """
    step1: dbz > 30 cluster selection
    step2: optional refine dbz > 40
    step3: optional refine eye dbz == 0

    If use_manual_range=True, the dbz > 30 selection follows:

        DBSCAN clustering
        -> manual circular truncation
        -> preview cluster ∩ circle
        -> interactive_cluster_view on truncated clusters
    """

    # =========================================
    # step1 必须：dbz > 30
    # =========================================
    if use_manual_range:
        (
            selected_region30,
            selected_radar30,
            measure30,
            manual_region30,
            circle_params30,
            label_img30_view
        ) = interactive_manual_circle_view(
            radar=radar,
            dbz_thr=30,
            mode="greater",
            orientation="largest",
            title="Manual circular truncation for dbz > 30 clusters"
        )

        print("[Manual range 30] circle_params =", circle_params30)

    else:
        label_img30, default_label30 = Clustering(
            frame=radar,
            dbz_thr=30,
            mode="greater",
            orientation="largest"
        )

        selected30, measure30 = interactive_cluster_view(
            radar=radar,
            label_img=label_img30,
            largest_label=default_label30
        )

        selected_region30 = np.isin(label_img30, list(selected30))
        selected_radar30 = np.where(selected_region30, radar, np.nan)

    # =========================================
    # step2 可选：dbz > 40 refine
    # =========================================
    selected_region40 = None
    selected_radar40 = None
    measure40 = None

    do_refine40 = ask_yes_no(
        "Do you want to refine by selecting clusters with dbz > 40?",
        default="n"
    )

    if do_refine40:
        label_img40, default_label40 = Clustering(
            frame=radar,
            dbz_thr=40,
            mode="greater",
            orientation="largest"
        )

        selected40, measure40 = interactive_cluster_view(
            radar=radar,
            label_img=label_img40,
            largest_label=default_label40
        )

        selected_region40 = np.isin(label_img40, list(selected40))
        selected_radar40 = np.where(selected_region40, radar, np.nan)

    # =========================================
    # step3 可选：eye refine dbz == 0
    # =========================================
    selected_region00 = None
    selected_radar00 = None

    do_refine00 = ask_yes_no(
        "Do you have confidence to select eye with dbz == 0?",
        default="n"
    )

    if do_refine00:
        label_img00, default_label00 = Clustering(
            frame=radar,
            dbz_thr=0,
            mode="equal",
            orientation="smallest"
        )

        selected00, _ = interactive_cluster_view(
            radar=radar,
            label_img=label_img00,
            largest_label=default_label00
        )

        selected_region00 = np.isin(label_img00, list(selected00))
        selected_radar00 = np.where(selected_region00, radar, np.nan)

    return (
        selected_region30, selected_radar30, measure30,
        selected_region40, selected_radar40, measure40,
        selected_region00, selected_radar00
    )


def _circle_from(p1, p2):
    cx = (p1[0] + p2[0]) / 2.0
    cy = (p1[1] + p2[1]) / 2.0
    r = math.hypot(p1[0] - cx, p1[1] - cy)
    return (cx, cy, r)


def _circle_from3(p1, p2, p3):
    ax, ay = p1
    bx, by = p2
    cx, cy = p3
    d = 2.0 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
    if abs(d) < 1e-12:
        return None
    ux = ((ax * ax + ay * ay) * (by - cy) + (bx * bx + by * by) * (cy - ay) + (cx * cx + cy * cy) * (ay - by)) / d
    uy = ((ax * ax + ay * ay) * (cx - bx) + (bx * bx + by * by) * (ax - cx) + (cx * cx + cy * cy) * (bx - ax)) / d
    r = math.hypot(ax - ux, ay - uy)
    return (ux, uy, r)


def _is_in_circle(c, p, eps=1e-7):
    return math.hypot(p[0] - c[0], p[1] - c[1]) <= c[2] + eps


def minimum_enclosing_circle(points, seed=0):
    pts = points[:]
    random.Random(seed).shuffle(pts)

    c = None
    for i, p in enumerate(pts):
        if c is not None and _is_in_circle(c, p):
            continue
        c = (p[0], p[1], 0.0)
        for j in range(i):
            q = pts[j]
            if _is_in_circle(c, q):
                continue
            c = _circle_from(p, q)
            for k in range(j):
                r = pts[k]
                if _is_in_circle(c, r):
                    continue
                c3 = _circle_from3(p, q, r)
                if c3 is None:
                    c = min(
                        (_circle_from(p, q), _circle_from(p, r), _circle_from(q, r)),
                        key=lambda cc: cc[2]
                    )
                else:
                    c = c3
    return c


def mec_from_mask_via_hull(mask_bool, seed=0):
    ys0, xs0 = np.where(mask_bool)
    if xs0.size == 0:
        raise ValueError("largest_mask 为空，无法求外接圆。")

    pts = np.column_stack([xs0, ys0]).astype(np.float64)

    if pts.shape[0] <= 3:
        return minimum_enclosing_circle([tuple(p) for p in pts], seed=seed)

    hull = ConvexHull(pts)
    hull_pts = pts[hull.vertices]
    return minimum_enclosing_circle([tuple(p) for p in hull_pts], seed=seed)


def circle_mask(cx, cy, r, shape=RADAR_SHAPE):
    yy, xx = np.indices(shape)
    return (xx - cx) ** 2 + (yy - cy) ** 2 <= r ** 2


def calc_iou(mask1, mask2):
    inter = np.sum(mask1 & mask2)
    union = np.sum(mask1 | mask2)
    if union == 0:
        return np.nan
    return inter / union


def tc_IoU_score(
    cx,
    cy,
    r30,
    r40,
    r00,
    selected_region30,
    selected_region40=None,
    selected_region00=None,
    radar=None,
    w30=1.0,
    w40=1.0,
    w00=1.0,
    core_alpha=0.5,
    dbz_thr=30,
    core_min_low_ratio=0.5,
):
    """
    Compute the unnormalized weighted IoU score for TC center localization.
    """
    score_sum = 0.0
    detail = {}
    shape = selected_region30.shape

    mask30 = circle_mask(cx, cy, r30, shape=shape)
    iou30 = calc_iou(mask30, selected_region30)
    detail["iou30"] = iou30
    if not np.isnan(iou30):
        score_sum += w30 * iou30

    if selected_region40 is not None and r40 is not None:
        if radar is None:
            raise ValueError("radar must be provided when using r40 core hollow constraint.")

        r_core = core_alpha * r40
        core_mask = circle_mask(cx, cy, r_core, shape=shape)

        n_core = int(np.sum(core_mask))
        if n_core == 0:
            detail["core_low_ratio40"] = np.nan
            detail["core_hollow_pass40"] = False
            detail["iou40"] = np.nan
            detail["score"] = np.nan
            return np.nan, detail

        core_low_ratio40 = np.mean(radar[core_mask] < dbz_thr)

        detail["core_radius40"] = float(r_core)
        detail["core_dbz_thr40"] = float(dbz_thr)
        detail["core_low_ratio40"] = float(core_low_ratio40)
        detail["core_hollow_pass40"] = bool(core_low_ratio40 >= core_min_low_ratio)

        if core_low_ratio40 < core_min_low_ratio:
            detail["iou40"] = np.nan
            detail["score"] = np.nan
            return np.nan, detail

        mask40 = circle_mask(cx, cy, r40, shape=shape)
        iou40 = calc_iou(mask40, selected_region40)
        detail["iou40"] = iou40
        if not np.isnan(iou40):
            score_sum += w40 * iou40
    else:
        detail["iou40"] = None
        detail["core_radius40"] = None
        detail["core_dbz_thr40"] = None
        detail["core_low_ratio40"] = None
        detail["core_hollow_pass40"] = None

    if selected_region00 is not None and r00 is not None:
        mask00 = circle_mask(cx, cy, r00, shape=shape)
        iou00 = calc_iou(mask00, selected_region00)
        detail["iou00"] = iou00
        if not np.isnan(iou00):
            score_sum += w00 * iou00
    else:
        detail["iou00"] = None

    detail["score"] = score_sum
    return score_sum, detail


def find_center(
    selected_region30,
    selected_region40=None,
    selected_region00=None,
    radar=None,
    use_opti=False,
    x_step=5,
    y_step=5,
    minr30=None,
    minr40=None,
    w30=1.0,
    w40=1.0,
    w00=1.0,
    de_maxiter=60,
    de_popsize=12,
    de_seed=0,
    core_alpha=0.5,
    dbz_thr=30,
    core_min_low_ratio=0.5,
):
    """
    Search for the TC center and characteristic radii.
    """
    yy_c, xx_c = np.where(selected_region30)
    if len(xx_c) == 0:
        raise ValueError("selected_region30 is empty.")

    xmin, xmax = xx_c.min(), xx_c.max()
    ymin, ymax = yy_c.min(), yy_c.max()

    has40 = selected_region40 is not None
    has00 = selected_region00 is not None

    if not use_opti:
        r30_list = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90]

        best_score = -np.inf
        best_result = None

        for cy in range(ymin, ymax + 1, y_step):
            for cx in range(xmin, xmax + 1, x_step):
                for r30 in r30_list:
                    r40_candidates = [None] if not has40 else range(5, r30 + 1)

                    for r40 in r40_candidates:
                        if not has00:
                            r00_candidates = [None]
                        else:
                            r00_upper = r40 if has40 else r30
                            r00_candidates = range(5, r00_upper + 1)

                        for r00 in r00_candidates:
                            score, detail = tc_IoU_score(
                                cx=cx,
                                cy=cy,
                                r30=r30,
                                r40=r40,
                                r00=r00,
                                selected_region30=selected_region30,
                                selected_region40=selected_region40,
                                selected_region00=selected_region00,
                                radar=radar,
                                w30=w30,
                                w40=w40,
                                w00=w00,
                                core_alpha=core_alpha,
                                dbz_thr=dbz_thr,
                                core_min_low_ratio=core_min_low_ratio
                            )

                            if np.isnan(score):
                                continue

                            if score > best_score:
                                best_score = score
                                best_result = {
                                    "cx": int(cx),
                                    "cy": int(cy),
                                    "r30": int(r30),
                                    "r40": None if r40 is None else int(r40),
                                    "r00": None if r00 is None else int(r00),
                                    "score": float(score),
                                    "detail": detail
                                }

        return best_result

    r30_min = 30
    if minr30 is not None:
        r30_max = minr30
    else:
        r30_max = 300

    bounds = [
        (xmin, xmax),
        (ymin, ymax),
        (r30_min, r30_max),
    ]
    if has40:
        bounds.append((0.0, 1.0))
    if has00:
        bounds.append((0.0, 1.0))

    def decode_params(p):
        idx = 0
        cx = int(round(p[idx])); idx += 1
        cy = int(round(p[idx])); idx += 1
        r30 = int(round(p[idx])); idx += 1

        r40 = None
        r00 = None

        if has40:
            u40 = p[idx]; idx += 1
            if minr40 is not None:
                r40_min_eff = max(30, minr40)
                r40_min_eff = min(r40_min_eff, r30 - 1)
                r40 = int(round(r40_min_eff + u40 * (r30 - r40_min_eff)))
            else:
                r40 = int(round(30 + u40 * (r30 - 30)))

        if has00:
            u00 = p[idx]; idx += 1
            if has40:
                r00 = int(round(5 + u00 * (r40 - 5)))
            else:
                r00 = int(round(5 + u00 * (r30 - 5)))

        return cx, cy, r30, r40, r00

    def objective(p):
        cx, cy, r30, r40, r00 = decode_params(p)

        score, detail = tc_IoU_score(
            cx=cx,
            cy=cy,
            r30=r30,
            r40=r40,
            r00=r00,
            selected_region30=selected_region30,
            selected_region40=selected_region40,
            selected_region00=selected_region00,
            radar=radar,
            w30=w30,
            w40=w40,
            w00=w00,
            core_alpha=core_alpha,
            dbz_thr=dbz_thr,
            core_min_low_ratio=core_min_low_ratio
        )

        if np.isnan(score):
            return 1e6

        return -score

    result = differential_evolution(
        objective,
        bounds=bounds,
        strategy='best1bin',
        maxiter=de_maxiter,
        popsize=de_popsize,
        tol=1e-3,
        mutation=(0.5, 1.0),
        recombination=0.7,
        polish=False,
        seed=de_seed,
        workers=1
    )

    print("result.fun =", result.fun)
    print("result.x   =", result.x)

    cx, cy, r30, r40, r00 = decode_params(result.x)

    best_score, detail = tc_IoU_score(
        cx=cx,
        cy=cy,
        r30=r30,
        r40=r40,
        r00=r00,
        selected_region30=selected_region30,
        selected_region40=selected_region40,
        selected_region00=selected_region00,
        radar=radar,
        w30=w30,
        w40=w40,
        w00=w00,
        core_alpha=core_alpha,
        dbz_thr=dbz_thr,
        core_min_low_ratio=core_min_low_ratio
    )

    best_result = {
        "cx": cx,
        "cy": cy,
        "r30": r30,
        "r40": r40,
        "r00": r00,
        "score": float(best_score),
        "detail": detail,
        "de_result": result
    }

    return best_result
