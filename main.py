# 作者：姚弘祎
# 修改日期：
# 版本： structured version

from config import (
    PATH_DEMO,
    USE_OPTI,
    DE_MAXITER,
    CORE_ALPHA,
    CORE_DBZ_THR,
    CORE_MIN_LOW_RATIO,
)
from data_utils import Load_Data, index_to_latlon, parse_time_from_radar_path, write_center_to_excel
from model_utils import Select_Clusters_With_Refine, find_center
from plot_utils import show_res


def main():
    # ------------------
    # 0. Load data
    # ------------------
    radar = Load_Data(PATH_DEMO)

    # ------------------
    # 1. Select radar regions interactively
    # ------------------
    (
        selected_region30,
        selected_radar30,
        dist30,
        selected_region40,
        selected_radar40,
        dist40,
        selected_region00,
        selected_radar00,
    ) = Select_Clusters_With_Refine(radar, use_manual_range=True)

    # ------------------
    # 2. Search TC center
    # ------------------
    res = find_center(
        selected_region30,
        selected_region40,
        selected_region00,
        radar=radar,
        use_opti=USE_OPTI,
        minr30=1.1 * dist30 / 2 if dist30 is not None else None,
        minr40=1.0 * dist40 / 2 if dist40 is not None else None,
        de_maxiter=DE_MAXITER,
        core_alpha=CORE_ALPHA,
        dbz_thr=CORE_DBZ_THR,
        core_min_low_ratio=CORE_MIN_LOW_RATIO,
    )

    print(res)
    print(res["cx"], res["cy"])

    # ------------------
    # 3. Convert index to lat/lon
    # ------------------
    lat, lon = index_to_latlon(res["cy"], res["cx"])
    print(lat, lon)

    xlsx_path = r"D:/Radar-Typhoon2/ibtracsforTCcenter.xlsx"
    sheet_name = "2309泰利"

    target_time = parse_time_from_radar_path(PATH_DEMO)

    # write_center_to_excel(
    #     xlsx_path=xlsx_path,
    #     sheet_name=sheet_name,
    #     target_time=target_time,
    #     lat=lat,
    #     lon=lon,
    #     tolerance_minutes=1
    # )

    # ------------------
    # 4. Show result
    # ------------------

    folder = PATH_DEMO.split("/")[-2]
    time_short = PATH_DEMO.split("_")[2][4:]
    save_name = f"{folder}_{time_short}_radius_profile.npy"

    show_res(
        radar=radar,
        Cx=res["cx"],
        Cy=res["cy"],
        r_min=10,
        r_max=150,
        dr=5,
        min_count=20,
        save_profile_path=save_name
    )


if __name__ == "__main__":
    main()
