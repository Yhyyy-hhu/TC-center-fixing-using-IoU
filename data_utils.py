# 作者：姚弘祎
# 修改日期：
# 版本： structured version

import numpy as np
import os
from datetime import datetime, date, time, timedelta
from openpyxl.utils.datetime import from_excel
from openpyxl import load_workbook
from datetime import timedelta
import re
from config import (
    RADAR_SHAPE,
    INVALID_VALUE_THRESHOLD,
    LON_MIN,
    LON_MAX,
    LAT_MIN,
    LAT_MAX,
)


def Load_Data(path, shape=RADAR_SHAPE, invalid_value_threshold=INVALID_VALUE_THRESHOLD):
    """
    Load one radar reflectivity frame from a binary .ref file.

    Parameters
    ----------
    path : str
        File path of the radar frame.
    shape : tuple
        Radar array shape, default is (700, 900).
    invalid_value_threshold : int or float
        Values larger than or equal to this threshold are set to zero.

    Returns
    -------
    radar : ndarray
        Radar reflectivity array.
    """
    radar = np.fromfile(path, dtype=np.uint8).reshape(shape)
    radar[radar >= invalid_value_threshold] = 0
    return radar


def index_to_latlon(
    row,
    col,
    lon_min=LON_MIN,
    lon_max=LON_MAX,
    lat_min=LAT_MIN,
    lat_max=LAT_MAX,
    nlat=RADAR_SHAPE[0],
    nlon=RADAR_SHAPE[1],
):
    """
    Convert array index to the lower-left geographic coordinate of the grid cell.

    Parameters
    ----------
    row, col : int
        Array index. The first row is assumed to be the northernmost row.

    Returns
    -------
    lat, lon : float
        Latitude and longitude of the grid cell.
    """
    dlon = (lon_max - lon_min) / nlon
    dlat = (lat_max - lat_min) / nlat

    if not (0 <= row < nlat):
        raise ValueError(f"row={row} 超出范围 [0, {nlat})")
    if not (0 <= col < nlon):
        raise ValueError(f"col={col} 超出范围 [0, {nlon})")

    row_from_south = nlat - 1 - row

    lat = lat_min + row_from_south * dlat
    lon = lon_min + col * dlon

    return lat, lon


def parse_time_from_radar_path(path):
    """
    从 radar 文件名中解析时间。

    示例：
    cappi_ref_201007220136_2500_7.ref
    -> datetime(2010, 7, 22, 1, 36)
    """

    filename = os.path.basename(path)

    # cappi_ref_201007220136_2500_7.ref
    time_str = filename.split("_")[2]

    return datetime.strptime(time_str, "%Y%m%d%H%M")


def normalize_excel_time(value, excel_epoch=None):
    """
    将 Excel C 列中的时间统一转成 datetime，并去掉秒和微秒。

    兼容：
    1. datetime 对象
    2. date 对象
    3. Excel serial number
    4. 字符串：2010/7/22 1:00, 2010/7/22 1:00:00
    5. 字符串中存在多个空格、不可见空格
    """

    if value is None:
        return None

    # ---------- 1. datetime ----------
    if isinstance(value, datetime):
        return value.replace(second=0, microsecond=0)

    # ---------- 2. date ----------
    if isinstance(value, date):
        return datetime.combine(value, time.min).replace(second=0, microsecond=0)

    # ---------- 3. Excel serial number ----------
    if isinstance(value, (int, float)):
        try:
            if excel_epoch is not None:
                dt = from_excel(value, epoch=excel_epoch)
            else:
                dt = from_excel(value)

            return dt.replace(second=0, microsecond=0)
        except Exception:
            return None

    # ---------- 4. string ----------
    if isinstance(value, str):
        s = value.strip()

        # 替换各种不可见空格 / 非标准空格
        s = s.replace("\xa0", " ")
        s = s.replace("\u3000", " ")

        # 多个空格压成一个空格
        s = re.sub(r"\s+", " ", s)

        # 有些 Excel 字符串可能是 '2010/7/22 1:00:00 '
        fmts = [
            "%Y/%m/%d %H:%M",
            "%Y/%m/%d %H:%M:%S",
            "%Y-%m-%d %H:%M",
            "%Y-%m-%d %H:%M:%S",
            "%Y/%m/%d %I:%M:%S",
            "%Y/%m/%d %I:%M",
        ]

        for fmt in fmts:
            try:
                return datetime.strptime(s, fmt).replace(second=0, microsecond=0)
            except ValueError:
                pass

    return None


def write_center_to_excel(
    xlsx_path,
    sheet_name,
    target_time,
    lat,
    lon,
    time_col=3,
    lat_col=4,
    lon_col=5,
    start_row=2,
    save=True,
    tolerance_minutes=0
):
    """
    根据 target_time 匹配 Excel 中 C 列时间，并将 lat/lon 写入 D/E 列。

    默认要求精确匹配到分钟。
    如果 tolerance_minutes > 0，则允许一定分钟误差。
    """

    target_time = target_time.replace(second=0, microsecond=0)

    wb = load_workbook(xlsx_path)

    if sheet_name not in wb.sheetnames:
        raise ValueError(f"Sheet '{sheet_name}' not found in {xlsx_path}")

    ws = wb[sheet_name]

    matched_row = None
    candidate_times = []

    for row in range(start_row, ws.max_row + 1):
        raw_value = ws.cell(row=row, column=time_col).value

        excel_time = normalize_excel_time(
            raw_value,
            excel_epoch=wb.epoch
        )

        if excel_time is None:
            continue

        candidate_times.append((row, raw_value, excel_time))

        if tolerance_minutes == 0:
            if excel_time == target_time:
                matched_row = row
                break
        else:
            dt_diff = abs(excel_time - target_time)
            if dt_diff <= timedelta(minutes=tolerance_minutes):
                matched_row = row
                break

    if matched_row is None:
        print("\n[Debug] target_time =", target_time)
        print("[Debug] Nearby parsed times in Excel:")

        nearby = []
        for row, raw_value, excel_time in candidate_times:
            diff = abs(excel_time - target_time)
            if diff <= timedelta(hours=2):
                nearby.append((row, raw_value, excel_time, diff))

        for row, raw_value, excel_time, diff in nearby[:20]:
            print(
                f"row={row}, raw={repr(raw_value)}, "
                f"parsed={excel_time}, diff={diff}"
            )

        raise ValueError(
            f"Time {target_time} not found in sheet '{sheet_name}' column C."
        )

    ws.cell(row=matched_row, column=lat_col, value=float(lat))
    ws.cell(row=matched_row, column=lon_col, value=float(lon))

    if save:
        wb.save(xlsx_path)

    print(
        f"[Excel updated] row={matched_row}, "
        f"time={target_time}, lat={lat:.6f}, lon={lon:.6f}"
    )

    return matched_row