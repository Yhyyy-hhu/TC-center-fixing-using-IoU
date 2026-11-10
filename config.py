# 作者：姚弘祎
# 修改日期：
# 版本： structured version

"""
Configuration file for TC center localization.

Only keep parameters that are likely to be changed by users.
Method-specific hyperparameters can still be passed directly in main.py.
"""

# -----------------------------
# Demo data path
# -----------------------------
# PATH_DEMO = r'D:/Radar-Typhoon2/newdata/20232309苏拉/cappi_ref_202309021300_2500_11.ref'
PATH_DEMO = r'D:/Radar-Typhoon2/newdata/20171707洛克/cappi_ref_201707232000_2500_11.ref'  # Discussion 4.1 用
# -----------------------------
# Radar grid setting
# -----------------------------
RADAR_SHAPE = (700, 900)  # (nlat, nlon)
INVALID_VALUE_THRESHOLD = 81

# -----------------------------
# Geographic extent
# -----------------------------
LON_MIN = 108.5
LON_MAX = 117.5
LAT_MIN = 19.0
LAT_MAX = 26.0

# -----------------------------
# Default optimization setting
# -----------------------------
USE_OPTI = True
DE_MAXITER = 100
CORE_ALPHA = 0.5
CORE_DBZ_THR = 30.0
CORE_MIN_LOW_RATIO = 0.6
