"""管道全局参数。标 ⏳ 的留实测调。"""

# --- 路径 ---
RAW_PANEL = "data/panel_long.parquet"      # download_us.py --consolidate 产出（raw OHLCV）
CLEAN_DIR = "data/clean"
CLEAN_PANEL = f"{CLEAN_DIR}/clean_panel.parquet"
UNIVERSE = f"{CLEAN_DIR}/universe.parquet"
V1_TENSOR = f"{CLEAN_DIR}/v1_tensor.npz"
SPLITS_JSON = f"{CLEAN_DIR}/splits.json"

# --- 阶段 0 日历 ---
CAL_MIN_SYMBOLS = 20        # 当日 >= 这么多标的交易，才算有效交易日（滤乌龙日期）

# --- 阶段 1 删错 ---
OHLC_TOL = 0.001            # OHLC 自洽容差（相对），超出视为记录错误

# --- 阶段 2 身份切分 ---
GAP_THRESH = 60            # 内部空洞 > 这么多交易日 → 切成新身份 segment
MIN_SEG_LEN = 20          # segment 短于这么多 bar → 丢弃（太短没法归一/建窗口）

# --- 阶段 3 拆股还原 ---
# open 比是主判据：拆股日开盘已反映纯拆股(无隔夜真实跳)，会精确落在干净分数上；
# 财报跳空不会。所以 open 容差收紧、close 放松(容当日真实涨跌)。
SPLIT_TOL_OPEN = 0.03     # open 比与"干净分数"的相对容差（主判据，收紧防误判）
SPLIT_TOL_CLOSE = 0.15    # close 比容差（放松，容当日真实 intraday 移动）
SPLIT_MIN_MOVE = 0.30     # |log open 比| 至少这么大（≥~30%，只碰明确大拆股）
SPLIT_MAX_K = 12          # 候选拆股比例的最大整数（n:1 / 1:n）
MAX_SPLITS_PER_SEG = 15   # 一段检出>这么多拆股 = 误判主导(仙股/权证)，不调整并标 drop
                          # （真实公司一生拆股 0~6 次，>15 必是假阳）

# --- 阶段 4 质量 ---
QUALITY_MIN_MED_VOL = 1.0   # 中位成交量 < 1 股 → vol≈0 不可信报价，drop（只删这一档）

# --- 阶段 6 归一化 ---
SLOW_WIN = 252            # 慢速因果窗口（~1 交易年）
NORM_MIN_PERIODS = 20     # 慢窗最少样本数才出统计

# --- 阶段 8 切分（可配）---
# 为"厚横截面"把训练起点设在 2012：512 universe(按整体成交额选) 多数 2012 后才活跃，
# 早年只有 ~97 slot/天会让模型学不到横截面联合结构。代价=丢 2008，换公平的信号检测。
TRAIN_START = "2012-01-01"
TRAIN_END = "2020-12-31"
VAL_END = "2022-12-31"
# holdout = VAL_END 之后 ~ 至今

# --- v1 ---
V1_N = 512                # v1 universe slot 数（成交额前 512）

# --- 建模侧（dataset 会用；具体值留实测）---
T_WINDOW = 128            # ⏳ context 窗口长度（交易日）
DT_MIN = 1               # ⏳ Δt 下限（交易日）
DT_MAX = 20              # ⏳ Δt 上限（交易日）

# 输入通道顺序（同构 [.,.,.,5]）
CHANNELS = ["open", "high", "low", "close", "volume"]
