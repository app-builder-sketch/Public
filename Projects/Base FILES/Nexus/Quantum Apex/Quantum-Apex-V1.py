// This Pine Script® code is subject to the terms of the Mozilla Public License 2.0 at https://mozilla.org/MPL/2.0/
// © DarkPoolCrypto
//@version=6
indicator("Quantum Apex: Unified Field [SMC + Vector + MCM Brain] v7.4", shorttitle="Quantum Apex", overlay=true, precision=2,
     max_boxes_count=500, max_lines_count=500, max_labels_count=500, explicit_plot_zorder=true)

//──────────────────────────────────────────────────────────────────────────────
// SECTION: UI & STYLE SETTINGS
//──────────────────────────────────────────────────────────────────────────────
string GRP_UI = "🎨 UI & HUD"
string TABLE_POS   = input.string("Bottom Right", "HUD Position", options=["Top Right", "Bottom Right", "Top Left", "Bottom Left"], group=GRP_UI)
string TABLE_SIZE  = input.string("Small", "HUD Size", options=["Tiny", "Small", "Normal"], group=GRP_UI)
bool   SHOW_LABELS = input.bool(false, "Show Chart Labels", group=GRP_UI, tooltip="Toggle text for BOS/FVG. Keep OFF for clean look.")
bool   SHOW_BARCOL = input.bool(true, "Color Bars by Vector", group=GRP_UI)
bool   SHOW_WTRMRK = input.bool(false, "Show Vector Watermark", group=GRP_UI, tooltip="Plots a scaled marker at the bottom of the chart.")

//──────────────────────────────────────────────────────────────────────────────
// SECTION: STABILITY / OBJECT LIMITS
//──────────────────────────────────────────────────────────────────────────────
string G_STAB = "🧱 Stability / Object Limits"
int MAX_WORMHOLE_BOXES = input.int(120, "Max Wormhole Boxes (Total)", minval=0, group=G_STAB, tooltip="0 disables wormhole boxes.")
int MAX_STRUCT_LINES   = input.int(120, "Max Structure Lines", minval=0, group=G_STAB, tooltip="0 disables structure lines.")
int MAX_STRUCT_LABELS  = input.int(120, "Max Structure Labels", minval=0, group=G_STAB, tooltip="0 disables structure labels.")
int MAX_EVENT_LABELS   = input.int(120, "Max Event Horizon Labels", minval=0, group=G_STAB, tooltip="0 disables event horizon dots.")

//──────────────────────────────────────────────────────────────────────────────
// SECTION: APEX PHYSICS ENGINE (The Brain)
//──────────────────────────────────────────────────────────────────────────────
string G_PHYS = "⚛️ Apex Physics Engine"
float EFF_SUPER  = input.float(0.60, "Superconductor Threshold", minval=0.1, step=0.05, group=G_PHYS, tooltip="The efficiency required to trigger a trend state.")
float EFF_RESIST = input.float(0.30, "Resistive Threshold", minval=0.0, step=0.05, group=G_PHYS, tooltip="Below this value, the market is considered in a chop/range.")
int   LEN_VEC    = input.int(14, "Vector Lookback", minval=2, group=G_PHYS)
int   VOL_NORM   = input.int(55, "Volume Normalization", minval=10, group=G_PHYS)
string SM_TYPE   = input.string("EMA", "Smoothing", options=["EMA", "SMA", "RMA", "VWMA"], group=G_PHYS)
int   LEN_SM     = input.int(5, "Smoothing Length", minval=1, group=G_PHYS)

//──────────────────────────────────────────────────────────────────────────────
// SECTION: SPACETIME GEOMETRY (SMC)
//──────────────────────────────────────────────────────────────────────────────
string G_SMC = "🌌 Spacetime (SMC)"
bool WORMHOLES      = input.bool(true, "Wormholes (FVG)", group=G_SMC)
bool EVENT_HORIZONS = input.bool(true, "Event Horizons (OB)", group=G_SMC)
bool STRUCTURE      = input.bool(true, "Structure (BOS/CHoCH)", group=G_SMC)
int  SMC_LOOKBACK   = input.int(5, "Pivot Lookback", minval=2, group=G_SMC)

//──────────────────────────────────────────────────────────────────────────────
// SECTION: MARKET CYCLE MASTER (SUPERTRND + MTF + FILTERS + RISK)
//──────────────────────────────────────────────────────────────────────────────
string G_MCM = "🌊 Market Cycle Master (Integrated)"

// --- Trend Engine (SuperTrend-style line) ---
float  st_mult   = input.float(3.0, "Volatility Factor", minval=1.0, maxval=10.0, step=0.1, group=G_MCM)
int    st_period = input.int(10, "Volatility Lookback", minval=1, group=G_MCM)
source st_src    = input.source(hlc3, "Source Price", group=G_MCM)

// --- Institutional Context (MTF) ---
bool   use_mtf = input.bool(true, "Filter with Higher Timeframe", group=G_MCM, tooltip="Hard or soft MTF alignment (see mode).")
string mtf_tf  = input.timeframe("240", "Higher Timeframe", group=G_MCM)
string MTF_MODE = input.string("Hard", "MTF Mode", options=["Hard", "Soft"], group=G_MCM, tooltip="Hard: gate entries by HTF. Soft: penalize confidence when misaligned.")
float  MTF_PENALTY = input.float(0.25, "MTF Soft Penalty (0..0.9)", minval=0.0, maxval=0.9, step=0.05, group=G_MCM, tooltip="Applies only when MTF Mode=Soft and use_mtf=true.")

// --- Filters ---
bool use_mfi = input.bool(true, "Use MFI Filter", group=G_MCM)
int  mfi_len = input.int(14, "MFI Length", minval=1, group=G_MCM)
int  mfi_min_bull = input.int(30, "MFI Min (Bull)", minval=1, maxval=99, group=G_MCM)
int  mfi_max_bear = input.int(70, "MFI Max (Bear)", minval=1, maxval=99, group=G_MCM)

bool  use_adx = input.bool(false, "Use ADX Filter", group=G_MCM)
int   adx_len = input.int(14, "ADX Length", minval=1, group=G_MCM)
float adx_min = input.float(18, "ADX Min", minval=0, step=0.5, group=G_MCM)

bool use_rsi = input.bool(false, "Use RSI Filter", group=G_MCM)
int  rsi_len = input.int(14, "RSI Length", minval=1, group=G_MCM)
int  rsi_min_bull = input.int(45, "RSI Min (Bull)", minval=1, maxval=99, group=G_MCM)
int  rsi_max_bear = input.int(55, "RSI Max (Bear)", minval=1, maxval=99, group=G_MCM)

// --- Signal behavior ---
bool confirm_on_close = input.bool(true, "Confirm Unified Signals On Bar Close", group=G_MCM)

// --- Risk Management Engine ---
bool  show_risk   = input.bool(true, "Show Position Size in HUD", group=G_MCM)
float acc_size    = input.float(10000, "Account Size ($)", minval=0, group=G_MCM)
float risk_pct    = input.float(1.0, "Risk Per Trade (%)", minval=0, step=0.1, group=G_MCM)
float point_value = input.float(1.0, "Point Value ($ per 1.0 move per 1 unit/contract)", minval=0.00000001, group=G_MCM)
float min_sl_dist = input.float(0.0, "Min Stop Distance (price units)", minval=0.0, group=G_MCM, tooltip="If >0, sizing returns 0 when stop distance is below this value.")
float qty_min = input.float(0.0, "Min Qty (0 = off)", minval=0.0, group=G_MCM)
float qty_max = input.float(0.0, "Max Qty (0 = off)", minval=0.0, group=G_MCM)

//──────────────────────────────────────────────────────────────────────────────
// SECTION: COLORS & THEME
//──────────────────────────────────────────────────────────────────────────────
string G_VIS = "🎨 Theme Colors"
color C_SUPER_UP = input.color(#00E676, "Superconductor Bull", group=G_VIS)
color C_SUPER_DN = input.color(#FF1744, "Superconductor Bear", group=G_VIS)
color C_RESIST   = input.color(#546E7A, "Resistive / Chop", group=G_VIS)
color C_HEAT     = input.color(#FFD600, "High Heat / Reversal", group=G_VIS)
color C_DIV      = input.color(#00B0FF, "Divergence Signal", group=G_VIS)

//──────────────────────────────────────────────────────────────────────────────
// HELPER FUNCTIONS
//──────────────────────────────────────────────────────────────────────────────
f_bar(float _val, float _limit, int _len) =>
    float norm = math.min(math.abs(_val) / _limit, 1.0)
    int filled = int(math.round(norm * _len))
    string s = ""
    if filled > 0
        for i = 1 to filled
            s := s + "■"
    if filled < _len
        for i = 1 to (_len - filled)
            s := s + "▫"
    s

f_clamp01(float x) =>
    math.max(0.0, math.min(1.0, x))

f_push_box(var box[] a, box b, int maxN) =>
    array.push(a, b)
    if array.size(a) > maxN
        box old = array.shift(a)
        if not na(old)
            box.delete(old)

f_push_line(var line[] a, line l, int maxN) =>
    array.push(a, l)
    if array.size(a) > maxN
        line old = array.shift(a)
        if not na(old)
            line.delete(old)

f_push_label(var label[] a, label lb, int maxN) =>
    array.push(a, lb)
    if array.size(a) > maxN
        label old = array.shift(a)
        if not na(old)
            label.delete(old)

f_supertrend_dir(_src, _len, _mult) =>
    _atr = ta.atr(_len)
    _up = _src - (_mult * _atr)
    _dn = _src + (_mult * _atr)
    var float _upt = na
    var float _dnt = na
    _upt := (close[1] > nz(_upt[1], _up)) ? math.max(_up, nz(_upt[1], _up)) : _up
    _dnt := (close[1] < nz(_dnt[1], _dn)) ? math.min(_dn, nz(_dnt[1], _dn)) : _dn
    var int _t = 1
    _t := (close > nz(_dnt[1], _dn)) ? 1 : (close < nz(_upt[1], _up)) ? -1 : nz(_t[1], 1)
    _t

f_clamp_qty(float q) =>
    float out = q
    out := qty_min > 0 ? math.max(out, qty_min) : out
    out := qty_max > 0 ? math.min(out, qty_max) : out
    out

//──────────────────────────────────────────────────────────────────────────────
// CALCULATION: APEX VECTOR
//──────────────────────────────────────────────────────────────────────────────
float range_abs = high - low
float body_abs  = math.abs(close - open)
float raw_eff   = range_abs == 0 ? 0.0 : body_abs / range_abs
float efficiency = ta.ema(raw_eff, LEN_VEC)

float vol_avg  = ta.sma(volume, VOL_NORM)
float vol_fact = (vol_avg == 0) ? 1.0 : (volume / vol_avg)

float direction = math.sign(close - open)
float vector_raw = direction * efficiency * vol_fact

float apex_flux = switch SM_TYPE
    "EMA"  => ta.ema(vector_raw, LEN_SM)
    "SMA"  => ta.sma(vector_raw, LEN_SM)
    "RMA"  => ta.rma(vector_raw, LEN_SM)
    "VWMA" => ta.vwma(vector_raw, LEN_SM)
    => ta.ema(vector_raw, LEN_SM)

bool is_super_bull = apex_flux > EFF_SUPER
bool is_super_bear = apex_flux < -EFF_SUPER
bool is_resistive  = math.abs(apex_flux) < EFF_RESIST
bool is_heat       = not is_super_bull and not is_super_bear and not is_resistive

//──────────────────────────────────────────────────────────────────────────────
// CALCULATION: DIVERGENCE ENGINE
//──────────────────────────────────────────────────────────────────────────────
int div_look = 5
float ph = ta.pivothigh(apex_flux, div_look, div_look)
float pl = ta.pivotlow(apex_flux, div_look, div_look)

var float prev_pl_flux  = na
var float prev_pl_price = na
var float prev_ph_flux  = na
var float prev_ph_price = na

bool div_bull = false
bool div_bear = false
string div_type = ""

if not na(pl)
    float price_at_pivot = low[div_look]
    if not na(prev_pl_flux)
        if (price_at_pivot < prev_pl_price and pl > prev_pl_flux) or (price_at_pivot > prev_pl_price and pl < prev_pl_flux)
            div_bull := true
            div_type := (price_at_pivot < prev_pl_price) ? "REG" : "HID"
    prev_pl_flux  := pl
    prev_pl_price := price_at_pivot

if not na(ph)
    float price_at_pivot = high[div_look]
    if not na(prev_ph_flux)
        if (price_at_pivot > prev_ph_price and ph < prev_ph_flux) or (price_at_pivot < prev_ph_price and ph > prev_ph_flux)
            div_bear := true
            div_type := (price_at_pivot > prev_ph_price) ? "REG" : "HID"
    prev_ph_flux  := ph
    prev_ph_price := price_at_pivot

//──────────────────────────────────────────────────────────────────────────────
// CALCULATION: SPACETIME (SMC)
//──────────────────────────────────────────────────────────────────────────────
bool wormhole_bull = WORMHOLES and (low > high[2])
bool wormhole_bear = WORMHOLES and (high < low[2])

bool event_horizon_bull = EVENT_HORIZONS and close[1] < open[1] and close > high[1] and is_super_bull
bool event_horizon_bear = EVENT_HORIZONS and close[1] > open[1] and close < low[1] and is_super_bear

float st_ph = ta.pivothigh(high, SMC_LOOKBACK, SMC_LOOKBACK)
float st_pl = ta.pivotlow(low, SMC_LOOKBACK, SMC_LOOKBACK)
var float last_st_high = na
var float last_st_low  = na
if not na(st_ph)
    last_st_high := high[SMC_LOOKBACK]
if not na(st_pl)
    last_st_low := low[SMC_LOOKBACK]

bool bos_bull   = STRUCTURE and not na(last_st_high) and close > last_st_high and high[1] <= last_st_high
bool bos_bear   = STRUCTURE and not na(last_st_low)  and close < last_st_low and low[1] >= last_st_low
bool choch_bull = STRUCTURE and not na(last_st_low)  and close > last_st_low  and apex_flux[1] < -EFF_RESIST
bool choch_bear = STRUCTURE and not na(last_st_high) and close < last_st_high and apex_flux[1] >  EFF_RESIST

//──────────────────────────────────────────────────────────────────────────────
// CALCULATION: MCM SUPERTRND + MTF + FILTERS + RISK
//──────────────────────────────────────────────────────────────────────────────
float st_atr = ta.atr(st_period)
float st_up  = st_src - (st_mult * st_atr)
float st_dn  = st_src + (st_mult * st_atr)

var float st_up_trend = na
var float st_dn_trend = na

st_up_trend := (close[1] > nz(st_up_trend[1], st_up)) ? math.max(st_up, nz(st_up_trend[1], st_up)) : st_up
st_dn_trend := (close[1] < nz(st_dn_trend[1], st_dn)) ? math.min(st_dn, nz(st_dn_trend[1], st_dn)) : st_dn

var int st_trend = 1
st_trend := (close > nz(st_dn_trend[1], st_dn)) ? 1 : (close < nz(st_up_trend[1], st_up)) ? -1 : nz(st_trend[1], 1)

float st_line = st_trend == 1 ? st_up_trend : st_dn_trend

int mtf_trend_dir = request.security(
     syminfo.tickerid,
     mtf_tf,
     f_supertrend_dir(st_src, st_period, st_mult),
     gaps=barmerge.gaps_off,
     lookahead=barmerge.lookahead_off
)

// Filters
float mfi = ta.mfi(hlc3, mfi_len)
bool mfi_ok = not use_mfi or ((st_trend == 1 and mfi > mfi_min_bull) or (st_trend == -1 and mfi < mfi_max_bear))

float adx = ta.adx(adx_len)
bool adx_ok = not use_adx or (adx >= adx_min)

float rsi = ta.rsi(close, rsi_len)
bool rsi_ok = not use_rsi or ((st_trend == 1 and rsi >= rsi_min_bull) or (st_trend == -1 and rsi <= rsi_max_bear))

// Risk sizing against SuperTrend line
float sl_price = st_line
float sl_dist  = math.abs(close - sl_price)
float risk_amt = acc_size * (risk_pct / 100.0)

// Guard: minimum stop distance
bool sl_ok = (min_sl_dist <= 0) or (sl_dist >= min_sl_dist)

float pos_size_raw = (sl_ok and sl_dist > 0 and point_value > 0) ? (risk_amt / (sl_dist * point_value)) : 0.0
float pos_size = f_clamp_qty(pos_size_raw)

bool confirmed_ok = confirm_on_close ? barstate.isconfirmed : true

// MTF gating mode
bool mtf_ok_bull = (not use_mtf) or (MTF_MODE == "Soft") or (mtf_trend_dir == 1)
bool mtf_ok_bear = (not use_mtf) or (MTF_MODE == "Soft") or (mtf_trend_dir == -1)

//──────────────────────────────────────────────────────────────────────────────
// UNIFIED BRAIN (Correctness + Side Thresholds + Dominance + Cooldown)
//──────────────────────────────────────────────────────────────────────────────
string G_BRAIN = "🧠 Unified Brain (Apex + SMC + MTF)"
bool   BRAIN_ON = input.bool(true, "Enable Unified Brain Signals", group=G_BRAIN)

float ENTRY_TH_BULL = input.float(0.70, "Entry Threshold (Bull) 0..1", minval=0.50, maxval=0.95, step=0.01, group=G_BRAIN)
float EXIT_TH_BULL  = input.float(0.55, "Exit Threshold (Bull) 0..1",  minval=0.30, maxval=0.90, step=0.01, group=G_BRAIN)
float ENTRY_TH_BEAR = input.float(0.70, "Entry Threshold (Bear) 0..1", minval=0.50, maxval=0.95, step=0.01, group=G_BRAIN)
float EXIT_TH_BEAR  = input.float(0.55, "Exit Threshold (Bear) 0..1",  minval=0.30, maxval=0.90, step=0.01, group=G_BRAIN)

bool REQUIRE_DOMINANCE = input.bool(true, "Require Conf Dominance (Bull>Bear / Bear>Bull)", group=G_BRAIN)
bool REQUIRE_SMC_EVENT = input.bool(false, "Require SMC Event For Entry", group=G_BRAIN,
     tooltip="If ON, entries require at least one SMC event OR divergence confirmation on that side.")
bool STRETCH_GUARD = input.bool(true, "Stretch Guard (Avoid Late Entries)", group=G_BRAIN)
int  STRETCH_LB    = input.int(200, "Stretch Lookback", minval=20, group=G_BRAIN)
float STRETCH_MAX  = input.float(0.80, "Max Stretch (0..1)", minval=0.10, maxval=1.50, step=0.05, group=G_BRAIN)

int COOLDOWN_BARS = input.int(0, "Cooldown Bars After State Change", minval=0, group=G_BRAIN,
     tooltip="If >0, prevents new entries from neutral until this many bars pass after the last state change.")

// Volatility regime via ATR percentile
float atr_min = ta.lowest(st_atr, STRETCH_LB)
float atr_max = ta.highest(st_atr, STRETCH_LB)
float vol_reg = atr_max == atr_min ? 0.5 : f_clamp01((st_atr - atr_min) / (atr_max - atr_min))

// Stretch of price vs stop reference
float dist = sl_dist
float dist_min = ta.lowest(dist, STRETCH_LB)
float dist_max = ta.highest(dist, STRETCH_LB)
float stretch = dist_max == dist_min ? 0.0 : f_clamp01((dist - dist_min) / (dist_max - dist_min))
bool stretch_ok = not STRETCH_GUARD or (stretch <= STRETCH_MAX)

// Evidence: LTF/HTF
float ltf_bull     = st_trend == 1 ? 1.0 : 0.0
float ltf_bear     = st_trend == -1 ? 1.0 : 0.0
float fractal_bull = mtf_trend_dir == 1 ? 1.0 : 0.0
float fractal_bear = mtf_trend_dir == -1 ? 1.0 : 0.0

// Evidence: oscillators (normalized using provided thresholds)
float mfi_den = math.max(1.0, float(mfi_max_bear) - float(mfi_min_bull))
float rsi_den = math.max(1.0, float(rsi_max_bear) - float(rsi_min_bull))
float mfi_bull = f_clamp01((mfi - float(mfi_min_bull)) / mfi_den)
float rsi_bull = f_clamp01((rsi - float(rsi_min_bull)) / rsi_den)
float mfi_bear = 1.0 - mfi_bull
float rsi_bear = 1.0 - rsi_bull

float adx_den = math.max(1.0, adx_min)
float adx_str = f_clamp01(adx / adx_den)

// Evidence: Apex
float apex_abs  = math.abs(apex_flux)
float apex_norm = f_clamp01(apex_abs / (EFF_SUPER * 1.5))
float apex_dir  = apex_flux >= 0 ? 1.0 : -1.0
float apex_bull = f_clamp01(0.5 + 0.5 * apex_dir * apex_norm)
float apex_bear = 1.0 - apex_bull

// Evidence: SMC/Divergence (symmetric)
bool smc_event_bull = bos_bull or choch_bull or wormhole_bull or event_horizon_bull or div_bull
bool smc_event_bear = bos_bear or choch_bear or wormhole_bear or event_horizon_bear or div_bear
float smc_bull = smc_event_bull ? 1.0 : 0.0
float smc_bear = smc_event_bear ? 1.0 : 0.0

// Adaptive weights
float w_trend   = 0.22 + 0.18 * vol_reg
float w_fractal = 0.20 + 0.20 * vol_reg
float w_adx     = 0.10 + 0.18 * vol_reg
float w_rsi     = 0.18 - 0.08 * vol_reg
float w_mfi     = 0.18 - 0.08 * vol_reg
float w_stretch = 0.12 - 0.12 * vol_reg
float w_apex    = 0.22
float w_smc     = 0.18

// Anti-double-counting: reduce Apex weight when MFI is ON (volume already represented)
float w_apex_eff = use_mfi ? (w_apex * 0.75) : w_apex

float wsum = w_trend + w_fractal + w_adx + w_rsi + w_mfi + w_stretch + w_apex_eff + w_smc
wt(x) => x / wsum

float conf_bull_raw =
     wt(w_trend)    * ltf_bull +
     wt(w_fractal)  * fractal_bull +
     wt(w_adx)      * adx_str +
     wt(w_rsi)      * rsi_bull +
     wt(w_mfi)      * mfi_bull +
     wt(w_stretch)  * (1.0 - stretch) +
     wt(w_apex_eff) * apex_bull +
     wt(w_smc)      * smc_bull

float conf_bear_raw =
     wt(w_trend)    * ltf_bear +
     wt(w_fractal)  * fractal_bear +
     wt(w_adx)      * adx_str +
     wt(w_rsi)      * rsi_bear +
     wt(w_mfi)      * mfi_bear +
     wt(w_stretch)  * (1.0 - stretch) +
     wt(w_apex_eff) * apex_bear +
     wt(w_smc)      * smc_bear

// Soft MTF penalty (applies only when enabled)
bool mtf_mismatch_bull = use_mtf and (mtf_trend_dir != 1)
bool mtf_mismatch_bear = use_mtf and (mtf_trend_dir != -1)

float conf_bull_raw_adj = (use_mtf and MTF_MODE == "Soft" and mtf_mismatch_bull) ? (conf_bull_raw * (1.0 - MTF_PENALTY)) : conf_bull_raw
float conf_bear_raw_adj = (use_mtf and MTF_MODE == "Soft" and mtf_mismatch_bear) ? (conf_bear_raw * (1.0 - MTF_PENALTY)) : conf_bear_raw

float denom = math.max(conf_bull_raw_adj + conf_bear_raw_adj, 0.000001)
float conf_bull = conf_bull_raw_adj / denom
float conf_bear = conf_bear_raw_adj / denom

bool dominance_long  = not REQUIRE_DOMINANCE or (conf_bull > conf_bear)
bool dominance_short = not REQUIRE_DOMINANCE or (conf_bear > conf_bull)

// Gates (explicit filters preserved + MTF mode preserved)
bool filters_ok_long  = mfi_ok and adx_ok and rsi_ok and mtf_ok_bull
bool filters_ok_short = mfi_ok and adx_ok and rsi_ok and mtf_ok_bear

bool smc_gate_long  = not REQUIRE_SMC_EVENT or smc_event_bull
bool smc_gate_short = not REQUIRE_SMC_EVENT or smc_event_bear

// Cooldown (after any state change)
var int last_state_change_bar = na
bool cooldown_ok = (COOLDOWN_BARS <= 0) or na(last_state_change_bar) or ((bar_index - last_state_change_bar) > COOLDOWN_BARS)

// Entries/exits per side
bool enter_bull = BRAIN_ON and cooldown_ok and (conf_bull >= ENTRY_TH_BULL) and filters_ok_long  and stretch_ok and smc_gate_long  and dominance_long  and confirmed_ok
bool enter_bear = BRAIN_ON and cooldown_ok and (conf_bear >= ENTRY_TH_BEAR) and filters_ok_short and stretch_ok and smc_gate_short and dominance_short and confirmed_ok

bool exit_bull  = BRAIN_ON and (conf_bull <= EXIT_TH_BULL) and confirmed_ok
bool exit_bear  = BRAIN_ON and (conf_bear <= EXIT_TH_BEAR) and confirmed_ok

var int brain_state = 0 // 1 bull, -1 bear, 0 neutral

int prev_state = brain_state
if BRAIN_ON
    if brain_state == 0
        brain_state := enter_bull ? 1 : enter_bear ? -1 : 0
    else if brain_state == 1
        brain_state := exit_bull ? 0 : 1
    else
        brain_state := exit_bear ? 0 : -1

if brain_state != prev_state
    last_state_change_bar := bar_index

bool unified_long  = BRAIN_ON and (brain_state == 1  and prev_state != 1)
bool unified_short = BRAIN_ON and (brain_state == -1 and prev_state != -1)

//──────────────────────────────────────────────────────────────────────────────
// VISUALIZATION: CHART OVERLAY (Stable on all TFs)
//──────────────────────────────────────────────────────────────────────────────
color active_color = is_super_bull ? C_SUPER_UP : is_super_bear ? C_SUPER_DN : is_resistive ? C_RESIST : C_HEAT
barcolor(SHOW_BARCOL ? active_color : na)

// Object pools
var box[]   wh_boxes = array.new_box()
var line[]  st_lines = array.new_line()
var label[] st_labs  = array.new_label()
var label[] ev_labs  = array.new_label()

// Wormholes (FVG) boxes (now properly disabled when MAX=0)
if wormhole_bull and MAX_WORMHOLE_BOXES > 0
    box b = box.new(bar_index[2], high[2], bar_index, low,
        border_color=color.new(C_SUPER_UP, 70), border_style=line.style_dotted,
        bgcolor=color.new(C_SUPER_UP, 92), extend=extend.right,
        text=SHOW_LABELS ? "WH" : na, text_size=size.tiny, text_color=color.new(C_SUPER_UP, 50))
    f_push_box(wh_boxes, b, MAX_WORMHOLE_BOXES)

if wormhole_bear and MAX_WORMHOLE_BOXES > 0
    box b = box.new(bar_index[2], high, bar_index, low[2],
        border_color=color.new(C_SUPER_DN, 70), border_style=line.style_dotted,
        bgcolor=color.new(C_SUPER_DN, 92), extend=extend.right,
        text=SHOW_LABELS ? "WH" : na, text_size=size.tiny, text_color=color.new(C_SUPER_DN, 50))
    f_push_box(wh_boxes, b, MAX_WORMHOLE_BOXES)

// Event Horizon Dots
if event_horizon_bull and MAX_EVENT_LABELS > 0
    label lb = label.new(bar_index[1], high[1], "•", color=color.new(C_SUPER_UP, 100), textcolor=C_SUPER_UP, style=label.style_label_down, size=size.small)
    f_push_label(ev_labs, lb, MAX_EVENT_LABELS)

if event_horizon_bear and MAX_EVENT_LABELS > 0
    label lb = label.new(bar_index[1], low[1], "•", color=color.new(C_SUPER_DN, 100), textcolor=C_SUPER_DN, style=label.style_label_up, size=size.small)
    f_push_label(ev_labs, lb, MAX_EVENT_LABELS)

// Structure Lines/Labels (BOS)
if bos_bull and MAX_STRUCT_LINES > 0
    line l = line.new(bar_index - 10, last_st_high, bar_index, last_st_high, color=color.new(C_SUPER_UP, 50), width=1, style=line.style_dotted)
    f_push_line(st_lines, l, MAX_STRUCT_LINES)
if bos_bull and MAX_STRUCT_LABELS > 0
    label lb = label.new(bar_index, last_st_high, SHOW_LABELS ? "BOS" : "↑", color=color.new(C_SUPER_UP, 100), textcolor=C_SUPER_UP, style=label.style_label_down, size=size.tiny)
    f_push_label(st_labs, lb, MAX_STRUCT_LABELS)

if bos_bear and MAX_STRUCT_LINES > 0
    line l = line.new(bar_index - 10, last_st_low, bar_index, last_st_low, color=color.new(C_SUPER_DN, 50), width=1, style=line.style_dotted)
    f_push_line(st_lines, l, MAX_STRUCT_LINES)
if bos_bear and MAX_STRUCT_LABELS > 0
    label lb = label.new(bar_index, last_st_low, SHOW_LABELS ? "BOS" : "↓", color=color.new(C_SUPER_DN, 100), textcolor=C_SUPER_DN, style=label.style_label_up, size=size.tiny)
    f_push_label(st_labs, lb, MAX_STRUCT_LABELS)

// Watermark Visualizer (bug fixed: no `var`)
float scale_max = ta.highest(high, 100)
float scale_min = ta.lowest(low, 100)
float chart_range = scale_max - scale_min
float watermark_y = low - (chart_range * 0.05)
plotchar(SHOW_WTRMRK ? watermark_y : na, "Watermark", "■", location.absolute, color=color.new(active_color, 60), size=size.tiny)

// MCM Visuals
string G_MCM_VIS = "🌊 MCM Visuals (Integrated)"
bool  SHOW_ST_LINE = input.bool(true, "Show SuperTrend Line", group=G_MCM_VIS)
color C_ST_BULL = input.color(#00E676, "ST Bull Color", group=G_MCM_VIS)
color C_ST_BEAR = input.color(#FF1744, "ST Bear Color", group=G_MCM_VIS)
plot(SHOW_ST_LINE ? st_line : na, "SuperTrend Line", color=st_trend == 1 ? C_ST_BULL : C_ST_BEAR, linewidth=2)

// Unified signals
plotshape(unified_long,  "Unified BUY",  shape.labelup,   location.belowbar, C_SUPER_UP, 0, "BUY",  color.white, size=size.tiny)
plotshape(unified_short, "Unified SELL", shape.labeldown, location.abovebar, C_SUPER_DN, 0, "SELL", color.white, size=size.tiny)

//──────────────────────────────────────────────────────────────────────────────
// HUD (kept; corrected conf + sizing + state)
//──────────────────────────────────────────────────────────────────────────────
position pos_in = switch TABLE_POS
    "Top Right"    => position.top_right
    "Bottom Right" => position.bottom_right
    "Top Left"     => position.top_left
    "Bottom Left"  => position.bottom_left
    => position.bottom_right

size size_in = switch TABLE_SIZE
    "Tiny"   => size.tiny
    "Small"  => size.small
    "Normal" => size.normal
    => size.small

var table hud = table.new(pos_in, 2, 10, bgcolor=color.new(#000000, 50), border_width=0)

if barstate.islast
    table.cell(hud, 0, 0, "⚛ APEX UNIFIED", text_color=color.new(#7C4DFF, 0), text_size=size_in, text_halign=text.align_left)
    string state_icon = is_super_bull ? "🚀" : is_super_bear ? "🔻" : is_resistive ? "💤" : "🔥"
    table.cell(hud, 1, 0, state_icon, text_color=color.white, text_size=size_in, text_halign=text.align_right)

    string bar = f_bar(apex_flux, EFF_SUPER * 1.5, 6)
    table.cell(hud, 0, 1, "Vector", text_color=color.gray, text_size=size_in, text_halign=text.align_left)
    table.cell(hud, 1, 1, bar, text_color=active_color, text_size=size_in, text_halign=text.align_right)

    table.cell(hud, 0, 2, "Efficiency", text_color=color.gray, text_size=size_in, text_halign=text.align_left)
    table.cell(hud, 1, 2, str.tostring(efficiency * 100, "#") + "%", text_color=color.silver, text_size=size_in, text_halign=text.align_right)

    string s_txt = bos_bull ? "BOS+" : bos_bear ? "BOS-" : choch_bull ? "CH+" : choch_bear ? "CH-" : "--"
    color s_col  = (bos_bull or choch_bull) ? C_SUPER_UP : (bos_bear or choch_bear) ? C_SUPER_DN : color.gray
    table.cell(hud, 0, 3, "Struct", text_color=color.gray, text_size=size_in, text_halign=text.align_left)
    table.cell(hud, 1, 3, s_txt, text_color=s_col, text_size=size_in, text_halign=text.align_right)

    string d_txt = div_bull ? "BULL " + div_type : div_bear ? "BEAR " + div_type : "NONE"
    color d_col  = (div_bull or div_bear) ? C_DIV : color.gray
    table.cell(hud, 0, 4, "Divergence", text_color=color.gray, text_size=size_in, text_halign=text.align_left)
    table.cell(hud, 1, 4, d_txt, text_color=d_col, text_size=size_in, text_halign=text.align_right)

    string rec = "WAIT"
    color rec_c = color.gray
    if unified_long
        rec := "BUY"
        rec_c := C_SUPER_UP
    else if unified_short
        rec := "SELL"
        rec_c := C_SUPER_DN
    else
        if is_super_bull and bos_bull
            rec := "LONG"
            rec_c := C_SUPER_UP
        else if is_super_bear and bos_bear
            rec := "SHORT"
            rec_c := C_SUPER_DN

    table.cell(hud, 0, 5, "Signal", text_color=color.gray, text_size=size_in, text_halign=text.align_left)
    table.cell(hud, 1, 5, rec, text_color=rec_c, text_size=size_in, text_halign=text.align_right, bgcolor=color.new(rec_c, 80))

    table.cell(hud, 0, 6, "ST / HTF", text_color=color.gray, text_size=size_in, text_halign=text.align_left)
    string sttxt = st_trend == 1 ? "BULL" : "BEAR"
    string htftxt = mtf_trend_dir == 1 ? "BULL" : "BEAR"
    table.cell(hud, 1, 6, sttxt + " / " + htftxt, text_color=color.white, text_size=size_in, text_halign=text.align_right, bgcolor=color.new(mtf_trend_dir == 1 ? C_ST_BULL : C_ST_BEAR, 85))

    table.cell(hud, 0, 7, "MFI/ADX/RSI", text_color=color.gray, text_size=size_in, text_halign=text.align_left)
    table.cell(hud, 1, 7, str.tostring(mfi, "#.0") + " / " + str.tostring(adx, "#.0") + " / " + str.tostring(rsi, "#.0"),
         text_color=color.silver, text_size=size_in, text_halign=text.align_right)

    table.cell(hud, 0, 8, "Conf (B/S)", text_color=color.gray, text_size=size_in, text_halign=text.align_left)
    string st_state = brain_state == 1 ? "BULL" : brain_state == -1 ? "BEAR" : "NEUT"
    color conf_col = brain_state == 1 ? C_SUPER_UP : brain_state == -1 ? C_SUPER_DN : C_RESIST
    table.cell(hud, 1, 8, str.tostring(conf_bull * 100, "#") + "% / " + str.tostring(conf_bear * 100, "#") + "% " + st_state,
         text_color=conf_col, text_size=size_in, text_halign=text.align_right)

    table.cell(hud, 0, 9, "Stop / Size", text_color=color.gray, text_size=size_in, text_halign=text.align_left)
    string sz = show_risk ? str.tostring(pos_size, "#.####") : "--"
    table.cell(hud, 1, 9, str.tostring(sl_price, format.mintick) + " / " + sz,
         text_color=show_risk ? color.aqua : color.gray, text_size=size_in, text_halign=text.align_right)

//──────────────────────────────────────────────────────────────────────────────
// ALERTS (Original + Unified)
//──────────────────────────────────────────────────────────────────────────────
alertcondition(is_super_bull and not is_super_bull[1], "Apex: Superconductor Bull", "Bullish Superconductor Detected")
alertcondition(is_super_bear and not is_super_bear[1], "Apex: Superconductor Bear", "Bearish Superconductor Detected")
alertcondition(bos_bull, "SMC: Bullish Break", "Bullish BOS Detected")
alertcondition(div_bull, "Apex: Bullish Divergence", "Bullish Divergence Detected")

f_payload(_side) =>
    str.format('{{"script":"QuantumApex v7.4","symbol":"{0}","tf":"{1}","side":"{2}","price":{3},"stop":{4},"qty":{5},"risk_amt":{6},"mtf_tf":"{7}","mtf_dir":{8},"mtf_mode":"{9}","conf_bull":{10},"conf_bear":{11},"state":{12}}}',
      syminfo.ticker, timeframe.period, _side,
      str.tostring(close, format.mintick),
      str.tostring(sl_price, format.mintick),
      str.tostring(pos_size, "#.####"),
      str.tostring(risk_amt, "#.##"),
      mtf_tf,
      mtf_trend_dir,
      MTF_MODE,
      str.tostring(conf_bull, "#.####"),
      str.tostring(conf_bear, "#.####"),
      str.tostring(brain_state)
    )

alertcondition(unified_long,  "Unified: BUY",  f_payload("BUY"))
alertcondition(unified_short, "Unified: SELL", f_payload("SELL"))

if barstate.isconfirmed
    if is_super_bull and not is_super_bull[1]
        alert("🚀 SUPERCONDUCTOR BULL | Vector: " + str.tostring(apex_flux), alert.freq_once_per_bar_close)
    if div_bull or div_bear
        alert("⚠️ DIVERGENCE DETECTED", alert.freq_once_per_bar_close)

    if unified_long
        alert("✅ UNIFIED BUY | Conf(B): " + str.tostring(conf_bull * 100, "#") + "% | Stop: " + str.tostring(sl_price, format.mintick) + " | Size: " + str.tostring(pos_size, "#.####"),
             alert.freq_once_per_bar_close)
    if unified_short
        alert("✅ UNIFIED SELL | Conf(S): " + str.tostring(conf_bear * 100, "#") + "% | Stop: " + str.tostring(sl_price, format.mintick) + " | Size: " + str.tostring(pos_size, "#.####"),
             alert.freq_once_per_bar_close)
