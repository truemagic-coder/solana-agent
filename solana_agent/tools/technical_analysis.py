"""
Technical Analysis Tool - Comprehensive technical indicators for token analysis.

Provides technical analysis indicators for any token using Birdeye OHLCV data.
Returns raw indicator values without interpretation - the consuming application
should interpret the data based on its own logic.

Requires minimum 200 candles for reliable indicator calculation.
"""

import logging
import time
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import httpx
import pandas as pd
from solana_agent import AutoTool, ToolRegistry

try:
    import pandas_ta as ta

    PANDAS_TA_AVAILABLE = True
except ImportError:  # pragma: no cover
    ta = None  # type: ignore[assignment]
    PANDAS_TA_AVAILABLE = False

logger = logging.getLogger(__name__)

# Minimum candles required for reliable TA calculation
MIN_CANDLES_REQUIRED = 200

# Supported timeframes and their mapping to Birdeye OHLCV V3 types
TIMEFRAME_MAP = {
    "1m": "1m",
    "5m": "5m",
    "15m": "15m",
    "30m": "30m",
    "1h": "1H",
    "2h": "2H",
    "4h": "4H",
    "8h": "8H",
    "1d": "1D",
}


def calculate_indicators(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Calculate all technical indicators from OHLCV data.

    Args:
        df: DataFrame with columns: open, high, low, close, volume

    Returns:
        Dictionary with all indicator values
    """
    if not PANDAS_TA_AVAILABLE:
        raise RuntimeError("pandas_ta is required for technical analysis calculations.")

    # Ensure datetime index for indicators that expect sorted dates (e.g., VWAP)
    if "timestamp" in df.columns:
        df = df.copy()
        df.index = pd.to_datetime(df["timestamp"], unit="s", utc=True, errors="coerce")
        df = df.sort_index()

    # Get the latest values for each indicator
    latest_idx = -1

    # === TREND INDICATORS ===
    # EMAs
    ema_9 = ta.ema(df["close"], length=9)
    ema_21 = ta.ema(df["close"], length=21)
    ema_50 = ta.ema(df["close"], length=50)
    ema_200 = ta.ema(df["close"], length=200)

    # SMAs
    sma_20 = ta.sma(df["close"], length=20)
    sma_50 = ta.sma(df["close"], length=50)
    sma_200 = ta.sma(df["close"], length=200)

    # MACD
    macd_result = ta.macd(df["close"], fast=12, slow=26, signal=9)
    macd_line = None
    macd_signal = None
    macd_histogram = None
    if macd_result is not None and not macd_result.empty:
        macd_line = _safe_get(macd_result, "MACD_12_26_9", latest_idx)
        macd_signal = _safe_get(macd_result, "MACDs_12_26_9", latest_idx)
        macd_histogram = _safe_get(macd_result, "MACDh_12_26_9", latest_idx)

    # ADX
    adx_result = ta.adx(df["high"], df["low"], df["close"], length=14)
    adx_value = None
    adx_pos = None
    adx_neg = None
    if adx_result is not None and not adx_result.empty:
        adx_value = _safe_get(adx_result, "ADX_14", latest_idx)
        adx_pos = _safe_get(adx_result, "DMP_14", latest_idx)
        adx_neg = _safe_get(adx_result, "DMN_14", latest_idx)

    # === MOMENTUM INDICATORS ===
    # RSI
    rsi_14 = ta.rsi(df["close"], length=14)

    # Stochastic
    stoch_result = ta.stoch(df["high"], df["low"], df["close"], k=14, d=3, smooth_k=3)
    stoch_k = None
    stoch_d = None
    if stoch_result is not None and not stoch_result.empty:
        stoch_k = _safe_get(stoch_result, "STOCHk_14_3_3", latest_idx)
        stoch_d = _safe_get(stoch_result, "STOCHd_14_3_3", latest_idx)

    # CCI
    cci_20 = ta.cci(df["high"], df["low"], df["close"], length=20)

    # Williams %R
    willr_14 = ta.willr(df["high"], df["low"], df["close"], length=14)

    # ROC (Rate of Change)
    roc_12 = ta.roc(df["close"], length=12)

    # MFI (Money Flow Index - volume-weighted RSI)
    mfi_14 = ta.mfi(df["high"], df["low"], df["close"], df["volume"], length=14)

    # === VOLATILITY INDICATORS ===
    # Bollinger Bands
    bbands = ta.bbands(df["close"], length=20, std=2)
    bb_upper = None
    bb_middle = None
    bb_lower = None
    bb_bandwidth = None
    bb_percent_b = None
    if bbands is not None and not bbands.empty:
        bb_upper = _safe_get(bbands, "BBU_20_2.0_2.0", latest_idx)
        bb_middle = _safe_get(bbands, "BBM_20_2.0_2.0", latest_idx)
        bb_lower = _safe_get(bbands, "BBL_20_2.0_2.0", latest_idx)
        bb_bandwidth = _safe_get(bbands, "BBB_20_2.0_2.0", latest_idx)
        bb_percent_b = _safe_get(bbands, "BBP_20_2.0_2.0", latest_idx)

    # ATR
    atr_14 = ta.atr(df["high"], df["low"], df["close"], length=14)
    current_price = df["close"].iloc[latest_idx]
    atr_value = _safe_get_series(atr_14, latest_idx)
    atr_percent = None
    if atr_value is not None and current_price > 0:
        atr_percent = (atr_value / current_price) * 100

    # Keltner Channels
    keltner = ta.kc(df["high"], df["low"], df["close"], length=20, scalar=2)
    kc_upper = None
    kc_middle = None
    kc_lower = None
    if keltner is not None and not keltner.empty:
        kc_upper = _safe_get(keltner, "KCUe_20_2", latest_idx)
        kc_middle = _safe_get(keltner, "KCBe_20_2", latest_idx)
        kc_lower = _safe_get(keltner, "KCLe_20_2", latest_idx)

    # === VOLUME INDICATORS ===
    # OBV
    obv = ta.obv(df["close"], df["volume"])
    obv_value = _safe_get_series(obv, latest_idx)

    # OBV EMA
    obv_ema_21 = None
    if obv is not None:
        obv_ema = ta.ema(obv, length=21)
        obv_ema_21 = _safe_get_series(obv_ema, latest_idx)

    # Volume SMA
    vol_sma_20 = ta.sma(df["volume"], length=20)
    vol_sma_value = _safe_get_series(vol_sma_20, latest_idx)
    current_volume = df["volume"].iloc[latest_idx]

    # VWAP (session-based, use last 24 bars as approximation)
    vwap = ta.vwap(df["high"], df["low"], df["close"], df["volume"])
    vwap_value = _safe_get_series(vwap, latest_idx)

    # === PRICE VS INDICATORS ===
    price_vs = {}
    if current_price is not None and current_price > 0:
        if _safe_get_series(ema_9, latest_idx) is not None:
            price_vs["vs_ema_9_percent"] = _calc_percent_diff(
                current_price, _safe_get_series(ema_9, latest_idx)
            )
        if _safe_get_series(ema_21, latest_idx) is not None:
            price_vs["vs_ema_21_percent"] = _calc_percent_diff(
                current_price, _safe_get_series(ema_21, latest_idx)
            )
        if _safe_get_series(ema_50, latest_idx) is not None:
            price_vs["vs_ema_50_percent"] = _calc_percent_diff(
                current_price, _safe_get_series(ema_50, latest_idx)
            )
        if _safe_get_series(ema_200, latest_idx) is not None:
            price_vs["vs_ema_200_percent"] = _calc_percent_diff(
                current_price, _safe_get_series(ema_200, latest_idx)
            )
        if _safe_get_series(sma_200, latest_idx) is not None:
            price_vs["vs_sma_200_percent"] = _calc_percent_diff(
                current_price, _safe_get_series(sma_200, latest_idx)
            )
        if vwap_value is not None:
            price_vs["vs_vwap_percent"] = _calc_percent_diff(current_price, vwap_value)
        if bb_middle is not None:
            price_vs["vs_bb_middle_percent"] = _calc_percent_diff(
                current_price, bb_middle
            )

    support_resistance = _calculate_support_resistance(
        df,
        current_price=current_price,
        atr_value=atr_value,
    )

    return {
        "trend": {
            "ema_9": _safe_get_series(ema_9, latest_idx),
            "ema_21": _safe_get_series(ema_21, latest_idx),
            "ema_50": _safe_get_series(ema_50, latest_idx),
            "ema_200": _safe_get_series(ema_200, latest_idx),
            "sma_20": _safe_get_series(sma_20, latest_idx),
            "sma_50": _safe_get_series(sma_50, latest_idx),
            "sma_200": _safe_get_series(sma_200, latest_idx),
            "macd": {
                "macd": macd_line,
                "signal": macd_signal,
                "histogram": macd_histogram,
            },
            "adx": adx_value,
            "adx_pos": adx_pos,
            "adx_neg": adx_neg,
        },
        "momentum": {
            "rsi_14": _safe_get_series(rsi_14, latest_idx),
            "stochastic": {
                "k": stoch_k,
                "d": stoch_d,
            },
            "cci_20": _safe_get_series(cci_20, latest_idx),
            "williams_r_14": _safe_get_series(willr_14, latest_idx),
            "roc_12": _safe_get_series(roc_12, latest_idx),
            "mfi_14": _safe_get_series(mfi_14, latest_idx),
        },
        "volatility": {
            "bollinger": {
                "upper": bb_upper,
                "middle": bb_middle,
                "lower": bb_lower,
                "bandwidth": bb_bandwidth,
                "percent_b": bb_percent_b,
            },
            "atr_14": atr_value,
            "atr_percent": atr_percent,
            "keltner": {
                "upper": kc_upper,
                "middle": kc_middle,
                "lower": kc_lower,
            },
        },
        "volume": {
            "obv": obv_value,
            "obv_ema_21": obv_ema_21,
            "volume_sma_20": vol_sma_value,
            "current_volume": current_volume,
            "vwap": vwap_value,
        },
        "support_resistance": support_resistance,
        "price_vs_indicators": price_vs,
    }


def _cluster_levels(levels: list[float], tolerance: float) -> list[float]:
    """Cluster nearby levels within tolerance and return cluster means."""
    if not levels:
        return []
    levels_sorted = sorted(levels)
    clusters: list[list[float]] = [[levels_sorted[0]]]
    for level in levels_sorted[1:]:
        current_cluster = clusters[-1]
        cluster_mean = sum(current_cluster) / len(current_cluster)
        if abs(level - cluster_mean) <= tolerance:
            current_cluster.append(level)
        else:
            clusters.append([level])
    return [float(sum(cluster) / len(cluster)) for cluster in clusters]


def _calculate_support_resistance(
    df: pd.DataFrame,
    current_price: Optional[float],
    atr_value: Optional[float],
    lookback: int = 200,
    pivot_window: int = 5,
    max_levels: int = 3,
    tolerance_percent: float = 0.005,
) -> Dict[str, Any]:
    """Calculate support and resistance levels using pivot highs/lows."""
    if df is None or df.empty:
        return {
            "supports": [],
            "resistances": [],
            "pivot_window": pivot_window,
            "lookback": 0,
            "tolerance": None,
        }

    data = df.tail(lookback)
    if len(data) < (pivot_window * 2 + 1):
        return {
            "supports": [],
            "resistances": [],
            "pivot_window": pivot_window,
            "lookback": len(data),
            "tolerance": None,
        }

    pivot_highs: list[float] = []
    pivot_lows: list[float] = []

    for i in range(pivot_window, len(data) - pivot_window):
        window_high = data["high"].iloc[i - pivot_window : i + pivot_window + 1]
        window_low = data["low"].iloc[i - pivot_window : i + pivot_window + 1]

        high_val = data["high"].iloc[i]
        low_val = data["low"].iloc[i]

        if pd.notna(high_val) and high_val == window_high.max():
            pivot_highs.append(float(high_val))
        if pd.notna(low_val) and low_val == window_low.min():
            pivot_lows.append(float(low_val))

    tolerance = None
    if atr_value is not None and atr_value > 0:
        tolerance = atr_value * 0.5
    elif current_price is not None and current_price > 0:
        tolerance = current_price * tolerance_percent
    else:
        tolerance = 0.0

    supports = _cluster_levels(pivot_lows, tolerance)
    resistances = _cluster_levels(pivot_highs, tolerance)

    if current_price is not None:
        supports = [level for level in supports if level <= current_price]
        resistances = [level for level in resistances if level >= current_price]

        supports = sorted(supports, key=lambda x: abs(current_price - x))
        resistances = sorted(resistances, key=lambda x: abs(current_price - x))
    else:
        supports = sorted(supports)
        resistances = sorted(resistances)

    return {
        "supports": supports[:max_levels],
        "resistances": resistances[:max_levels],
        "pivot_window": pivot_window,
        "lookback": len(data),
        "tolerance": tolerance,
    }


def _safe_get(df: pd.DataFrame, column: str, idx: int) -> Optional[float]:
    """Safely get a value from a DataFrame column."""
    try:
        if df is None or df.empty:
            return None
        if column not in df.columns:
            return None
        val = df[column].iloc[idx]
        if pd.isna(val):
            return None
        return float(val)
    except (IndexError, KeyError):
        return None


def _safe_get_series(series: Optional[pd.Series], idx: int) -> Optional[float]:
    """Safely get a value from a Series."""
    try:
        if series is None or series.empty:
            return None
        val = series.iloc[idx]
        if pd.isna(val):
            return None
        return float(val)
    except (IndexError, KeyError):
        return None


def _calc_percent_diff(current: float, reference: float) -> Optional[float]:
    """Calculate percentage difference from reference."""
    if reference is None or reference == 0:
        return None
    return ((current - reference) / reference) * 100


class TechnicalAnalysisTool(AutoTool):
    """
    Technical Analysis Tool using Birdeye OHLCV data.

    Provides comprehensive technical indicators for any token:
    - Trend: EMA (9, 21, 50, 200), SMA (20, 50, 200), MACD, ADX
    - Momentum: RSI, Stochastic, CCI, Williams %R, ROC, MFI
    - Volatility: Bollinger Bands, ATR, Keltner Channels
    - Volume: OBV, VWAP, Volume SMA
    - Support/Resistance: Pivot-based levels

    Requires minimum 200 candles for reliable calculation.
    """

    def __init__(self, registry: Optional[ToolRegistry] = None):
        super().__init__(
            name="technical_analysis",
            description=(
                "Get comprehensive technical analysis indicators for a token. "
                "Provides trend indicators (EMA, SMA, MACD, ADX), momentum indicators "
                "(RSI, Stochastic, CCI, Williams %R, ROC, MFI), volatility indicators "
                "(Bollinger Bands, ATR, Keltner Channels), volume indicators "
                "(OBV, VWAP), and support/resistance levels. Requires token address. "
                "Returns raw indicator values "
                "without interpretation. Use timeframe parameter to specify candle "
                "interval (default: 4h). Requires minimum 200 candles of data."
            ),
            registry=registry,
        )
        self.birdeye_base_url = "https://public-api.birdeye.so"
        self.api_key = ""
        self.default_chain = "solana"

    def get_schema(self) -> Dict[str, Any]:
        """Return the JSON schema for the tool parameters."""
        return {
            "type": "object",
            "properties": {
                "address": {
                    "type": "string",
                    "description": "The token mint address to analyze.",
                },
                "timeframe": {
                    "type": "string",
                    "description": (
                        "Candle timeframe for analysis. Options: 1m, 5m, 15m, 30m, "
                        "1h, 2h, 4h, 8h, 1d. Default: 4h. Shorter timeframes show "
                        "recent trends, longer timeframes show macro trends."
                    ),
                    "default": "4h",
                },
            },
            "required": ["address", "timeframe"],
            "additionalProperties": False,
        }

    def configure(self, config: Dict[str, Any]) -> None:
        """Configure the tool with API credentials."""
        # Get API key and chain from tools.technical_analysis config
        tools_config = config.get("tools", {})
        ta_config = tools_config.get("technical_analysis", {})
        if isinstance(ta_config, dict):
            self.api_key = ta_config.get("api_key", "")
            if ta_config.get("chain"):
                self.default_chain = ta_config.get("chain")

    async def _get_ohlcv_data(
        self, address: str, timeframe: str, chain: str
    ) -> Dict[str, Any]:
        """Fetch OHLCV data from Birdeye V3 API."""
        birdeye_type = TIMEFRAME_MAP.get(timeframe, "4H")

        # Calculate time range to get 500+ candles (we need 200 minimum)
        now = int(time.time())

        # Map timeframe to seconds
        timeframe_seconds = {
            "1m": 60,
            "5m": 300,
            "15m": 900,
            "30m": 1800,
            "1h": 3600,
            "2h": 7200,
            "4h": 14400,
            "8h": 28800,
            "1d": 86400,
        }
        interval_seconds = timeframe_seconds.get(timeframe, 14400)

        # Request 500 candles worth of data
        time_from = now - (500 * interval_seconds)

        url = f"{self.birdeye_base_url}/defi/v3/ohlcv"
        params = {
            "address": address,
            "type": birdeye_type,
            "time_from": time_from,
            "time_to": now,
            "currency": "usd",
        }
        headers = {
            "accept": "application/json",
            "X-API-KEY": self.api_key,
            "x-chain": chain,
        }

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(url, params=params, headers=headers)
            response.raise_for_status()
            return response.json()

    async def _get_token_overview(self, address: str, chain: str) -> Dict[str, Any]:
        """Fetch token overview from Birdeye API."""
        url = f"{self.birdeye_base_url}/defi/token_overview"
        params = {"address": address}
        headers = {
            "accept": "application/json",
            "X-API-KEY": self.api_key,
            "x-chain": chain,
        }

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(url, params=params, headers=headers)
            response.raise_for_status()
            return response.json()

    async def execute(
        self,
        address: str,
        timeframe: str = "4h",
    ) -> Dict[str, Any]:
        """
        Execute technical analysis for a token.

        Args:
            address: Token mint address
            timeframe: Candle interval (1m, 5m, 15m, 30m, 1h, 2h, 4h, 8h, 1d)

        Returns:
            Dictionary with technical indicators or error
        """
        if not PANDAS_TA_AVAILABLE:
            return {
                "status": "error",
                "error": "missing_dependency",
                "message": "pandas_ta is not installed. Install pandas_ta to use technical_analysis.",
            }

        # Validate timeframe
        timeframe = timeframe.lower()
        if timeframe not in TIMEFRAME_MAP:
            return {
                "status": "error",
                "error": "invalid_timeframe",
                "message": f"Invalid timeframe '{timeframe}'. Valid options: {list(TIMEFRAME_MAP.keys())}",
            }

        chain = self.default_chain

        try:
            # Fetch OHLCV data - this can raise HTTPStatusError
            ohlcv_response = await self._get_ohlcv_data(address, timeframe, chain)

            # Fetch overview separately - non-critical, can fail silently
            overview_data = None
            try:
                overview_response = await self._get_token_overview(address, chain)
                if overview_response.get("success"):
                    overview_data = overview_response.get("data", {})
            except Exception as e:
                logger.warning(f"Token overview fetch error: {e}")

            # Check if OHLCV request was successful
            if not ohlcv_response.get("success"):
                return {
                    "status": "error",
                    "error": "api_error",
                    "message": ohlcv_response.get("message", "OHLCV request failed"),
                }

            # Extract candle data
            items = ohlcv_response.get("data", {}).get("items", [])
            if not items:
                return {
                    "status": "error",
                    "error": "no_data",
                    "message": "No OHLCV data available for this token",
                }

            # Check minimum candle requirement
            if len(items) < MIN_CANDLES_REQUIRED:
                return {
                    "status": "error",
                    "error": "insufficient_data",
                    "candles_available": len(items),
                    "candles_required": MIN_CANDLES_REQUIRED,
                    "message": f"Insufficient data: {len(items)} candles available, {MIN_CANDLES_REQUIRED} required for reliable technical analysis",
                }

            # Convert to DataFrame
            df = pd.DataFrame(items)
            df = df.rename(
                columns={
                    "o": "open",
                    "h": "high",
                    "l": "low",
                    "c": "close",
                    "v": "volume",
                    "unix_time": "timestamp",
                }
            )

            # Ensure proper types
            for col in ["open", "high", "low", "close", "volume"]:
                df[col] = pd.to_numeric(df[col], errors="coerce")

            # Sort by timestamp ascending
            df = df.sort_values("timestamp").reset_index(drop=True)

            # Calculate indicators
            indicators = calculate_indicators(df)

            # Get current price and time info
            current_price = df["close"].iloc[-1]
            first_timestamp = df["timestamp"].iloc[0]
            last_timestamp = df["timestamp"].iloc[-1]

            # Build token info
            token_info = {
                "address": address,
                "symbol": overview_data.get("symbol") if overview_data else None,
                "name": overview_data.get("name") if overview_data else None,
                "decimals": overview_data.get("decimals") if overview_data else None,
            }

            # Build current market data
            current_data = {
                "price": current_price,
            }
            if overview_data:
                current_data.update(
                    {
                        "price_24h_ago": overview_data.get("history24hPrice"),
                        "price_change_24h_percent": overview_data.get(
                            "priceChange24hPercent"
                        ),
                        "market_cap": overview_data.get("marketCap"),
                        "liquidity": overview_data.get("liquidity"),
                    }
                )

            return {
                "status": "success",
                "token": token_info,
                "analysis": {
                    "timeframe": timeframe,
                    "candles_analyzed": len(items),
                    "data_start": datetime.fromtimestamp(
                        first_timestamp, timezone.utc
                    ).isoformat(),
                    "data_end": datetime.fromtimestamp(
                        last_timestamp, timezone.utc
                    ).isoformat(),
                },
                "current": current_data,
                **indicators,
            }

        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error: {e}")
            if e.response.status_code == 401:
                return {
                    "status": "error",
                    "error": "unauthorized",
                    "message": "Invalid or missing Birdeye API key",
                }
            elif e.response.status_code == 404:
                return {
                    "status": "error",
                    "error": "token_not_found",
                    "address": address,
                    "message": "Token not found",
                }
            return {
                "status": "error",
                "error": "api_error",
                "message": f"API error: {e.response.status_code}",
            }
        except Exception as e:
            logger.error(f"Technical analysis error: {e}")
            return {
                "status": "error",
                "error": "internal_error",
                "message": str(e),
            }


class TechnicalAnalysisPlugin:
    """Plugin for Technical Analysis tool."""

    def __init__(self):
        self.name = "technical_analysis"
        self.config = None
        self.tool_registry = None
        self._tool = None

    @property
    def description(self):
        return "Plugin for comprehensive technical analysis of tokens."

    def initialize(self, tool_registry: ToolRegistry) -> None:
        self.tool_registry = tool_registry
        self._tool = TechnicalAnalysisTool(registry=tool_registry)

    def configure(self, config: Dict[str, Any]) -> None:  # pragma: no cover
        self.config = config
        if self._tool:
            self._tool.configure(self.config)

    def get_tools(self) -> list[AutoTool]:  # pragma: no cover
        return [self._tool] if self._tool else []


def get_plugin():  # pragma: no cover
    return TechnicalAnalysisPlugin()
