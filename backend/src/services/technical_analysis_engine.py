"""
技术分析引擎

提供各种技术指标计算和市场分析功能。
"""

import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass

from ..core.exchange_integration import KlineData
from ..core.logging import BusinessLogger
from ..core.exceptions import ValidationError, BusinessLogicError
from ..models.technical_analysis import TechnicalAnalysis

logger = logging.getLogger(__name__)
business_logger = BusinessLogger("technical_analysis_engine")


@dataclass
class TechnicalIndicators:
    """技术指标结果"""
    rsi: Optional[float] = None
    macd_line: Optional[float] = None
    macd_signal: Optional[float] = None
    macd_histogram: Optional[float] = None
    bollinger_upper: Optional[float] = None
    bollinger_middle: Optional[float] = None
    bollinger_lower: Optional[float] = None
    sma_20: Optional[float] = None
    sma_50: Optional[float] = None
    ema_12: Optional[float] = None
    ema_26: Optional[float] = None
    volume_sma: Optional[float] = None
    volume_ratio: Optional[float] = None
    atr: Optional[float] = None  # Average True Range
    stochastic_k: Optional[float] = None
    stochastic_d: Optional[float] = None
    williams_r: Optional[float] = None
    cci: Optional[float] = None  # Commodity Channel Index


@dataclass
class MarketAnalysis:
    """市场分析结果"""
    trend_direction: str  # 'uptrend', 'downtrend', 'sideways'
    trend_strength: float  # 0.0 - 1.0
    volatility_level: str  # 'low', 'medium', 'high'
    volume_profile: str  # 'increasing', 'decreasing', 'stable'
    market_sentiment: str  # 'bullish', 'bearish', 'neutral'
    support_levels: List[float]
    resistance_levels: List[float]
    key_levels: Dict[str, float]
    pattern_signals: List[str]


class TechnicalAnalysisEngine:
    """技术分析引擎"""

    def __init__(self):
        self.logger = logger
        self.business_logger = business_logger

        # 默认参数
        self.default_rsi_period = 14
        self.default_macd_fast = 12
        self.default_macd_slow = 26
        self.default_macd_signal = 9
        self.default_bb_period = 20
        self.default_bb_std = 2
        self.default_sma_periods = [20, 50, 200]
        self.default_ema_periods = [12, 26]

    async def analyze(
        self,
        klines: List[KlineData],
        symbol: str = None,
        timeframe: str = None
    ) -> Dict[str, Any]:
        """完整技术分析"""
        try:
            if len(klines) < 50:
                raise ValidationError("数据不足，至少需要50个K线数据点")

            # 转换为DataFrame便于计算
            df = self._klines_to_dataframe(klines)

            # 计算技术指标
            indicators = self._calculate_all_indicators(df)

            # 市场分析
            market_analysis = self._analyze_market_conditions(df, indicators)

            # 支撑阻力位分析
            support_resistance = self._find_support_resistance_levels(df)

            # 形态识别
            patterns = self._identify_patterns(df)

            # 综合分析结果
            analysis_result = {
                "symbol": symbol,
                "timeframe": timeframe,
                "current_price": float(df['close'].iloc[-1]),
                "analysis_timestamp": datetime.utcnow(),
                "indicators": indicators,
                "market_analysis": market_analysis,
                "support_resistance": support_resistance,
                "patterns": patterns,
                "data_points_count": len(klines),
                "analysis_period_hours": self._estimate_period_hours(klines),
                "technical_signals": self._generate_technical_signals(indicators, market_analysis)
            }

            self.business_logger.log_system_event(
                event_type="technical_analysis_completed",
                severity="info",
                message=f"技术分析完成: {symbol} {timeframe}",
                details={
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "data_points": len(klines),
                    "current_price": analysis_result["current_price"],
                    "trend_direction": market_analysis.trend_direction,
                    "market_sentiment": market_analysis.market_sentiment
                }
            )

            return analysis_result

        except Exception as e:
            self.logger.error(f"技术分析失败: {e}")
            raise

    def calculate_rsi(self, klines: List[KlineData], period: int = 14) -> float:
        """计算RSI指标"""
        try:
            if len(klines) < period + 1:
                raise ValidationError(f"RSI计算需要至少 {period + 1} 个数据点")

            closes = [kline.close for kline in klines]
            df = pd.Series(closes)

            # 计算价格变化
            delta = df.diff()

            # 分离涨跌
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)

            # 计算平均涨跌幅
            avg_gain = gain.rolling(window=period).mean()
            avg_loss = loss.rolling(window=period).mean()

            # 计算RS和RSI
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))

            return float(rsi.iloc[-1])

        except Exception as e:
            self.logger.error(f"RSI计算失败: {e}")
            raise

    def calculate_macd(
        self,
        klines: List[KlineData],
        fast_period: int = 12,
        slow_period: int = 26,
        signal_period: int = 9
    ) -> Dict[str, float]:
        """计算MACD指标"""
        try:
            if len(klines) < slow_period + signal_period:
                raise ValidationError("数据不足，无法计算MACD")

            closes = [kline.close for kline in klines]
            df = pd.Series(closes)

            # 计算EMA
            ema_fast = df.ewm(span=fast_period).mean()
            ema_slow = df.ewm(span=slow_period).mean()

            # 计算MACD线
            macd_line = ema_fast - ema_slow

            # 计算信号线
            signal_line = macd_line.ewm(span=signal_period).mean()

            # 计算柱状图
            histogram = macd_line - signal_line

            return {
                "macd_line": float(macd_line.iloc[-1]),
                "signal_line": float(signal_line.iloc[-1]),
                "histogram": float(histogram.iloc[-1])
            }

        except Exception as e:
            self.logger.error(f"MACD计算失败: {e}")
            raise

    def calculate_bollinger_bands(
        self,
        klines: List[KlineData],
        period: int = 20,
        std_dev: float = 2
    ) -> Dict[str, float]:
        """计算布林带"""
        try:
            if len(klines) < period:
                raise ValidationError(f"布林带计算需要至少 {period} 个数据点")

            closes = [kline.close for kline in klines]
            df = pd.Series(closes)

            # 计算移动平均线
            middle_band = df.rolling(window=period).mean()
            std = df.rolling(window=period).std()

            # 计算上下轨
            upper_band = middle_band + (std * std_dev)
            lower_band = middle_band - (std * std_dev)

            return {
                "upper_band": float(upper_band.iloc[-1]),
                "middle_band": float(middle_band.iloc[-1]),
                "lower_band": float(lower_band.iloc[-1])
            }

        except Exception as e:
            self.logger.error(f"布林带计算失败: {e}")
            raise

    def calculate_sma(self, klines: List[KlineData], period: int) -> List[Optional[float]]:
        """计算简单移动平均线"""
        try:
            if len(klines) < period:
                return [None] * len(klines)

            closes = [kline.close for kline in klines]
            df = pd.Series(closes)

            sma = df.rolling(window=period).mean()
            return [float(val) if pd.notna(val) else None for val in sma]

        except Exception as e:
            self.logger.error(f"SMA计算失败: {e}")
            raise

    def calculate_ema(self, klines: List[KlineData], period: int) -> List[Optional[float]]:
        """计算指数移动平均线"""
        try:
            if len(klines) < period:
                return [None] * len(klines)

            closes = [kline.close for kline in klines]
            df = pd.Series(closes)

            ema = df.ewm(span=period).mean()
            return [float(val) if pd.notna(val) else None for val in ema]

        except Exception as e:
            self.logger.error(f"EMA计算失败: {e}")
            raise

    def _klines_to_dataframe(self, klines: List[KlineData]) -> pd.DataFrame:
        """将K线数据转换为DataFrame"""
        data = {
            'timestamp': [kline.timestamp for kline in klines],
            'open': [kline.open for kline in klines],
            'high': [kline.high for kline in klines],
            'low': [kline.low for kline in klines],
            'close': [kline.close for kline in klines],
            'volume': [kline.volume for kline in klines]
        }

        df = pd.DataFrame(data)
        df.set_index('timestamp', inplace=True)
        return df

    def _calculate_all_indicators(self, df: pd.DataFrame) -> TechnicalIndicators:
        """计算所有技术指标"""
        indicators = TechnicalIndicators()

        try:
            # RSI
            indicators.rsi = self._calculate_rsi_dataframe(df, self.default_rsi_period)

            # MACD
            macd_result = self._calculate_macd_dataframe(
                df, self.default_macd_fast, self.default_macd_slow, self.default_macd_signal
            )
            indicators.macd_line = macd_result["macd_line"]
            indicators.macd_signal = macd_result["signal_line"]
            indicators.macd_histogram = macd_result["histogram"]

            # 布林带
            bb_result = self._calculate_bollinger_bands_dataframe(
                df, self.default_bb_period, self.default_bb_std
            )
            indicators.bollinger_upper = bb_result["upper_band"]
            indicators.bollinger_middle = bb_result["middle_band"]
            indicators.bollinger_lower = bb_result["lower_band"]

            # 移动平均线
            indicators.sma_20 = float(df['close'].rolling(window=20).mean().iloc[-1])
            indicators.sma_50 = float(df['close'].rolling(window=50).mean().iloc[-1])
            indicators.ema_12 = float(df['close'].ewm(span=12).mean().iloc[-1])
            indicators.ema_26 = float(df['close'].ewm(span=26).mean().iloc[-1])

            # 成交量指标
            indicators.volume_sma = float(df['volume'].rolling(window=20).mean().iloc[-1])
            current_volume = float(df['volume'].iloc[-1])
            indicators.volume_ratio = current_volume / indicators.volume_sma if indicators.volume_sma > 0 else 1.0

            # ATR (Average True Range)
            indicators.atr = self._calculate_atr(df, 14)

            # 随机指标
            stochastic_result = self._calculate_stochastic(df, 14, 3)
            indicators.stochastic_k = stochastic_result["k"]
            indicators.stochastic_d = stochastic_result["d"]

            # Williams %R
            indicators.williams_r = self._calculate_williams_r(df, 14)

            # CCI
            indicators.cci = self._calculate_cci(df, 20)

        except Exception as e:
            self.logger.error(f"技术指标计算失败: {e}")

        return indicators

    def _calculate_rsi_dataframe(self, df: pd.DataFrame, period: int = 14) -> Optional[float]:
        """使用DataFrame计算RSI"""
        try:
            delta = df['close'].diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)

            avg_gain = gain.rolling(window=period).mean()
            avg_loss = loss.rolling(window=period).mean()

            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))

            return float(rsi.iloc[-1]) if pd.notna(rsi.iloc[-1]) else None

        except Exception:
            return None

    def _calculate_macd_dataframe(
        self,
        df: pd.DataFrame,
        fast_period: int = 12,
        slow_period: int = 26,
        signal_period: int = 9
    ) -> Dict[str, Optional[float]]:
        """使用DataFrame计算MACD"""
        try:
            ema_fast = df['close'].ewm(span=fast_period).mean()
            ema_slow = df['close'].ewm(span=slow_period).mean()

            macd_line = ema_fast - ema_slow
            signal_line = macd_line.ewm(span=signal_period).mean()
            histogram = macd_line - signal_line

            return {
                "macd_line": float(macd_line.iloc[-1]) if pd.notna(macd_line.iloc[-1]) else None,
                "signal_line": float(signal_line.iloc[-1]) if pd.notna(signal_line.iloc[-1]) else None,
                "histogram": float(histogram.iloc[-1]) if pd.notna(histogram.iloc[-1]) else None
            }

        except Exception:
            return {"macd_line": None, "signal_line": None, "histogram": None}

    def _calculate_bollinger_bands_dataframe(
        self,
        df: pd.DataFrame,
        period: int = 20,
        std_dev: float = 2
    ) -> Dict[str, Optional[float]]:
        """使用DataFrame计算布林带"""
        try:
            middle_band = df['close'].rolling(window=period).mean()
            std = df['close'].rolling(window=period).std()

            upper_band = middle_band + (std * std_dev)
            lower_band = middle_band - (std * std_dev)

            return {
                "upper_band": float(upper_band.iloc[-1]) if pd.notna(upper_band.iloc[-1]) else None,
                "middle_band": float(middle_band.iloc[-1]) if pd.notna(middle_band.iloc[-1]) else None,
                "lower_band": float(lower_band.iloc[-1]) if pd.notna(lower_band.iloc[-1]) else None
            }

        except Exception:
            return {"upper_band": None, "middle_band": None, "lower_band": None}

    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> Optional[float]:
        """计算平均真实范围"""
        try:
            high_low = df['high'] - df['low']
            high_close = abs(df['high'] - df['close'].shift())
            low_close = abs(df['low'] - df['close'].shift())

            true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            atr = true_range.rolling(window=period).mean()

            return float(atr.iloc[-1]) if pd.notna(atr.iloc[-1]) else None

        except Exception:
            return None

    def _calculate_stochastic(self, df: pd.DataFrame, k_period: int = 14, d_period: int = 3) -> Dict[str, Optional[float]]:
        """计算随机指标"""
        try:
            lowest_low = df['low'].rolling(window=k_period).min()
            highest_high = df['high'].rolling(window=k_period).max()

            k_percent = 100 * ((df['close'] - lowest_low) / (highest_high - lowest_low))
            d_percent = k_percent.rolling(window=d_period).mean()

            return {
                "k": float(k_percent.iloc[-1]) if pd.notna(k_percent.iloc[-1]) else None,
                "d": float(d_percent.iloc[-1]) if pd.notna(d_percent.iloc[-1]) else None
            }

        except Exception:
            return {"k": None, "d": None}

    def _calculate_williams_r(self, df: pd.DataFrame, period: int = 14) -> Optional[float]:
        """计算Williams %R"""
        try:
            highest_high = df['high'].rolling(window=period).max()
            lowest_low = df['low'].rolling(window=period).min()

            williams_r = -100 * ((highest_high - df['close']) / (highest_high - lowest_low))
            return float(williams_r.iloc[-1]) if pd.notna(williams_r.iloc[-1]) else None

        except Exception:
            return None

    def _calculate_cci(self, df: pd.DataFrame, period: int = 20) -> Optional[float]:
        """计算商品通道指标"""
        try:
            typical_price = (df['high'] + df['low'] + df['close']) / 3
            sma_tp = typical_price.rolling(window=period).mean()
            mean_deviation = typical_price.rolling(window=period).apply(lambda x: np.abs(x - x.mean()).mean())

            cci = (typical_price - sma_tp) / (0.015 * mean_deviation)
            return float(cci.iloc[-1]) if pd.notna(cci.iloc[-1]) else None

        except Exception:
            return None

    def _analyze_market_conditions(
        self,
        df: pd.DataFrame,
        indicators: TechnicalIndicators
    ) -> MarketAnalysis:
        """分析市场条件"""
        try:
            current_price = df['close'].iloc[-1]

            # 趋势分析
            trend_direction, trend_strength = self._analyze_trend(df, indicators)

            # 波动性分析
            volatility_level = self._analyze_volatility(df, indicators)

            # 成交量分析
            volume_profile = self._analyze_volume(df, indicators)

            # 市场情绪分析
            market_sentiment = self._analyze_sentiment(indicators)

            # 支撑阻力位
            support_resistance = self._find_support_resistance_levels(df)

            return MarketAnalysis(
                trend_direction=trend_direction,
                trend_strength=trend_strength,
                volatility_level=volatility_level,
                volume_profile=volume_profile,
                market_sentiment=market_sentiment,
                support_levels=support_resistance["support_levels"],
                resistance_levels=support_resistance["resistance_levels"],
                key_levels=support_resistance["key_levels"],
                pattern_signals=[]
            )

        except Exception as e:
            self.logger.error(f"市场条件分析失败: {e}")
            # 返回默认值
            return MarketAnalysis(
                trend_direction="neutral",
                trend_strength=0.0,
                volatility_level="medium",
                volume_profile="stable",
                market_sentiment="neutral",
                support_levels=[],
                resistance_levels=[],
                key_levels={},
                pattern_signals=[]
            )

    def _analyze_trend(
        self,
        df: pd.DataFrame,
        indicators: TechnicalIndicators
    ) -> Tuple[str, float]:
        """分析趋势"""
        try:
            current_price = df['close'].iloc[-1]

            # 基于移动平均线的趋势判断
            sma_20 = indicators.sma_20
            sma_50 = indicators.sma_50

            if sma_20 and sma_50:
                if current_price > sma_20 > sma_50:
                    return "uptrend", 0.8
                elif current_price < sma_20 < sma_50:
                    return "downtrend", 0.8
                else:
                    return "sideways", 0.5

            # 基于EMA的趋势判断
            ema_12 = indicators.ema_12
            ema_26 = indicators.ema_26

            if ema_12 and ema_26:
                if ema_12 > ema_26:
                    return "uptrend", 0.6
                else:
                    return "downtrend", 0.6

            return "neutral", 0.3

        except Exception:
            return "neutral", 0.0

    def _analyze_volatility(
        self,
        df: pd.DataFrame,
        indicators: TechnicalIndicators
    ) -> str:
        """分析波动性"""
        try:
            atr = indicators.atr
            current_price = df['close'].iloc[-1]

            if atr and current_price:
                atr_ratio = atr / current_price

                if atr_ratio > 0.05:  # 5%以上
                    return "high"
                elif atr_ratio > 0.02:  # 2%-5%
                    return "medium"
                else:
                    return "low"

            return "medium"

        except Exception:
            return "medium"

    def _analyze_volume(self, df: pd.DataFrame, indicators: TechnicalIndicators) -> str:
        """分析成交量"""
        try:
            volume_ratio = indicators.volume_ratio

            if volume_ratio:
                if volume_ratio > 1.5:
                    return "increasing"
                elif volume_ratio < 0.7:
                    return "decreasing"
                else:
                    return "stable"

            return "stable"

        except Exception:
            return "stable"

    def _analyze_sentiment(self, indicators: TechnicalIndicators) -> str:
        """分析市场情绪"""
        try:
            rsi = indicators.rsi
            macd_histogram = indicators.macd_histogram

            bullish_signals = 0
            bearish_signals = 0

            # RSI信号
            if rsi:
                if rsi < 30:
                    bullish_signals += 2  # 超卖
                elif rsi > 70:
                    bearish_signals += 2  # 超买
                elif 30 <= rsi <= 70:
                    bullish_signals += 1  # 中性偏多

            # MACD信号
            if macd_histogram:
                if macd_histogram > 0:
                    bullish_signals += 1
                else:
                    bearish_signals += 1

            # 基于信号数量判断情绪
            if bullish_signals > bearish_signals:
                return "bullish"
            elif bearish_signals > bullish_signals:
                return "bearish"
            else:
                return "neutral"

        except Exception:
            return "neutral"

    def _find_support_resistance_levels(self, df: pd.DataFrame) -> Dict[str, Any]:
        """寻找支撑阻力位"""
        try:
            # 简单的支撑阻力位识别
            recent_highs = df['high'].rolling(window=10).max()
            recent_lows = df['low'].rolling(window=10).min()

            current_price = df['close'].iloc[-1]

            # 寻找最近的支撑阻力位
            resistance_levels = []
            support_levels = []

            for i in range(len(df) - 20, len(df)):
                high_val = df['high'].iloc[i]
                low_val = df['low'].iloc[i]

                # 检查是否为局部高点
                if i > 0 and i < len(df) - 1:
                    if (high_val > df['high'].iloc[i-1] and
                        high_val > df['high'].iloc[i+1] and
                        high_val > current_price):
                        resistance_levels.append(high_val)

                    # 检查是否为局部低点
                    if (low_val < df['low'].iloc[i-1] and
                        low_val < df['low'].iloc[i+1] and
                        low_val < current_price):
                        support_levels.append(low_val)

            # 去重并排序
            resistance_levels = sorted(list(set(resistance_levels)), reverse=True)[:5]
            support_levels = sorted(list(set(support_levels)))[:5]

            # 找到最接近的支撑阻力位
            nearest_resistance = min(resistance_levels) if resistance_levels else None
            nearest_support = max(support_levels) if support_levels else None

            return {
                "support_levels": support_levels,
                "resistance_levels": resistance_levels,
                "key_levels": {
                    "nearest_support": nearest_support,
                    "nearest_resistance": nearest_support
                }
            }

        except Exception as e:
            self.logger.error(f"支撑阻力位分析失败: {e}")
            return {
                "support_levels": [],
                "resistance_levels": [],
                "key_levels": {}
            }

    def _identify_patterns(self, df: pd.DataFrame) -> List[str]:
        """识别技术形态"""
        patterns = []

        try:
            # 简单的形态识别逻辑
            # 这里可以扩展更多复杂的形态识别算法

            # 锤头线
            latest_candle = df.iloc[-1]
            body_size = abs(latest_candle['close'] - latest_candle['open'])
            upper_shadow = latest_candle['high'] - max(latest_candle['close'], latest_candle['open'])
            lower_shadow = min(latest_candle['close'], latest_candle['open']) - latest_candle['low']

            if (lower_shadow > 2 * body_size and upper_shadow < body_size * 0.1):
                if latest_candle['close'] > latest_candle['open']:
                    patterns.append("bullish_hammer")
                else:
                    patterns.append("bearish_hammer")

            # 十字星
            if body_size < (latest_candle['high'] - latest_candle['low']) * 0.1:
                patterns.append("doji")

        except Exception as e:
            self.logger.error(f"形态识别失败: {e}")

        return patterns

    def _generate_technical_signals(
        self,
        indicators: TechnicalIndicators,
        market_analysis: MarketAnalysis
    ) -> List[Dict[str, Any]]:
        """生成技术信号"""
        signals = []

        try:
            # RSI信号
            if indicators.rsi:
                if indicators.rsi < 30:
                    signals.append({
                        "type": "buy",
                        "indicator": "RSI",
                        "strength": "strong",
                        "reason": f"RSI超卖 ({indicators.rsi:.1f})"
                    })
                elif indicators.rsi > 70:
                    signals.append({
                        "type": "sell",
                        "indicator": "RSI",
                        "strength": "strong",
                        "reason": f"RSI超买 ({indicators.rsi:.1f})"
                    })

            # MACD信号
            if indicators.macd_histogram:
                if indicators.macd_histogram > 0:
                    signals.append({
                        "type": "buy",
                        "indicator": "MACD",
                        "strength": "medium",
                        "reason": "MACD柱状图为正"
                    })
                else:
                    signals.append({
                        "type": "sell",
                        "indicator": "MACD",
                        "strength": "medium",
                        "reason": "MACD柱状图为负"
                    })

            # 布林带信号
            if (indicators.bollinger_upper and indicators.bollinger_lower and
                indicators.bollinger_middle):
                current_price = indicators.bollinger_middle  # 这里应该传入当前价格

                if current_price > indicators.bollinger_upper:
                    signals.append({
                        "type": "sell",
                        "indicator": "Bollinger Bands",
                        "strength": "medium",
                        "reason": "价格突破布林带上轨"
                    })
                elif current_price < indicators.bollinger_lower:
                    signals.append({
                        "type": "buy",
                        "indicator": "Bollinger Bands",
                        "strength": "medium",
                        "reason": "价格跌破布林带下轨"
                    })

        except Exception as e:
            self.logger.error(f"技术信号生成失败: {e}")

        return signals

    def _estimate_period_hours(self, klines: List[KlineData]) -> int:
        """估算数据覆盖的时间范围（小时）"""
        if len(klines) < 2:
            return 0

        time_diff = klines[-1].timestamp - klines[0].timestamp
        return int(time_diff.total_seconds() / 3600)


# 便捷函数
def analyze_klines(klines: List[KlineData]) -> Dict[str, Any]:
    """分析K线数据的便捷函数"""
    engine = TechnicalAnalysisEngine()
    return asyncio.run(engine.analyze(klines))


def calculate_rsi(klines: List[KlineData], period: int = 14) -> float:
    """计算RSI的便捷函数"""
    engine = TechnicalAnalysisEngine()
    return engine.calculate_rsi(klines, period)


def calculate_macd(klines: List[KlineData]) -> Dict[str, float]:
    """计算MACD的便捷函数"""
    engine = TechnicalAnalysisEngine()
    return engine.calculate_macd(klines)