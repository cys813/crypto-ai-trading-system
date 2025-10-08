"""
技术分析数据模型

包含技术分析结果、指标计算和策略分析相关的数据库模型。
"""

from datetime import datetime
from typing import Optional, Dict, Any
from sqlalchemy import Column, String, DateTime, Float, Integer, Text, JSON, Boolean, ForeignKey
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import UUID, ARRAY

from .base import BaseModel


class TechnicalAnalysis(BaseModel):
    """技术分析结果模型"""

    __tablename__ = "technical_analysis"

    # 基本信息
    symbol = Column(String(20), nullable=False, index=True)
    exchange = Column(String(50), nullable=False, default="binance")
    timeframe = Column(String(10), nullable=False)  # 1m, 5m, 15m, 1h, 4h, 1d

    # 分析结果
    signal_type = Column(String(20), nullable=False)  # 'long', 'short', 'neutral'
    signal_strength = Column(Float, nullable=False)  # 0.0 - 1.0
    confidence_score = Column(Float, nullable=False)  # 0.0 - 1.0

    # 关键价格点
    current_price = Column(Float, nullable=False)
    support_level = Column(Float)  # 支撑位
    resistance_level = Column(Float)  # 阻力位
    stop_loss_level = Column(Float)  # 止损位
    take_profit_level = Column(Float)  # 止盈位

    # 技术指标
    indicators = Column(JSON, nullable=False)  # 存储各种技术指标
    trend_analysis = Column(JSON)  # 趋势分析结果
    pattern_recognition = Column(JSON)  # 形态识别结果

    # 策略建议
    entry_conditions = Column(JSON)  # 入场条件
    exit_conditions = Column(JSON)  # 出场条件
    risk_factors = Column(JSON)  # 风险因素
    position_sizing = Column(JSON)  # 仓位建议

    # 元数据
    analysis_version = Column(String(20), default="1.0")
    analysis_period_hours = Column(Integer, default=24)  # 分析周期（小时）
    data_points_count = Column(Integer)  # 使用的数据点数量

    # 时间信息
    analysis_timestamp = Column(DateTime(timezone=True), nullable=False, index=True)
    expires_at = Column(DateTime(timezone=True))  # 分析结果过期时间

    # 外键关联
    trading_symbol_id = Column(UUID(as_uuid=True), ForeignKey("trading_symbols.id"))

    def __repr__(self):
        return f"<TechnicalAnalysis(symbol={self.symbol}, signal={self.signal_type}, confidence={self.confidence_score})>"

    @property
    def is_valid(self) -> bool:
        """检查分析结果是否仍然有效"""
        if not self.expires_at:
            return True
        return datetime.utcnow() < self.expires_at

    @property
    def risk_reward_ratio(self) -> Optional[float]:
        """计算风险回报比"""
        if not all([self.take_profit_level, self.stop_loss_level, self.current_price]):
            return None

        if self.signal_type == "long":
            profit_potential = self.take_profit_level - self.current_price
            risk_amount = self.current_price - self.stop_loss_level
        elif self.signal_type == "short":
            profit_potential = self.current_price - self.take_profit_level
            risk_amount = self.stop_loss_level - self.current_price
        else:
            return None

        return profit_potential / risk_amount if risk_amount > 0 else None

    def get_indicator_value(self, indicator_name: str, default=None):
        """获取特定技术指标的值"""
        if not self.indicators:
            return default
        return self.indicators.get(indicator_name, default)

    def get_risk_level(self) -> str:
        """获取风险等级"""
        confidence = self.confidence_score or 0

        if confidence >= 0.8:
            return "low"
        elif confidence >= 0.6:
            return "medium"
        else:
            return "high"


class TechnicalIndicator(BaseModel):
    """单个技术指标模型"""

    __tablename__ = "technical_indicators"

    # 基本信息
    name = Column(String(50), nullable=False)
    symbol = Column(String(20), nullable=False, index=True)
    timeframe = Column(String(10), nullable=False)

    # 指标值
    value = Column(Float, nullable=False)
    previous_value = Column(Float)  # 前一个值
    change = Column(Float)  # 变化量
    change_percent = Column(Float)  # 变化百分比

    # 指标参数
    parameters = Column(JSON)  # 指标计算参数
    signal = Column(String(20))  # 'buy', 'sell', 'neutral'
    signal_strength = Column(Float)  # 信号强度

    # 分类信息
    category = Column(String(30))  # 'trend', 'momentum', 'volatility', 'volume'
    subcategory = Column(String(30))

    # 时间信息
    timestamp = Column(DateTime(timezone=True), nullable=False, index=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    def __repr__(self):
        return f"<TechnicalIndicator(name={self.name}, symbol={self.symbol}, value={self.value})>"


class KlineData(BaseModel):
    """K线数据模型 - TimescaleDB超表"""

    __tablename__ = "kline_data"

    # 时间和标识
    timestamp = Column(DateTime(timezone=True), nullable=False, primary_key=True)
    symbol_id = Column(UUID(as_uuid=True), ForeignKey("trading_symbols.id"), primary_key=True)
    exchange_id = Column(UUID(as_uuid=True), ForeignKey("exchanges.id"), primary_key=True)

    # OHLCV数据
    open_price = Column(Float, nullable=False)
    high_price = Column(Float, nullable=False)
    low_price = Column(Float, nullable=False)
    close_price = Column(Float, nullable=False)
    volume = Column(Float, nullable=False)
    quote_volume = Column(Float)  # 成交额
    trades_count = Column(Integer)  # 成交笔数

    # 技术指标（预计算）
    sma_20 = Column(Float)
    ema_12 = Column(Float)
    ema_26 = Column(Float)
    rsi = Column(Float)
    macd = Column(Float)
    macd_signal = Column(Float)
    macd_histogram = Column(Float)
    bollinger_upper = Column(Float)
    bollinger_middle = Column(Float)
    bollinger_lower = Column(Float)

    # 统计信息
    price_change = Column(Float)
    price_change_percent = Column(Float)
    volume_change = Column(Float)
    volume_change_percent = Column(Float)

    # 数据质量标记
    is_anomaly = Column(Boolean, default=False)
    data_quality_score = Column(Float, default=1.0)  # 0.0 - 1.0
    interpolation_used = Column(Boolean, default=False)

    def __repr__(self):
        return f"<KlineData(symbol={self.symbol_id}, time={self.timestamp}, close={self.close_price})>"

    @property
    def is_bullish(self) -> bool:
        """是否为阳线"""
        return self.close_price > self.open_price

    @property
    def is_bearish(self) -> bool:
        """是否为阴线"""
        return self.close_price < self.open_price

    @property
    def body_size(self) -> float:
        """实体大小"""
        return abs(self.close_price - self.open_price)

    @property
    def upper_shadow(self) -> float:
        """上影线长度"""
        return self.high_price - max(self.open_price, self.close_price)

    @property
    def lower_shadow(self) -> float:
        """下影线长度"""
        return min(self.open_price, self.close_price) - self.low_price

    @property
    def typical_price(self) -> float:
        """典型价格 (H+L+C)/3"""
        return (self.high_price + self.low_price + self.close_price) / 3

    @property
    def weighted_price(self) -> float:
        """加权平均价格 (H+L+C+C)/4"""
        return (self.high_price + self.low_price + 2 * self.close_price) / 4


class AnalysisSession(BaseModel):
    """分析会话模型"""

    __tablename__ = "analysis_sessions"

    # 会话信息
    session_id = Column(String(100), nullable=False, unique=True, index=True)
    analysis_type = Column(String(50), nullable=False)  # 'long_strategy', 'short_strategy'
    status = Column(String(20), nullable=False, default='pending')  # 'pending', 'running', 'completed', 'failed'

    # 分析参数
    symbols = Column(ARRAY(String))  # 分析的交易符号
    timeframes = Column(ARRAY(String))  # 分析的时间框架
    analysis_period_hours = Column(Integer, default=24)
    confidence_threshold = Column(Float, default=0.7)

    # 执行信息
    started_at = Column(DateTime(timezone=True))
    completed_at = Column(DateTime(timezone=True))
    execution_time_seconds = Column(Float)

    # 结果统计
    total_analyses = Column(Integer, default=0)
    successful_analyses = Column(Integer, default=0)
    failed_analyses = Column(Integer, default=0)

    # 错误信息
    error_message = Column(Text)
    error_details = Column(JSON)

    # 元数据
    llm_provider = Column(String(50))  # 使用的LLM提供商
    llm_model = Column(String(100))
    total_tokens_used = Column(Integer)
    total_cost_usd = Column(Float)

    def __repr__(self):
        return f"<AnalysisSession(id={self.session_id}, type={self.analysis_type}, status={self.status})>"

    @property
    def success_rate(self) -> float:
        """成功率"""
        if self.total_analyses == 0:
            return 0.0
        return self.successful_analyses / self.total_analyses

    @property
    def is_completed(self) -> bool:
        """是否已完成"""
        return self.status in ['completed', 'failed']

    @property
    def duration_seconds(self) -> Optional[float]:
        """执行时长（秒）"""
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        return None


class StrategyTemplate(BaseModel):
    """策略模板模型"""

    __tablename__ = "strategy_templates"

    # 基本信息
    name = Column(String(100), nullable=False, unique=True)
    description = Column(Text)
    category = Column(String(50), nullable=False)  # 'trend_following', 'mean_reversion', 'breakout'
    strategy_type = Column(String(20), nullable=False)  # 'long', 'short', 'neutral'

    # 策略参数
    entry_conditions = Column(JSON, nullable=False)
    exit_conditions = Column(JSON, nullable=False)
    risk_management = Column(JSON, nullable=False)

    # 技术指标要求
    required_indicators = Column(ARRAY(String))
    indicator_parameters = Column(JSON)

    # 性能统计
    total_trades = Column(Integer, default=0)
    winning_trades = Column(Integer, default=0)
    profit_factor = Column(Float)
    max_drawdown = Column(Float)
    sharpe_ratio = Column(Float)

    # 配置信息
    is_active = Column(Boolean, default=True)
    is_default = Column(Boolean, default=False)
    min_confidence_threshold = Column(Float, default=0.7)

    # 时间信息
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now())
    last_used_at = Column(DateTime(timezone=True))

    def __repr__(self):
        return f"<StrategyTemplate(name={self.name}, type={self.strategy_type}, active={self.is_active})>"

    @property
    def win_rate(self) -> float:
        """胜率"""
        if self.total_trades == 0:
            return 0.0
        return self.winning_trades / self.total_trades

    def is_suitable_for_signal(self, signal_type: str, confidence: float) -> bool:
        """判断策略是否适合当前信号"""
        if not self.is_active:
            return False

        if confidence < self.min_confidence_threshold:
            return False

        # 可以根据策略类型和信号类型进行更复杂的匹配逻辑
        return True