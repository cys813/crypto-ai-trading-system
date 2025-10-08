"""
市场数据模型

包含交易所、交易对、K线数据和技术分析相关的数据模型。
"""

from sqlalchemy import Column, String, Boolean, DECIMAL, Integer, Text, ForeignKey, DateTime
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import UUID, JSONB
from .base import BaseModel


class Exchange(BaseModel):
    """交易所模型"""

    __tablename__ = "exchanges"

    name = Column(String(100), unique=True, nullable=False)
    code = Column(String(20), unique=True, nullable=False)  # binance, coinbase, etc.

    # API配置
    api_base_url = Column(String(500), nullable=False)
    api_version = Column(String(20))
    is_testnet = Column(Boolean, default=False)

    # 限流配置
    rate_limit_requests_per_minute = Column(Integer)
    rate_limit_orders_per_second = Column(Integer)
    rate_limit_weight_per_minute = Column(Integer)

    # 状态信息
    is_active = Column(Boolean, default=True)
    last_heartbeat = Column(DateTime(timezone=True))
    status = Column(String(20), default="online")  # online, offline, maintenance

    # 关系
    kline_data = relationship("KlineData", back_populates="exchange")
    technical_analysis = relationship("TechnicalAnalysis", back_populates="exchange")
    trading_orders = relationship("TradingOrder", back_populates="exchange")
    positions = relationship("Position", back_populates="exchange")

    def __repr__(self):
        return f"<Exchange(code='{self.code}', status='{self.status}')>"


class TradingSymbol(BaseModel):
    """交易符号模型"""

    __tablename__ = "trading_symbols"

    symbol = Column(String(20), unique=True, nullable=False, index=True)
    base_asset = Column(String(10), nullable=False)
    quote_asset = Column(String(10), nullable=False)

    # 符号信息
    status = Column(String(20), default="active")
    is_spot_trading = Column(Boolean, default=True)
    is_margin_trading = Column(Boolean, default=False)

    # 交易限制
    min_qty = Column(DECIMAL(20, 8))
    max_qty = Column(DECIMAL(20, 8))
    step_size = Column(DECIMAL(20, 8))
    min_price = Column(DECIMAL(20, 8))
    max_price = Column(DECIMAL(20, 8))
    tick_size = Column(DECIMAL(20, 8))

    # 关系
    trading_strategies = relationship("TradingStrategy", back_populates="symbol")
    kline_data = relationship("KlineData", back_populates="symbol")
    technical_analysis = relationship("TechnicalAnalysis", back_populates="symbol")

    def __repr__(self):
        return f"<TradingSymbol(symbol='{self.symbol}', status='{self.status}')>"


class KlineData(BaseModel):
    """K线数据模型 - TimescaleDB超表"""

    __tablename__ = "kline_data"

    time = Column(DateTime(timezone=True), nullable=False, primary_key=True)
    symbol_id = Column(UUID(as_uuid=True), ForeignKey("trading_symbols.id"), nullable=False, primary_key=True)
    exchange_id = Column(UUID(as_uuid=True), ForeignKey("exchanges.id"), nullable=False, primary_key=True)

    # OHLCV数据
    open_price = Column(DECIMAL(20, 8), nullable=False)
    high_price = Column(DECIMAL(20, 8), nullable=False)
    low_price = Column(DECIMAL(20, 8), nullable=False)
    close_price = Column(DECIMAL(20, 8), nullable=False)
    volume = Column(DECIMAL(20, 8), nullable=False)
    quote_volume = Column(DECIMAL(20, 8))

    # 技术指标
    sma_20 = Column(DECIMAL(20, 8))
    ema_12 = Column(DECIMAL(20, 8))
    ema_26 = Column(DECIMAL(20, 8))
    rsi = Column(DECIMAL(5, 2))
    macd = Column(DECIMAL(20, 8))
    macd_signal = Column(DECIMAL(20, 8))
    bollinger_upper = Column(DECIMAL(20, 8))
    bollinger_lower = Column(DECIMAL(20, 8))

    # 关系
    symbol = relationship("TradingSymbol", back_populates="kline_data")
    exchange = relationship("Exchange", back_populates="kline_data")
    technical_analysis = relationship("TechnicalAnalysis", back_populates="kline_data")

    def __repr__(self):
        return f"<KlineData(time='{self.time}', symbol_id='{self.symbol_id}')>"


class TechnicalAnalysis(BaseModel):
    """技术分析结果模型"""

    __tablename__ = "technical_analysis"

    # 分析信息
    symbol_id = Column(UUID(as_uuid=True), ForeignKey("trading_symbols.id"), nullable=False)
    exchange_id = Column(UUID(as_uuid=True), ForeignKey("exchanges.id"), nullable=False)
    timeframe = Column(String(10), nullable=False)  # 1m, 5m, 15m, 1h, 4h, 1d

    # 分析结果
    signal_type = Column(String(20), nullable=False)  # long, short, neutral
    signal_strength = Column(DECIMAL(3, 2))  # 0.00-1.00

    # 关键价格点
    support_level = Column(DECIMAL(20, 8))
    resistance_level = Column(DECIMAL(20, 8))

    # 策略建议
    entry_conditions = Column(JSONB)
    exit_conditions = Column(JSONB)
    risk_factors = Column(JSONB)

    # 元数据
    analysis_version = Column(String(20), default="1.0")
    confidence_score = Column(DECIMAL(3, 2))

    # 关系
    symbol = relationship("TradingSymbol", back_populates="technical_analysis")
    exchange = relationship("Exchange", back_populates="technical_analysis")
    kline_data = relationship("KlineData", back_populates="technical_analysis")

    def __repr__(self):
        return f"<TechnicalAnalysis(signal='{self.signal_type}', strength={self.signal_strength})>"