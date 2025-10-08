"""
交易相关模型

包含交易策略、订单和持仓相关的数据模型。
"""

from sqlalchemy import Column, String, DECIMAL, Integer, DateTime, Text, ForeignKey, Boolean
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import UUID
from .base import BaseModel


class TradingStrategy(BaseModel):
    """交易策略模型"""

    __tablename__ = "trading_strategies"

    name = Column(String(100), nullable=False)
    description = Column(Text)
    strategy_type = Column(String(50), nullable=False)  # long, short, neutral
    status = Column(String(20), default="active")  # active, paused, archived

    # LLM生成信息
    llm_provider = Column(String(50), nullable=False)
    llm_model = Column(String(100), nullable=False)
    confidence_score = Column(DECIMAL(3, 2))  # 0.00-1.00

    # 策略参数
    entry_price = Column(DECIMAL(20, 8))
    position_size = Column(DECIMAL(20, 8))
    stop_loss_price = Column(DECIMAL(20, 8))
    take_profit_price = Column(DECIMAL(20, 8))

    # 时间信息
    expires_at = Column(DateTime(timezone=True))

    # 外键
    created_by_id = Column(UUID(as_uuid=True), ForeignKey("users.id"))
    symbol_id = Column(UUID(as_uuid=True), ForeignKey("trading_symbols.id"))

    # 关系
    creator = relationship("User", back_populates="trading_strategies")
    symbol = relationship("TradingSymbol", back_populates="trading_strategies")
    trading_orders = relationship("TradingOrder", back_populates="strategy")

    def __repr__(self):
        return f"<TradingStrategy(name='{self.name}', type='{self.strategy_type}')>"


class TradingOrder(BaseModel):
    """交易订单模型"""

    __tablename__ = "trading_orders"

    order_id = Column(String(100), nullable=False)  # 交易所订单ID

    # 订单基本信息
    symbol = Column(String(20), nullable=False)
    side = Column(String(10), nullable=False)  # buy, sell
    order_type = Column(String(20), nullable=False)  # market, limit, stop, stop_limit
    amount = Column(DECIMAL(20, 8), nullable=False)
    price = Column(DECIMAL(20, 8))

    # 状态信息
    status = Column(String(20), default="pending")  # pending, filled, cancelled, failed
    filled_amount = Column(DECIMAL(20, 8), default=0)
    average_price = Column(DECIMAL(20, 8))

    # 风险管理
    stop_loss_price = Column(DECIMAL(20, 8))
    take_profit_price = Column(DECIMAL(20, 8))
    timeout_seconds = Column(Integer, default=300)

    # 时间信息
    filled_at = Column(DateTime(timezone=True))
    cancelled_at = Column(DateTime(timezone=True))

    # 外键
    exchange_id = Column(UUID(as_uuid=True), ForeignKey("exchanges.id"))
    strategy_id = Column(UUID(as_uuid=True), ForeignKey("trading_strategies.id"))
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"))

    # 关系
    exchange = relationship("Exchange", back_populates="trading_orders")
    strategy = relationship("TradingStrategy", back_populates="trading_orders")
    user = relationship("User", back_populates="trading_orders")

    def __repr__(self):
        return f"<TradingOrder(order_id='{self.order_id}', status='{self.status}')>"


class Position(BaseModel):
    """持仓信息模型"""

    __tablename__ = "positions"

    # 持仓基本信息
    symbol = Column(String(20), nullable=False)
    side = Column(String(10), nullable=False)  # long, short
    amount = Column(DECIMAL(20, 8), nullable=False)
    average_cost = Column(DECIMAL(20, 8), nullable=False)

    # 盈亏信息
    current_price = Column(DECIMAL(20, 8))
    unrealized_pnl = Column(DECIMAL(20, 8))
    realized_pnl = Column(DECIMAL(20, 8), default=0)

    # 风险指标
    risk_exposure = Column(DECIMAL(20, 8))
    margin_used = Column(DECIMAL(20, 8))

    # 时间信息
    opened_at = Column(DateTime(timezone=True), nullable=False)
    closed_at = Column(DateTime(timezone=True))

    # 外键
    exchange_id = Column(UUID(as_uuid=True), ForeignKey("exchanges.id"))
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"))

    # 关系
    exchange = relationship("Exchange", back_populates="positions")
    user = relationship("User", back_populates="positions")

    def __repr__(self):
        return f"<Position(symbol='{self.symbol}', side='{self.side}', amount={self.amount})>"