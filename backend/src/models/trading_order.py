"""
交易订单数据模型

定义交易订单的数据库模型和相关操作，支持多种订单类型和高级功能。
"""

import uuid
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, Any, Optional, List
from enum import Enum

from sqlalchemy import Column, String, DECIMAL, Integer, DateTime, Text, ForeignKey, Boolean, JSON, Index
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.sql import func

from .base import BaseModel


class OrderSide(str, Enum):
    """订单方向枚举"""
    BUY = "buy"
    SELL = "sell"


class OrderType(str, Enum):
    """订单类型枚举"""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"
    STOP_MARKET = "stop_market"
    LIMIT_MAKER = "limit_maker"


class OrderStatus(str, Enum):
    """订单状态枚举"""
    PENDING = "pending"
    OPEN = "open"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"
    FAILED = "failed"


class TimeInForce(str, Enum):
    """订单有效期枚举"""
    GTC = "GTC"  # Good Till Canceled
    IOC = "IOC"  # Immediate Or Cancel
    FOK = "FOK"  # Fill Or Kill
    GTD = "GTD"  # Good Till Date


class TradingOrder(BaseModel):
    """交易订单模型"""

    __tablename__ = "trading_orders"

    # 订单标识
    order_id = Column(String(100), nullable=False, unique=True, index=True)
    client_order_id = Column(String(100), nullable=True, unique=True, index=True)
    exchange_order_id = Column(String(100), nullable=True, index=True)

    # 基本信息
    symbol = Column(String(50), nullable=False, index=True)
    side = Column(String(10), nullable=False)  # buy, sell
    order_type = Column(String(20), nullable=False)  # market, limit, stop, etc.
    amount = Column(DECIMAL(20, 8), nullable=False)
    price = Column(DECIMAL(20, 8), nullable=True)
    filled_amount = Column(DECIMAL(20, 8), default=0, nullable=False)
    remaining_amount = Column(DECIMAL(20, 8), nullable=False)

    # 执行信息
    average_price = Column(DECIMAL(20, 8), nullable=True)
    status = Column(String(20), nullable=False, default=OrderStatus.PENDING.value)
    time_in_force = Column(String(10), default=TimeInForce.GTC.value)

    # 风险管理
    stop_loss_price = Column(DECIMAL(20, 8), nullable=True)
    take_profit_price = Column(DECIMAL(20, 8), nullable=True)
    trailing_stop_amount = Column(DECIMAL(20, 8), nullable=True)
    trailing_stop_percent = Column(DECIMAL(5, 2), nullable=True)

    # 高级订单参数
    iceberg_amount = Column(DECIMAL(20, 8), nullable=True)
    visible_amount = Column(DECIMAL(20, 8), nullable=True)
    min_amount = Column(DECIMAL(20, 8), nullable=True)
    max_amount = Column(DECIMAL(20, 8), nullable=True)

    # 时间信息
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)
    expires_at = Column(DateTime(timezone=True), nullable=True)
    filled_at = Column(DateTime(timezone=True), nullable=True)
    cancelled_at = Column(DateTime(timezone=True), nullable=True)

    # 交易所和执行信息
    exchange = Column(String(50), nullable=False)
    exchange_fee = Column(DECIMAL(20, 8), default=0)
    exchange_fee_asset = Column(String(20), nullable=True)

    # 错误信息
    error_code = Column(String(50), nullable=True)
    error_message = Column(Text, nullable=True)
    retry_count = Column(Integer, default=0)

    # 元数据
    metadata = Column(JSON, nullable=True)
    tags = Column(JSON, nullable=True)  # 标签列表

    # 外键
    strategy_id = Column(UUID(as_uuid=True), ForeignKey("trading_strategies.id"), nullable=True)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=True)
    parent_order_id = Column(UUID(as_uuid=True), ForeignKey("trading_orders.id"), nullable=True)

    # 关系
    strategy = relationship("TradingStrategy", back_populates="orders")
    user = relationship("User", back_populates="orders")
    child_orders = relationship("TradingOrder", backref="parent_order", remote_side=[BaseModel.id])
    fills = relationship("OrderFill", back_populates="order", cascade="all, delete-orphan")

    # 索引
    __table_args__ = (
        Index('idx_trading_orders_symbol_status', 'symbol', 'status'),
        Index('idx_trading_orders_user_status', 'user_id', 'status'),
        Index('idx_trading_orders_strategy_status', 'strategy_id', 'status'),
        Index('idx_trading_orders_created_at', 'created_at'),
        Index('idx_trading_orders_exchange_symbol', 'exchange', 'symbol'),
    )

    def __repr__(self):
        return f"<TradingOrder(order_id='{self.order_id}', symbol='{self.symbol}', side='{self.side}', status='{self.status}')>"

    @property
    def is_active(self) -> bool:
        """检查订单是否为活跃状态"""
        return self.status in [
            OrderStatus.PENDING.value,
            OrderStatus.OPEN.value,
            OrderStatus.PARTIALLY_FILLED.value
        ]

    @property
    def is_completed(self) -> bool:
        """检查订单是否已完成"""
        return self.status in [
            OrderStatus.FILLED.value,
            OrderStatus.CANCELLED.value,
            OrderStatus.REJECTED.value,
            OrderStatus.EXPIRED.value,
            OrderStatus.FAILED.value
        ]

    @property
    def fill_percentage(self) -> float:
        """计算订单成交百分比"""
        if float(self.amount) == 0:
            return 0.0
        return float(self.filled_amount) / float(self.amount) * 100

    @property
    def is_expired(self) -> bool:
        """检查订单是否过期"""
        if self.expires_at is None:
            return False
        return datetime.utcnow() > self.expires_at

    def calculate_total_value(self) -> Optional[Decimal]:
        """计算订单总价值"""
        if self.price is None:
            return None
        return self.amount * self.price

    def calculate_filled_value(self) -> Optional[Decimal]:
        """计算已成交价值"""
        if self.average_price is None or self.filled_amount == 0:
            return None
        return self.filled_amount * self.average_price

    def update_fill(self, filled_amount: Decimal, fill_price: Decimal, fee: Decimal = None) -> None:
        """更新订单成交信息"""
        self.filled_amount += filled_amount
        self.remaining_amount = self.amount - self.filled_amount

        # 计算平均成交价格
        if self.filled_amount > 0:
            total_value = self.calculate_filled_value() or 0
            new_value = filled_amount * fill_price
            self.average_price = (total_value + new_value) / self.filled_amount

        # 更新状态
        if self.filled_amount >= self.amount:
            self.status = OrderStatus.FILLED.value
            self.filled_at = datetime.utcnow()
        elif self.filled_amount > 0:
            self.status = OrderStatus.PARTIALLY_FILLED.value
        else:
            self.status = OrderStatus.OPEN.value

        # 更新手续费
        if fee is not None:
            self.exchange_fee += fee

        self.updated_at = datetime.utcnow()

    def cancel(self, reason: str = None) -> None:
        """取消订单"""
        if self.is_active:
            self.status = OrderStatus.CANCELLED.value
            self.cancelled_at = datetime.utcnow()
            if reason:
                self.error_message = reason
            self.updated_at = datetime.utcnow()

    def reject(self, error_code: str, error_message: str) -> None:
        """拒绝订单"""
        self.status = OrderStatus.REJECTED.value
        self.error_code = error_code
        self.error_message = error_message
        self.updated_at = datetime.utcnow()

    def expire(self) -> None:
        """使订单过期"""
        if self.is_active:
            self.status = OrderStatus.EXPIRED.value
            self.updated_at = datetime.utcnow()

    def can_modify(self) -> bool:
        """检查订单是否可以修改"""
        return self.status in [OrderStatus.PENDING.value, OrderStatus.OPEN.value]

    def can_cancel(self) -> bool:
        """检查订单是否可以取消"""
        return self.is_active and not self.is_expired

    def update_trailing_stop(self, current_price: Decimal) -> bool:
        """更新移动止损价格"""
        if self.trailing_stop_amount is None or self.side == OrderSide.BUY.value:
            return False

        current_stop = self.stop_loss_price
        if self.side == OrderSide.SELL.value:
            # 卖单：止损价格只能向上移动
            new_stop = current_price - self.trailing_stop_amount
            if current_stop is None or new_stop > current_stop:
                self.stop_loss_price = new_stop
                self.updated_at = datetime.utcnow()
                return True
        else:
            # 买单：止损价格只能向下移动
            new_stop = current_price + self.trailing_stop_amount
            if current_stop is None or new_stop < current_stop:
                self.stop_loss_price = new_stop
                self.updated_at = datetime.utcnow()
                return True
        return False

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "id": str(self.id),
            "order_id": self.order_id,
            "client_order_id": self.client_order_id,
            "exchange_order_id": self.exchange_order_id,
            "symbol": self.symbol,
            "side": self.side,
            "order_type": self.order_type,
            "amount": float(self.amount),
            "price": float(self.price) if self.price else None,
            "filled_amount": float(self.filled_amount),
            "remaining_amount": float(self.remaining_amount),
            "average_price": float(self.average_price) if self.average_price else None,
            "status": self.status,
            "time_in_force": self.time_in_force,
            "stop_loss_price": float(self.stop_loss_price) if self.stop_loss_price else None,
            "take_profit_price": float(self.take_profit_price) if self.take_profit_price else None,
            "trailing_stop_amount": float(self.trailing_stop_amount) if self.trailing_stop_amount else None,
            "trailing_stop_percent": float(self.trailing_stop_percent) if self.trailing_stop_percent else None,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "filled_at": self.filled_at.isoformat() if self.filled_at else None,
            "cancelled_at": self.cancelled_at.isoformat() if self.cancelled_at else None,
            "exchange": self.exchange,
            "exchange_fee": float(self.exchange_fee) if self.exchange_fee else None,
            "exchange_fee_asset": self.exchange_fee_asset,
            "error_code": self.error_code,
            "error_message": self.error_message,
            "retry_count": self.retry_count,
            "metadata": self.metadata,
            "tags": self.tags,
            "strategy_id": str(self.strategy_id) if self.strategy_id else None,
            "user_id": str(self.user_id) if self.user_id else None,
            "parent_order_id": str(self.parent_order_id) if self.parent_order_id else None,
            "is_active": self.is_active,
            "is_completed": self.is_completed,
            "fill_percentage": self.fill_percentage,
            "is_expired": self.is_expired,
            "total_value": float(self.calculate_total_value()) if self.calculate_total_value() else None,
            "filled_value": float(self.calculate_filled_value()) if self.calculate_filled_value() else None
        }


class OrderFill(BaseModel):
    """订单成交记录模型"""

    __tablename__ = "order_fills"

    # 成交标识
    fill_id = Column(String(100), nullable=False, unique=True, index=True)
    trade_id = Column(String(100), nullable=True, index=True)

    # 成交信息
    order_id = Column(UUID(as_uuid=True), ForeignKey("trading_orders.id"), nullable=False)
    symbol = Column(String(50), nullable=False)
    side = Column(String(10), nullable=False)
    amount = Column(DECIMAL(20, 8), nullable=False)
    price = Column(DECIMAL(20, 8), nullable=False)
    quote_amount = Column(DECIMAL(20, 8), nullable=False)

    # 手续费信息
    fee = Column(DECIMAL(20, 8), default=0)
    fee_asset = Column(String(20), nullable=True)

    # 交易所信息
    exchange = Column(String(50), nullable=False)
    timestamp = Column(DateTime(timezone=True), nullable=False)

    # 外键
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=True)
    strategy_id = Column(UUID(as_uuid=True), ForeignKey("trading_strategies.id"), nullable=True)

    # 关系
    order = relationship("TradingOrder", back_populates="fills")
    user = relationship("User", back_populates="order_fills")
    strategy = relationship("TradingStrategy", back_populates="order_fills")

    def __repr__(self):
        return f"<OrderFill(fill_id='{self.fill_id}', order_id='{self.order_id}', amount={self.amount}, price={self.price})>"

    @property
    def total_value(self) -> Decimal:
        """计算成交总价值"""
        return self.amount * self.price

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "id": str(self.id),
            "fill_id": self.fill_id,
            "trade_id": self.trade_id,
            "order_id": str(self.order_id),
            "symbol": self.symbol,
            "side": self.side,
            "amount": float(self.amount),
            "price": float(self.price),
            "quote_amount": float(self.quote_amount),
            "fee": float(self.fee) if self.fee else None,
            "fee_asset": self.fee_asset,
            "exchange": self.exchange,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "user_id": str(self.user_id) if self.user_id else None,
            "strategy_id": str(self.strategy_id) if self.strategy_id else None,
            "total_value": float(self.total_value)
        }