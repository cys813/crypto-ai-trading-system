"""
持仓数据模型

定义交易持仓的数据库模型和相关操作，支持多种持仓类型和风险管理。
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


class PositionSide(str, Enum):
    """持仓方向枚举"""
    LONG = "long"
    SHORT = "short"


class PositionStatus(str, Enum):
    """持仓状态枚举"""
    OPEN = "open"
    CLOSED = "closed"
    CLOSING = "closing"
    LIQUIDATED = "liquidated"
    ADJUSTING = "adjusting"


class PositionType(str, Enum):
    """持仓类型枚举"""
    SPOT = "spot"
    MARGIN = "margin"
    FUTURES = "futures"
    ISOLATED = "isolated"
    CROSS = "cross"


class Position(BaseModel):
    """持仓信息模型"""

    __tablename__ = "positions"

    # 基本信息
    symbol = Column(String(50), nullable=False, index=True)
    side = Column(String(10), nullable=False)  # long, short
    amount = Column(DECIMAL(20, 8), nullable=False)
    average_cost = Column(DECIMAL(20, 8), nullable=False)
    current_price = Column(DECIMAL(20, 8), nullable=True)

    # 持仓标识
    position_id = Column(String(100), nullable=True, unique=True, index=True)
    exchange_position_id = Column(String(100), nullable=True, index=True)

    # 状态信息
    status = Column(String(20), nullable=False, default=PositionStatus.OPEN.value)
    position_type = Column(String(20), default=PositionType.SPOT.value)
    is_active = Column(Boolean, default=True)

    # 盈亏信息
    unrealized_pnl = Column(DECIMAL(20, 8), default=0)
    realized_pnl = Column(DECIMAL(20, 8), default=0)
    total_pnl = Column(DECIMAL(20, 8), default=0)
    unrealized_pnl_percent = Column(DECIMAL(10, 4), default=0)
    realized_pnl_percent = Column(DECIMAL(10, 4), default=0)

    # 风险指标
    risk_exposure = Column(DECIMAL(20, 8), nullable=True)
    margin_used = Column(DECIMAL(20, 8), default=0)
    margin_ratio = Column(DECIMAL(5, 4), nullable=True)
    maintenance_margin = Column(DECIMAL(20, 8), nullable=True)
    liquidation_price = Column(DECIMAL(20, 8), nullable=True)

    # 杠杆和保证金
    leverage = Column(DECIMAL(5, 2), default=1)
    initial_margin = Column(DECIMAL(20, 8), default=0)
    maintenance_margin_requirement = Column(DECIMAL(5, 4), nullable=True)

    # 止损止盈
    stop_loss_price = Column(DECIMAL(20, 8), nullable=True)
    take_profit_price = Column(DECIMAL(20, 8), nullable=True)
    trailing_stop_amount = Column(DECIMAL(20, 8), nullable=True)
    trailing_stop_percent = Column(DECIMAL(5, 2), nullable=True)

    # 时间信息
    opened_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    closed_at = Column(DateTime(timezone=True), nullable=True)
    last_updated = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)

    # 交易所信息
    exchange = Column(String(50), nullable=False)
    exchange_fee_total = Column(DECIMAL(20, 8), default=0)

    # 成本信息
    entry_value = Column(DECIMAL(20, 8), nullable=True)
    current_value = Column(DECIMAL(20, 8), nullable=True)
    cost_basis = Column(DECIMAL(20, 8), nullable=True)

    # 统计信息
    total_fees = Column(DECIMAL(20, 8), default=0)
    total_volume = Column(DECIMAL(20, 8), default=0)
    trade_count = Column(Integer, default=0)
    win_count = Column(Integer, default=0)
    loss_count = Column(Integer, default=0)

    # 元数据
    metadata = Column(JSON, nullable=True)
    tags = Column(JSON, nullable=True)

    # 外键
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=True)
    strategy_id = Column(UUID(as_uuid=True), ForeignKey("trading_strategies.id"), nullable=True)
    parent_position_id = Column(UUID(as_uuid=True), ForeignKey("positions.id"), nullable=True)

    # 关系
    user = relationship("User", back_populates="positions")
    strategy = relationship("TradingStrategy", back_populates="positions")
    child_positions = relationship("Position", backref="parent_position", remote_side=[BaseModel.id])
    trades = relationship("PositionTrade", back_populates="position", cascade="all, delete-orphan")

    # 索引
    __table_args__ = (
        Index('idx_positions_symbol_status', 'symbol', 'status'),
        Index('idx_positions_user_status', 'user_id', 'status'),
        Index('idx_positions_strategy_status', 'strategy_id', 'status'),
        Index('idx_positions_opened_at', 'opened_at'),
        Index('idx_positions_exchange_symbol', 'exchange', 'symbol'),
    )

    def __repr__(self):
        return f"<Position(symbol='{self.symbol}', side='{self.side}', amount={self.amount}, status='{self.status}')>"

    @property
    def is_open(self) -> bool:
        """检查持仓是否开放"""
        return self.status == PositionStatus.OPEN.value

    @property
    def is_closed(self) -> bool:
        """检查持仓是否已关闭"""
        return self.status == PositionStatus.CLOSED.value

    @property
    def is_liquidated(self) -> bool:
        """检查持仓是否被清算"""
        return self.status == PositionStatus.LIQUIDATED.value

    @property
    def duration(self) -> Optional[timedelta]:
        """计算持仓持续时间"""
        if self.opened_at is None:
            return None
        end_time = self.closed_at or datetime.utcnow()
        return end_time - self.opened_at

    @property
    def pnl_percent(self) -> float:
        """计算总盈亏百分比"""
        if self.cost_basis == 0:
            return 0.0
        return float(self.total_pnl / self.cost_basis * 100)

    @property
    def win_rate(self) -> float:
        """计算胜率"""
        if self.trade_count == 0:
            return 0.0
        return float(self.win_count / self.trade_count * 100)

    @property
    def risk_reward_ratio(self) -> Optional[float]:
        """计算风险回报比"""
        if not all([self.stop_loss_price, self.take_profit_price, self.average_cost]):
            return None

        if self.side == PositionSide.LONG.value:
            profit_potential = float(self.take_profit_price - self.average_cost)
            risk_amount = float(self.average_cost - self.stop_loss_price)
        else:
            profit_potential = float(self.average_cost - self.take_profit_price)
            risk_amount = float(self.stop_loss_price - self.average_cost)

        return profit_potential / risk_amount if risk_amount > 0 else None

    def update_market_price(self, new_price: Decimal) -> None:
        """更新市场价格并重新计算盈亏"""
        old_price = self.current_price or Decimal('0')
        self.current_price = new_price

        # 计算当前价值
        self.current_value = self.amount * new_price

        # 计算未实现盈亏
        if self.side == PositionSide.LONG.value:
            self.unrealized_pnl = self.current_value - self.cost_basis
        else:
            self.unrealized_pnl = self.cost_basis - self.current_value

        # 计算未实现盈亏百分比
        if self.cost_basis > 0:
            self.unrealized_pnl_percent = self.unrealized_pnl / self.cost_basis * 100

        # 计算总盈亏
        self.total_pnl = self.unrealized_pnl + self.realized_pnl

        # 重新计算清算价格
        self.calculate_liquidation_price()

        self.last_updated = datetime.utcnow()

    def calculate_liquidation_price(self) -> None:
        """计算清算价格"""
        if self.leverage <= 1 or self.maintenance_margin_requirement is None:
            return

        if self.side == PositionSide.LONG.value:
            # 多头清算价格
            liquidation_value = self.cost_basis * (1 - 1 / self.leverage + self.maintenance_margin_requirement)
            self.liquidation_price = liquidation_value / self.amount
        else:
            # 空头清算价格
            liquidation_value = self.cost_basis * (1 + 1 / self.leverage + self.maintenance_margin_requirement)
            self.liquidation_price = liquidation_value / self.amount

    def add_trade(self, amount: Decimal, price: Decimal, fee: Decimal = None) -> None:
        """添加交易记录并更新持仓信息"""
        trade_value = amount * price
        old_amount = self.amount
        old_cost_basis = self.cost_basis

        # 更新持仓数量
        if self.side == PositionSide.LONG.value:
            self.amount += amount
        else:
            self.amount -= amount

        # 重新计算平均成本
        if self.side == PositionSide.LONG.value:
            self.cost_basis = old_cost_basis + trade_value
        else:
            self.cost_basis = old_cost_basis - trade_value

        if self.amount > 0:
            self.average_cost = self.cost_basis / self.amount

        # 更新交易统计
        self.trade_count += 1
        self.total_volume += trade_value
        if fee:
            self.total_fees += fee

        # 更新入场价值
        self.entry_value = self.cost_basis

        self.last_updated = datetime.utcnow()

    def close_position(self, close_price: Decimal, close_amount: Decimal = None) -> Decimal:
        """平仓并计算已实现盈亏"""
        close_amount = close_amount or self.amount
        close_value = close_amount * close_price

        # 计算已实现盈亏
        cost_basis_for_close = self.cost_basis * (close_amount / self.amount)
        if self.side == PositionSide.LONG.value:
            realized_pnl = close_value - cost_basis_for_close
        else:
            realized_pnl = cost_basis_for_close - close_value

        # 更新已实现盈亏
        self.realized_pnl += realized_pnl
        self.total_pnl = self.realized_pnl + self.unrealized_pnl

        # 更新已实现盈亏百分比
        if cost_basis_for_close > 0:
            realized_pnl_percent = realized_pnl / cost_basis_for_close * 100
            self.realized_pnl_percent += realized_pnl_percent

        # 更新持仓数量
        self.amount -= close_amount
        self.cost_basis -= cost_basis_for_close

        # 如果完全平仓
        if self.amount <= 0:
            self.status = PositionStatus.CLOSED.value
            self.closed_at = datetime.utcnow()
            self.amount = Decimal('0')
            self.cost_basis = Decimal('0')
            self.unrealized_pnl = Decimal('0')

        # 更新统计
        if realized_pnl > 0:
            self.win_count += 1
        else:
            self.loss_count += 1

        self.last_updated = datetime.utcnow()
        return realized_pnl

    def adjust_position(self, new_amount: Decimal, new_price: Decimal) -> None:
        """调整持仓"""
        self.status = PositionStatus.ADJUSTING.value
        self.add_trade(new_amount - self.amount, new_price)
        self.status = PositionStatus.OPEN.value
        self.last_updated = datetime.utcnow()

    def set_stop_loss(self, stop_loss_price: Decimal) -> None:
        """设置止损价格"""
        self.stop_loss_price = stop_loss_price
        self.last_updated = datetime.utcnow()

    def set_take_profit(self, take_profit_price: Decimal) -> None:
        """设置止盈价格"""
        self.take_profit_price = take_profit_price
        self.last_updated = datetime.utcnow()

    def update_trailing_stop(self, current_price: Decimal) -> bool:
        """更新移动止损"""
        if self.trailing_stop_amount is None:
            return False

        current_stop = self.stop_loss_price
        if self.side == PositionSide.LONG.value:
            # 多头：止损价格只能向上移动
            new_stop = current_price - self.trailing_stop_amount
            if current_stop is None or new_stop > current_stop:
                self.stop_loss_price = new_stop
                self.last_updated = datetime.utcnow()
                return True
        else:
            # 空头：止损价格只能向下移动
            new_stop = current_price + self.trailing_stop_amount
            if current_stop is None or new_stop < current_stop:
                self.stop_loss_price = new_stop
                self.last_updated = datetime.utcnow()
                return True
        return False

    def calculate_risk_metrics(self) -> Dict[str, Any]:
        """计算风险指标"""
        return {
            "risk_exposure": float(self.risk_exposure) if self.risk_exposure else None,
            "margin_used": float(self.margin_used),
            "margin_ratio": float(self.margin_ratio) if self.margin_ratio else None,
            "liquidation_price": float(self.liquidation_price) if self.liquidation_price else None,
            "distance_to_liquidation": self.calculate_distance_to_liquidation(),
            "risk_reward_ratio": self.risk_reward_ratio,
            "max_loss": float(self.calculate_max_loss()),
            "position_value": float(self.current_value) if self.current_value else None
        }

    def calculate_distance_to_liquidation(self) -> Optional[float]:
        """计算距离清算价格的百分比"""
        if self.liquidation_price is None or self.current_price is None:
            return None

        if self.side == PositionSide.LONG.value:
            return float((self.current_price - self.liquidation_price) / self.current_price * 100)
        else:
            return float((self.liquidation_price - self.current_price) / self.current_price * 100)

    def calculate_max_loss(self) -> Decimal:
        """计算最大可能损失"""
        if self.stop_loss_price is None:
            return Decimal('0')

        if self.side == PositionSide.LONG.value:
            return self.cost_basis - (self.amount * self.stop_loss_price)
        else:
            return (self.amount * self.stop_loss_price) - self.cost_basis

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "id": str(self.id),
            "position_id": self.position_id,
            "exchange_position_id": self.exchange_position_id,
            "symbol": self.symbol,
            "side": self.side,
            "amount": float(self.amount),
            "average_cost": float(self.average_cost),
            "current_price": float(self.current_price) if self.current_price else None,
            "status": self.status,
            "position_type": self.position_type,
            "is_active": self.is_active,
            "unrealized_pnl": float(self.unrealized_pnl),
            "realized_pnl": float(self.realized_pnl),
            "total_pnl": float(self.total_pnl),
            "unrealized_pnl_percent": float(self.unrealized_pnl_percent),
            "realized_pnl_percent": float(self.realized_pnl_percent),
            "pnl_percent": self.pnl_percent,
            "risk_exposure": float(self.risk_exposure) if self.risk_exposure else None,
            "margin_used": float(self.margin_used),
            "margin_ratio": float(self.margin_ratio) if self.margin_ratio else None,
            "maintenance_margin": float(self.maintenance_margin) if self.maintenance_margin else None,
            "liquidation_price": float(self.liquidation_price) if self.liquidation_price else None,
            "leverage": float(self.leverage),
            "initial_margin": float(self.initial_margin),
            "maintenance_margin_requirement": float(self.maintenance_margin_requirement) if self.maintenance_margin_requirement else None,
            "stop_loss_price": float(self.stop_loss_price) if self.stop_loss_price else None,
            "take_profit_price": float(self.take_profit_price) if self.take_profit_price else None,
            "trailing_stop_amount": float(self.trailing_stop_amount) if self.trailing_stop_amount else None,
            "trailing_stop_percent": float(self.trailing_stop_percent) if self.trailing_stop_percent else None,
            "opened_at": self.opened_at.isoformat() if self.opened_at else None,
            "closed_at": self.closed_at.isoformat() if self.closed_at else None,
            "last_updated": self.last_updated.isoformat() if self.last_updated else None,
            "exchange": self.exchange,
            "exchange_fee_total": float(self.exchange_fee_total),
            "entry_value": float(self.entry_value) if self.entry_value else None,
            "current_value": float(self.current_value) if self.current_value else None,
            "cost_basis": float(self.cost_basis),
            "total_fees": float(self.total_fees),
            "total_volume": float(self.total_volume),
            "trade_count": self.trade_count,
            "win_count": self.win_count,
            "loss_count": self.loss_count,
            "win_rate": self.win_rate,
            "duration": str(self.duration) if self.duration else None,
            "risk_reward_ratio": self.risk_reward_ratio,
            "risk_metrics": self.calculate_risk_metrics(),
            "metadata": self.metadata,
            "tags": self.tags,
            "user_id": str(self.user_id) if self.user_id else None,
            "strategy_id": str(self.strategy_id) if self.strategy_id else None,
            "parent_position_id": str(self.parent_position_id) if self.parent_position_id else None
        }


class PositionTrade(BaseModel):
    """持仓交易记录模型"""

    __tablename__ = "position_trades"

    # 交易标识
    trade_id = Column(String(100), nullable=False, unique=True, index=True)
    order_id = Column(String(100), nullable=True, index=True)

    # 基本信息
    position_id = Column(UUID(as_uuid=True), ForeignKey("positions.id"), nullable=False)
    symbol = Column(String(50), nullable=False)
    side = Column(String(10), nullable=False)
    amount = Column(DECIMAL(20, 8), nullable=False)
    price = Column(DECIMAL(20, 8), nullable=False)
    quote_amount = Column(DECIMAL(20, 8), nullable=False)

    # 手续费信息
    fee = Column(DECIMAL(20, 8), default=0)
    fee_asset = Column(String(20), nullable=True)

    # 交易类型
    trade_type = Column(String(20), default="position")  # position, open, close, adjust
    pnl = Column(DECIMAL(20, 8), nullable=True)  # 该笔交易的盈亏

    # 交易所信息
    exchange = Column(String(50), nullable=False)
    timestamp = Column(DateTime(timezone=True), nullable=False)

    # 外键
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=True)
    strategy_id = Column(UUID(as_uuid=True), ForeignKey("trading_strategies.id"), nullable=True)

    # 关系
    position = relationship("Position", back_populates="trades")
    user = relationship("User", back_populates="position_trades")
    strategy = relationship("TradingStrategy", back_populates="position_trades")

    def __repr__(self):
        return f"<PositionTrade(trade_id='{self.trade_id}', position_id='{self.position_id}', amount={self.amount}, price={self.price})>"

    @property
    def total_value(self) -> Decimal:
        """计算交易总价值"""
        return self.amount * self.price

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "id": str(self.id),
            "trade_id": self.trade_id,
            "order_id": self.order_id,
            "position_id": str(self.position_id),
            "symbol": self.symbol,
            "side": self.side,
            "amount": float(self.amount),
            "price": float(self.price),
            "quote_amount": float(self.quote_amount),
            "fee": float(self.fee) if self.fee else None,
            "fee_asset": self.fee_asset,
            "trade_type": self.trade_type,
            "pnl": float(self.pnl) if self.pnl else None,
            "exchange": self.exchange,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "user_id": str(self.user_id) if self.user_id else None,
            "strategy_id": str(self.strategy_id) if self.strategy_id else None,
            "total_value": float(self.total_value)
        }