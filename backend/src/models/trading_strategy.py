"""
交易策略数据模型

定义交易策略的数据库模型和相关操作。
"""

from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from enum import Enum
from sqlalchemy import Column, String, Text, DateTime, DECIMAL, Integer, JSON, Boolean, ForeignKey
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
import uuid

from ..core.database import BaseModel


class StrategyType(str, Enum):
    """策略类型枚举"""
    LONG = "long"
    SHORT = "short"
    NEUTRAL = "neutral"
    COMPREHENSIVE = "comprehensive"


class StrategyStatus(str, Enum):
    """策略状态枚举"""
    ACTIVE = "active"
    PAUSED = "paused"
    ARCHIVED = "archived"
    EXPIRED = "expired"
    EXECUTED = "executed"
    CANCELLED = "cancelled"


class RiskLevel(str, Enum):
    """风险等级枚举"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class TimeHorizon(str, Enum):
    """时间范围枚举"""
    SHORT_TERM = "short_term"    # < 1天
    MEDIUM_TERM = "medium_term"  # 1-7天
    LONG_TERM = "long_term"      # > 7天


class TradingStrategy(BaseModel):
    """交易策略模型"""

    __tablename__ = "trading_strategies"

    # 主键
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    # 基本信息
    name = Column(String(100), nullable=False)
    description = Column(Text)
    strategy_type = Column(String(50), nullable=False)  # 'long', 'short', 'neutral', 'comprehensive'
    status = Column(String(20), nullable=False, default=StrategyStatus.ACTIVE.value)

    # LLM生成信息
    llm_provider = Column(String(50), nullable=False)
    llm_model = Column(String(100), nullable=False)
    confidence_score = Column(DECIMAL(3, 2), nullable=True)  # 0.00-1.00

    # 策略参数
    final_recommendation = Column(String(20), nullable=False)  # 'long', 'short', 'hold'
    entry_price = Column(DECIMAL(20, 8), nullable=True)
    stop_loss_price = Column(DECIMAL(20, 8), nullable=True)
    take_profit_price = Column(DECIMAL(20, 8), nullable=True)
    position_size_percent = Column(DECIMAL(5, 2), nullable=True)  # 0.00-100.00

    # 策略特征
    risk_level = Column(String(20), default=RiskLevel.MEDIUM.value)
    time_horizon = Column(String(20), default=TimeHorizon.MEDIUM_TERM.value)
    strategy_style = Column(String(20))  # 'aggressive', 'moderate', 'conservative'

    # 分析结果（JSON格式存储）
    market_analysis = Column(JSON, nullable=True)
    risk_assessment = Column(JSON, nullable=True)
    execution_plan = Column(JSON, nullable=True)
    technical_signals = Column(JSON, nullable=True)
    news_analysis = Column(JSON, nullable=True)

    # 元数据
    analysis_types = Column(JSON, nullable=True)  # ['long', 'short', 'news', 'technical']
    analysis_parameters = Column(JSON, nullable=True)  # 分析时使用的参数
    custom_parameters = Column(JSON, nullable=True)  # 用户自定义参数

    # 时间信息
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    expires_at = Column(DateTime(timezone=True), nullable=True)
    executed_at = Column(DateTime(timezone=True), nullable=True)

    # 外键
    symbol_id = Column(UUID(as_uuid=True), ForeignKey("trading_symbols.id"), nullable=True)
    created_by = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=True)

    # 关系
    symbol = relationship("TradingSymbol", back_populates="strategies")
    creator = relationship("User", back_populates="strategies")
    orders = relationship("TradingOrder", back_populates="strategy")

    def __repr__(self):
        return f"<TradingStrategy(id={self.id}, symbol={self.symbol}, type={self.strategy_type}, recommendation={self.final_recommendation})>"

    @property
    def is_expired(self) -> bool:
        """检查策略是否过期"""
        if self.expires_at is None:
            return False
        return datetime.utcnow() > self.expires_at

    @property
    def is_active(self) -> bool:
        """检查策略是否活跃"""
        return (
            self.status == StrategyStatus.ACTIVE.value and
            not self.is_expired and
            self.created_at > datetime.utcnow() - timedelta(days=30)  # 30天内的策略为活跃
        )

    @property
    def risk_reward_ratio(self) -> Optional[float]:
        """计算风险回报比"""
        if not all([self.entry_price, self.stop_loss_price, self.take_profit_price]):
            return None

        if self.final_recommendation == StrategyType.LONG.value:
            profit_potential = float(self.take_profit_price - self.entry_price)
            risk_amount = float(self.entry_price - self.stop_loss_price)
        elif self.final_recommendation == StrategyType.SHORT.value:
            profit_potential = float(self.entry_price - self.take_profit_price)
            risk_amount = float(self.stop_loss_price - self.entry_price)
        else:
            return None

        return profit_potential / risk_amount if risk_amount > 0 else None

    def update_status(self, new_status: StrategyStatus) -> None:
        """更新策略状态"""
        self.status = new_status.value
        if new_status == StrategyStatus.EXECUTED:
            self.executed_at = datetime.utcnow()

    def extend_expiry(self, hours: int = 24) -> None:
        """延长策略过期时间"""
        if self.expires_at is None:
            self.expires_at = datetime.utcnow() + timedelta(hours=hours)
        else:
            self.expires_at = self.expires_at + timedelta(hours=hours)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "id": str(self.id),
            "name": self.name,
            "description": self.description,
            "strategy_type": self.strategy_type,
            "status": self.status,
            "final_recommendation": self.final_recommendation,
            "confidence_score": float(self.confidence_score) if self.confidence_score else None,
            "entry_price": float(self.entry_price) if self.entry_price else None,
            "stop_loss_price": float(self.stop_loss_price) if self.stop_loss_price else None,
            "take_profit_price": float(self.take_profit_price) if self.take_profit_price else None,
            "position_size_percent": float(self.position_size_percent) if self.position_size_percent else None,
            "risk_level": self.risk_level,
            "time_horizon": self.time_horizon,
            "strategy_style": self.strategy_style,
            "market_analysis": self.market_analysis,
            "risk_assessment": self.risk_assessment,
            "execution_plan": self.execution_plan,
            "technical_signals": self.technical_signals,
            "news_analysis": self.news_analysis,
            "analysis_types": self.analysis_types,
            "analysis_parameters": self.analysis_parameters,
            "custom_parameters": self.custom_parameters,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "executed_at": self.executed_at.isoformat() if self.executed_at else None,
            "risk_reward_ratio": self.risk_reward_ratio,
            "is_active": self.is_active,
            "is_expired": self.is_expired,
            "symbol": self.symbol.symbol if self.symbol else None,
            "created_by": self.creator.username if self.creator else None
        }


class StrategyAnalysis(BaseModel):
    """策略分析结果模型"""

    __tablename__ = "strategy_analysis"

    # 主键
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    # 关联信息
    strategy_id = Column(UUID(as_uuid=True), ForeignKey("trading_strategies.id"), nullable=False)

    # 分析类型
    analysis_type = Column(String(50), nullable=False)  # 'long', 'short', 'technical', 'news'

    # 分析结果
    analysis_data = Column(JSON, nullable=False)
    confidence_score = Column(DECIMAL(3, 2), nullable=True)

    # 分析元数据
    analyzer_version = Column(String(20), default="1.0")
    analysis_duration_ms = Column(Integer, nullable=True)

    # 时间信息
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # 关系
    strategy = relationship("TradingStrategy", back_populates="analysis_results")

    def __repr__(self):
        return f"<StrategyAnalysis(id={self.id}, strategy_id={self.strategy_id}, type={self.analysis_type})>"


class StrategyPerformance(BaseModel):
    """策略表现跟踪模型"""

    __tablename__ = "strategy_performance"

    # 主键
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    # 关联信息
    strategy_id = Column(UUID(as_uuid=True), ForeignKey("trading_strategies.id"), nullable=False)

    # 表现指标
    entry_price = Column(DECIMAL(20, 8), nullable=False)
    exit_price = Column(DECIMAL(20, 8), nullable=True)
    current_price = Column(DECIMAL(20, 8), nullable=True)

    # 盈亏信息
    unrealized_pnl_percent = Column(DECIMAL(10, 4), default=0)
    realized_pnl_percent = Column(DECIMAL(10, 4), nullable=True)

    # 状态信息
    status = Column(String(20), default="open")  # 'open', 'closed', 'stopped_out', 'profit_taken'

    # 时间信息
    entry_time = Column(DateTime(timezone=True), server_default=func.now())
    exit_time = Column(DateTime(timezone=True), nullable=True)
    last_updated = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    # 关系
    strategy = relationship("TradingStrategy", back_populates="performance_records")

    def __repr__(self):
        return f"<StrategyPerformance(id={self.id}, strategy_id={self.strategy_id}, status={self.status})>"

    @property
    def pnl_percent(self) -> Optional[float]:
        """获取盈亏百分比"""
        if self.status == "closed" and self.exit_price:
            return float((self.exit_price - self.entry_price) / self.entry_price * 100)
        elif self.current_price:
            return float((self.current_price - self.entry_price) / self.entry_price * 100)
        return None

    def update_current_price(self, new_price: float) -> None:
        """更新当前价格并计算未实现盈亏"""
        self.current_price = new_price
        if self.status == "open":
            self.unrealized_pnl_percent = (new_price - self.entry_price) / self.entry_price * 100
        self.last_updated = datetime.utcnow()

    def close_position(self, exit_price: float) -> None:
        """平仓并计算已实现盈亏"""
        self.exit_price = exit_price
        self.exit_time = datetime.utcnow()
        self.realized_pnl_percent = (exit_price - self.entry_price) / self.entry_price * 100
        self.status = "closed"
        self.last_updated = datetime.utcnow()


# 扩展TradingStrategy模型的关系
TradingStrategy.analysis_results = relationship("StrategyAnalysis", back_populates="strategy")
TradingStrategy.performance_records = relationship("StrategyPerformance", back_populates="strategy")