"""
动态资金管理服务

负责资金的动态分配、仓位管理、资金优化和风险控制。
支持多种资金分配策略和自适应调整机制。
"""

import logging
import asyncio
from datetime import datetime, timedelta
from decimal import Decimal
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum

from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, func

from ..core.cache import get_cache, CacheKeys
from ..core.database import SessionLocal
from ..core.logging import BusinessLogger
from ..core.exceptions import FundError, ValidationError
from ..models.position import Position, PositionStatus, PositionSide
from ..models.trading_strategy import TradingStrategy
from ..models.trading_order import TradingOrder, OrderStatus

logger = logging.getLogger(__name__)
business_logger = BusinessLogger("dynamic_fund_manager")


class FundAllocationStrategy(str, Enum):
    """资金分配策略"""
    FIXED_PERCENT = "fixed_percent"  # 固定百分比
    KELLY_CRITERION = "kelly_criterion"  # 凯利公式
    RISK_PARITY = "risk_parity"  # 风险平价
    EQUAL_WEIGHT = "equal_weight"  # 等权重
    VOLATILITY_TARGET = "volatility_target"  # 目标波动率
    ADAPTIVE = "adaptive"  # 自适应


class RebalanceTrigger(str, Enum):
    """再平衡触发条件"""
    TIME_BASED = "time_based"  # 基于时间
    DEVIATION_BASED = "deviation_based"  # 基于偏差
    RISK_BASED = "risk_based"  # 基于风险
    PERFORMANCE_BASED = "performance_based"  # 基于表现


@dataclass
class FundAllocation:
    """资金分配"""
    strategy_id: str
    strategy_name: str
    allocated_amount: Decimal
    allocated_percent: float
    target_amount: Decimal
    target_percent: float
    current_value: Decimal
    pnl: Decimal
    pnl_percent: float
    risk_score: float
    last_updated: datetime


@dataclass
class FundMetrics:
    """资金指标"""
    total_funds: Decimal
    allocated_funds: Decimal
    available_funds: Decimal
    allocation_efficiency: float
    diversification_score: float
    total_pnl: Decimal
    total_pnl_percent: float
    sharpe_ratio: Optional[float]
    max_drawdown: float
    var_95: float


@dataclass
class RebalanceRecommendation:
    """再平衡建议"""
    strategy_id: str
    action: str  # increase, decrease, maintain
    amount: Decimal
    reason: str
    priority: int  # 1-5, 5为最高优先级


@dataclass
class FundManagementConfig:
    """资金管理配置"""
    total_funds: Decimal
    min_reserve_percent: float = 10.0  # 最小储备百分比
    max_allocation_percent: float = 30.0  # 单个策略最大分配百分比
    rebalance_threshold: float = 5.0  # 再平衡阈值百分比
    rebalance_frequency: int = 24  # 再平衡频率（小时）
    allocation_strategy: FundAllocationStrategy = FundAllocationStrategy.FIXED_PERCENT
    enable_auto_rebalance: bool = True
    performance_lookback_days: int = 30  # 表现回看天数


class DynamicFundManager:
    """动态资金管理器"""

    def __init__(self, config: FundManagementConfig):
        self.logger = logger
        self.business_logger = business_logger
        self.cache = get_cache()
        self.config = config

        # 再平衡任务
        self._rebalance_task = None
        self._monitoring_active = False

    async def allocate_funds(
        self,
        strategy_id: str,
        requested_amount: Decimal,
        user_id: str,
        db: Session
    ) -> Tuple[bool, str, Decimal]:
        """
        分配资金给策略

        Args:
            strategy_id: 策略ID
            requested_amount: 请求金额
            user_id: 用户ID
            db: 数据库会话

        Returns:
            Tuple[bool, str, Decimal]: (是否成功, 消息, 实际分配金额)
        """
        try:
            # 获取当前资金分配情况
            current_allocations = await self._get_current_allocations(user_id, db)
            fund_metrics = await self._calculate_fund_metrics(user_id, current_allocations, db)

            # 验证资金可用性
            available_funds = fund_metrics.available_funds
            if requested_amount > available_funds:
                return False, f"资金不足，可用资金: {available_funds}，请求: {requested_amount}", Decimal('0')

            # 检查单个策略分配限制
            current_allocation = next(
                (a for a in current_allocations if a.strategy_id == strategy_id),
                None
            )

            new_total_allocation = requested_amount
            if current_allocation:
                new_total_allocation += current_allocation.allocated_amount

            max_allowed = self.config.total_funds * (self.config.max_allocation_percent / 100)
            if new_total_allocation > max_allowed:
                return False, f"超出单策略分配限制，最大允许: {max_allowed}", Decimal('0')

            # 执行资金分配
            allocation_result = await self._execute_allocation(
                strategy_id, requested_amount, user_id, db
            )

            if allocation_result:
                await self.business_logger.log_event(
                    "funds_allocated",
                    user_id=user_id,
                    strategy_id=strategy_id,
                    amount=float(requested_amount)
                )

                return True, "资金分配成功", requested_amount
            else:
                return False, "资金分配失败", Decimal('0')

        except Exception as e:
            self.logger.error(f"资金分配失败: {str(e)}", exc_info=True)
            return False, f"资金分配失败: {str(e)}", Decimal('0')

    async def deallocate_funds(
        self,
        strategy_id: str,
        amount: Decimal,
        user_id: str,
        db: Session
    ) -> Tuple[bool, str]:
        """
        从策略回收资金

        Args:
            strategy_id: 策略ID
            amount: 回收金额
            user_id: 用户ID
            db: 数据库会话

        Returns:
            Tuple[bool, str]: (是否成功, 消息)
        """
        try:
            # 获取当前分配
            current_allocation = await self._get_strategy_allocation(strategy_id, user_id, db)

            if not current_allocation:
                return False, "未找到该策略的资金分配记录"

            if amount > current_allocation.allocated_amount:
                return False, f"回收金额超过已分配金额，已分配: {current_allocation.allocated_amount}"

            # 检查策略是否有未平仓头寸
            active_positions = await self._get_strategy_positions(strategy_id, db)
            if active_positions:
                position_value = sum(p.current_value or 0 for p in active_positions)
                allocated_after_deallocation = current_allocation.allocated_amount - amount

                if allocated_after_deallocation < position_value:
                    return False, f"无法回收资金，策略有未平仓头寸，头寸价值: {position_value}"

            # 执行资金回收
            deallocation_result = await self._execute_deallocation(
                strategy_id, amount, user_id, db
            )

            if deallocation_result:
                await self.business_logger.log_event(
                    "funds_deallocated",
                    user_id=user_id,
                    strategy_id=strategy_id,
                    amount=float(amount)
                )

                return True, "资金回收成功"
            else:
                return False, "资金回收失败"

        except Exception as e:
            self.logger.error(f"资金回收失败: {str(e)}", exc_info=True)
            return False, f"资金回收失败: {str(e)}"

    async def get_optimal_allocation(
        self,
        user_id: str,
        strategies: List[str],
        db: Session
    ) -> List[FundAllocation]:
        """
        获取最优资金分配方案

        Args:
            user_id: 用户ID
            strategies: 策略ID列表
            db: 数据库会话

        Returns:
            List[FundAllocation]: 最优分配方案
        """
        try:
            # 获取策略表现数据
            strategy_performance = await self._get_strategy_performance(strategies, db)

            # 根据配置的策略计算最优分配
            if self.config.allocation_strategy == FundAllocationStrategy.FIXED_PERCENT:
                allocations = await self._calculate_fixed_percent_allocation(strategies, strategy_performance)
            elif self.config.allocation_strategy == FundAllocationStrategy.KELLY_CRITERION:
                allocations = await self._calculate_kelly_allocation(strategies, strategy_performance)
            elif self.config.allocation_strategy == FundAllocationStrategy.RISK_PARITY:
                allocations = await self._calculate_risk_parity_allocation(strategies, strategy_performance)
            elif self.config.allocation_strategy == FundAllocationStrategy.EQUAL_WEIGHT:
                allocations = await self._calculate_equal_weight_allocation(strategies, strategy_performance)
            elif self.config.allocation_strategy == FundAllocationStrategy.VOLATILITY_TARGET:
                allocations = await self._calculate_volatility_target_allocation(strategies, strategy_performance)
            else:
                allocations = await self._calculate_adaptive_allocation(strategies, strategy_performance, db)

            # 应用约束条件
            allocations = await self._apply_allocation_constraints(allocations)

            await self.business_logger.log_event(
                "optimal_allocation_calculated",
                user_id=user_id,
                strategy_count=len(strategies),
                allocation_strategy=self.config.allocation_strategy.value
            )

            return allocations

        except Exception as e:
            self.logger.error(f"最优分配计算失败: {str(e)}", exc_info=True)
            raise FundError(f"最优分配计算失败: {str(e)}")

    async def rebalance_portfolio(
        self,
        user_id: str,
        trigger: RebalanceTrigger = RebalanceTrigger.DEVIATION_BASED,
        db: Session = None
    ) -> List[RebalanceRecommendation]:
        """
        再平衡投资组合

        Args:
            user_id: 用户ID
            trigger: 触发条件
            db: 数据库会话

        Returns:
            List[RebalanceRecommendation]: 再平衡建议列表
        """
        try:
            if db is None:
                db = SessionLocal()

            # 获取当前分配和目标分配
            current_allocations = await self._get_current_allocations(user_id, db)
            strategies = [a.strategy_id for a in current_allocations]

            optimal_allocations = await self.get_optimal_allocation(user_id, strategies, db)

            # 生成再平衡建议
            recommendations = await self._generate_rebalance_recommendations(
                current_allocations, optimal_allocations, trigger
            )

            # 按优先级排序
            recommendations.sort(key=lambda x: x.priority, reverse=True)

            # 记录再平衡事件
            await self.business_logger.log_event(
                "portfolio_rebalanced",
                user_id=user_id,
                trigger=trigger.value,
                recommendations_count=len(recommendations)
            )

            return recommendations

        except Exception as e:
            self.logger.error(f"投资组合再平衡失败: {str(e)}", exc_info=True)
            raise FundError(f"投资组合再平衡失败: {str(e)}")

    async def get_fund_metrics(self, user_id: str, db: Session = None) -> FundMetrics:
        """
        获取资金指标

        Args:
            user_id: 用户ID
            db: 数据库会话

        Returns:
            FundMetrics: 资金指标
        """
        try:
            if db is None:
                db = SessionLocal()

            # 获取当前分配
            current_allocations = await self._get_current_allocations(user_id, db)

            # 计算资金指标
            metrics = await self._calculate_fund_metrics(user_id, current_allocations, db)

            return metrics

        except Exception as e:
            self.logger.error(f"资金指标计算失败: {str(e)}", exc_info=True)
            raise FundError(f"资金指标计算失败: {str(e)}")

    async def start_auto_monitoring(self, user_id: str):
        """启动自动监控"""
        if not self._monitoring_active:
            self._monitoring_active = True
            self._rebalance_task = asyncio.create_task(self._auto_rebalance_monitor(user_id))

            await self.business_logger.log_event(
                "auto_fund_monitoring_started",
                user_id=user_id
            )

    async def stop_auto_monitoring(self):
        """停止自动监控"""
        if self._monitoring_active:
            self._monitoring_active = False
            if self._rebalance_task:
                self._rebalance_task.cancel()

            await self.business_logger.log_event("auto_fund_monitoring_stopped")

    async def _auto_rebalance_monitor(self, user_id: str):
        """自动再平衡监控"""
        while self._monitoring_active:
            try:
                await asyncio.sleep(self.config.rebalance_frequency * 3600)  # 按频率检查

                with SessionLocal() as db:
                    # 检查是否需要再平衡
                    recommendations = await self.rebalance_portfolio(
                        user_id, RebalanceTrigger.DEVIATION_BASED, db
                    )

                    # 如果有再平衡建议，执行高优先级的建议
                    high_priority_recommendations = [
                        r for r in recommendations if r.priority >= 4
                    ]

                    if high_priority_recommendations:
                        for rec in high_priority_recommendations[:3]:  # 最多执行3个
                            await self._execute_rebalance_recommendation(rec, user_id, db)

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"自动再平衡监控异常: {str(e)}", exc_info=True)

    async def _get_current_allocations(self, user_id: str, db: Session) -> List[FundAllocation]:
        """获取当前资金分配"""
        try:
            # 这里应该从资金分配表获取数据
            # 暂时返回模拟数据
            allocations = [
                FundAllocation(
                    strategy_id="strategy_1",
                    strategy_name="Strategy 1",
                    allocated_amount=Decimal('1000'),
                    allocated_percent=20.0,
                    target_amount=Decimal('1200'),
                    target_percent=24.0,
                    current_value=Decimal('1100'),
                    pnl=Decimal('100'),
                    pnl_percent=10.0,
                    risk_score=0.6,
                    last_updated=datetime.utcnow()
                )
            ]

            return allocations

        except Exception as e:
            self.logger.error(f"获取当前分配失败: {str(e)}")
            return []

    async def _get_strategy_allocation(
        self,
        strategy_id: str,
        user_id: str,
        db: Session
    ) -> Optional[FundAllocation]:
        """获取特定策略的资金分配"""
        allocations = await self._get_current_allocations(user_id, db)
        return next(
            (a for a in allocations if a.strategy_id == strategy_id),
            None
        )

    async def _get_strategy_positions(self, strategy_id: str, db: Session) -> List[Position]:
        """获取策略的活跃持仓"""
        return db.query(Position).filter(
            and_(
                Position.strategy_id == uuid.UUID(strategy_id),
                Position.status == PositionStatus.OPEN.value
            )
        ).all()

    async def _calculate_fund_metrics(
        self,
        user_id: str,
        allocations: List[FundAllocation],
        db: Session
    ) -> FundMetrics:
        """计算资金指标"""
        try:
            total_funds = self.config.total_funds
            allocated_funds = sum(a.allocated_amount for a in allocations)
            available_funds = total_funds - allocated_funds

            # 计算总盈亏
            total_pnl = sum(a.pnl for a in allocations)
            total_pnl_percent = float(total_pnl / total_funds * 100) if total_funds > 0 else 0

            # 计算夏普比率
            sharpe_ratio = await self._calculate_sharpe_ratio(allocations)

            # 计算最大回撤
            max_drawdown = await self._calculate_max_drawdown(user_id, db)

            # 计算VaR
            var_95 = await self._calculate_portfolio_var(allocations)

            # 计算分配效率
            allocation_efficiency = await self._calculate_allocation_efficiency(allocations)

            # 计算多样化得分
            diversification_score = await self._calculate_diversification_score(allocations)

            return FundMetrics(
                total_funds=total_funds,
                allocated_funds=allocated_funds,
                available_funds=available_funds,
                allocation_efficiency=allocation_efficiency,
                diversification_score=diversification_score,
                total_pnl=total_pnl,
                total_pnl_percent=total_pnl_percent,
                sharpe_ratio=sharpe_ratio,
                max_drawdown=max_drawdown,
                var_95=var_95
            )

        except Exception as e:
            self.logger.error(f"资金指标计算失败: {str(e)}")
            # 返回默认值
            return FundMetrics(
                total_funds=self.config.total_funds,
                allocated_funds=Decimal('0'),
                available_funds=self.config.total_funds,
                allocation_efficiency=0.0,
                diversification_score=0.0,
                total_pnl=Decimal('0'),
                total_pnl_percent=0.0,
                sharpe_ratio=None,
                max_drawdown=0.0,
                var_95=0.0
            )

    async def _get_strategy_performance(
        self,
        strategies: List[str],
        db: Session
    ) -> Dict[str, Dict[str, Any]]:
        """获取策略表现数据"""
        performance = {}

        for strategy_id in strategies:
            try:
                # 这里应该从数据库计算策略表现
                # 暂时返回模拟数据
                performance[strategy_id] = {
                    'return': 0.15,
                    'volatility': 0.2,
                    'sharpe_ratio': 0.75,
                    'max_drawdown': 0.1,
                    'win_rate': 0.6,
                    'profit_factor': 1.5
                }
            except Exception as e:
                self.logger.error(f"获取策略表现失败 {strategy_id}: {str(e)}")
                performance[strategy_id] = {
                    'return': 0.0,
                    'volatility': 0.2,
                    'sharpe_ratio': 0.0,
                    'max_drawdown': 0.0,
                    'win_rate': 0.5,
                    'profit_factor': 1.0
                }

        return performance

    async def _calculate_fixed_percent_allocation(
        self,
        strategies: List[str],
        performance: Dict[str, Dict[str, Any]]
    ) -> List[FundAllocation]:
        """计算固定百分比分配"""
        equal_percent = 100.0 / len(strategies)
        allocations = []

        for strategy_id in strategies:
            amount = self.config.total_funds * (equal_percent / 100)
            perf = performance.get(strategy_id, {})

            allocation = FundAllocation(
                strategy_id=strategy_id,
                strategy_name=f"Strategy {strategy_id}",
                allocated_amount=Decimal('0'),
                allocated_percent=0.0,
                target_amount=amount,
                target_percent=equal_percent,
                current_value=Decimal('0'),
                pnl=Decimal('0'),
                pnl_percent=0.0,
                risk_score=1.0 - perf.get('sharpe_ratio', 0) / 2,  # 简化风险评分
                last_updated=datetime.utcnow()
            )
            allocations.append(allocation)

        return allocations

    async def _calculate_kelly_allocation(
        self,
        strategies: List[str],
        performance: Dict[str, Dict[str, Any]]
    ) -> List[FundAllocation]:
        """计算凯利公式分配"""
        allocations = []
        total_kelly = 0

        # 计算凯利比例
        kelly_fractions = {}
        for strategy_id in strategies:
            perf = performance.get(strategy_id, {})
            win_rate = perf.get('win_rate', 0.5)
            avg_win = perf.get('avg_win', 1.0)
            avg_loss = perf.get('avg_loss', 1.0)

            if avg_loss > 0:
                kelly = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_loss
                kelly = max(0, min(kelly, 0.25))  # 限制在0-25%之间
            else:
                kelly = 0

            kelly_fractions[strategy_id] = kelly
            total_kelly += kelly

        # 标准化并创建分配
        for strategy_id in strategies:
            if total_kelly > 0:
                percent = (kelly_fractions[strategy_id] / total_kelly) * 100
            else:
                percent = 100.0 / len(strategies)

            amount = self.config.total_funds * (percent / 100)
            perf = performance.get(strategy_id, {})

            allocation = FundAllocation(
                strategy_id=strategy_id,
                strategy_name=f"Strategy {strategy_id}",
                allocated_amount=Decimal('0'),
                allocated_percent=0.0,
                target_amount=amount,
                target_percent=percent,
                current_value=Decimal('0'),
                pnl=Decimal('0'),
                pnl_percent=0.0,
                risk_score=1.0 - perf.get('sharpe_ratio', 0) / 2,
                last_updated=datetime.utcnow()
            )
            allocations.append(allocation)

        return allocations

    async def _calculate_risk_parity_allocation(
        self,
        strategies: List[str],
        performance: Dict[str, Dict[str, Any]]
    ) -> List[FundAllocation]:
        """计算风险平价分配"""
        allocations = []

        # 计算风险贡献倒数
        risk_contributions = {}
        total_inverse_risk = 0

        for strategy_id in strategies:
            perf = performance.get(strategy_id, {})
            volatility = perf.get('volatility', 0.2)

            # 风险贡献与波动率成正比，分配权重与风险贡献成反比
            inverse_risk = 1.0 / volatility if volatility > 0 else 1.0
            risk_contributions[strategy_id] = inverse_risk
            total_inverse_risk += inverse_risk

        # 创建分配
        for strategy_id in strategies:
            percent = (risk_contributions[strategy_id] / total_inverse_risk) * 100
            amount = self.config.total_funds * (percent / 100)
            perf = performance.get(strategy_id, {})

            allocation = FundAllocation(
                strategy_id=strategy_id,
                strategy_name=f"Strategy {strategy_id}",
                allocated_amount=Decimal('0'),
                allocated_percent=0.0,
                target_amount=amount,
                target_percent=percent,
                current_value=Decimal('0'),
                pnl=Decimal('0'),
                pnl_percent=0.0,
                risk_score=perf.get('volatility', 0.2),
                last_updated=datetime.utcnow()
            )
            allocations.append(allocation)

        return allocations

    async def _calculate_equal_weight_allocation(
        self,
        strategies: List[str],
        performance: Dict[str, Dict[str, Any]]
    ) -> List[FundAllocation]:
        """计算等权重分配"""
        return await self._calculate_fixed_percent_allocation(strategies, performance)

    async def _calculate_volatility_target_allocation(
        self,
        strategies: List[str],
        performance: Dict[str, Dict[str, Any]]
    ) -> List[FundAllocation]:
        """计算目标波动率分配"""
        target_volatility = 0.15  # 15%目标波动率
        allocations = []

        for strategy_id in strategies:
            perf = performance.get(strategy_id, {})
            strategy_volatility = perf.get('volatility', 0.2)

            # 根据目标波动率调整权重
            weight = target_volatility / strategy_volatility if strategy_volatility > 0 else 1.0
            weight = min(weight, 2.0)  # 限制最大权重

            percent = weight * (100.0 / len(strategies))
            amount = self.config.total_funds * (percent / 100)

            allocation = FundAllocation(
                strategy_id=strategy_id,
                strategy_name=f"Strategy {strategy_id}",
                allocated_amount=Decimal('0'),
                allocated_percent=0.0,
                target_amount=amount,
                target_percent=percent,
                current_value=Decimal('0'),
                pnl=Decimal('0'),
                pnl_percent=0.0,
                risk_score=strategy_volatility,
                last_updated=datetime.utcnow()
            )
            allocations.append(allocation)

        return allocations

    async def _calculate_adaptive_allocation(
        self,
        strategies: List[str],
        performance: Dict[str, Dict[str, Any]],
        db: Session
    ) -> List[FundAllocation]:
        """计算自适应分配"""
        # 结合多种方法的综合分配
        fixed_allocations = await self._calculate_fixed_percent_allocation(strategies, performance)
        kelly_allocations = await self._calculate_kelly_allocation(strategies, performance)

        # 加权平均
        adaptive_allocations = []
        for i, strategy_id in enumerate(strategies):
            fixed = fixed_allocations[i]
            kelly = kelly_allocations[i]

            # 50%固定分配 + 50%凯利分配
            combined_percent = (fixed.target_percent + kelly.target_percent) / 2
            amount = self.config.total_funds * (combined_percent / 100)
            perf = performance.get(strategy_id, {})

            allocation = FundAllocation(
                strategy_id=strategy_id,
                strategy_name=f"Strategy {strategy_id}",
                allocated_amount=Decimal('0'),
                allocated_percent=0.0,
                target_amount=amount,
                target_percent=combined_percent,
                current_value=Decimal('0'),
                pnl=Decimal('0'),
                pnl_percent=0.0,
                risk_score=1.0 - perf.get('sharpe_ratio', 0) / 2,
                last_updated=datetime.utcnow()
            )
            adaptive_allocations.append(allocation)

        return adaptive_allocations

    async def _apply_allocation_constraints(
        self,
        allocations: List[FundAllocation]
    ) -> List[FundAllocation]:
        """应用分配约束"""
        # 重新标准化以满足约束条件
        total_percent = sum(a.target_percent for a in allocations)

        if total_percent > (100 - self.config.min_reserve_percent):
            # 按比例缩减
            scale_factor = (100 - self.config.min_reserve_percent) / total_percent
            for allocation in allocations:
                allocation.target_percent *= scale_factor
                allocation.target_amount = self.config.total_funds * (allocation.target_percent / 100)

        # 应用单个策略最大限制
        for allocation in allocations:
            if allocation.target_percent > self.config.max_allocation_percent:
                allocation.target_percent = self.config.max_allocation_percent
                allocation.target_amount = self.config.total_funds * (allocation.target_percent / 100)

        return allocations

    async def _generate_rebalance_recommendations(
        self,
        current: List[FundAllocation],
        target: List[FundAllocation],
        trigger: RebalanceTrigger
    ) -> List[RebalanceRecommendation]:
        """生成再平衡建议"""
        recommendations = []

        for target_alloc in target:
            current_alloc = next(
                (c for c in current if c.strategy_id == target_alloc.strategy_id),
                None
            )

            if current_alloc:
                # 计算偏差
                deviation = abs(current_alloc.allocated_percent - target_alloc.target_percent)

                if deviation > self.config.rebalance_threshold:
                    if current_alloc.allocated_percent < target_alloc.target_percent:
                        action = "increase"
                        amount = target_alloc.target_amount - current_alloc.allocated_amount
                        reason = f"当前分配{current_alloc.allocated_percent:.1f}%，目标{target_alloc.target_percent:.1f}%"
                    else:
                        action = "decrease"
                        amount = current_alloc.allocated_amount - target_alloc.target_amount
                        reason = f"当前分配{current_alloc.allocated_percent:.1f}%，目标{target_alloc.target_percent:.1f}%"

                    # 计算优先级
                    priority = min(int(deviation / 10) + 1, 5)

                    recommendation = RebalanceRecommendation(
                        strategy_id=target_alloc.strategy_id,
                        action=action,
                        amount=amount,
                        reason=reason,
                        priority=priority
                    )
                    recommendations.append(recommendation)

        return recommendations

    async def _execute_allocation(
        self,
        strategy_id: str,
        amount: Decimal,
        user_id: str,
        db: Session
    ) -> bool:
        """执行资金分配"""
        try:
            # 这里应该更新资金分配表
            # 暂时返回成功
            return True

        except Exception as e:
            self.logger.error(f"执行资金分配失败: {str(e)}")
            return False

    async def _execute_deallocation(
        self,
        strategy_id: str,
        amount: Decimal,
        user_id: str,
        db: Session
    ) -> bool:
        """执行资金回收"""
        try:
            # 这里应该更新资金分配表
            # 暂时返回成功
            return True

        except Exception as e:
            self.logger.error(f"执行资金回收失败: {str(e)}")
            return False

    async def _execute_rebalance_recommendation(
        self,
        recommendation: RebalanceRecommendation,
        user_id: str,
        db: Session
    ):
        """执行再平衡建议"""
        try:
            if recommendation.action == "increase":
                await self._execute_allocation(
                    recommendation.strategy_id,
                    recommendation.amount,
                    user_id,
                    db
                )
            elif recommendation.action == "decrease":
                await self._execute_deallocation(
                    recommendation.strategy_id,
                    recommendation.amount,
                    user_id,
                    db
                )

            await self.business_logger.log_event(
                "rebalance_executed",
                user_id=user_id,
                strategy_id=recommendation.strategy_id,
                action=recommendation.action,
                amount=float(recommendation.amount)
            )

        except Exception as e:
            self.logger.error(f"执行再平衡建议失败: {str(e)}")

    async def _calculate_sharpe_ratio(self, allocations: List[FundAllocation]) -> Optional[float]:
        """计算夏普比率"""
        try:
            if not allocations:
                return None

            total_return = sum(a.pnl_percent for a in allocations) / len(allocations)
            total_volatility = sum(a.risk_score for a in allocations) / len(allocations)

            if total_volatility == 0:
                return None

            # 假设无风险利率为2%
            risk_free_rate = 0.02
            sharpe_ratio = (total_return / 100 - risk_free_rate) / total_volatility

            return sharpe_ratio

        except Exception as e:
            self.logger.error(f"夏普比率计算失败: {str(e)}")
            return None

    async def _calculate_max_drawdown(self, user_id: str, db: Session) -> float:
        """计算最大回撤"""
        try:
            # 这里应该从历史数据计算最大回撤
            # 暂时返回模拟值
            return 8.5

        except Exception as e:
            self.logger.error(f"最大回撤计算失败: {str(e)}")
            return 0.0

    async def _calculate_portfolio_var(self, allocations: List[FundAllocation]) -> float:
        """计算投资组合VaR"""
        try:
            if not allocations:
                return 0.0

            total_value = sum(a.current_value for a in allocations if a.current_value > 0)
            if total_value == 0:
                return 0.0

            # 简化的VaR计算：假设95% VaR为总价值的2%
            var_95 = float(total_value * 0.02)

            return var_95

        except Exception as e:
            self.logger.error(f"投资组合VaR计算失败: {str(e)}")
            return 0.0

    async def _calculate_allocation_efficiency(self, allocations: List[FundAllocation]) -> float:
        """计算分配效率"""
        try:
            if not allocations:
                return 0.0

            # 效率 = 正收益策略的比例
            positive_allocations = [a for a in allocations if a.pnl > 0]
            efficiency = len(positive_allocations) / len(allocations)

            return efficiency

        except Exception as e:
            self.logger.error(f"分配效率计算失败: {str(e)}")
            return 0.0

    async def _calculate_diversification_score(self, allocations: List[FundAllocation]) -> float:
        """计算多样化得分"""
        try:
            if not allocations:
                return 0.0

            # 使用赫芬达尔指数计算集中度，多样化得分 = 1 - 集中度
            total_allocation = sum(a.allocated_percent for a in allocations)
            if total_allocation == 0:
                return 0.0

            herfindahl_index = sum((a.allocated_percent / total_allocation) ** 2 for a in allocations)
            diversification_score = 1 - herfindahl_index

            return diversification_score

        except Exception as e:
            self.logger.error(f"多样化得分计算失败: {str(e)}")
            return 0.0