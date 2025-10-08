"""
风险管理服务

负责交易风险评估、仓位控制、止损管理和风险监控。
支持多种风险指标和动态风险管理策略。
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
from ..core.exceptions import RiskError, ValidationError
from ..models.trading_order import TradingOrder, OrderStatus
from ..models.position import Position, PositionStatus, PositionSide
from ..models.trading_strategy import TradingStrategy
from ..models.user import User

logger = logging.getLogger(__name__)
business_logger = BusinessLogger("risk_manager")


class RiskLevel(str, Enum):
    """风险等级"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class RiskAction(str, Enum):
    """风险措施"""
    NONE = "none"
    WARNING = "warning"
    REDUCE_POSITION = "reduce_position"
    CLOSE_POSITION = "close_position"
    STOP_NEW_ORDERS = "stop_new_orders"
    EMERGENCY_EXIT = "emergency_exit"


@dataclass
class RiskParameters:
    """风险参数"""
    max_position_size_percent: float = 20.0  # 最大单个持仓占总资金百分比
    max_total_exposure_percent: float = 80.0  # 最大总风险敞口百分比
    max_leverage: float = 3.0  # 最大杠杆倍数
    max_correlation_exposure: float = 0.7  # 最大相关性敞口
    max_daily_loss_percent: float = 5.0  # 最大日亏损百分比
    max_drawdown_percent: float = 15.0  # 最大回撤百分比
    stop_loss_percent: float = 2.0  # 默认止损百分比
    take_profit_percent: float = 6.0  # 默认止盈百分比
    position_size_method: str = "fixed_percent"  # 仓位计算方法
    volatility_adjustment: bool = True  # 是否根据波动率调整
    correlation_adjustment: bool = True  # 是否根据相关性调整


@dataclass
class RiskMetrics:
    """风险指标"""
    current_exposure: float = 0.0
    total_exposure_percent: float = 0.0
    leverage_ratio: float = 0.0
    daily_pnl: float = 0.0
    daily_pnl_percent: float = 0.0
    max_drawdown: float = 0.0
    current_drawdown: float = 0.0
    var_95: float = 0.0  # 95% VaR
    position_concentration: float = 0.0
    correlation_risk: float = 0.0
    liquidity_risk: float = 0.0


@dataclass
class RiskAssessment:
    """风险评估结果"""
    risk_level: RiskLevel
    risk_score: float  # 0-100
    risk_metrics: RiskMetrics
    warnings: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    required_actions: List[RiskAction] = field(default_factory=list)


@dataclass
class PositionRiskResult:
    """持仓风险评估结果"""
    position_id: str
    symbol: str
    risk_score: float
    risk_level: RiskLevel
    max_loss: float
    risk_reward_ratio: Optional[float]
    stop_loss_price: Optional[float]
    take_profit_price: Optional[float]
    warnings: List[str] = field(default_factory=list)


class RiskManager:
    """风险管理器"""

    def __init__(self, risk_params: RiskParameters = None):
        self.logger = logger
        self.business_logger = business_logger
        self.cache = get_cache()
        self.risk_params = risk_params or RiskParameters()

        # 风险监控任务
        self._monitoring_active = False
        self._monitoring_task = None

    async def assess_order_risk(
        self,
        order_request: Dict[str, Any],
        user_id: str,
        db: Session
    ) -> RiskAssessment:
        """
        评估订单风险

        Args:
            order_request: 订单请求
            user_id: 用户ID
            db: 数据库会话

        Returns:
            RiskAssessment: 风险评估结果
        """
        try:
            # 获取用户当前持仓和订单
            current_positions = await self._get_user_positions(user_id, db)
            pending_orders = await self._get_user_orders(user_id, OrderStatus.PENDING.value, db)

            # 计算风险指标
            risk_metrics = await self._calculate_risk_metrics(
                user_id, current_positions, pending_orders, order_request, db
            )

            # 评估风险等级
            risk_level, risk_score = self._assess_risk_level(risk_metrics)

            # 生成警告和建议
            warnings, recommendations, actions = self._generate_risk_response(
                risk_level, risk_metrics, order_request
            )

            assessment = RiskAssessment(
                risk_level=risk_level,
                risk_score=risk_score,
                risk_metrics=risk_metrics,
                warnings=warnings,
                recommendations=recommendations,
                required_actions=actions
            )

            # 记录风险评估
            await self.business_logger.log_event(
                "order_risk_assessed",
                user_id=user_id,
                symbol=order_request.get('symbol'),
                side=order_request.get('side'),
                amount=order_request.get('amount'),
                risk_level=risk_level.value,
                risk_score=risk_score
            )

            return assessment

        except Exception as e:
            self.logger.error(f"订单风险评估失败: {str(e)}", exc_info=True)
            raise RiskError(f"风险评估失败: {str(e)}")

    async def assess_position_risk(
        self,
        position: Position,
        db: Session
    ) -> PositionRiskResult:
        """
        评估单个持仓风险

        Args:
            position: 持仓对象
            db: 数据库会话

        Returns:
            PositionRiskResult: 持仓风险评估结果
        """
        try:
            # 获取市场数据
            market_data = await self._get_market_data(position.symbol, position.exchange)

            # 计算风险指标
            risk_score = await self._calculate_position_risk_score(position, market_data)

            # 评估风险等级
            risk_level = self._get_risk_level_from_score(risk_score)

            # 计算最大损失
            max_loss = position.calculate_max_loss()

            # 计算风险回报比
            risk_reward_ratio = position.risk_reward_ratio

            # 生成警告
            warnings = self._generate_position_warnings(position, market_data)

            result = PositionRiskResult(
                position_id=str(position.id),
                symbol=position.symbol,
                risk_score=risk_score,
                risk_level=risk_level,
                max_loss=float(max_loss),
                risk_reward_ratio=risk_reward_ratio,
                stop_loss_price=float(position.stop_loss_price) if position.stop_loss_price else None,
                take_profit_price=float(position.take_profit_price) if position.take_profit_price else None,
                warnings=warnings
            )

            # 记录持仓风险评估
            await self.business_logger.log_event(
                "position_risk_assessed",
                position_id=str(position.id),
                symbol=position.symbol,
                risk_level=risk_level.value,
                risk_score=risk_score
            )

            return result

        except Exception as e:
            self.logger.error(f"持仓风险评估失败: {str(e)}", exc_info=True)
            raise RiskError(f"持仓风险评估失败: {str(e)}")

    async def calculate_position_size(
        self,
        symbol: str,
        side: str,
        entry_price: Decimal,
        stop_loss_price: Decimal,
        account_balance: Decimal,
        risk_per_trade: float = 2.0,
        db: Session = None
    ) -> Decimal:
        """
        计算建议仓位大小

        Args:
            symbol: 交易符号
            side: 交易方向
            entry_price: 入场价格
            stop_loss_price: 止损价格
            account_balance: 账户余额
            risk_per_trade: 每笔交易风险百分比
            db: 数据库会话

        Returns:
            Decimal: 建议仓位大小
        """
        try:
            if db is None:
                db = SessionLocal()

            # 计算风险金额
            risk_amount = account_balance * (risk_per_trade / 100)

            # 计算每单位风险
            if side.lower() == 'buy':
                price_risk = entry_price - stop_loss_price
            else:
                price_risk = stop_loss_price - entry_price

            if price_risk <= 0:
                raise ValidationError("止损价格设置错误")

            # 基础仓位大小
            position_size = risk_amount / price_risk

            # 应用风险管理参数
            max_position_value = account_balance * (self.risk_params.max_position_size_percent / 100)
            max_position_size = max_position_value / entry_price

            # 取较小值
            position_size = min(position_size, max_position_size)

            # 波动率调整
            if self.risk_params.volatility_adjustment:
                volatility_factor = await self._calculate_volatility_adjustment(symbol, db)
                position_size *= volatility_factor

            # 相关性调整
            if self.risk_params.correlation_adjustment:
                correlation_factor = await self._calculate_correlation_adjustment(symbol, db)
                position_size *= correlation_factor

            # 确保最小交易量
            min_amount = await self._get_min_amount(symbol)
            position_size = max(position_size, min_amount)

            return position_size

        except Exception as e:
            self.logger.error(f"仓位大小计算失败: {str(e)}", exc_info=True)
            raise RiskError(f"仓位大小计算失败: {str(e)}")

    async def update_stop_loss(
        self,
        position: Position,
        current_price: Decimal,
        db: Session
    ) -> bool:
        """
        更新移动止损

        Args:
            position: 持仓对象
            current_price: 当前价格
            db: 数据库会话

        Returns:
            bool: 是否更新成功
        """
        try:
            updated = position.update_trailing_stop(current_price)

            if updated:
                db.commit()

                await self.business_logger.log_event(
                    "trailing_stop_updated",
                    position_id=str(position.id),
                    symbol=position.symbol,
                    new_stop_loss=float(position.stop_loss_price),
                    current_price=float(current_price)
                )

            return updated

        except Exception as e:
            self.logger.error(f"移动止损更新失败: {str(e)}", exc_info=True)
            return False

    async def check_risk_limits(self, user_id: str, db: Session) -> RiskAssessment:
        """
        检查用户风险限制

        Args:
            user_id: 用户ID
            db: 数据库会话

        Returns:
            RiskAssessment: 风险评估结果
        """
        try:
            # 获取用户持仓和订单
            current_positions = await self._get_user_positions(user_id, db)
            pending_orders = await self._get_user_orders(user_id, OrderStatus.PENDING.value, db)

            # 计算当前风险指标
            risk_metrics = await self._calculate_risk_metrics(
                user_id, current_positions, pending_orders, None, db
            )

            # 评估风险等级
            risk_level, risk_score = self._assess_risk_level(risk_metrics)

            # 生成警告和建议
            warnings, recommendations, actions = self._generate_risk_response(
                risk_level, risk_metrics, None
            )

            assessment = RiskAssessment(
                risk_level=risk_level,
                risk_score=risk_score,
                risk_metrics=risk_metrics,
                warnings=warnings,
                recommendations=recommendations,
                required_actions=actions
            )

            # 记录风险检查
            await self.business_logger.log_event(
                "risk_limits_checked",
                user_id=user_id,
                risk_level=risk_level.value,
                risk_score=risk_score
            )

            return assessment

        except Exception as e:
            self.logger.error(f"风险限制检查失败: {str(e)}", exc_info=True)
            raise RiskError(f"风险限制检查失败: {str(e)}")

    async def start_monitoring(self, user_id: str):
        """启动风险监控"""
        if not self._monitoring_active:
            self._monitoring_active = True
            self._monitoring_task = asyncio.create_task(self._monitor_risk(user_id))

            await self.business_logger.log_event(
                "risk_monitoring_started",
                user_id=user_id
            )

    async def stop_monitoring(self):
        """停止风险监控"""
        if self._monitoring_active:
            self._monitoring_active = False
            if self._monitoring_task:
                self._monitoring_task.cancel()

            await self.business_logger.log_event("risk_monitoring_stopped")

    async def _monitor_risk(self, user_id: str):
        """风险监控任务"""
        while self._monitoring_active:
            try:
                await asyncio.sleep(30)  # 每30秒检查一次

                with SessionLocal() as db:
                    assessment = await self.check_risk_limits(user_id, db)

                    # 执行必要的风险措施
                    for action in assessment.required_actions:
                        await self._execute_risk_action(action, user_id, assessment, db)

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"风险监控异常: {str(e)}", exc_info=True)

    async def _execute_risk_action(
        self,
        action: RiskAction,
        user_id: str,
        assessment: RiskAssessment,
        db: Session
    ):
        """执行风险措施"""
        try:
            if action == RiskAction.NONE:
                return

            elif action == RiskAction.WARNING:
                # 发送风险警告
                await self._send_risk_warning(user_id, assessment)

            elif action == RiskAction.REDUCE_POSITION:
                # 建议减仓
                await self._suggest_position_reduction(user_id, assessment, db)

            elif action == RiskAction.CLOSE_POSITION:
                # 建议平仓
                await self._suggest_position_closure(user_id, assessment, db)

            elif action == RiskAction.STOP_NEW_ORDERS:
                # 停止新订单
                await self._stop_new_orders(user_id)

            elif action == RiskAction.EMERGENCY_EXIT:
                # 紧急退出
                await self._emergency_exit(user_id, db)

            await self.business_logger.log_event(
                "risk_action_executed",
                user_id=user_id,
                action=action.value,
                risk_level=assessment.risk_level.value
            )

        except Exception as e:
            self.logger.error(f"风险措施执行失败 {action}: {str(e)}", exc_info=True)

    async def _get_user_positions(self, user_id: str, db: Session) -> List[Position]:
        """获取用户持仓"""
        return db.query(Position).filter(
            and_(
                Position.user_id == uuid.UUID(user_id),
                Position.status == PositionStatus.OPEN.value
            )
        ).all()

    async def _get_user_orders(
        self,
        user_id: str,
        status: str,
        db: Session
    ) -> List[TradingOrder]:
        """获取用户订单"""
        return db.query(TradingOrder).filter(
            and_(
                TradingOrder.user_id == uuid.UUID(user_id),
                TradingOrder.status == status
            )
        ).all()

    async def _calculate_risk_metrics(
        self,
        user_id: str,
        positions: List[Position],
        orders: List[TradingOrder],
        new_order: Optional[Dict[str, Any]],
        db: Session
    ) -> RiskMetrics:
        """计算风险指标"""
        metrics = RiskMetrics()

        try:
            # 计算总风险敞口
            total_exposure = sum(
                position.current_value or 0 for position in positions
            )

            # 包含新订单的风险敞口
            if new_order:
                new_order_value = Decimal(str(new_order.get('amount', 0))) * \
                                Decimal(str(new_order.get('price', 0)))
                total_exposure += new_order_value

            # 获取账户总价值
            account_value = await self._get_account_value(user_id, db)

            if account_value > 0:
                metrics.total_exposure_percent = float(total_exposure / account_value * 100)
                metrics.current_exposure = float(total_exposure)

            # 计算杠杆比率
            total_margin = sum(position.margin_used for position in positions if position.margin_used)
            if total_margin > 0:
                metrics.leverage_ratio = float(total_exposure / total_margin)

            # 计算日盈亏
            today = datetime.utcnow().date()
            daily_pnl = sum(
                position.total_pnl for position in positions
                if position.last_updated.date() == today
            )

            metrics.daily_pnl = float(daily_pnl)
            if account_value > 0:
                metrics.daily_pnl_percent = float(daily_pnl / account_value * 100)

            # 计算回撤
            metrics.max_drawdown, metrics.current_drawdown = await self._calculate_drawdown(
                user_id, account_value, db
            )

            # 计算VaR
            metrics.var_95 = await self._calculate_var(user_id, positions, db)

            # 计算持仓集中度
            metrics.position_concentration = await self._calculate_concentration(positions)

            # 计算相关性风险
            metrics.correlation_risk = await self._calculate_correlation_risk(positions, db)

            # 计算流动性风险
            metrics.liquidity_risk = await self._calculate_liquidity_risk(positions, db)

        except Exception as e:
            self.logger.error(f"风险指标计算失败: {str(e)}", exc_info=True)

        return metrics

    def _assess_risk_level(self, metrics: RiskMetrics) -> Tuple[RiskLevel, float]:
        """评估风险等级"""
        risk_score = 0

        # 敞口风险评分 (0-25分)
        if metrics.total_exposure_percent > 90:
            risk_score += 25
        elif metrics.total_exposure_percent > 80:
            risk_score += 20
        elif metrics.total_exposure_percent > 60:
            risk_score += 15
        elif metrics.total_exposure_percent > 40:
            risk_score += 10
        elif metrics.total_exposure_percent > 20:
            risk_score += 5

        # 杠杆风险评分 (0-20分)
        if metrics.leverage_ratio > 5:
            risk_score += 20
        elif metrics.leverage_ratio > 3:
            risk_score += 15
        elif metrics.leverage_ratio > 2:
            risk_score += 10
        elif metrics.leverage_ratio > 1:
            risk_score += 5

        # 日亏损评分 (0-20分)
        if metrics.daily_pnl_percent < -10:
            risk_score += 20
        elif metrics.daily_pnl_percent < -7:
            risk_score += 15
        elif metrics.daily_pnl_percent < -5:
            risk_score += 10
        elif metrics.daily_pnl_percent < -3:
            risk_score += 5

        # 回撤评分 (0-20分)
        if metrics.current_drawdown > 20:
            risk_score += 20
        elif metrics.current_drawdown > 15:
            risk_score += 15
        elif metrics.current_drawdown > 10:
            risk_score += 10
        elif metrics.current_drawdown > 5:
            risk_score += 5

        # 集中度评分 (0-15分)
        if metrics.position_concentration > 0.8:
            risk_score += 15
        elif metrics.position_concentration > 0.6:
            risk_score += 10
        elif metrics.position_concentration > 0.4:
            risk_score += 5

        # 确定风险等级
        if risk_score >= 70:
            risk_level = RiskLevel.CRITICAL
        elif risk_score >= 50:
            risk_level = RiskLevel.HIGH
        elif risk_score >= 30:
            risk_level = RiskLevel.MEDIUM
        else:
            risk_level = RiskLevel.LOW

        return risk_level, risk_score

    def _generate_risk_response(
        self,
        risk_level: RiskLevel,
        metrics: RiskMetrics,
        new_order: Optional[Dict[str, Any]]
    ) -> Tuple[List[str], List[str], List[RiskAction]]:
        """生成风险响应"""
        warnings = []
        recommendations = []
        actions = []

        if risk_level == RiskLevel.CRITICAL:
            warnings.append("风险等级：危急")
            warnings.append("当前风险敞口过大，可能导致重大损失")
            recommendations.append("立即停止所有新交易")
            recommendations.append("考虑紧急平仓以降低风险")
            actions.extend([RiskAction.STOP_NEW_ORDERS, RiskAction.EMERGENCY_EXIT])

        elif risk_level == RiskLevel.HIGH:
            warnings.append("风险等级：高")
            warnings.append("风险指标超出正常范围")
            recommendations.append("减少持仓规模")
            recommendations.append("加强止损管理")
            actions.extend([RiskAction.REDUCE_POSITION, RiskAction.WARNING])

        elif risk_level == RiskLevel.MEDIUM:
            warnings.append("风险等级：中等")
            warnings.append("需要关注风险指标变化")
            recommendations.append("谨慎开新仓")
            recommendations.append("定期检查持仓状况")
            actions.append(RiskAction.WARNING)

        # 具体风险警告
        if metrics.total_exposure_percent > 80:
            warnings.append(f"总风险敞口过高：{metrics.total_exposure_percent:.1f}%")

        if metrics.leverage_ratio > 3:
            warnings.append(f"杠杆比率过高：{metrics.leverage_ratio:.1f}倍")

        if metrics.daily_pnl_percent < -5:
            warnings.append(f"当日亏损较大：{metrics.daily_pnl_percent:.1f}%")

        if metrics.current_drawdown > 10:
            warnings.append(f"当前回撤较深：{metrics.current_drawdown:.1f}%")

        if metrics.position_concentration > 0.6:
            warnings.append(f"持仓过于集中：{metrics.position_concentration:.1f}%")

        return warnings, recommendations, actions

    async def _calculate_position_risk_score(
        self,
        position: Position,
        market_data: Dict[str, Any]
    ) -> float:
        """计算持仓风险评分"""
        risk_score = 0

        try:
            # 基于盈亏的风险评分 (0-30分)
            pnl_percent = position.pnl_percent
            if pnl_percent < -20:
                risk_score += 30
            elif pnl_percent < -15:
                risk_score += 25
            elif pnl_percent < -10:
                risk_score += 20
            elif pnl_percent < -5:
                risk_score += 15
            elif pnl_percent < 0:
                risk_score += 10

            # 基于持仓时间的风险评分 (0-20分)
            if position.duration:
                duration_hours = position.duration.total_seconds() / 3600
                if duration_hours > 168:  # 超过7天
                    risk_score += 20
                elif duration_hours > 72:  # 超过3天
                    risk_score += 15
                elif duration_hours > 24:  # 超过1天
                    risk_score += 10
                elif duration_hours > 12:
                    risk_score += 5

            # 基于波动率的风险评分 (0-25分)
            if market_data.get('volatility'):
                volatility = market_data['volatility']
                if volatility > 0.1:  # 10%以上波动率
                    risk_score += 25
                elif volatility > 0.07:  # 7-10%波动率
                    risk_score += 20
                elif volatility > 0.05:  # 5-7%波动率
                    risk_score += 15
                elif volatility > 0.03:  # 3-5%波动率
                    risk_score += 10

            # 基于流动性的风险评分 (0-15分)
            if market_data.get('liquidity_score'):
                liquidity = market_data['liquidity_score']
                if liquidity < 0.3:  # 低流动性
                    risk_score += 15
                elif liquidity < 0.5:  # 中低流动性
                    risk_score += 10
                elif liquidity < 0.7:  # 中等流动性
                    risk_score += 5

            # 基于价格偏离的风险评分 (0-10分)
            if position.current_price and position.average_cost:
                deviation = abs(position.current_price - position.average_cost) / position.average_cost
                if deviation > 0.2:  # 偏离20%以上
                    risk_score += 10
                elif deviation > 0.15:  # 偏离15-20%
                    risk_score += 8
                elif deviation > 0.1:  # 偏离10-15%
                    risk_score += 5
                elif deviation > 0.05:  # 偏离5-10%
                    risk_score += 3

        except Exception as e:
            self.logger.error(f"持仓风险评分计算失败: {str(e)}", exc_info=True)
            risk_score = 50  # 默认中等风险

        return min(risk_score, 100)

    def _get_risk_level_from_score(self, score: float) -> RiskLevel:
        """根据评分获取风险等级"""
        if score >= 70:
            return RiskLevel.CRITICAL
        elif score >= 50:
            return RiskLevel.HIGH
        elif score >= 30:
            return RiskLevel.MEDIUM
        else:
            return RiskLevel.LOW

    def _generate_position_warnings(
        self,
        position: Position,
        market_data: Dict[str, Any]
    ) -> List[str]:
        """生成持仓风险警告"""
        warnings = []

        # 盈亏警告
        pnl_percent = position.pnl_percent
        if pnl_percent < -15:
            warnings.append(f"严重亏损：{pnl_percent:.1f}%")
        elif pnl_percent < -10:
            warnings.append(f"较大亏损：{pnl_percent:.1f}%")
        elif pnl_percent < -5:
            warnings.append(f"出现亏损：{pnl_percent:.1f}%")

        # 止损警告
        if position.stop_loss_price and position.current_price:
            if position.side == PositionSide.LONG.value:
                distance_to_stop = float((position.current_price - position.stop_loss_price) / position.current_price * 100)
                if distance_to_stop < 2:
                    warnings.append(f"接近止损价格：{distance_to_stop:.1f}%")
            else:
                distance_to_stop = float((position.stop_loss_price - position.current_price) / position.current_price * 100)
                if distance_to_stop < 2:
                    warnings.append(f"接近止损价格：{distance_to_stop:.1f}%")

        # 时间警告
        if position.duration:
            duration_hours = position.duration.total_seconds() / 3600
            if duration_hours > 168:
                warnings.append(f"持仓时间过长：{duration_hours:.0f}小时")

        # 波动率警告
        if market_data.get('volatility', 0) > 0.08:
            warnings.append(f"高波动率风险：{market_data['volatility']*100:.1f}%")

        # 流动性警告
        if market_data.get('liquidity_score', 1) < 0.4:
            warnings.append("流动性不足风险")

        return warnings

    async def _get_market_data(self, symbol: str, exchange: str) -> Dict[str, Any]:
        """获取市场数据"""
        cache_key = f"market_data:{exchange}:{symbol}"
        cached_data = await self.cache.get(cache_key)

        if cached_data:
            return cached_data

        try:
            # 这里应该调用交易所API获取市场数据
            # 暂时返回模拟数据
            market_data = {
                'volatility': 0.05,
                'liquidity_score': 0.8,
                'price': 50000,
                'volume': 1000
            }

            # 缓存5分钟
            await self.cache.set(cache_key, market_data, expire=300)

            return market_data

        except Exception as e:
            self.logger.error(f"获取市场数据失败 {symbol}: {str(e)}")
            return {}

    async def _calculate_volatility_adjustment(self, symbol: str, db: Session) -> float:
        """计算波动率调整因子"""
        try:
            # 获取历史价格数据计算波动率
            # 这里使用简化逻辑
            market_data = await self._get_market_data(symbol, "binance")
            volatility = market_data.get('volatility', 0.05)

            # 波动率越高，仓位越小
            if volatility > 0.1:
                return 0.5
            elif volatility > 0.07:
                return 0.7
            elif volatility > 0.05:
                return 0.85
            else:
                return 1.0

        except Exception as e:
            self.logger.error(f"波动率调整计算失败: {str(e)}")
            return 1.0

    async def _calculate_correlation_adjustment(self, symbol: str, db: Session) -> float:
        """计算相关性调整因子"""
        try:
            # 计算与现有持仓的相关性
            # 这里使用简化逻辑
            return 0.9  # 暂时返回固定值

        except Exception as e:
            self.logger.error(f"相关性调整计算失败: {str(e)}")
            return 1.0

    async def _get_min_amount(self, symbol: str) -> Decimal:
        """获取最小交易量"""
        # 这里应该从交易所获取最小交易量限制
        return Decimal('0.001')

    async def _get_account_value(self, user_id: str, db: Session) -> Decimal:
        """获取账户总价值"""
        # 这里应该计算用户总资产价值
        # 暂时返回模拟值
        return Decimal('10000')

    async def _calculate_drawdown(
        self,
        user_id: str,
        current_value: Decimal,
        db: Session
    ) -> Tuple[float, float]:
        """计算回撤指标"""
        try:
            # 获取历史最高价值
            # 这里使用简化逻辑
            peak_value = current_value * Decimal('1.1')  # 假设历史最高值为当前值的110%

            max_drawdown = 0.0
            current_drawdown = 0.0

            if peak_value > 0:
                current_drawdown = float((peak_value - current_value) / peak_value * 100)
                max_drawdown = current_drawdown  # 简化处理

            return max_drawdown, current_drawdown

        except Exception as e:
            self.logger.error(f"回撤计算失败: {str(e)}")
            return 0.0, 0.0

    async def _calculate_var(
        self,
        user_id: str,
        positions: List[Position],
        db: Session
    ) -> float:
        """计算VaR (Value at Risk)"""
        try:
            # 使用历史模拟法计算95% VaR
            # 这里使用简化逻辑
            if not positions:
                return 0.0

            total_value = sum(position.current_value or 0 for position in positions)
            # 假设日VaR为总价值的2%
            var_95 = float(total_value * 0.02)

            return var_95

        except Exception as e:
            self.logger.error(f"VaR计算失败: {str(e)}")
            return 0.0

    async def _calculate_concentration(self, positions: List[Position]) -> float:
        """计算持仓集中度"""
        if not positions:
            return 0.0

        total_value = sum(position.current_value or 0 for position in positions)
        if total_value == 0:
            return 0.0

        # 计算最大持仓占比
        max_position_value = max(position.current_value or 0 for position in positions)
        concentration = float(max_position_value / total_value)

        return concentration

    async def _calculate_correlation_risk(self, positions: List[Position], db: Session) -> float:
        """计算相关性风险"""
        # 这里使用简化逻辑
        return 0.5

    async def _calculate_liquidity_risk(self, positions: List[Position], db: Session) -> float:
        """计算流动性风险"""
        try:
            if not positions:
                return 0.0

            total_liquidity_risk = 0.0
            for position in positions:
                market_data = await self._get_market_data(position.symbol, position.exchange)
                liquidity_score = market_data.get('liquidity_score', 1.0)
                position_risk = 1.0 - liquidity_score
                total_liquidity_risk += position_risk

            return total_liquidity_risk / len(positions)

        except Exception as e:
            self.logger.error(f"流动性风险计算失败: {str(e)}")
            return 0.5

    async def _send_risk_warning(self, user_id: str, assessment: RiskAssessment):
        """发送风险警告"""
        # 这里应该实现通知逻辑
        await self.business_logger.log_event(
            "risk_warning_sent",
            user_id=user_id,
            risk_level=assessment.risk_level.value,
            warnings=assessment.warnings
        )

    async def _suggest_position_reduction(self, user_id: str, assessment: RiskAssessment, db: Session):
        """建议减仓"""
        await self.business_logger.log_event(
            "position_reduction_suggested",
            user_id=user_id,
            risk_level=assessment.risk_level.value
        )

    async def _suggest_position_closure(self, user_id: str, assessment: RiskAssessment, db: Session):
        """建议平仓"""
        await self.business_logger.log_event(
            "position_closure_suggested",
            user_id=user_id,
            risk_level=assessment.risk_level.value
        )

    async def _stop_new_orders(self, user_id: str):
        """停止新订单"""
        # 这里应该实现停止新订单的逻辑
        await self.cache.set(f"stop_orders:{user_id}", True, expire=3600)
        await self.business_logger.log_event("new_orders_stopped", user_id=user_id)

    async def _emergency_exit(self, user_id: str, db: Session):
        """紧急退出"""
        # 这里应该实现紧急平仓的逻辑
        await self.business_logger.log_event("emergency_exit_triggered", user_id=user_id)