"""
持仓监控服务

负责实时监控持仓状态、盈亏变化、风险指标和自动止损止盈。
支持多种监控策略和自动化风险管理。
"""

import logging
import asyncio
from datetime import datetime, timedelta
from decimal import Decimal
from typing import List, Dict, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum

from sqlalchemy.orm import Session
from sqlalchemy import and_, or_

from ..core.cache import get_cache, CacheKeys
from ..core.database import SessionLocal
from ..core.logging import BusinessLogger
from ..core.exceptions import PositionError, ValidationError
from ..models.position import Position, PositionStatus, PositionSide
from ..models.trading_order import TradingOrder, OrderType, OrderSide
from ..services.order_manager import OrderManager
from ..services.risk_manager import RiskManager

logger = logging.getLogger(__name__)
business_logger = BusinessLogger("position_monitor")


class MonitorAction(str, Enum):
    """监控措施"""
    NONE = "none"
    UPDATE_STOP_LOSS = "update_stop_loss"
    UPDATE_TAKE_PROFIT = "update_take_profit"
    CLOSE_POSITION = "close_position"
    PARTIAL_CLOSE = "partial_close"
    SEND_ALERT = "send_alert"
    ADJUST_POSITION = "adjust_position"


class MonitorTrigger(str, Enum):
    """监控触发条件"""
    PRICE_LEVEL = "price_level"
    TIME_BASED = "time_based"
    RISK_LEVEL = "risk_level"
    VOLATILITY = "volatility"
    CORRELATION = "correlation"
    LIQUIDITY = "liquidity"


@dataclass
class MonitorConfig:
    """监控配置"""
    update_interval: int = 30  # 更新间隔（秒）
    price_check_interval: int = 5  # 价格检查间隔（秒）
    risk_check_interval: int = 60  # 风险检查间隔（秒）
    enable_auto_stop_loss: bool = True
    enable_auto_take_profit: bool = True
    enable_trailing_stop: bool = True
    enable_risk_monitoring: bool = True
    max_monitoring_positions: int = 100  # 最大监控持仓数量
    alert_threshold_percent: float = 5.0  # 警报阈值百分比
    force_close_threshold: float = 20.0  # 强制平仓阈值百分比


@dataclass
class PositionAlert:
    """持仓警报"""
    position_id: str
    symbol: str
    alert_type: str
    message: str
    severity: str  # low, medium, high, critical
    current_price: Decimal
    current_pnl_percent: float
    triggered_at: datetime
    suggested_action: Optional[MonitorAction] = None
    action_parameters: Optional[Dict[str, Any]] = None


@dataclass
class MonitorResult:
    """监控结果"""
    position_id: str
    symbol: str
    current_price: Decimal
    current_pnl: Decimal
    current_pnl_percent: float
    unrealized_pnl: Decimal
    risk_score: float
    alerts: List[PositionAlert] = field(default_factory=list)
    actions_taken: List[Tuple[MonitorAction, Dict[str, Any]]] = field(default_factory=list)
    last_updated: datetime = field(default_factory=datetime.utcnow)


class PositionMonitor:
    """持仓监控器"""

    def __init__(self, config: MonitorConfig = None):
        self.logger = logger
        self.business_logger = business_logger
        self.cache = get_cache()
        self.config = config or MonitorConfig()

        # 依赖服务
        self.order_manager = OrderManager()
        self.risk_manager = RiskManager()

        # 监控状态
        self._monitoring_active = False
        self._monitoring_tasks = {}
        self._position_monitors = {}

        # 监控统计
        self._monitoring_stats = {
            'total_positions_monitored': 0,
            'alerts_generated': 0,
            'actions_executed': 0,
            'last_update': None
        }

    async def start_monitoring(self, user_id: str = None):
        """启动持仓监控"""
        if not self._monitoring_active:
            self._monitoring_active = True

            # 启动主监控任务
            self._monitoring_tasks['main'] = asyncio.create_task(
                self._monitor_positions_loop(user_id)
            )

            # 启动价格检查任务
            self._monitoring_tasks['price'] = asyncio.create_task(
                self._price_monitor_loop(user_id)
            )

            # 启动风险检查任务
            if self.config.enable_risk_monitoring:
                self._monitoring_tasks['risk'] = asyncio.create_task(
                    self._risk_monitor_loop(user_id)
                )

            await self.business_logger.log_event(
                "position_monitoring_started",
                user_id=user_id,
                config=self.config.__dict__
            )

    async def stop_monitoring(self):
        """停止持仓监控"""
        if self._monitoring_active:
            self._monitoring_active = False

            # 取消所有监控任务
            for task_name, task in self._monitoring_tasks.items():
                if task and not task.done():
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass

            self._monitoring_tasks.clear()
            self._position_monitors.clear()

            await self.business_logger.log_event("position_monitoring_stopped")

    async def add_position_monitor(self, position_id: str, custom_config: Dict[str, Any] = None):
        """添加持仓监控"""
        try:
            if position_id in self._position_monitors:
                self.logger.warning(f"持仓 {position_id} 已在监控中")
                return

            # 创建监控配置
            monitor_config = {
                'position_id': position_id,
                'added_at': datetime.utcnow(),
                'custom_config': custom_config or {},
                'last_price_check': None,
                'last_risk_check': None,
                'alert_count': 0,
                'action_count': 0
            }

            self._position_monitors[position_id] = monitor_config

            await self.business_logger.log_event(
                "position_monitor_added",
                position_id=position_id
            )

        except Exception as e:
            self.logger.error(f"添加持仓监控失败 {position_id}: {str(e)}")
            raise PositionError(f"添加持仓监控失败: {str(e)}")

    async def remove_position_monitor(self, position_id: str):
        """移除持仓监控"""
        try:
            if position_id in self._position_monitors:
                del self._position_monitors[position_id]

                await self.business_logger.log_event(
                    "position_monitor_removed",
                    position_id=position_id
                )

        except Exception as e:
            self.logger.error(f"移除持仓监控失败 {position_id}: {str(e)}")

    async def monitor_single_position(self, position_id: str, db: Session = None) -> MonitorResult:
        """
        监控单个持仓

        Args:
            position_id: 持仓ID
            db: 数据库会话

        Returns:
            MonitorResult: 监控结果
        """
        try:
            if db is None:
                db = SessionLocal()

            # 获取持仓信息
            position = db.query(Position).filter(
                Position.id == uuid.UUID(position_id)
            ).first()

            if not position:
                raise PositionError(f"持仓不存在: {position_id}")

            if position.status != PositionStatus.OPEN.value:
                return MonitorResult(
                    position_id=position_id,
                    symbol=position.symbol,
                    current_price=position.current_price or Decimal('0'),
                    current_pnl=position.total_pnl,
                    current_pnl_percent=position.pnl_percent,
                    unrealized_pnl=position.unrealized_pnl,
                    risk_score=0.0,
                    alerts=[PositionAlert(
                        position_id=position_id,
                        symbol=position.symbol,
                        alert_type="position_closed",
                        message=f"持仓状态为 {position.status}，无需监控",
                        severity="low",
                        current_price=position.current_price or Decimal('0'),
                        current_pnl_percent=position.pnl_percent,
                        triggered_at=datetime.utcnow()
                    )]
                )

            # 获取当前市场价格
            current_price = await self._get_current_price(position.symbol, position.exchange)
            if current_price:
                position.update_market_price(current_price)

            # 生成监控结果
            result = await self._generate_monitor_result(position, db)

            # 执行监控措施
            await self._execute_monitor_actions(position, result, db)

            # 更新监控统计
            self._update_monitoring_stats(result)

            return result

        except Exception as e:
            self.logger.error(f"持仓监控失败 {position_id}: {str(e)}", exc_info=True)
            raise PositionError(f"持仓监控失败: {str(e)}")

    async def get_monitoring_stats(self) -> Dict[str, Any]:
        """获取监控统计信息"""
        return {
            **self._monitoring_stats,
            'active_monitors': len(self._position_monitors),
            'monitoring_active': self._monitoring_active,
            'config': self.config.__dict__
        }

    async def _monitor_positions_loop(self, user_id: str = None):
        """主持仓监控循环"""
        while self._monitoring_active:
            try:
                await asyncio.sleep(self.config.update_interval)

                with SessionLocal() as db:
                    # 获取需要监控的持仓
                    positions = await self._get_active_positions(user_id, db)

                    # 限制监控数量
                    if len(positions) > self.config.max_monitoring_positions:
                        positions = positions[:self.config.max_monitoring_positions]

                    # 并发监控持仓
                    tasks = []
                    for position in positions:
                        if str(position.id) in self._position_monitors:
                            task = asyncio.create_task(
                                self.monitor_single_position(str(position.id), db)
                            )
                            tasks.append(task)

                    if tasks:
                        results = await asyncio.gather(*tasks, return_exceptions=True)

                        # 处理监控结果
                        for result in results:
                            if isinstance(result, Exception):
                                self.logger.error(f"监控任务异常: {str(result)}")

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"持仓监控循环异常: {str(e)}", exc_info=True)

    async def _price_monitor_loop(self, user_id: str = None):
        """价格监控循环"""
        while self._monitoring_active:
            try:
                await asyncio.sleep(self.config.price_check_interval)

                with SessionLocal() as db:
                    # 获取监控中的持仓
                    position_ids = list(self._position_monitors.keys())
                    if not position_ids:
                        continue

                    positions = db.query(Position).filter(
                        and_(
                            Position.id.in_(position_ids),
                            Position.status == PositionStatus.OPEN.value
                        )
                    ).all()

                    # 更新价格并检查价格触发条件
                    for position in positions:
                        await self._check_price_triggers(position, db)

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"价格监控循环异常: {str(e)}", exc_info=True)

    async def _risk_monitor_loop(self, user_id: str = None):
        """风险监控循环"""
        while self._monitoring_active:
            try:
                await asyncio.sleep(self.config.risk_check_interval)

                with SessionLocal() as db:
                    # 获取监控中的持仓
                    position_ids = list(self._position_monitors.keys())
                    if not position_ids:
                        continue

                    positions = db.query(Position).filter(
                        and_(
                            Position.id.in_(position_ids),
                            Position.status == PositionStatus.OPEN.value
                        )
                    ).all()

                    # 检查风险触发条件
                    for position in positions:
                        await self._check_risk_triggers(position, db)

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"风险监控循环异常: {str(e)}", exc_info=True)

    async def _get_active_positions(self, user_id: str = None, db: Session = None) -> List[Position]:
        """获取活跃持仓"""
        query = db.query(Position).filter(
            Position.status == PositionStatus.OPEN.value
        )

        if user_id:
            query = query.filter(Position.user_id == uuid.UUID(user_id))

        return query.all()

    async def _get_current_price(self, symbol: str, exchange: str) -> Optional[Decimal]:
        """获取当前市场价格"""
        try:
            # 这里应该调用交易所API获取实时价格
            # 暂时使用缓存数据
            cache_key = f"ticker:{exchange}:{symbol}"
            cached_price = await self.cache.get(cache_key)

            if cached_price:
                return Decimal(str(cached_price.get('price', 0)))

            # 模拟价格获取
            return Decimal('50000')

        except Exception as e:
            self.logger.error(f"获取价格失败 {symbol}: {str(e)}")
            return None

    async def _generate_monitor_result(self, position: Position, db: Session) -> MonitorResult:
        """生成监控结果"""
        alerts = []
        actions_taken = []

        # 评估风险
        risk_assessment = await self.risk_manager.assess_position_risk(position, db)

        # 生成警报
        alerts.extend(await self._generate_position_alerts(position, risk_assessment))

        # 创建监控结果
        result = MonitorResult(
            position_id=str(position.id),
            symbol=position.symbol,
            current_price=position.current_price or Decimal('0'),
            current_pnl=position.total_pnl,
            current_pnl_percent=position.pnl_percent,
            unrealized_pnl=position.unrealized_pnl,
            risk_score=risk_assessment.risk_score,
            alerts=alerts,
            actions_taken=actions_taken
        )

        return result

    async def _generate_position_alerts(
        self,
        position: Position,
        risk_assessment
    ) -> List[PositionAlert]:
        """生成持仓警报"""
        alerts = []

        try:
            # 亏损警报
            if position.pnl_percent < -self.config.alert_threshold_percent:
                alerts.append(PositionAlert(
                    position_id=str(position.id),
                    symbol=position.symbol,
                    alert_type="loss_warning",
                    message=f"持仓亏损 {position.pnl_percent:.2f}%",
                    severity="medium" if position.pnl_percent > -10 else "high",
                    current_price=position.current_price or Decimal('0'),
                    current_pnl_percent=position.pnl_percent,
                    triggered_at=datetime.utcnow(),
                    suggested_action=MonitorAction.SEND_ALERT
                ))

            # 严重亏损警报
            if position.pnl_percent < -self.config.force_close_threshold:
                alerts.append(PositionAlert(
                    position_id=str(position.id),
                    symbol=position.symbol,
                    alert_type="critical_loss",
                    message=f"严重亏损 {position.pnl_percent:.2f}%，建议强制平仓",
                    severity="critical",
                    current_price=position.current_price or Decimal('0'),
                    current_pnl_percent=position.pnl_percent,
                    triggered_at=datetime.utcnow(),
                    suggested_action=MonitorAction.CLOSE_POSITION
                ))

            # 利润警报
            if position.pnl_percent > self.config.alert_threshold_percent:
                alerts.append(PositionAlert(
                    position_id=str(position.id),
                    symbol=position.symbol,
                    alert_type="profit_warning",
                    message=f"持仓盈利 {position.pnl_percent:.2f}%",
                    severity="low",
                    current_price=position.current_price or Decimal('0'),
                    current_pnl_percent=position.pnl_percent,
                    triggered_at=datetime.utcnow()
                ))

            # 止损价格接近警报
            if position.stop_loss_price and position.current_price:
                if position.side == PositionSide.LONG.value:
                    distance_to_stop = float((position.current_price - position.stop_loss_price) / position.current_price * 100)
                else:
                    distance_to_stop = float((position.stop_loss_price - position.current_price) / position.current_price * 100)

                if distance_to_stop < 2:
                    alerts.append(PositionAlert(
                        position_id=str(position.id),
                        symbol=position.symbol,
                        alert_type="stop_loss_near",
                        message=f"接近止损价格，距离 {distance_to_stop:.2f}%",
                        severity="high",
                        current_price=position.current_price,
                        current_pnl_percent=position.pnl_percent,
                        triggered_at=datetime.utcnow(),
                        suggested_action=MonitorAction.UPDATE_STOP_LOSS if self.config.enable_trailing_stop else None
                    ))

            # 止盈价格接近警报
            if position.take_profit_price and position.current_price:
                if position.side == PositionSide.LONG.value:
                    distance_to_target = float((position.take_profit_price - position.current_price) / position.current_price * 100)
                else:
                    distance_to_target = float((position.current_price - position.take_profit_price) / position.current_price * 100)

                if distance_to_target < 2:
                    alerts.append(PositionAlert(
                        position_id=str(position.id),
                        symbol=position.symbol,
                        alert_type="take_profit_near",
                        message=f"接近止盈价格，距离 {distance_to_target:.2f}%",
                        severity="medium",
                        current_price=position.current_price,
                        current_pnl_percent=position.pnl_percent,
                        triggered_at=datetime.utcnow()
                    ))

            # 风险等级警报
            if risk_assessment.risk_level.value in ['high', 'critical']:
                alerts.append(PositionAlert(
                    position_id=str(position.id),
                    symbol=position.symbol,
                    alert_type="risk_warning",
                    message=f"风险等级: {risk_assessment.risk_level.value}，评分: {risk_assessment.risk_score}",
                    severity=risk_assessment.risk_level.value,
                    current_price=position.current_price or Decimal('0'),
                    current_pnl_percent=position.pnl_percent,
                    triggered_at=datetime.utcnow(),
                    suggested_action=MonitorAction.SEND_ALERT
                ))

            # 持仓时间警报
            if position.duration:
                duration_hours = position.duration.total_seconds() / 3600
                if duration_hours > 168:  # 超过7天
                    alerts.append(PositionAlert(
                        position_id=str(position.id),
                        symbol=position.symbol,
                        alert_type="long_duration",
                        message=f"持仓时间过长: {duration_hours:.0f}小时",
                        severity="medium",
                        current_price=position.current_price or Decimal('0'),
                        current_pnl_percent=position.pnl_percent,
                        triggered_at=datetime.utcnow()
                    ))

        except Exception as e:
            self.logger.error(f"生成警报失败: {str(e)}")
            alerts.append(PositionAlert(
                position_id=str(position.id),
                symbol=position.symbol,
                alert_type="monitor_error",
                message=f"监控异常: {str(e)}",
                severity="low",
                current_price=position.current_price or Decimal('0'),
                current_pnl_percent=position.pnl_percent,
                triggered_at=datetime.utcnow()
            ))

        return alerts

    async def _execute_monitor_actions(
        self,
        position: Position,
        result: MonitorResult,
        db: Session
    ):
        """执行监控措施"""
        try:
            for alert in result.alerts:
                if not alert.suggested_action:
                    continue

                action_params = {}

                if alert.suggested_action == MonitorAction.UPDATE_STOP_LOSS:
                    if self.config.enable_auto_stop_loss and self.config.enable_trailing_stop:
                        updated = await self.risk_manager.update_stop_loss(
                            position, result.current_price, db
                        )
                        if updated:
                            result.actions_taken.append((
                                MonitorAction.UPDATE_STOP_LOSS,
                                {'new_stop_loss': float(position.stop_loss_price)}
                            ))

                elif alert.suggested_action == MonitorAction.CLOSE_POSITION:
                    if alert.severity == 'critical':
                        # 严重亏损时强制平仓
                        success = await self._force_close_position(position, db)
                        if success:
                            result.actions_taken.append((
                                MonitorAction.CLOSE_POSITION,
                                {'reason': 'critical_loss', 'pnl_percent': position.pnl_percent}
                            ))

                elif alert.suggested_action == MonitorAction.SEND_ALERT:
                    # 发送警报通知
                    await self._send_position_alert(alert)
                    result.actions_taken.append((
                        MonitorAction.SEND_ALERT,
                        {'alert_type': alert.alert_type, 'severity': alert.severity}
                    ))

            # 更新监控统计
            if str(position.id) in self._position_monitors:
                self._position_monitors[str(position.id)]['alert_count'] += len(result.alerts)
                self._position_monitors[str(position.id)]['action_count'] += len(result.actions_taken)

        except Exception as e:
            self.logger.error(f"执行监控措施失败: {str(e)}")

    async def _check_price_triggers(self, position: Position, db: Session):
        """检查价格触发条件"""
        try:
            # 更新市场价格
            current_price = await self._get_current_price(position.symbol, position.exchange)
            if current_price:
                position.update_market_price(current_price)

                # 检查止损
                if self.config.enable_auto_stop_loss and position.stop_loss_price:
                    if await self._check_stop_loss_trigger(position, current_price):
                        await self._execute_stop_loss(position, current_price, db)

                # 检查止盈
                if self.config.enable_auto_take_profit and position.take_profit_price:
                    if await self._check_take_profit_trigger(position, current_price):
                        await self._execute_take_profit(position, current_price, db)

                # 更新移动止损
                if self.config.enable_trailing_stop and position.trailing_stop_amount:
                    await self.risk_manager.update_stop_loss(position, current_price, db)

                db.commit()

        except Exception as e:
            self.logger.error(f"检查价格触发条件失败 {position.symbol}: {str(e)}")

    async def _check_risk_triggers(self, position: Position, db: Session):
        """检查风险触发条件"""
        try:
            # 评估风险
            risk_assessment = await self.risk_manager.assess_position_risk(position, db)

            # 根据风险等级执行措施
            if risk_assessment.risk_level.value == 'critical':
                # 严重风险：考虑强制平仓
                if position.pnl_percent < -self.config.force_close_threshold:
                    await self._force_close_position(position, db)

        except Exception as e:
            self.logger.error(f"检查风险触发条件失败 {position.symbol}: {str(e)}")

    async def _check_stop_loss_trigger(self, position: Position, current_price: Decimal) -> bool:
        """检查止损触发"""
        if not position.stop_loss_price:
            return False

        if position.side == PositionSide.LONG.value:
            return current_price <= position.stop_loss_price
        else:
            return current_price >= position.stop_loss_price

    async def _check_take_profit_trigger(self, position: Position, current_price: Decimal) -> bool:
        """检查止盈触发"""
        if not position.take_profit_price:
            return False

        if position.side == PositionSide.LONG.value:
            return current_price >= position.take_profit_price
        else:
            return current_price <= position.take_profit_price

    async def _execute_stop_loss(self, position: Position, current_price: Decimal, db: Session):
        """执行止损"""
        try:
            # 创建市价平仓订单
            order_side = OrderSide.SELL.value if position.side == PositionSide.LONG.value else OrderSide.BUY.value

            order_request = {
                'symbol': position.symbol,
                'side': order_side,
                'order_type': OrderType.MARKET.value,
                'amount': position.amount,
                'exchange': position.exchange,
                'user_id': str(position.user_id),
                'strategy_id': str(position.strategy_id),
                'metadata': {'trigger_type': 'stop_loss', 'trigger_price': float(current_price)}
            }

            # 通过订单管理器执行
            order_result = await self.order_manager.create_order(order_request, db)

            if order_result.success:
                position.status = PositionStatus.CLOSED.value
                position.closed_at = datetime.utcnow()

                await self.business_logger.log_event(
                    "stop_loss_executed",
                    position_id=str(position.id),
                    symbol=position.symbol,
                    trigger_price=float(current_price),
                    order_id=order_result.order_id
                )

        except Exception as e:
            self.logger.error(f"执行止损失败: {str(e)}")

    async def _execute_take_profit(self, position: Position, current_price: Decimal, db: Session):
        """执行止盈"""
        try:
            # 创建市价平仓订单
            order_side = OrderSide.SELL.value if position.side == PositionSide.LONG.value else OrderSide.BUY.value

            order_request = {
                'symbol': position.symbol,
                'side': order_side,
                'order_type': OrderType.MARKET.value,
                'amount': position.amount,
                'exchange': position.exchange,
                'user_id': str(position.user_id),
                'strategy_id': str(position.strategy_id),
                'metadata': {'trigger_type': 'take_profit', 'trigger_price': float(current_price)}
            }

            # 通过订单管理器执行
            order_result = await self.order_manager.create_order(order_request, db)

            if order_result.success:
                position.status = PositionStatus.CLOSED.value
                position.closed_at = datetime.utcnow()

                await self.business_logger.log_event(
                    "take_profit_executed",
                    position_id=str(position.id),
                    symbol=position.symbol,
                    trigger_price=float(current_price),
                    order_id=order_result.order_id
                )

        except Exception as e:
            self.logger.error(f"执行止盈失败: {str(e)}")

    async def _force_close_position(self, position: Position, db: Session) -> bool:
        """强制平仓"""
        try:
            # 创建市价平仓订单
            order_side = OrderSide.SELL.value if position.side == PositionSide.LONG.value else OrderSide.BUY.value

            order_request = {
                'symbol': position.symbol,
                'side': order_side,
                'order_type': OrderType.MARKET.value,
                'amount': position.amount,
                'exchange': position.exchange,
                'user_id': str(position.user_id),
                'strategy_id': str(position.strategy_id),
                'metadata': {'trigger_type': 'force_close', 'reason': 'critical_loss'}
            }

            # 通过订单管理器执行
            order_result = await self.order_manager.create_order(order_request, db)

            if order_result.success:
                position.status = PositionStatus.CLOSED.value
                position.closed_at = datetime.utcnow()

                await self.business_logger.log_event(
                    "position_force_closed",
                    position_id=str(position.id),
                    symbol=position.symbol,
                    pnl_percent=position.pnl_percent,
                    order_id=order_result.order_id
                )

                return True

        except Exception as e:
            self.logger.error(f"强制平仓失败: {str(e)}")

        return False

    async def _send_position_alert(self, alert: PositionAlert):
        """发送持仓警报"""
        try:
            # 这里应该实现通知逻辑（邮件、短信、推送等）
            await self.business_logger.log_event(
                "position_alert_sent",
                position_id=alert.position_id,
                symbol=alert.symbol,
                alert_type=alert.alert_type,
                severity=alert.severity,
                message=alert.message,
                current_price=float(alert.current_price),
                pnl_percent=alert.current_pnl_percent
            )

        except Exception as e:
            self.logger.error(f"发送警报失败: {str(e)}")

    def _update_monitoring_stats(self, result: MonitorResult):
        """更新监控统计"""
        self._monitoring_stats['total_positions_monitored'] += 1
        self._monitoring_stats['alerts_generated'] += len(result.alerts)
        self._monitoring_stats['actions_executed'] += len(result.actions_taken)
        self._monitoring_stats['last_update'] = datetime.utcnow()

        # 更新缓存
        asyncio.create_task(self.cache.set(
            f"position_monitor_stats",
            self._monitoring_stats,
            expire=300
        ))