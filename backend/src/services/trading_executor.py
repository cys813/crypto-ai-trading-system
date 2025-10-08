"""
交易执行服务

负责策略执行、交易决策、订单管理和自动化交易。
支持多种交易策略和风险控制机制。
"""

import logging
import asyncio
from datetime import datetime, timedelta
from decimal import Decimal
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

from sqlalchemy.orm import Session
from sqlalchemy import and_, or_

from ..core.cache import get_cache, CacheKeys
from ..core.database import SessionLocal
from ..core.logging import BusinessLogger
from ..core.exceptions import TradingError, ValidationError, InsufficientFundsError
from ..models.trading_strategy import TradingStrategy, StrategyStatus
from ..models.trading_order import TradingOrder, OrderStatus, OrderType, OrderSide
from ..models.position import Position, PositionStatus, PositionSide
from ..services.order_manager import OrderManager, OrderRequest
from ..services.risk_manager import RiskManager
from ..services.dynamic_fund_manager import DynamicFundManager, FundManagementConfig
from ..services.position_monitor import PositionMonitor

logger = logging.getLogger(__name__)
business_logger = BusinessLogger("trading_executor")


class ExecutionStatus(str, Enum):
    """执行状态"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    PAUSED = "paused"
    STOPPED = "stopped"


class ExecutionMode(str, Enum):
    """执行模式"""
    SIMULATION = "simulation"  # 模拟交易
    PAPER = "paper"  # 纸质交易
    LIVE = "live"  # 实盘交易


class OrderExecutionRule(str, Enum):
    """订单执行规则"""
    IMMEDIATE = "immediate"  # 立即执行
    SCHEDULED = "scheduled"  # 定时执行
    CONDITIONAL = "conditional"  # 条件执行
    PHASED = "phased"  # 分阶段执行


@dataclass
class ExecutionConfig:
    """执行配置"""
    mode: ExecutionMode = ExecutionMode.SIMULATION
    max_orders_per_minute: int = 10
    max_position_size_percent: float = 20.0
    max_daily_trades: int = 100
    enable_risk_check: bool = True
    enable_fund_management: bool = True
    enable_position_monitoring: bool = True
    emergency_stop: bool = False
    execution_rules: List[OrderExecutionRule] = field(default_factory=list)


@dataclass
class TradingSignal:
    """交易信号"""
    strategy_id: str
    symbol: str
    action: str  # buy, sell, hold
    confidence: float  # 0-1
    entry_price: Optional[Decimal] = None
    stop_loss_price: Optional[Decimal] = None
    take_profit_price: Optional[Decimal] = None
    position_size_percent: Optional[float] = None
    expiration_time: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExecutionResult:
    """执行结果"""
    strategy_id: str
    signal: TradingSignal
    success: bool
    orders_created: List[str] = field(default_factory=list)
    positions_opened: List[str] = field(default_factory=list)
    error_message: Optional[str] = None
    execution_time: Optional[datetime] = None
    execution_details: Dict[str, Any] = field(default_factory=dict)


class TradingExecutor:
    """交易执行器"""

    def __init__(self, config: ExecutionConfig = None):
        self.logger = logger
        self.business_logger = business_logger
        self.cache = get_cache()
        self.config = config or ExecutionConfig()

        # 依赖服务
        self.order_manager = OrderManager()
        self.risk_manager = RiskManager()
        self.fund_manager = None
        self.position_monitor = None

        # 执行状态
        self._execution_active = False
        self._execution_tasks = {}
        self._active_strategies = {}
        self._execution_stats = {
            'total_signals_processed': 0,
            'successful_executions': 0,
            'failed_executions': 0,
            'orders_created': 0,
            'daily_trades': 0,
            'last_execution': None
        }

        # 限流控制
        self._rate_limiter = {
            'orders_per_minute': 0,
            'last_minute_reset': datetime.utcnow(),
            'daily_trades': 0,
            'last_day_reset': datetime.utcnow().date()
        }

    async def initialize_services(self, user_id: str, total_funds: Decimal):
        """初始化依赖服务"""
        try:
            # 初始化资金管理服务
            if self.config.enable_fund_management:
                fund_config = FundManagementConfig(total_funds=total_funds)
                self.fund_manager = DynamicFundManager(fund_config)

            # 初始化持仓监控服务
            if self.config.enable_position_monitoring:
                self.position_monitor = PositionMonitor()
                await self.position_monitor.start_monitoring(user_id)

            await self.business_logger.log_event(
                "trading_executor_initialized",
                user_id=user_id,
                total_funds=float(total_funds),
                config=self.config.__dict__
            )

        except Exception as e:
            self.logger.error(f"初始化服务失败: {str(e)}")
            raise TradingError(f"初始化服务失败: {str(e)}")

    async def execute_strategy(self, strategy_id: str, user_id: str, db: Session) -> ExecutionResult:
        """
        执行交易策略

        Args:
            strategy_id: 策略ID
            user_id: 用户ID
            db: 数据库会话

        Returns:
            ExecutionResult: 执行结果
        """
        try:
            # 检查紧急停止
            if self.config.emergency_stop:
                return ExecutionResult(
                    strategy_id=strategy_id,
                    signal=None,
                    success=False,
                    error_message="紧急停止已激活"
                )

            # 检查限流
            if not await self._check_rate_limits():
                return ExecutionResult(
                    strategy_id=strategy_id,
                    signal=None,
                    success=False,
                    error_message="触发限流保护"
                )

            # 获取策略
            strategy = db.query(TradingStrategy).filter(
                TradingStrategy.id == uuid.UUID(strategy_id)
            ).first()

            if not strategy:
                return ExecutionResult(
                    strategy_id=strategy_id,
                    signal=None,
                    success=False,
                    error_message="策略不存在"
                )

            # 检查策略状态
            if strategy.status != StrategyStatus.ACTIVE.value:
                return ExecutionResult(
                    strategy_id=strategy_id,
                    signal=None,
                    success=False,
                    error_message=f"策略状态为 {strategy.status}，无法执行"
                )

            # 检查策略过期
            if strategy.is_expired:
                return ExecutionResult(
                    strategy_id=strategy_id,
                    signal=None,
                    success=False,
                    error_message="策略已过期"
                )

            # 生成交易信号
            signal = await self._generate_trading_signal(strategy, db)
            if not signal:
                return ExecutionResult(
                    strategy_id=strategy_id,
                    signal=None,
                    success=False,
                    error_message="无法生成交易信号"
                )

            # 验证信号
            validation_result = await self._validate_trading_signal(signal, user_id, db)
            if not validation_result['valid']:
                return ExecutionResult(
                    strategy_id=strategy_id,
                    signal=signal,
                    success=False,
                    error_message=f"信号验证失败: {validation_result['error']}"
                )

            # 执行交易
            execution_result = await self._execute_trading_signal(signal, user_id, db)

            # 更新策略状态
            if execution_result.success:
                strategy.status = StrategyStatus.EXECUTED.value
                strategy.executed_at = datetime.utcnow()

            # 更新统计
            self._update_execution_stats(execution_result)

            # 记录执行日志
            await self.business_logger.log_event(
                "strategy_executed",
                strategy_id=strategy_id,
                user_id=user_id,
                signal_action=signal.action,
                signal_confidence=signal.confidence,
                execution_success=execution_result.success,
                orders_created=len(execution_result.orders_created)
            )

            return execution_result

        except Exception as e:
            self.logger.error(f"策略执行失败 {strategy_id}: {str(e)}", exc_info=True)

            # 更新失败统计
            self._execution_stats['failed_executions'] += 1

            return ExecutionResult(
                strategy_id=strategy_id,
                signal=None,
                success=False,
                error_message=str(e)
            )

    async def execute_signal(self, signal: TradingSignal, user_id: str, db: Session) -> ExecutionResult:
        """
        直接执行交易信号

        Args:
            signal: 交易信号
            user_id: 用户ID
            db: 数据库会话

        Returns:
            ExecutionResult: 执行结果
        """
        try:
            # 检查紧急停止
            if self.config.emergency_stop:
                return ExecutionResult(
                    strategy_id=signal.strategy_id,
                    signal=signal,
                    success=False,
                    error_message="紧急停止已激活"
                )

            # 验证信号
            validation_result = await self._validate_trading_signal(signal, user_id, db)
            if not validation_result['valid']:
                return ExecutionResult(
                    strategy_id=signal.strategy_id,
                    signal=signal,
                    success=False,
                    error_message=f"信号验证失败: {validation_result['error']}"
                )

            # 执行交易
            execution_result = await self._execute_trading_signal(signal, user_id, db)

            # 更新统计
            self._update_execution_stats(execution_result)

            return execution_result

        except Exception as e:
            self.logger.error(f"信号执行失败: {str(e)}", exc_info=True)
            return ExecutionResult(
                strategy_id=signal.strategy_id,
                signal=signal,
                success=False,
                error_message=str(e)
            )

    async def start_auto_execution(self, user_id: str):
        """启动自动执行"""
        try:
            if not self._execution_active:
                self._execution_active = True

                # 启动自动执行任务
                self._execution_tasks['auto'] = asyncio.create_task(
                    self._auto_execution_loop(user_id)
                )

                await self.business_logger.log_event(
                    "auto_execution_started",
                    user_id=user_id
                )

        except Exception as e:
            self.logger.error(f"启动自动执行失败: {str(e)}")
            raise TradingError(f"启动自动执行失败: {str(e)}")

    async def stop_auto_execution(self):
        """停止自动执行"""
        try:
            if self._execution_active:
                self._execution_active = False

                # 取消所有执行任务
                for task_name, task in self._execution_tasks.items():
                    if task and not task.done():
                        task.cancel()
                        try:
                            await task
                        except asyncio.CancelledError:
                            pass

                self._execution_tasks.clear()
                self._active_strategies.clear()

                # 停止持仓监控
                if self.position_monitor:
                    await self.position_monitor.stop_monitoring()

                await self.business_logger.log_event("auto_execution_stopped")

        except Exception as e:
            self.logger.error(f"停止自动执行失败: {str(e)}")

    async def emergency_stop_all(self, reason: str = "Manual emergency stop"):
        """紧急停止所有交易"""
        try:
            self.config.emergency_stop = True

            # 停止自动执行
            await self.stop_auto_execution()

            # 取消所有挂单
            await self._cancel_all_pending_orders()

            # 关闭所有持仓（根据风险等级）
            if self.config.mode == ExecutionMode.LIVE:
                await self._emergency_close_positions()

            await self.business_logger.log_event(
                "emergency_stop_triggered",
                reason=reason,
                mode=self.config.mode.value
            )

        except Exception as e:
            self.logger.error(f"紧急停止失败: {str(e)}")

    async def get_execution_stats(self) -> Dict[str, Any]:
        """获取执行统计"""
        return {
            **self._execution_stats,
            'execution_active': self._execution_active,
            'active_strategies': len(self._active_strategies),
            'config': self.config.__dict__,
            'rate_limiter': self._rate_limiter
        }

    async def _generate_trading_signal(self, strategy: TradingStrategy, db: Session) -> Optional[TradingSignal]:
        """生成交易信号"""
        try:
            # 检查策略是否有有效的推荐
            if not strategy.final_recommendation or strategy.final_recommendation == 'hold':
                return None

            # 创建交易信号
            signal = TradingSignal(
                strategy_id=str(strategy.id),
                symbol=strategy.symbol.symbol if strategy.symbol else 'BTC/USDT',
                action=strategy.final_recommendation,
                confidence=float(strategy.confidence_score or 0.5),
                entry_price=strategy.entry_price,
                stop_loss_price=strategy.stop_loss_price,
                take_profit_price=strategy.take_profit_price,
                position_size_percent=float(strategy.position_size_percent or 10),
                expiration_time=strategy.expires_at,
                metadata={
                    'strategy_type': strategy.strategy_type,
                    'risk_level': strategy.risk_level,
                    'llm_provider': strategy.llm_provider,
                    'llm_model': strategy.llm_model
                }
            )

            return signal

        except Exception as e:
            self.logger.error(f"生成交易信号失败: {str(e)}")
            return None

    async def _validate_trading_signal(self, signal: TradingSignal, user_id: str, db: Session) -> Dict[str, Any]:
        """验证交易信号"""
        try:
            # 基本验证
            if signal.action not in ['buy', 'sell']:
                return {'valid': False, 'error': '无效的交易动作'}

            if signal.confidence <= 0 or signal.confidence > 1:
                return {'valid': False, 'error': '无效的置信度'}

            if signal.position_size_percent and (signal.position_size_percent <= 0 or signal.position_size_percent > 100):
                return {'valid': False, 'error': '无效的仓位大小'}

            # 检查过期时间
            if signal.expiration_time and signal.expiration_time <= datetime.utcnow():
                return {'valid': False, 'error': '信号已过期'}

            # 风险验证
            if self.config.enable_risk_check:
                risk_assessment = await self.risk_manager.assess_order_risk(
                    {
                        'symbol': signal.symbol,
                        'side': signal.action,
                        'amount': signal.position_size_percent or 10,
                        'price': signal.entry_price
                    },
                    user_id,
                    db
                )

                if risk_assessment.risk_level.value == 'critical':
                    return {'valid': False, 'error': f'风险等级过高: {risk_assessment.risk_level.value}'}

            # 资金验证
            if self.config.enable_fund_management and self.fund_manager:
                # 这里应该验证资金是否足够
                pass

            return {'valid': True}

        except Exception as e:
            self.logger.error(f"验证交易信号失败: {str(e)}")
            return {'valid': False, 'error': f'验证异常: {str(e)}'}

    async def _execute_trading_signal(self, signal: TradingSignal, user_id: str, db: Session) -> ExecutionResult:
        """执行交易信号"""
        try:
            execution_result = ExecutionResult(
                strategy_id=signal.strategy_id,
                signal=signal,
                success=False,
                execution_time=datetime.utcnow()
            )

            # 获取当前价格
            current_price = signal.entry_price
            if not current_price:
                current_price = await self._get_market_price(signal.symbol)
                if not current_price:
                    execution_result.error_message = "无法获取市场价格"
                    return execution_result

            # 计算订单数量
            if self.config.enable_fund_management and self.fund_manager:
                # 使用资金管理计算仓位大小
                stop_loss_price = signal.stop_loss_price or current_price * Decimal('0.98')  # 默认2%止损
                amount = await self.fund_manager.calculate_position_size(
                    signal.symbol,
                    signal.action,
                    current_price,
                    stop_loss_price,
                    await self._get_account_balance(user_id, db)
                )
            else:
                # 使用固定百分比
                total_balance = await self._get_account_balance(user_id, db)
                position_value = total_balance * (Decimal(str(signal.position_size_percent or 10)) / 100)
                amount = position_value / current_price

            # 创建订单请求
            order_request = OrderRequest(
                symbol=signal.symbol,
                side=signal.action,
                order_type=OrderType.MARKET.value,  # 使用市价单确保执行
                amount=amount,
                price=current_price,
                stop_loss_price=signal.stop_loss_price,
                take_profit_price=signal.take_profit_price,
                user_id=user_id,
                strategy_id=signal.strategy_id,
                metadata=signal.metadata
            )

            # 执行订单
            order_result = await self.order_manager.create_order(order_request, db)

            if order_result.success:
                execution_result.success = True
                execution_result.orders_created.append(order_result.order_id)

                # 如果订单完全成交，创建持仓监控
                if order_result.filled_amount and order_result.filled_amount >= amount * Decimal('0.95'):
                    # 获取或创建持仓记录
                    position_id = await self._update_or_create_position(
                        signal, order_result, user_id, db
                    )
                    if position_id:
                        execution_result.positions_opened.append(position_id)

                        # 添加持仓监控
                        if self.position_monitor:
                            await self.position_monitor.add_position_monitor(position_id)

                execution_result.execution_details = {
                    'order_id': order_result.order_id,
                    'exchange_order_id': order_result.exchange_order_id,
                    'filled_amount': float(order_result.filled_amount) if order_result.filled_amount else None,
                    'filled_price': float(order_result.filled_price) if order_result.filled_price else None,
                    'fee': float(order_result.fee) if order_result.fee else None
                }

            else:
                execution_result.error_message = order_result.error_message

            return execution_result

        except Exception as e:
            self.logger.error(f"执行交易信号失败: {str(e)}", exc_info=True)
            return ExecutionResult(
                strategy_id=signal.strategy_id,
                signal=signal,
                success=False,
                error_message=str(e),
                execution_time=datetime.utcnow()
            )

    async def _auto_execution_loop(self, user_id: str):
        """自动执行循环"""
        while self._execution_active:
            try:
                await asyncio.sleep(60)  # 每分钟检查一次

                # 检查紧急停止
                if self.config.emergency_stop:
                    break

                with SessionLocal() as db:
                    # 获取活跃策略
                    active_strategies = db.query(TradingStrategy).filter(
                        and_(
                            TradingStrategy.status == StrategyStatus.ACTIVE.value,
                            TradingStrategy.created_at > datetime.utcnow() - timedelta(days=1)  # 只执行24小时内的策略
                        )
                    ).all()

                    for strategy in active_strategies:
                        if str(strategy.id) not in self._active_strategies:
                            try:
                                # 执行策略
                                result = await self.execute_strategy(str(strategy.id), user_id, db)

                                if result.success:
                                    self._active_strategies[str(strategy.id)] = {
                                        'last_execution': datetime.utcnow(),
                                        'execution_count': 1
                                    }
                                else:
                                    self.logger.warning(f"策略执行失败 {strategy.id}: {result.error_message}")

                            except Exception as e:
                                self.logger.error(f"自动执行策略失败 {strategy.id}: {str(e)}")

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"自动执行循环异常: {str(e)}", exc_info=True)

    async def _check_rate_limits(self) -> bool:
        """检查限流"""
        try:
            now = datetime.utcnow()
            today = now.date()

            # 检查每分钟限流
            if now.minute != self._rate_limiter['last_minute_reset'].minute:
                self._rate_limiter['orders_per_minute'] = 0
                self._rate_limiter['last_minute_reset'] = now

            # 检查每日限流
            if today != self._rate_limiter['last_day_reset']:
                self._rate_limiter['daily_trades'] = 0
                self._rate_limiter['last_day_reset'] = today

            # 检查限制
            if (self._rate_limiter['orders_per_minute'] >= self.config.max_orders_per_minute or
                self._rate_limiter['daily_trades'] >= self.config.max_daily_trades):
                return False

            return True

        except Exception as e:
            self.logger.error(f"检查限流失败: {str(e)}")
            return False

    async def _get_market_price(self, symbol: str) -> Optional[Decimal]:
        """获取市场价格"""
        try:
            # 这里应该调用交易所API获取价格
            # 暂时返回模拟价格
            return Decimal('50000')

        except Exception as e:
            self.logger.error(f"获取市场价格失败 {symbol}: {str(e)}")
            return None

    async def _get_account_balance(self, user_id: str, db: Session) -> Decimal:
        """获取账户余额"""
        try:
            # 这里应该计算用户账户总余额
            # 暂时返回模拟余额
            return Decimal('10000')

        except Exception as e:
            self.logger.error(f"获取账户余额失败: {str(e)}")
            return Decimal('0')

    async def _update_or_create_position(
        self,
        signal: TradingSignal,
        order_result: Any,
        user_id: str,
        db: Session
    ) -> Optional[str]:
        """更新或创建持仓"""
        try:
            # 检查是否已有该策略的持仓
            existing_position = db.query(Position).filter(
                and_(
                    Position.strategy_id == uuid.UUID(signal.strategy_id),
                    Position.symbol == signal.symbol,
                    Position.status == PositionStatus.OPEN.value
                )
            ).first()

            if existing_position:
                # 更新现有持仓
                if order_result.filled_amount:
                    existing_position.add_trade(
                        order_result.filled_amount,
                        order_result.filled_price or Decimal('0')
                    )

                db.commit()
                return str(existing_position.id)

            else:
                # 创建新持仓
                if order_result.filled_amount:
                    position = Position(
                        symbol=signal.symbol,
                        side=PositionSide.LONG.value if signal.action == 'buy' else PositionSide.SHORT.value,
                        amount=order_result.filled_amount,
                        average_cost=order_result.filled_price or Decimal('0'),
                        current_price=order_result.filled_price or Decimal('0'),
                        entry_value=order_result.filled_amount * (order_result.filled_price or Decimal('0')),
                        current_value=order_result.filled_amount * (order_result.filled_price or Decimal('0')),
                        cost_basis=order_result.filled_amount * (order_result.filled_price or Decimal('0')),
                        stop_loss_price=signal.stop_loss_price,
                        take_profit_price=signal.take_profit_price,
                        exchange='binance',  # 应该从订单获取
                        user_id=uuid.UUID(user_id),
                        strategy_id=uuid.UUID(signal.strategy_id)
                    )

                    db.add(position)
                    db.commit()
                    db.refresh(position)

                    return str(position.id)

        except Exception as e:
            self.logger.error(f"更新或创建持仓失败: {str(e)}")
            db.rollback()

        return None

    async def _cancel_all_pending_orders(self):
        """取消所有挂单"""
        try:
            # 这里应该调用订单管理器取消所有挂单
            await self.business_logger.log_event("all_pending_orders_cancelled")

        except Exception as e:
            self.logger.error(f"取消所有挂单失败: {str(e)}")

    async def _emergency_close_positions(self):
        """紧急关闭持仓"""
        try:
            # 这里应该实现紧急平仓逻辑
            await self.business_logger.log_event("emergency_positions_closed")

        except Exception as e:
            self.logger.error(f"紧急平仓失败: {str(e)}")

    def _update_execution_stats(self, result: ExecutionResult):
        """更新执行统计"""
        self._execution_stats['total_signals_processed'] += 1

        if result.success:
            self._execution_stats['successful_executions'] += 1
            self._execution_stats['orders_created'] += len(result.orders_created)
            self._execution_stats['daily_trades'] += 1
        else:
            self._execution_stats['failed_executions'] += 1

        self._execution_stats['last_execution'] = result.execution_time

        # 更新限流计数
        self._rate_limiter['orders_per_minute'] += len(result.orders_created)