"""
订单管理服务

负责处理交易订单的创建、执行、监控和管理。
支持多种订单类型、风险控制和批量操作。
"""

import logging
import asyncio
import uuid
from datetime import datetime, timedelta
from decimal import Decimal
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass, field
from enum import Enum

from sqlalchemy.orm import Session
from sqlalchemy import and_, or_

from ..core.exchange_integration import get_exchange_manager, ExchangeCredentials
from ..core.cache import get_cache, CacheKeys
from ..core.database import SessionLocal
from ..core.logging import BusinessLogger
from ..core.exceptions import ExchangeAPIError, ValidationError, InsufficientFundsError, OrderError
from ..models.trading_order import (
    TradingOrder, OrderFill, OrderSide, OrderType, OrderStatus, TimeInForce
)
from ..models.position import Position, PositionSide, PositionStatus
from ..models.trading_strategy import TradingStrategy

logger = logging.getLogger(__name__)
business_logger = BusinessLogger("order_manager")


@dataclass
class OrderRequest:
    """订单请求"""
    symbol: str
    side: str  # buy, sell
    order_type: str  # market, limit, stop, stop_limit
    amount: Union[Decimal, float, str]
    price: Optional[Union[Decimal, float, str]] = None
    exchange: str = "binance"
    strategy_id: Optional[str] = None
    user_id: Optional[str] = None

    # 高级参数
    stop_loss_price: Optional[Union[Decimal, float, str]] = None
    take_profit_price: Optional[Union[Decimal, float, str]] = None
    trailing_stop_amount: Optional[Union[Decimal, float, str]] = None
    time_in_force: str = TimeInForce.GTC.value
    expire_time: Optional[datetime] = None

    # 冰山订单参数
    iceberg_amount: Optional[Union[Decimal, float, str]] = None
    visible_amount: Optional[Union[Decimal, float, str]] = None

    # 元数据
    metadata: Optional[Dict[str, Any]] = None
    tags: Optional[List[str]] = None


@dataclass
class OrderResult:
    """订单执行结果"""
    success: bool
    order_id: Optional[str] = None
    exchange_order_id: Optional[str] = None
    error_code: Optional[str] = None
    error_message: Optional[str] = None
    filled_amount: Optional[Decimal] = None
    filled_price: Optional[Decimal] = None
    fee: Optional[Decimal] = None


class OrderValidationResult:
    """订单验证结果"""
    def __init__(self, is_valid: bool, errors: List[str] = None, warnings: List[str] = None):
        self.is_valid = is_valid
        self.errors = errors or []
        self.warnings = warnings or []


class OrderManager:
    """订单管理器"""

    def __init__(self):
        self.logger = logger
        self.business_logger = business_logger
        self.exchange_manager = get_exchange_manager()
        self.cache = get_cache()

        # 订单状态监控任务
        self._monitoring_tasks = {}

    async def create_order(self, order_request: OrderRequest, db: Session) -> OrderResult:
        """
        创建并执行订单

        Args:
            order_request: 订单请求
            db: 数据库会话

        Returns:
            OrderResult: 订单执行结果
        """
        try:
            # 验证订单请求
            validation_result = await self.validate_order(order_request, db)
            if not validation_result.is_valid:
                return OrderResult(
                    success=False,
                    error_code="VALIDATION_ERROR",
                    error_message="; ".join(validation_result.errors)
                )

            # 创建订单记录
            order = await self._create_order_record(order_request, db)

            # 提交到数据库
            db.add(order)
            db.commit()
            db.refresh(order)

            # 记录业务日志
            await self.business_logger.log_event(
                "order_created",
                order_id=order.order_id,
                symbol=order.symbol,
                side=order.side,
                amount=float(order.amount),
                price=float(order.price) if order.price else None,
                strategy_id=str(order.strategy_id) if order.strategy_id else None
            )

            # 执行订单
            execution_result = await self._execute_order(order, db)

            if execution_result.success:
                # 启动订单监控
                await self._start_order_monitoring(order)

                await self.business_logger.log_event(
                    "order_executed",
                    order_id=order.order_id,
                    exchange_order_id=execution_result.exchange_order_id,
                    filled_amount=float(execution_result.filled_amount) if execution_result.filled_amount else None
                )
            else:
                # 标记订单失败
                order.reject(
                    error_code=execution_result.error_code or "EXECUTION_FAILED",
                    error_message=execution_result.error_message or "Order execution failed"
                )
                db.commit()

            return OrderResult(
                success=execution_result.success,
                order_id=order.order_id,
                exchange_order_id=execution_result.exchange_order_id,
                error_code=execution_result.error_code,
                error_message=execution_result.error_message,
                filled_amount=execution_result.filled_amount,
                filled_price=execution_result.filled_price,
                fee=execution_result.fee
            )

        except Exception as e:
            self.logger.error(f"创建订单失败: {str(e)}", exc_info=True)
            db.rollback()

            await self.business_logger.log_event(
                "order_creation_failed",
                error=str(e),
                symbol=order_request.symbol,
                side=order_request.side
            )

            return OrderResult(
                success=False,
                error_code="INTERNAL_ERROR",
                error_message=str(e)
            )

    async def validate_order(self, order_request: OrderRequest, db: Session) -> OrderValidationResult:
        """
        验证订单请求

        Args:
            order_request: 订单请求
            db: 数据库会话

        Returns:
            OrderValidationResult: 验证结果
        """
        errors = []
        warnings = []

        # 基本参数验证
        if not order_request.symbol:
            errors.append("交易符号不能为空")

        if order_request.side not in [OrderSide.BUY.value, OrderSide.SELL.value]:
            errors.append("订单方向必须是 'buy' 或 'sell'")

        if order_request.order_type not in [t.value for t in OrderType]:
            errors.append(f"无效的订单类型: {order_request.order_type}")

        # 数量和价格验证
        try:
            amount = Decimal(str(order_request.amount))
            if amount <= 0:
                errors.append("订单数量必须大于0")

            # 限制订单数量精度
            if amount.as_tuple().exponent < -8:
                errors.append("订单数量精度不能超过8位小数")
        except (ValueError, TypeError):
            errors.append("无效的订单数量")

        # 价格验证（限价单必须有价格）
        if order_request.order_type in [OrderType.LIMIT.value, OrderType.STOP_LIMIT.value]:
            if order_request.price is None:
                errors.append("限价单必须指定价格")
            else:
                try:
                    price = Decimal(str(order_request.price))
                    if price <= 0:
                        errors.append("订单价格必须大于0")
                except (ValueError, TypeError):
                    errors.append("无效的订单价格")

        # 止损止盈价格验证
        if order_request.stop_loss_price is not None:
            try:
                stop_loss = Decimal(str(order_request.stop_loss_price))
                if stop_loss <= 0:
                    errors.append("止损价格必须大于0")

                # 验证止损价格逻辑
                if order_request.price is not None:
                    price = Decimal(str(order_request.price))
                    if order_request.side == OrderSide.BUY.value and stop_loss >= price:
                        errors.append("买单止损价格必须低于订单价格")
                    elif order_request.side == OrderSide.SELL.value and stop_loss <= price:
                        errors.append("卖单止损价格必须高于订单价格")
            except (ValueError, TypeError):
                errors.append("无效的止损价格")

        # 过期时间验证
        if order_request.expire_time is not None and order_request.expire_time <= datetime.utcnow():
            errors.append("订单过期时间必须在当前时间之后")

        # 交易所验证
        supported_exchanges = self.exchange_manager.get_supported_exchanges()
        if order_request.exchange not in supported_exchanges:
            errors.append(f"不支持的交易所: {order_request.exchange}")

        # 风险检查
        risk_warnings = await self._check_order_risk(order_request, db)
        warnings.extend(risk_warnings)

        return OrderValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings
        )

    async def cancel_order(self, order_id: str, user_id: str = None, db: Session = None) -> OrderResult:
        """
        取消订单

        Args:
            order_id: 订单ID
            user_id: 用户ID（可选）
            db: 数据库会话

        Returns:
            OrderResult: 取消结果
        """
        if db is None:
            db = SessionLocal()

        try:
            # 查询订单
            order = db.query(TradingOrder).filter(
                TradingOrder.order_id == order_id
            ).first()

            if not order:
                return OrderResult(
                    success=False,
                    error_code="ORDER_NOT_FOUND",
                    error_message="订单不存在"
                )

            # 用户权限检查
            if user_id and str(order.user_id) != user_id:
                return OrderResult(
                    success=False,
                    error_code="PERMISSION_DENIED",
                    error_message="无权限取消此订单"
                )

            # 检查订单是否可以取消
            if not order.can_cancel():
                return OrderResult(
                    success=False,
                    error_code="ORDER_NOT_CANCELLABLE",
                    error_message=f"订单状态为 {order.status}，无法取消"
                )

            # 调用交易所API取消订单
            exchange_result = await self.exchange_manager.cancel_order(
                exchange=order.exchange,
                order_id=order.exchange_order_id,
                symbol=order.symbol
            )

            if exchange_result.get('success'):
                # 更新订单状态
                order.cancel()
                db.commit()

                # 停止监控
                await self._stop_order_monitoring(order_id)

                await self.business_logger.log_event(
                    "order_cancelled",
                    order_id=order.order_id,
                    exchange_order_id=order.exchange_order_id,
                    reason="user_request"
                )

                return OrderResult(success=True, order_id=order.order_id)
            else:
                return OrderResult(
                    success=False,
                    error_code=exchange_result.get('error_code', 'CANCEL_FAILED'),
                    error_message=exchange_result.get('error_message', '取消订单失败')
                )

        except Exception as e:
            self.logger.error(f"取消订单失败: {str(e)}", exc_info=True)
            return OrderResult(
                success=False,
                error_code="INTERNAL_ERROR",
                error_message=str(e)
            )

    async def get_order_status(self, order_id: str, db: Session = None) -> Optional[Dict[str, Any]]:
        """
        获取订单状态

        Args:
            order_id: 订单ID
            db: 数据库会话

        Returns:
            订单状态信息或None
        """
        if db is None:
            db = SessionLocal()

        try:
            order = db.query(TradingOrder).filter(
                TradingOrder.order_id == order_id
            ).first()

            if not order:
                return None

            # 从缓存获取实时状态
            cached_status = await self.cache.get(f"order_status:{order_id}")
            if cached_status:
                return cached_status

            # 从交易所获取最新状态
            if order.is_active and order.exchange_order_id:
                exchange_status = await self.exchange_manager.get_order_status(
                    exchange=order.exchange,
                    order_id=order.exchange_order_id,
                    symbol=order.symbol
                )

                if exchange_status:
                    # 更新订单状态
                    await self._update_order_from_exchange(order, exchange_status, db)
                    db.commit()

            return order.to_dict()

        except Exception as e:
            self.logger.error(f"获取订单状态失败: {str(e)}", exc_info=True)
            return None

    async def get_user_orders(
        self,
        user_id: str,
        symbol: str = None,
        status: str = None,
        limit: int = 100,
        offset: int = 0,
        db: Session = None
    ) -> List[Dict[str, Any]]:
        """
        获取用户订单列表

        Args:
            user_id: 用户ID
            symbol: 交易符号过滤
            status: 状态过滤
            limit: 限制数量
            offset: 偏移量
            db: 数据库会话

        Returns:
            订单列表
        """
        if db is None:
            db = SessionLocal()

        try:
            query = db.query(TradingOrder).filter(TradingOrder.user_id == user_id)

            if symbol:
                query = query.filter(TradingOrder.symbol == symbol)

            if status:
                query = query.filter(TradingOrder.status == status)

            orders = query.order_by(TradingOrder.created_at.desc()).offset(offset).limit(limit).all()

            return [order.to_dict() for order in orders]

        except Exception as e:
            self.logger.error(f"获取用户订单失败: {str(e)}", exc_info=True)
            return []

    async def _create_order_record(self, order_request: OrderRequest, db: Session) -> TradingOrder:
        """创建订单记录"""
        # 生成订单ID
        order_id = f"order_{uuid.uuid4().hex[:12]}"
        client_order_id = f"client_{uuid.uuid4().hex[:12]}"

        # 计算剩余数量
        amount = Decimal(str(order_request.amount))

        # 计算过期时间
        expire_time = None
        if order_request.expire_time:
            expire_time = order_request.expire_time
        elif order_request.time_in_force == TimeInForce.GTD.value:
            expire_time = datetime.utcnow() + timedelta(hours=24)

        # 创建订单对象
        order = TradingOrder(
            order_id=order_id,
            client_order_id=client_order_id,
            symbol=order_request.symbol,
            side=order_request.side,
            order_type=order_request.order_type,
            amount=amount,
            remaining_amount=amount,
            price=Decimal(str(order_request.price)) if order_request.price else None,
            time_in_force=order_request.time_in_force,
            expires_at=expire_time,
            exchange=order_request.exchange,
            stop_loss_price=Decimal(str(order_request.stop_loss_price)) if order_request.stop_loss_price else None,
            take_profit_price=Decimal(str(order_request.take_profit_price)) if order_request.take_profit_price else None,
            trailing_stop_amount=Decimal(str(order_request.trailing_stop_amount)) if order_request.trailing_stop_amount else None,
            iceberg_amount=Decimal(str(order_request.iceberg_amount)) if order_request.iceberg_amount else None,
            visible_amount=Decimal(str(order_request.visible_amount)) if order_request.visible_amount else None,
            metadata=order_request.metadata,
            tags=order_request.tags,
            strategy_id=uuid.UUID(order_request.strategy_id) if order_request.strategy_id else None,
            user_id=uuid.UUID(order_request.user_id) if order_request.user_id else None
        )

        return order

    async def _execute_order(self, order: TradingOrder, db: Session) -> OrderResult:
        """执行订单"""
        try:
            # 准备订单参数
            order_params = {
                'symbol': order.symbol,
                'side': order.side,
                'order_type': order.order_type,
                'amount': order.amount,
                'price': order.price,
                'time_in_force': order.time_in_force,
                'client_order_id': order.client_order_id
            }

            # 添加高级参数
            if order.iceberg_amount:
                order_params['iceberg_amount'] = order.iceberg_amount
            if order.visible_amount:
                order_params['visible_amount'] = order.visible_amount
            if order.stop_loss_price:
                order_params['stop_loss_price'] = order.stop_loss_price
            if order.take_profit_price:
                order_params['take_profit_price'] = order.take_profit_price

            # 调用交易所API
            exchange_result = await self.exchange_manager.create_order(
                exchange=order.exchange,
                **order_params
            )

            if exchange_result.get('success'):
                # 更新订单信息
                order.exchange_order_id = exchange_result.get('order_id')
                order.status = OrderStatus.OPEN.value if order.order_type != OrderType.MARKET.value else OrderStatus.FILLED.value

                # 如果是市价单或有立即成交
                if exchange_result.get('filled_amount'):
                    filled_amount = Decimal(str(exchange_result['filled_amount']))
                    filled_price = Decimal(str(exchange_result.get('average_price', 0)))
                    fee = Decimal(str(exchange_result.get('fee', 0)))

                    order.update_fill(filled_amount, filled_price, fee)

                    # 创建成交记录
                    fill = OrderFill(
                        fill_id=f"fill_{uuid.uuid4().hex[:12]}",
                        order_id=order.id,
                        symbol=order.symbol,
                        side=order.side,
                        amount=filled_amount,
                        price=filled_price,
                        quote_amount=filled_amount * filled_price,
                        fee=fee,
                        fee_asset=exchange_result.get('fee_asset'),
                        exchange=order.exchange,
                        timestamp=datetime.utcnow(),
                        user_id=order.user_id,
                        strategy_id=order.strategy_id
                    )
                    db.add(fill)

                db.commit()

                return OrderResult(
                    success=True,
                    order_id=order.order_id,
                    exchange_order_id=order.exchange_order_id,
                    filled_amount=exchange_result.get('filled_amount'),
                    filled_price=exchange_result.get('average_price'),
                    fee=exchange_result.get('fee')
                )
            else:
                return OrderResult(
                    success=False,
                    error_code=exchange_result.get('error_code', 'EXECUTION_FAILED'),
                    error_message=exchange_result.get('error_message', '订单执行失败')
                )

        except Exception as e:
            self.logger.error(f"订单执行失败: {str(e)}", exc_info=True)
            return OrderResult(
                success=False,
                error_code="EXECUTION_ERROR",
                error_message=str(e)
            )

    async def _start_order_monitoring(self, order: TradingOrder):
        """启动订单监控"""
        if order.order_id not in self._monitoring_tasks and order.is_active:
            task = asyncio.create_task(self._monitor_order(order.order_id))
            self._monitoring_tasks[order.order_id] = task

    async def _stop_order_monitoring(self, order_id: str):
        """停止订单监控"""
        if order_id in self._monitoring_tasks:
            task = self._monitoring_tasks.pop(order_id)
            task.cancel()

    async def _monitor_order(self, order_id: str):
        """监控订单状态"""
        while True:
            try:
                await asyncio.sleep(5)  # 每5秒检查一次

                # 获取数据库会话
                with SessionLocal() as db:
                    order = db.query(TradingOrder).filter(
                        TradingOrder.order_id == order_id
                    ).first()

                    if not order or order.is_completed:
                        await self._stop_order_monitoring(order_id)
                        break

                    # 从交易所获取最新状态
                    if order.exchange_order_id:
                        exchange_status = await self.exchange_manager.get_order_status(
                            exchange=order.exchange,
                            order_id=order.exchange_order_id,
                            symbol=order.symbol
                        )

                        if exchange_status:
                            await self._update_order_from_exchange(order, exchange_status, db)
                            db.commit()

                            # 如果订单完成，停止监控
                            if order.is_completed:
                                await self._stop_order_monitoring(order_id)
                                break

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"订单监控异常 {order_id}: {str(e)}", exc_info=True)

    async def _update_order_from_exchange(self, order: TradingOrder, exchange_status: Dict[str, Any], db: Session):
        """根据交易所状态更新订单"""
        status = exchange_status.get('status')
        filled_amount = Decimal(str(exchange_status.get('filled', 0)))
        average_price = Decimal(str(exchange_status.get('average_price', 0)))

        # 更新订单状态
        if status == 'filled':
            if filled_amount > order.filled_amount:
                additional_fill = filled_amount - order.filled_amount
                order.update_fill(additional_fill, average_price)

                # 创建成交记录
                fill = OrderFill(
                    fill_id=f"fill_{uuid.uuid4().hex[:12]}",
                    order_id=order.id,
                    symbol=order.symbol,
                    side=order.side,
                    amount=additional_fill,
                    price=average_price,
                    quote_amount=additional_fill * average_price,
                    exchange=order.exchange,
                    timestamp=datetime.utcnow(),
                    user_id=order.user_id,
                    strategy_id=order.strategy_id
                )
                db.add(fill)

                await self.business_logger.log_event(
                    "order_filled",
                    order_id=order.order_id,
                    filled_amount=float(additional_fill),
                    filled_price=float(average_price)
                )

        elif status == 'cancelled':
            order.cancel()
            await self.business_logger.log_event(
                "order_cancelled_by_exchange",
                order_id=order.order_id,
                reason=exchange_status.get('reason', 'Unknown')
            )

        elif status == 'expired':
            order.expire()
            await self.business_logger.log_event(
                "order_expired",
                order_id=order.order_id
            )

    async def _check_order_risk(self, order_request: OrderRequest, db: Session) -> List[str]:
        """检查订单风险"""
        warnings = []

        try:
            # 检查账户余额
            if order_request.user_id:
                balance_warnings = await self._check_account_balance(order_request, db)
                warnings.extend(balance_warnings)

            # 检查持仓风险
            position_warnings = await self._check_position_risk(order_request, db)
            warnings.extend(position_warnings)

            # 检查市场风险
            market_warnings = await self._check_market_risk(order_request)
            warnings.extend(market_warnings)

        except Exception as e:
            self.logger.error(f"风险检查失败: {str(e)}", exc_info=True)
            warnings.append("风险检查失败，请谨慎操作")

        return warnings

    async def _check_account_balance(self, order_request: OrderRequest, db: Session) -> List[str]:
        """检查账户余额"""
        warnings = []

        try:
            # 获取账户余额
            balance_info = await self.exchange_manager.get_account_balance(
                exchange=order_request.exchange,
                user_id=order_request.user_id
            )

            if not balance_info:
                warnings.append("无法获取账户余额信息")
                return warnings

            # 计算需要的资金
            amount = Decimal(str(order_request.amount))
            if order_request.price:
                required_value = amount * Decimal(str(order_request.price))
            else:
                # 市价单估算
                ticker = await self.exchange_manager.get_ticker(
                    exchange=order_request.exchange,
                    symbol=order_request.symbol
                )
                if ticker:
                    required_value = amount * Decimal(str(ticker.get('price', 0)))
                else:
                    required_value = Decimal('0')

            # 检查余额是否足够
            base_currency = order_request.symbol.split('/')[0]
            quote_currency = order_request.symbol.split('/')[1] if '/' in order_request.symbol else 'USDT'

            if order_request.side == OrderSide.BUY.value:
                # 买单需要检查报价货币余额
                available_balance = balance_info.get(quote_currency, {}).get('free', 0)
                if required_value > available_balance:
                    warnings.append(f"{quote_currency}余额不足，需要 {required_value}，可用 {available_balance}")

            else:
                # 卖单需要检查基础货币余额
                available_balance = balance_info.get(base_currency, {}).get('free', 0)
                if amount > available_balance:
                    warnings.append(f"{base_currency}余额不足，需要 {amount}，可用 {available_balance}")

        except Exception as e:
            self.logger.error(f"账户余额检查失败: {str(e)}", exc_info=True)
            warnings.append("账户余额检查失败")

        return warnings

    async def _check_position_risk(self, order_request: OrderRequest, db: Session) -> List[str]:
        """检查持仓风险"""
        warnings = []

        try:
            if not order_request.user_id:
                return warnings

            # 查询现有持仓
            positions = db.query(Position).filter(
                and_(
                    Position.user_id == uuid.UUID(order_request.user_id),
                    Position.symbol == order_request.symbol,
                    Position.status == PositionStatus.OPEN.value
                )
            ).all()

            for position in positions:
                # 检查是否会增加风险敞口
                if (position.side == PositionSide.LONG.value and order_request.side == OrderSide.BUY.value) or \
                   (position.side == PositionSide.SHORT.value and order_request.side == OrderSide.SELL.value):
                    warnings.append(f"当前已有 {position.side} 持仓，此订单会增加风险敞口")

                # 检查反向持仓（对冲）
                if (position.side == PositionSide.LONG.value and order_request.side == OrderSide.SELL.value) or \
                   (position.side == PositionSide.SHORT.value and order_request.side == OrderSide.BUY.value):
                    warnings.append(f"此订单将对冲现有 {position.side} 持仓")

        except Exception as e:
            self.logger.error(f"持仓风险检查失败: {str(e)}", exc_info=True)
            warnings.append("持仓风险检查失败")

        return warnings

    async def _check_market_risk(self, order_request: OrderRequest) -> List[str]:
        """检查市场风险"""
        warnings = []

        try:
            # 获取市场数据
            ticker = await self.exchange_manager.get_ticker(
                exchange=order_request.exchange,
                symbol=order_request.symbol
            )

            if ticker:
                # 检查价格波动性
                price_change_24h = ticker.get('price_change_24h_percent', 0)
                if abs(price_change_24h) > 10:  # 24小时涨跌幅超过10%
                    warnings.append(f"24小时价格波动较大：{price_change_24h:.2f}%")

                # 检查市场深度
                orderbook = await self.exchange_manager.get_orderbook(
                    exchange=order_request.exchange,
                    symbol=order_request.symbol,
                    limit=20
                )

                if orderbook:
                    # 计算买卖价差
                    best_bid = orderbook.get('bids', [])[0][0] if orderbook.get('bids') else 0
                    best_ask = orderbook.get('asks', [])[0][0] if orderbook.get('asks') else 0

                    if best_bid > 0 and best_ask > 0:
                        spread_percent = (best_ask - best_bid) / best_bid * 100
                        if spread_percent > 0.5:  # 买卖价差超过0.5%
                            warnings.append(f"买卖价差较大：{spread_percent:.2f}%")

        except Exception as e:
            self.logger.error(f"市场风险检查失败: {str(e)}", exc_info=True)
            warnings.append("市场风险检查失败")

        return warnings