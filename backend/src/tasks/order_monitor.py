"""
订单监控和管理任务

处理交易订单的实时监控、状态更新、风险控制和自动化管理。
"""

import logging
import asyncio
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, List, Any, Optional

from celery import Task
from celery.schedules import crontab
from sqlalchemy import and_, or_, func

from .celery_app import celery_app
from ..core.database import SessionLocal
from ..core.cache import get_cache
from ..core.logging import BusinessLogger
from ..core.exceptions import OrderError, ExchangeAPIError
from ..models.trading_order import TradingOrder, OrderStatus, OrderType
from ..models.position import Position, PositionStatus
from ..services.order_manager import OrderManager
from ..services.risk_manager import RiskManager
from ..services.position_monitor import PositionMonitor, MonitorConfig

logger = logging.getLogger(__name__)
business_logger = BusinessLogger("order_monitor_tasks")


class BaseOrderMonitorTask(Task):
    """订单监控任务基类"""

    def on_success(self, retval, task_id, args, kwargs):
        """任务成功回调"""
        business_logger.log_event(
            "order_monitor_task_success",
            task_id=task_id,
            result=retval
        )

    def on_failure(self, exc, task_id, args, kwargs, einfo):
        """任务失败回调"""
        business_logger.log_event(
            "order_monitor_task_failed",
            task_id=task_id,
            error=str(exc),
            traceback=str(einfo)
        )


@celery_app.task(bind=True, base=BaseOrderMonitorTask, name="src.tasks.order_monitor.monitor_active_orders")
def monitor_active_orders(self):
    """监控活跃订单任务"""
    try:
        logger.info("开始监控活跃订单")

        monitored_count = 0
        updated_count = 0
        error_count = 0

        with SessionLocal() as db:
            # 获取所有活跃订单
            active_orders = db.query(TradingOrder).filter(
                TradingOrder.status.in_([
                    OrderStatus.PENDING.value,
                    OrderStatus.OPEN.value,
                    OrderStatus.PARTIALLY_FILLED.value
                ])
            ).all()

            order_manager = OrderManager()

            for order in active_orders:
                try:
                    # 检查订单是否过期
                    if order.is_expired:
                        order.expire()
                        db.commit()
                        updated_count += 1

                        await business_logger.log_event(
                            "order_expired",
                            order_id=order.order_id,
                            symbol=order.symbol
                        )
                        continue

                    # 从交易所获取最新状态
                    if order.exchange_order_id:
                        exchange_status = await _get_exchange_order_status(order)
                        if exchange_status:
                            await _update_order_from_exchange(order, exchange_status, db)
                            updated_count += 1

                    # 检查订单超时
                    if await _check_order_timeout(order):
                        await _handle_timeout_order(order, db, order_manager)
                        updated_count += 1

                    monitored_count += 1

                except Exception as e:
                    logger.error(f"监控订单失败 {order.order_id}: {str(e)}")
                    error_count += 1
                    continue

        result = {
            "status": "success",
            "monitored_orders": monitored_count,
            "updated_orders": updated_count,
            "errors": error_count,
            "timestamp": datetime.utcnow().isoformat()
        }

        logger.info(f"订单监控完成: {result}")
        return result

    except Exception as e:
        logger.error(f"订单监控任务失败: {str(e)}", exc_info=True)
        raise self.retry(exc=e, countdown=30, max_retries=3)


@celery_app.task(bind=True, base=BaseOrderMonitorTask, name="src.tasks.order_monitor.monitor_order_fills")
def monitor_order_fills(self):
    """监控订单成交任务"""
    try:
        logger.info("开始监控订单成交")

        fills_processed = 0
        positions_updated = 0

        with SessionLocal() as db:
            # 获取部分成交的订单
            partial_orders = db.query(TradingOrder).filter(
                TradingOrder.status == OrderStatus.PARTIALLY_FILLED.value
            ).all()

            for order in partial_orders:
                try:
                    # 检查是否有新的成交
                    new_fills = await _get_new_fills(order)
                    if new_fills:
                        await _process_order_fills(order, new_fills, db)
                        fills_processed += len(new_fills)

                        # 更新相关持仓
                        await _update_related_positions(order, db)
                        positions_updated += 1

                except Exception as e:
                    logger.error(f"处理订单成交失败 {order.order_id}: {str(e)}")
                    continue

        result = {
            "status": "success",
            "fills_processed": fills_processed,
            "positions_updated": positions_updated,
            "timestamp": datetime.utcnow().isoformat()
        }

        logger.info(f"订单成交监控完成: {result}")
        return result

    except Exception as e:
        logger.error(f"订单成交监控任务失败: {str(e)}", exc_info=True)
        raise self.retry(exc=e, countdown=15, max_retries=3)


@celery_app.task(bind=True, base=BaseOrderMonitorTask, name="src.tasks.order_monitor.cleanup_old_orders")
def cleanup_old_orders(self):
    """清理旧订单任务"""
    try:
        logger.info("开始清理旧订单")

        cleanup_count = 0

        with SessionLocal() as db:
            # 清理7天前的已完成订单
            cutoff_date = datetime.utcnow() - timedelta(days=7)

            old_orders = db.query(TradingOrder).filter(
                and_(
                    TradingOrder.status.in_([
                        OrderStatus.FILLED.value,
                        OrderStatus.CANCELLED.value,
                        OrderStatus.REJECTED.value,
                        OrderStatus.EXPIRED.value
                    ]),
                    TradingOrder.updated_at < cutoff_date
                )
            ).all()

            for order in old_orders:
                try:
                    # 记录清理日志
                    await business_logger.log_event(
                        "order_cleanup",
                        order_id=order.order_id,
                        status=order.status,
                        age_days=(datetime.utcnow() - order.updated_at).days
                    )

                    # 这里可以选择归档或删除订单
                    # 暂时只记录日志，不删除
                    cleanup_count += 1

                except Exception as e:
                    logger.error(f"清理订单失败 {order.order_id}: {str(e)}")
                    continue

        result = {
            "status": "success",
            "cleanup_count": cleanup_count,
            "cutoff_date": cutoff_date.isoformat(),
            "timestamp": datetime.utcnow().isoformat()
        }

        logger.info(f"旧订单清理完成: {result}")
        return result

    except Exception as e:
        logger.error(f"旧订单清理任务失败: {str(e)}", exc_info=True)
        raise self.retry(exc=e, countdown=3600, max_retries=2)  # 1小时后重试


@celery_app.task(bind=True, base=BaseOrderMonitorTask, name="src.tasks.order_monitor.sync_order_status")
def sync_order_status(self, exchange_name: str = None, symbol: str = None):
    """同步订单状态任务"""
    try:
        logger.info(f"开始同步订单状态: exchange={exchange_name}, symbol={symbol}")

        sync_count = 0
        error_count = 0

        with SessionLocal() as db:
            # 构建查询条件
            query = db.query(TradingOrder).filter(
                TradingOrder.status.in_([
                    OrderStatus.PENDING.value,
                    OrderStatus.OPEN.value,
                    OrderStatus.PARTIALLY_FILLED.value
                ])
            )

            if exchange_name:
                query = query.filter(TradingOrder.exchange == exchange_name)

            if symbol:
                query = query.filter(TradingOrder.symbol == symbol)

            orders = query.all()

            order_manager = OrderManager()

            for order in orders:
                try:
                    if order.exchange_order_id:
                        # 从交易所获取最新状态
                        order_status = await order_manager.get_order_status(order.order_id, db)

                        if order_status:
                            sync_count += 1
                            await business_logger.log_event(
                                "order_status_synced",
                                order_id=order.order_id,
                                new_status=order_status.get('status')
                            )

                except Exception as e:
                    logger.error(f"同步订单状态失败 {order.order_id}: {str(e)}")
                    error_count += 1
                    continue

        result = {
            "status": "success",
            "sync_count": sync_count,
            "error_count": error_count,
            "exchange": exchange_name,
            "symbol": symbol,
            "timestamp": datetime.utcnow().isoformat()
        }

        logger.info(f"订单状态同步完成: {result}")
        return result

    except Exception as e:
        logger.error(f"订单状态同步任务失败: {str(e)}", exc_info=True)
        raise self.retry(exc=e, countdown=60, max_retries=3)


@celery_app.task(bind=True, base=BaseOrderMonitorTask, name="src.tasks.order_monitor.monitor_order_risks")
def monitor_order_risks(self):
    """监控订单风险任务"""
    try:
        logger.info("开始监控订单风险")

        risk_alerts = 0
        actions_taken = 0

        with SessionLocal() as db:
            # 获取活跃订单
            active_orders = db.query(TradingOrder).filter(
                TradingOrder.status.in_([
                    OrderStatus.PENDING.value,
                    OrderStatus.OPEN.value,
                    OrderStatus.PARTIALLY_FILLED.value
                ])
            ).all()

            risk_manager = RiskManager()

            for order in active_orders:
                try:
                    # 评估订单风险
                    risk_assessment = await risk_manager.assess_order_risk(
                        {
                            'symbol': order.symbol,
                            'side': order.side,
                            'order_type': order.order_type,
                            'amount': float(order.amount),
                            'price': float(order.price) if order.price else None,
                            'exchange': order.exchange
                        },
                        str(order.user_id) if order.user_id else None,
                        db
                    )

                    # 处理高风险订单
                    if risk_assessment.risk_level.value in ['high', 'critical']:
                        await _handle_risky_order(order, risk_assessment, db)
                        risk_alerts += 1

                        if risk_assessment.required_actions:
                            actions_taken += len(risk_assessment.required_actions)

                except Exception as e:
                    logger.error(f"监控订单风险失败 {order.order_id}: {str(e)}")
                    continue

        result = {
            "status": "success",
            "risk_alerts": risk_alerts,
            "actions_taken": actions_taken,
            "timestamp": datetime.utcnow().isoformat()
        }

        logger.info(f"订单风险监控完成: {result}")
        return result

    except Exception as e:
        logger.error(f"订单风险监控任务失败: {str(e)}", exc_info=True)
        raise self.retry(exc=e, countdown=30, max_retries=3)


@celery_app.task(bind=True, base=BaseOrderMonitorTask, name="src.tasks.order_monitor.batch_order_cancel")
def batch_order_cancel(self, user_id: str = None, symbol: str = None, order_type: str = None):
    """批量取消订单任务"""
    try:
        logger.info(f"开始批量取消订单: user={user_id}, symbol={symbol}, type={order_type}")

        cancel_count = 0
        error_count = 0

        with SessionLocal() as db:
            # 构建查询条件
            query = db.query(TradingOrder).filter(
                TradingOrder.status.in_([
                    OrderStatus.PENDING.value,
                    OrderStatus.OPEN.value
                ])
            )

            if user_id:
                query = query.filter(TradingOrder.user_id == user_id)

            if symbol:
                query = query.filter(TradingOrder.symbol == symbol)

            if order_type:
                query = query.filter(TradingOrder.order_type == order_type)

            orders = query.all()
            order_manager = OrderManager()

            for order in orders:
                try:
                    result = await order_manager.cancel_order(order.order_id, db)

                    if result.success:
                        cancel_count += 1
                        await business_logger.log_event(
                            "order_batch_cancelled",
                            order_id=order.order_id,
                            user_id=user_id,
                            symbol=symbol
                        )
                    else:
                        error_count += 1

                except Exception as e:
                    logger.error(f"批量取消订单失败 {order.order_id}: {str(e)}")
                    error_count += 1
                    continue

        result = {
            "status": "success",
            "cancel_count": cancel_count,
            "error_count": error_count,
            "user_id": user_id,
            "symbol": symbol,
            "order_type": order_type,
            "timestamp": datetime.utcnow().isoformat()
        }

        logger.info(f"批量取消订单完成: {result}")
        return result

    except Exception as e:
        logger.error(f"批量取消订单任务失败: {str(e)}", exc_info=True)
        raise self.retry(exc=e, countdown=60, max_retries=2)


# 辅助函数
async def _get_exchange_order_status(order: TradingOrder) -> Optional[Dict[str, Any]]:
    """从交易所获取订单状态"""
    try:
        # 这里应该调用交易所API
        # 暂时返回模拟数据
        return {
            'status': 'filled',
            'filled': float(order.amount),
            'average_price': 45000.0,
            'fee': 0.001
        }
    except Exception as e:
        logger.error(f"获取交易所订单状态失败 {order.order_id}: {str(e)}")
        return None


async def _update_order_from_exchange(order: TradingOrder, exchange_status: Dict[str, Any], db: Session):
    """根据交易所状态更新订单"""
    try:
        status = exchange_status.get('status')
        filled_amount = Decimal(str(exchange_status.get('filled', 0)))
        average_price = Decimal(str(exchange_status.get('average_price', 0)))

        if status == 'filled':
            if filled_amount > order.filled_amount:
                additional_fill = filled_amount - order.filled_amount
                order.update_fill(additional_fill, average_price)

                await business_logger.log_event(
                    "order_filled_from_exchange",
                    order_id=order.order_id,
                    filled_amount=float(additional_fill),
                    filled_price=float(average_price)
                )

        db.commit()

    except Exception as e:
        logger.error(f"更新订单状态失败 {order.order_id}: {str(e)}")
        db.rollback()


async def _check_order_timeout(order: TradingOrder) -> bool:
    """检查订单是否超时"""
    try:
        if order.timeout_seconds:
            timeout_time = order.created_at + timedelta(seconds=order.timeout_seconds)
            return datetime.utcnow() > timeout_time
        return False
    except Exception as e:
        logger.error(f"检查订单超时失败 {order.order_id}: {str(e)}")
        return False


async def _handle_timeout_order(order: TradingOrder, db: Session, order_manager: OrderManager):
    """处理超时订单"""
    try:
        result = await order_manager.cancel_order(order.order_id, db)

        if result.success:
            await business_logger.log_event(
                "order_timeout_cancelled",
                order_id=order.order_id,
                timeout_seconds=order.timeout_seconds
            )

    except Exception as e:
        logger.error(f"处理超时订单失败 {order.order_id}: {str(e)}")


async def _get_new_fills(order: TradingOrder) -> List[Dict[str, Any]]:
    """获取新的成交记录"""
    try:
        # 这里应该从交易所获取成交记录
        # 暂时返回空列表
        return []
    except Exception as e:
        logger.error(f"获取成交记录失败 {order.order_id}: {str(e)}")
        return []


async def _process_order_fills(order: TradingOrder, fills: List[Dict[str, Any]], db: Session):
    """处理订单成交"""
    try:
        for fill in fills:
            # 创建成交记录
            # 这里应该创建OrderFill记录
            pass

        db.commit()

    except Exception as e:
        logger.error(f"处理订单成交失败 {order.order_id}: {str(e)}")
        db.rollback()


async def _update_related_positions(order: TradingOrder, db: Session):
    """更新相关持仓"""
    try:
        # 根据订单更新或创建持仓
        # 这里应该实现持仓更新逻辑
        pass

    except Exception as e:
        logger.error(f"更新相关持仓失败 {order.order_id}: {str(e)}")


async def _handle_risky_order(order: TradingOrder, risk_assessment, db: Session):
    """处理高风险订单"""
    try:
        # 根据风险评估结果采取措施
        for action in risk_assessment.required_actions:
            if action.value == 'cancel_order':
                order_manager = OrderManager()
                await order_manager.cancel_order(order.order_id, db)

                await business_logger.log_event(
                    "risky_order_cancelled",
                    order_id=order.order_id,
                    risk_level=risk_assessment.risk_level.value,
                    risk_score=risk_assessment.risk_score
                )

    except Exception as e:
        logger.error(f"处理高风险订单失败 {order.order_id}: {str(e)}")


# 定时任务配置
def setup_periodic_tasks():
    """设置定时任务"""

    # 每30秒监控活跃订单
    celery_app.add_periodic_task(
        30.0,
        monitor_active_orders.s(),
        name='monitor_active_orders_periodic'
    )

    # 每分钟监控订单成交
    celery_app.add_periodic_task(
        60.0,
        monitor_order_fills.s(),
        name='monitor_order_fills_periodic'
    )

    # 每5分钟同步订单状态
    celery_app.add_periodic_task(
        300.0,
        sync_order_status.s(),
        name='sync_order_status_periodic'
    )

    # 每2分钟监控订单风险
    celery_app.add_periodic_task(
        120.0,
        monitor_order_risks.s(),
        name='monitor_order_risks_periodic'
    )

    # 每天凌晨2点清理旧订单
    celery_app.add_periodic_task(
        crontab(hour=2, minute=0),
        cleanup_old_orders.s(),
        name='cleanup_old_orders_daily'
    )