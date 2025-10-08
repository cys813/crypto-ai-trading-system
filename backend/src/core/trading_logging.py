"""
交易操作日志模块

提供交易相关的详细日志记录功能，包括订单、持仓、策略执行等操作的日志。
"""

import logging
import json
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, asdict
from enum import Enum

from sqlalchemy import Column, String, DateTime, Text, Integer, JSON, Float
from sqlalchemy.ext.declarative import declarative_base

from .database import SessionLocal
from .logging import BusinessLogger

logger = logging.getLogger(__name__)


class LogLevel(str, Enum):
    """日志级别"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class TradingEventType(str, Enum):
    """交易事件类型"""
    ORDER_CREATED = "order_created"
    ORDER_UPDATED = "order_updated"
    ORDER_FILLED = "order_filled"
    ORDER_CANCELLED = "order_cancelled"
    ORDER_EXPIRED = "order_expired"
    ORDER_FAILED = "order_failed"

    POSITION_OPENED = "position_opened"
    POSITION_CLOSED = "position_closed"
    POSITION_UPDATED = "position_updated"
    POSITION_ADJUSTED = "position_adjusted"

    STRATEGY_EXECUTED = "strategy_executed"
    STRATEGY_FAILED = "strategy_failed"
    STRATEGY_SIGNAL_GENERATED = "strategy_signal_generated"

    RISK_ALERT = "risk_alert"
    RISK_LIMIT_EXCEEDED = "risk_limit_exceeded"
    EMERGENCY_STOP = "emergency_stop"

    FUNDS_ALLOCATED = "funds_allocated"
    FUNDS_DEALLOCATED = "funds_deallocated"

    API_REQUEST = "api_request"
    API_RESPONSE = "api_response"

    SYSTEM_ERROR = "system_error"
    PERFORMANCE_METRIC = "performance_metric"


@dataclass
class TradingLogEntry:
    """交易日志条目"""
    event_type: TradingEventType
    level: LogLevel
    message: str
    user_id: Optional[str] = None
    order_id: Optional[str] = None
    position_id: Optional[str] = None
    strategy_id: Optional[str] = None
    symbol: Optional[str] = None
    exchange: Optional[str] = None
    amount: Optional[Union[float, Decimal]] = None
    price: Optional[Union[float, Decimal]] = None
    fee: Optional[Union[float, Decimal]] = None
    pnl: Optional[Union[float, Decimal]] = None
    risk_level: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    timestamp: datetime = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()


class TradingLogManager:
    """交易日志管理器"""

    def __init__(self):
        self.logger = BusinessLogger("trading_operations")
        self.file_logger = logging.getLogger("trading_detailed")

        # 配置文件日志
        if not self.file_logger.handlers:
            handler = logging.FileHandler("logs/trading_operations.log")
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.file_logger.addHandler(handler)
            self.file_logger.setLevel(logging.INFO)

    async def log_event(self, event_type: Union[str, TradingEventType], **kwargs):
        """
        记录交易事件

        Args:
            event_type: 事件类型
            **kwargs: 事件参数
        """
        try:
            # 转换事件类型
            if isinstance(event_type, str):
                try:
                    event_type = TradingEventType(event_type)
                except ValueError:
                    event_type = TradingEventType.SYSTEM_ERROR

            # 确定日志级别
            level = kwargs.get('level', LogLevel.INFO)
            if isinstance(level, str):
                level = LogLevel(level)

            # 创建日志条目
            log_entry = TradingLogEntry(
                event_type=event_type,
                level=level,
                message=kwargs.get('message', ''),
                user_id=kwargs.get('user_id'),
                order_id=kwargs.get('order_id'),
                position_id=kwargs.get('position_id'),
                strategy_id=kwargs.get('strategy_id'),
                symbol=kwargs.get('symbol'),
                exchange=kwargs.get('exchange'),
                amount=kwargs.get('amount'),
                price=kwargs.get('price'),
                fee=kwargs.get('fee'),
                pnl=kwargs.get('pnl'),
                risk_level=kwargs.get('risk_level'),
                metadata=kwargs.get('metadata', {})
            )

            # 写入不同日志
            await self._write_business_log(log_entry)
            await self._write_detailed_log(log_entry)
            await self._write_database_log(log_entry)

        except Exception as e:
            logger.error(f"记录交易日志失败: {str(e)}")

    async def log_order_lifecycle(self, order_data: Dict[str, Any]):
        """记录订单生命周期"""
        try:
            order_id = order_data.get('order_id')
            user_id = order_data.get('user_id')
            symbol = order_data.get('symbol')
            side = order_data.get('side')
            order_type = order_data.get('order_type')
            amount = order_data.get('amount')
            price = order_data.get('price')
            status = order_data.get('status')
            exchange = order_data.get('exchange')

            # 记录订单创建
            if status == 'pending':
                await self.log_event(
                    TradingEventType.ORDER_CREATED,
                    message=f"订单创建: {side} {amount} {symbol} @ {price}",
                    level=LogLevel.INFO,
                    user_id=user_id,
                    order_id=order_id,
                    symbol=symbol,
                    side=side,
                    order_type=order_type,
                    amount=amount,
                    price=price,
                    exchange=exchange,
                    metadata=order_data
                )

            # 记录订单成交
            elif status == 'filled':
                filled_amount = order_data.get('filled_amount')
                filled_price = order_data.get('filled_price')
                fee = order_data.get('fee')

                await self.log_event(
                    TradingEventType.ORDER_FILLED,
                    message=f"订单成交: {filled_amount} {symbol} @ {filled_price}",
                    level=LogLevel.INFO,
                    user_id=user_id,
                    order_id=order_id,
                    symbol=symbol,
                    amount=filled_amount,
                    price=filled_price,
                    fee=fee,
                    exchange=exchange,
                    metadata=order_data
                )

            # 记录订单取消
            elif status == 'cancelled':
                reason = order_data.get('cancel_reason', '用户取消')

                await self.log_event(
                    TradingEventType.ORDER_CANCELLED,
                    message=f"订单取消: {reason}",
                    level=LogLevel.WARNING,
                    user_id=user_id,
                    order_id=order_id,
                    symbol=symbol,
                    exchange=exchange,
                    metadata={'cancel_reason': reason, **order_data}
                )

            # 记录订单失败
            elif status == 'failed':
                error_code = order_data.get('error_code')
                error_message = order_data.get('error_message')

                await self.log_event(
                    TradingEventType.ORDER_FAILED,
                    message=f"订单失败: {error_message}",
                    level=LogLevel.ERROR,
                    user_id=user_id,
                    order_id=order_id,
                    symbol=symbol,
                    exchange=exchange,
                    metadata={
                        'error_code': error_code,
                        'error_message': error_message,
                        **order_data
                    }
                )

        except Exception as e:
            logger.error(f"记录订单生命周期失败: {str(e)}")

    async def log_position_lifecycle(self, position_data: Dict[str, Any]):
        """记录持仓生命周期"""
        try:
            position_id = position_data.get('position_id')
            user_id = position_data.get('user_id')
            symbol = position_data.get('symbol')
            side = position_data.get('side')
            amount = position_data.get('amount')
            entry_price = position_data.get('entry_price')
            current_price = position_data.get('current_price')
            pnl = position_data.get('pnl')
            status = position_data.get('status')

            # 记录持仓开仓
            if status == 'open':
                await self.log_event(
                    TradingEventType.POSITION_OPENED,
                    message=f"持仓开仓: {side} {amount} {symbol} @ {entry_price}",
                    level=LogLevel.INFO,
                    user_id=user_id,
                    position_id=position_id,
                    symbol=symbol,
                    amount=amount,
                    price=entry_price,
                    metadata=position_data
                )

            # 记录持仓平仓
            elif status == 'closed':
                exit_price = position_data.get('exit_price', current_price)
                realized_pnl = position_data.get('realized_pnl', pnl)

                await self.log_event(
                    TradingEventType.POSITION_CLOSED,
                    message=f"持仓平仓: {symbol} @ {exit_price}, 盈亏: {realized_pnl}",
                    level=LogLevel.INFO,
                    user_id=user_id,
                    position_id=position_id,
                    symbol=symbol,
                    price=exit_price,
                    pnl=realized_pnl,
                    metadata=position_data
                )

            # 记录持仓调整
            elif status == 'adjusted':
                adjustment_amount = position_data.get('adjustment_amount')
                adjustment_type = position_data.get('adjustment_type')

                await self.log_event(
                    TradingEventType.POSITION_ADJUSTED,
                    message=f"持仓调整: {adjustment_type} {adjustment_amount} {symbol}",
                    level=LogLevel.INFO,
                    user_id=user_id,
                    position_id=position_id,
                    symbol=symbol,
                    amount=adjustment_amount,
                    metadata=position_data
                )

        except Exception as e:
            logger.error(f"记录持仓生命周期失败: {str(e)}")

    async def log_strategy_execution(self, strategy_data: Dict[str, Any]):
        """记录策略执行"""
        try:
            strategy_id = strategy_data.get('strategy_id')
            user_id = strategy_data.get('user_id')
            strategy_name = strategy_data.get('strategy_name')
            symbol = strategy_data.get('symbol')
            signal = strategy_data.get('signal')
            confidence = strategy_data.get('confidence')
            execution_mode = strategy_data.get('execution_mode')
            success = strategy_data.get('success', True)

            if success:
                await self.log_event(
                    TradingEventType.STRATEGY_EXECUTED,
                    message=f"策略执行成功: {strategy_name} -> {signal} {symbol}",
                    level=LogLevel.INFO,
                    user_id=user_id,
                    strategy_id=strategy_id,
                    symbol=symbol,
                    metadata={
                        'strategy_name': strategy_name,
                        'signal': signal,
                        'confidence': confidence,
                        'execution_mode': execution_mode,
                        **strategy_data
                    }
                )
            else:
                error_message = strategy_data.get('error_message', '未知错误')

                await self.log_event(
                    TradingEventType.STRATEGY_FAILED,
                    message=f"策略执行失败: {strategy_name} - {error_message}",
                    level=LogLevel.ERROR,
                    user_id=user_id,
                    strategy_id=strategy_id,
                    symbol=symbol,
                    metadata={
                        'strategy_name': strategy_name,
                        'error_message': error_message,
                        **strategy_data
                    }
                )

        except Exception as e:
            logger.error(f"记录策略执行失败: {str(e)}")

    async def log_risk_event(self, risk_data: Dict[str, Any]):
        """记录风险事件"""
        try:
            user_id = risk_data.get('user_id')
            risk_type = risk_data.get('risk_type')
            risk_level = risk_data.get('risk_level')
            risk_score = risk_data.get('risk_score')
            message = risk_data.get('message')
            trigger_data = risk_data.get('trigger_data', {})

            if risk_level in ['high', 'critical']:
                await self.log_event(
                    TradingEventType.RISK_ALERT,
                    message=f"风险警报 [{risk_level.upper()}]: {message}",
                    level=LogLevel.WARNING if risk_level == 'high' else LogLevel.ERROR,
                    user_id=user_id,
                    risk_level=risk_level,
                    metadata={
                        'risk_type': risk_type,
                        'risk_score': risk_score,
                        'trigger_data': trigger_data,
                        **risk_data
                    }
                )

        except Exception as e:
            logger.error(f"记录风险事件失败: {str(e)}")

    async def log_api_call(self, api_data: Dict[str, Any]):
        """记录API调用"""
        try:
            user_id = api_data.get('user_id')
            endpoint = api_data.get('endpoint')
            method = api_data.get('method')
            status_code = api_data.get('status_code')
            response_time = api_data.get('response_time')
            request_data = api_data.get('request_data', {})
            error_message = api_data.get('error_message')

            if error_message:
                await self.log_event(
                    TradingEventType.API_RESPONSE,
                    message=f"API错误: {method} {endpoint} - {error_message}",
                    level=LogLevel.ERROR,
                    user_id=user_id,
                    metadata={
                        'endpoint': endpoint,
                        'method': method,
                        'status_code': status_code,
                        'response_time': response_time,
                        'error_message': error_message,
                        'request_data': request_data
                    }
                )
            else:
                await self.log_event(
                    TradingEventType.API_REQUEST,
                    message=f"API调用: {method} {endpoint}",
                    level=LogLevel.DEBUG,
                    user_id=user_id,
                    metadata={
                        'endpoint': endpoint,
                        'method': method,
                        'status_code': status_code,
                        'response_time': response_time,
                        'request_data': request_data
                    }
                )

        except Exception as e:
            logger.error(f"记录API调用失败: {str(e)}")

    async def log_performance_metrics(self, metrics_data: Dict[str, Any]):
        """记录性能指标"""
        try:
            metric_type = metrics_data.get('metric_type')
            value = metrics_data.get('value')
            unit = metrics_data.get('unit')
            user_id = metrics_data.get('user_id')

            await self.log_event(
                TradingEventType.PERFORMANCE_METRIC,
                message=f"性能指标: {metric_type} = {value} {unit}",
                level=LogLevel.DEBUG,
                user_id=user_id,
                metadata=metrics_data
            )

        except Exception as e:
            logger.error(f"记录性能指标失败: {str(e)}")

    async def get_trading_logs(
        self,
        user_id: Optional[str] = None,
        event_type: Optional[TradingEventType] = None,
        symbol: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 1000
    ) -> List[Dict[str, Any]]:
        """
        获取交易日志

        Args:
            user_id: 用户ID过滤
            event_type: 事件类型过滤
            symbol: 交易符号过滤
            start_time: 开始时间
            end_time: 结束时间
            limit: 限制数量

        Returns:
            List[Dict[str, Any]]: 日志列表
        """
        try:
            # 这里应该从数据库查询日志
            # 暂时返回空列表
            return []

        except Exception as e:
            logger.error(f"获取交易日志失败: {str(e)}")
            return []

    async def _write_business_log(self, log_entry: TradingLogEntry):
        """写入业务日志"""
        try:
            log_data = {
                'event_type': log_entry.event_type.value,
                'message': log_entry.message,
                'user_id': log_entry.user_id,
                'order_id': log_entry.order_id,
                'position_id': log_entry.position_id,
                'strategy_id': log_entry.strategy_id,
                'symbol': log_entry.symbol,
                'amount': float(log_entry.amount) if log_entry.amount else None,
                'price': float(log_entry.price) if log_entry.price else None,
                'pnl': float(log_entry.pnl) if log_entry.pnl else None,
                'risk_level': log_entry.risk_level,
                'metadata': log_entry.metadata
            }

            # 根据日志级别选择不同的业务日志方法
            if log_entry.level == LogLevel.DEBUG:
                self.logger.log_debug(log_entry.event_type.value, **log_data)
            elif log_entry.level == LogLevel.INFO:
                self.logger.log_info(log_entry.event_type.value, **log_data)
            elif log_entry.level == LogLevel.WARNING:
                self.logger.log_warning(log_entry.event_type.value, **log_data)
            elif log_entry.level == LogLevel.ERROR:
                self.logger.log_error(log_entry.event_type.value, **log_data)
            elif log_entry.level == LogLevel.CRITICAL:
                self.logger.log_critical(log_entry.event_type.value, **log_data)

        except Exception as e:
            logger.error(f"写入业务日志失败: {str(e)}")

    async def _write_detailed_log(self, log_entry: TradingLogEntry):
        """写入详细日志"""
        try:
            log_data = {
                'timestamp': log_entry.timestamp.isoformat(),
                'event_type': log_entry.event_type.value,
                'level': log_entry.level.value,
                'message': log_entry.message,
                'user_id': log_entry.user_id,
                'order_id': log_entry.order_id,
                'position_id': log_entry.position_id,
                'strategy_id': log_entry.strategy_id,
                'symbol': log_entry.symbol,
                'exchange': log_entry.exchange,
                'amount': float(log_entry.amount) if log_entry.amount else None,
                'price': float(log_entry.price) if log_entry.price else None,
                'fee': float(log_entry.fee) if log_entry.fee else None,
                'pnl': float(log_entry.pnl) if log_entry.pnl else None,
                'risk_level': log_entry.risk_level,
                'metadata': log_entry.metadata or {}
            }

            # 写入文件日志
            log_message = json.dumps(log_data, ensure_ascii=False)
            self.file_logger.info(log_message)

        except Exception as e:
            logger.error(f"写入详细日志失败: {str(e)}")

    async def _write_database_log(self, log_entry: TradingLogEntry):
        """写入数据库日志"""
        try:
            # 这里应该将日志写入数据库
            # 暂时跳过，避免对现有数据库结构的影响
            pass

        except Exception as e:
            logger.error(f"写入数据库日志失败: {str(e)}")


# 全局日志管理器实例
trading_log_manager = TradingLogManager()


# 便捷函数
async def log_trading_event(event_type: Union[str, TradingEventType], **kwargs):
    """记录交易事件的便捷函数"""
    await trading_log_manager.log_event(event_type, **kwargs)


async def log_order_event(order_data: Dict[str, Any]):
    """记录订单事件的便捷函数"""
    await trading_log_manager.log_order_lifecycle(order_data)


async def log_position_event(position_data: Dict[str, Any]):
    """记录持仓事件的便捷函数"""
    await trading_log_manager.log_position_lifecycle(position_data)


async def log_strategy_event(strategy_data: Dict[str, Any]):
    """记录策略事件的便捷函数"""
    await trading_log_manager.log_strategy_execution(strategy_data)


async def log_risk_event(risk_data: Dict[str, Any]):
    """记录风险事件的便捷函数"""
    await trading_log_manager.log_risk_event(risk_data)