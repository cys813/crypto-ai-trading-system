"""
交易验证和错误处理模块

提供交易操作的数据验证、错误处理和异常管理功能。
"""

import logging
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass
from enum import Enum

from pydantic import BaseModel, ValidationError, validator
from sqlalchemy.orm import Session

from .exceptions import (
    ValidationError as CustomValidationError,
    InsufficientFundsError,
    OrderError,
    PositionError,
    RiskError,
    ExchangeAPIError
)

logger = logging.getLogger(__name__)


class ValidationSeverity(str, Enum):
    """验证严重程度"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class ErrorCategory(str, Enum):
    """错误类别"""
    VALIDATION = "validation"
    BUSINESS = "business"
    TECHNICAL = "technical"
    EXTERNAL = "external"
    PERMISSION = "permission"


@dataclass
class ValidationResult:
    """验证结果"""
    is_valid: bool
    severity: ValidationSeverity
    field: Optional[str] = None
    message: str = ""
    code: Optional[str] = None
    data: Optional[Dict[str, Any]] = None


@dataclass
class ErrorInfo:
    """错误信息"""
    category: ErrorCategory
    code: str
    message: str
    details: Optional[Dict[str, Any]] = None
    timestamp: datetime = None
    user_id: Optional[str] = None
    request_id: Optional[str] = None
    context: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()


class TradingValidator:
    """交易验证器"""

    def __init__(self):
        self.logger = logger

    async def validate_order_request(self, order_data: Dict[str, Any], db: Session) -> List[ValidationResult]:
        """
        验证订单请求

        Args:
            order_data: 订单数据
            db: 数据库会话

        Returns:
            List[ValidationResult]: 验证结果列表
        """
        results = []

        try:
            # 基本字段验证
            results.extend(self._validate_basic_order_fields(order_data))

            # 价格和数量验证
            results.extend(self._validate_price_and_amount(order_data))

            # 时间验证
            results.extend(self._validate_order_timing(order_data))

            # 业务规则验证
            results.extend(await self._validate_business_rules(order_data, db))

            # 风险验证
            results.extend(await self._validate_risk_rules(order_data, db))

        except Exception as e:
            self.logger.error(f"订单验证异常: {str(e)}")
            results.append(ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.CRITICAL,
                message="验证过程发生异常",
                code="VALIDATION_EXCEPTION"
            ))

        return results

    def _validate_basic_order_fields(self, order_data: Dict[str, Any]) -> List[ValidationResult]:
        """验证基本订单字段"""
        results = []

        # 验证交易符号
        symbol = order_data.get('symbol')
        if not symbol:
            results.append(ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.ERROR,
                field='symbol',
                message="交易符号不能为空",
                code="SYMBOL_REQUIRED"
            ))
        elif not isinstance(symbol, str) or len(symbol.strip()) == 0:
            results.append(ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.ERROR,
                field='symbol',
                message="交易符号格式无效",
                code="SYMBOL_INVALID"
            ))

        # 验证订单方向
        side = order_data.get('side')
        if not side:
            results.append(ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.ERROR,
                field='side',
                message="订单方向不能为空",
                code="SIDE_REQUIRED"
            ))
        elif side not in ['buy', 'sell']:
            results.append(ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.ERROR,
                field='side',
                message="订单方向必须是 'buy' 或 'sell'",
                code="SIDE_INVALID"
            ))

        # 验证订单类型
        order_type = order_data.get('order_type')
        if not order_type:
            results.append(ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.ERROR,
                field='order_type',
                message="订单类型不能为空",
                code="ORDER_TYPE_REQUIRED"
            ))
        elif order_type not in ['market', 'limit', 'stop', 'stop_limit', 'stop_market', 'limit_maker']:
            results.append(ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.ERROR,
                field='order_type',
                message="无效的订单类型",
                code="ORDER_TYPE_INVALID"
            ))

        return results

    def _validate_price_and_amount(self, order_data: Dict[str, Any]) -> List[ValidationResult]:
        """验证价格和数量"""
        results = []

        # 验证订单数量
        amount = order_data.get('amount')
        if not amount:
            results.append(ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.ERROR,
                field='amount',
                message="订单数量不能为空",
                code="AMOUNT_REQUIRED"
            ))
        else:
            try:
                amount_decimal = Decimal(str(amount))
                if amount_decimal <= 0:
                    results.append(ValidationResult(
                        is_valid=False,
                        severity=ValidationSeverity.ERROR,
                        field='amount',
                        message="订单数量必须大于0",
                        code="AMOUNT_INVALID"
                    ))
                elif amount_decimal.as_tuple().exponent < -8:
                    results.append(ValidationResult(
                        is_valid=False,
                        severity=ValidationSeverity.ERROR,
                        field='amount',
                        message="订单数量精度不能超过8位小数",
                        code="AMOUNT_PRECISION_INVALID"
                    ))
            except (ValueError, TypeError):
                results.append(ValidationResult(
                    is_valid=False,
                    severity=ValidationSeverity.ERROR,
                    field='amount',
                    message="无效的订单数量格式",
                    code="AMOUNT_FORMAT_INVALID"
                ))

        # 验证价格（限价单必须有价格）
        order_type = order_data.get('order_type')
        price = order_data.get('price')

        if order_type in ['limit', 'stop_limit'] and not price:
            results.append(ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.ERROR,
                field='price',
                message="限价单必须指定价格",
                code="PRICE_REQUIRED"
            ))
        elif price:
            try:
                price_decimal = Decimal(str(price))
                if price_decimal <= 0:
                    results.append(ValidationResult(
                        is_valid=False,
                        severity=ValidationSeverity.ERROR,
                        field='price',
                        message="订单价格必须大于0",
                        code="PRICE_INVALID"
                    ))
                elif price_decimal.as_tuple().exponent < -8:
                    results.append(ValidationResult(
                        is_valid=False,
                        severity=ValidationSeverity.ERROR,
                        field='price',
                        message="订单价格精度不能超过8位小数",
                        code="PRICE_PRECISION_INVALID"
                    ))
            except (ValueError, TypeError):
                results.append(ValidationResult(
                    is_valid=False,
                    severity=ValidationSeverity.ERROR,
                    field='price',
                    message="无效的订单价格格式",
                    code="PRICE_FORMAT_INVALID"
                ))

        # 验证止损价格
        stop_loss_price = order_data.get('stop_loss_price')
        if stop_loss_price:
            try:
                stop_loss_decimal = Decimal(str(stop_loss_price))
                if stop_loss_decimal <= 0:
                    results.append(ValidationResult(
                        is_valid=False,
                        severity=ValidationSeverity.ERROR,
                        field='stop_loss_price',
                        message="止损价格必须大于0",
                        code="STOP_LOSS_PRICE_INVALID"
                    ))

                # 验证止损价格逻辑
                if price and order_type in ['limit', 'stop_limit']:
                    price_decimal = Decimal(str(price))
                    side = order_data.get('side')
                    if side == 'buy' and stop_loss_decimal >= price_decimal:
                        results.append(ValidationResult(
                            is_valid=False,
                            severity=ValidationSeverity.ERROR,
                            field='stop_loss_price',
                            message="买单止损价格必须低于订单价格",
                            code="STOP_LOSS_PRICE_LOGIC_BUY"
                        ))
                    elif side == 'sell' and stop_loss_decimal <= price_decimal:
                        results.append(ValidationResult(
                            is_valid=False,
                            severity=ValidationSeverity.ERROR,
                            field='stop_loss_price',
                            message="卖单止损价格必须高于订单价格",
                            code="STOP_LOSS_PRICE_LOGIC_SELL"
                        ))

            except (ValueError, TypeError):
                results.append(ValidationResult(
                    is_valid=False,
                    severity=ValidationSeverity.ERROR,
                    field='stop_loss_price',
                    message="无效的止损价格格式",
                    code="STOP_LOSS_PRICE_FORMAT_INVALID"
                ))

        # 验证止盈价格
        take_profit_price = order_data.get('take_profit_price')
        if take_profit_price:
            try:
                take_profit_decimal = Decimal(str(take_profit_price))
                if take_profit_decimal <= 0:
                    results.append(ValidationResult(
                        is_valid=False,
                        severity=ValidationSeverity.ERROR,
                        field='take_profit_price',
                        message="止盈价格必须大于0",
                        code="TAKE_PROFIT_PRICE_INVALID"
                    ))

                # 验证止盈价格逻辑
                if price and order_type in ['limit', 'stop_limit']:
                    price_decimal = Decimal(str(price))
                    side = order_data.get('side')
                    if side == 'buy' and take_profit_decimal <= price_decimal:
                        results.append(ValidationResult(
                            is_valid=False,
                            severity=ValidationSeverity.WARNING,
                            field='take_profit_price',
                            message="买单止盈价格应该高于订单价格",
                            code="TAKE_PROFIT_PRICE_LOGIC_BUY"
                        ))
                    elif side == 'sell' and take_profit_decimal >= price_decimal:
                        results.append(ValidationResult(
                            is_valid=False,
                            severity=ValidationSeverity.WARNING,
                            field='take_profit_price',
                            message="卖单止盈价格应该低于订单价格",
                            code="TAKE_PROFIT_PRICE_LOGIC_SELL"
                        ))

            except (ValueError, TypeError):
                results.append(ValidationResult(
                    is_valid=False,
                    severity=ValidationSeverity.ERROR,
                    field='take_profit_price',
                    message="无效的止盈价格格式",
                    code="TAKE_PROFIT_PRICE_FORMAT_INVALID"
                ))

        return results

    def _validate_order_timing(self, order_data: Dict[str, Any]) -> List[ValidationResult]:
        """验证订单时间"""
        results = []

        # 验证过期时间
        expire_time = order_data.get('expire_time')
        if expire_time:
            if isinstance(expire_time, str):
                try:
                    expire_time = datetime.fromisoformat(expire_time.replace('Z', '+00:00'))
                except ValueError:
                    results.append(ValidationResult(
                        is_valid=False,
                        severity=ValidationSeverity.ERROR,
                        field='expire_time',
                        message="无效的过期时间格式",
                        code="EXPIRE_TIME_FORMAT_INVALID"
                    ))
                    return results

            if expire_time <= datetime.utcnow():
                results.append(ValidationResult(
                    is_valid=False,
                    severity=ValidationSeverity.ERROR,
                    field='expire_time',
                    message="订单过期时间必须在当前时间之后",
                    code="EXPIRE_TIME_PAST"
                ))

        # 验证订单有效期类型
        time_in_force = order_data.get('time_in_force')
        if time_in_force and time_in_force not in ['GTC', 'IOC', 'FOK', 'GTD']:
            results.append(ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.ERROR,
                field='time_in_force',
                message="无效的订单有效期类型",
                code="TIME_IN_FORCE_INVALID"
            ))

        return results

    async def _validate_business_rules(self, order_data: Dict[str, Any], db: Session) -> List[ValidationResult]:
        """验证业务规则"""
        results = []

        try:
            # 验证交易时间
            if not self._is_trading_time_allowed():
                results.append(ValidationResult(
                    is_valid=False,
                    severity=ValidationSeverity.ERROR,
                    message="当前时间不允许交易",
                    code="TRADING_TIME_NOT_ALLOWED"
                ))

            # 验证最小交易金额
            min_amount = await self._get_min_trade_amount(order_data.get('symbol'), order_data.get('exchange'))
            if min_amount and Decimal(str(order_data.get('amount', 0))) < min_amount:
                results.append(ValidationResult(
                    is_valid=False,
                    severity=ValidationSeverity.ERROR,
                    field='amount',
                    message=f"订单数量小于最小交易量 {min_amount}",
                    code="AMOUNT_BELOW_MINIMUM",
                    data={'minimum_amount': float(min_amount)}
                ))

        except Exception as e:
            self.logger.error(f"业务规则验证失败: {str(e)}")
            results.append(ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.WARNING,
                message="业务规则验证失败",
                code="BUSINESS_RULE_VALIDATION_FAILED"
            ))

        return results

    async def _validate_risk_rules(self, order_data: Dict[str, Any], db: Session) -> List[ValidationResult]:
        """验证风险规则"""
        results = []

        try:
            # 这里可以添加风险相关的验证逻辑
            # 例如：检查用户风险等级、持仓限制等

            user_id = order_data.get('user_id')
            if user_id:
                # 检查用户日交易次数限制
                daily_trades = await self._get_user_daily_trade_count(user_id, db)
                if daily_trades >= 100:  # 假设每日最多100笔交易
                    results.append(ValidationResult(
                        is_valid=False,
                        severity=ValidationSeverity.ERROR,
                        message="已达到每日最大交易次数限制",
                        code="DAILY_TRADE_LIMIT_EXCEEDED",
                        data={'daily_trades': daily_trades, 'limit': 100}
                    ))

        except Exception as e:
            self.logger.error(f"风险规则验证失败: {str(e)}")
            results.append(ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.WARNING,
                message="风险规则验证失败",
                code="RISK_RULE_VALIDATION_FAILED"
            ))

        return results

    def _is_trading_time_allowed(self) -> bool:
        """检查是否允许交易时间"""
        # 7x24小时交易，暂时返回True
        return True

    async def _get_min_trade_amount(self, symbol: str, exchange: str) -> Optional[Decimal]:
        """获取最小交易量"""
        # 这里应该从交易所获取最小交易量
        # 暂时返回默认值
        return Decimal('0.001')

    async def _get_user_daily_trade_count(self, user_id: str, db: Session) -> int:
        """获取用户日交易次数"""
        try:
            from ..models.trading_order import TradingOrder

            today = datetime.utcnow().date()
            count = db.query(TradingOrder).filter(
                and_(
                    TradingOrder.user_id == user_id,
                    func.date(TradingOrder.created_at) == today
                )
            ).count()

            return count
        except Exception as e:
            self.logger.error(f"获取用户日交易次数失败: {str(e)}")
            return 0


class TradingErrorHandler:
    """交易错误处理器"""

    def __init__(self):
        self.logger = logger

    def handle_order_error(self, error: Exception, context: Dict[str, Any] = None) -> ErrorInfo:
        """
        处理订单错误

        Args:
            error: 异常对象
            context: 错误上下文

        Returns:
            ErrorInfo: 错误信息
        """
        context = context or {}

        if isinstance(error, (CustomValidationError, ValidationError)):
            return ErrorInfo(
                category=ErrorCategory.VALIDATION,
                code="VALIDATION_ERROR",
                message=str(error),
                details={'validation_errors': getattr(error, 'errors', None)},
                context=context
            )

        elif isinstance(error, InsufficientFundsError):
            return ErrorInfo(
                category=ErrorCategory.BUSINESS,
                code="INSUFFICIENT_FUNDS",
                message="资金不足",
                details={'required_amount': getattr(error, 'required_amount', None)},
                context=context
            )

        elif isinstance(error, OrderError):
            return ErrorInfo(
                category=ErrorCategory.BUSINESS,
                code="ORDER_ERROR",
                message="订单处理错误",
                details={'order_id': getattr(error, 'order_id', None)},
                context=context
            )

        elif isinstance(error, PositionError):
            return ErrorInfo(
                category=ErrorCategory.BUSINESS,
                code="POSITION_ERROR",
                message="持仓处理错误",
                details={'position_id': getattr(error, 'position_id', None)},
                context=context
            )

        elif isinstance(error, RiskError):
            return ErrorInfo(
                category=ErrorCategory.BUSINESS,
                code="RISK_ERROR",
                message="风险管理错误",
                details={'risk_level': getattr(error, 'risk_level', None)},
                context=context
            )

        elif isinstance(error, ExchangeAPIError):
            return ErrorInfo(
                category=ErrorCategory.EXTERNAL,
                code="EXCHANGE_API_ERROR",
                message="交易所API错误",
                details={'exchange': getattr(error, 'exchange', None), 'api_error': str(error)},
                context=context
            )

        else:
            return ErrorInfo(
                category=ErrorCategory.TECHNICAL,
                code="UNKNOWN_ERROR",
                message="未知错误",
                details={'error_type': type(error).__name__, 'error_message': str(error)},
                context=context
            )

    def get_error_response(self, error_info: ErrorInfo) -> Dict[str, Any]:
        """
        根据错误信息生成响应

        Args:
            error_info: 错误信息

        Returns:
            Dict[str, Any]: 错误响应
        """
        response = {
            "success": False,
            "error": {
                "code": error_info.code,
                "message": error_info.message,
                "category": error_info.category.value,
                "timestamp": error_info.timestamp.isoformat()
            }
        }

        if error_info.details:
            response["error"]["details"] = error_info.details

        if error_info.context:
            response["error"]["context"] = error_info.context

        # 根据错误类别添加HTTP状态码建议
        status_code_map = {
            ErrorCategory.VALIDATION: 400,
            ErrorCategory.BUSINESS: 422,
            ErrorCategory.PERMISSION: 403,
            ErrorCategory.EXTERNAL: 502,
            ErrorCategory.TECHNICAL: 500
        }

        response["status_code"] = status_code_map.get(error_info.category, 500)

        return response

    async def log_error(self, error_info: ErrorInfo):
        """记录错误日志"""
        log_data = {
            "error_code": error_info.code,
            "error_message": error_info.message,
            "error_category": error_info.category.value,
            "timestamp": error_info.timestamp.isoformat(),
            "context": error_info.context,
            "details": error_info.details
        }

        if error_info.category in [ErrorCategory.CRITICAL, ErrorCategory.TECHNICAL]:
            self.logger.error(f"交易错误: {log_data}")
        else:
            self.logger.warning(f"交易错误: {log_data}")


# 全局验证器和错误处理器实例
trading_validator = TradingValidator()
trading_error_handler = TradingErrorHandler()


# 装饰器函数
def validate_order_request(func):
    """订单请求验证装饰器"""
    async def wrapper(*args, **kwargs):
        # 提取订单数据和数据库会话
        order_data = kwargs.get('order_data') or args[1] if len(args) > 1 else None
        db = kwargs.get('db') or args[2] if len(args) > 2 else None

        if order_data and db:
            validation_results = await trading_validator.validate_order_request(order_data, db)

            # 检查是否有错误级别的验证结果
            error_results = [r for r in validation_results if r.severity == ValidationSeverity.ERROR]
            if error_results:
                error_info = ErrorInfo(
                    category=ErrorCategory.VALIDATION,
                    code="VALIDATION_FAILED",
                    message="订单验证失败",
                    details={'validation_errors': [r.__dict__ for r in error_results]}
                )
                await trading_error_handler.log_error(error_info)
                raise CustomValidationError(f"订单验证失败: {[r.message for r in error_results]}")

        return await func(*args, **kwargs)
    return wrapper


def handle_trading_errors(func):
    """交易错误处理装饰器"""
    async def wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            # 构建错误上下文
            context = {
                'function_name': func.__name__,
                'args_count': len(args),
                'kwargs_keys': list(kwargs.keys())
            }

            # 处理错误
            error_info = trading_error_handler.handle_order_error(e, context)
            await trading_error_handler.log_error(error_info)

            # 重新抛出异常
            raise e
    return wrapper