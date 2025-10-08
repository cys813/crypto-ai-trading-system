"""
异常处理模块

定义自定义异常类和全局异常处理器。
"""

import logging
import traceback
from typing import Any, Dict, Optional, Union
from fastapi import FastAPI, Request, HTTPException, status
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

from .config import settings

logger = logging.getLogger(__name__)


class BaseCustomException(Exception):
    """基础自定义异常类"""

    def __init__(
        self,
        message: str,
        error_code: str = None,
        details: Dict[str, Any] = None,
        cause: Exception = None
    ):
        self.message = message
        self.error_code = error_code or self.__class__.__name__
        self.details = details or {}
        self.cause = cause
        super().__init__(self.message)


class ValidationError(BaseCustomException):
    """验证错误"""
    pass


class AuthenticationError(BaseCustomException):
    """认证错误"""
    pass


class AuthorizationError(BaseCustomException):
    """授权错误"""
    pass


class DatabaseError(BaseCustomException):
    """数据库错误"""
    pass


class ExternalServiceError(BaseCustomException):
    """外部服务错误"""
    pass


class ExchangeAPIError(ExternalServiceError):
    """交易所API错误"""
    pass


class LLMServiceError(ExternalServiceError):
    """LLM服务错误"""
    pass


class NewsAPIError(ExternalServiceError):
    """新闻API错误"""
    pass


class NewsValidationError(ValidationError):
    """新闻验证错误"""
    pass


class NewsProcessingError(BusinessLogicError):
    """新闻处理错误"""
    pass


class CacheError(BaseCustomException):
    """缓存错误"""
    pass


class BusinessLogicError(BaseCustomException):
    """业务逻辑错误"""
    pass


class TradingError(BusinessLogicError):
    """交易错误"""
    pass


class InsufficientFundsError(TradingError):
    """资金不足错误"""
    pass


class OrderTimeoutError(TradingError):
    """订单超时错误"""
    pass


class RiskLimitExceededError(TradingError):
    """风险限制超出错误"""
    pass


class ConfigurationError(BaseCustomException):
    """配置错误"""
    pass


class RateLimitExceededError(BaseCustomException):
    """限流超出错误"""
    pass


class ResourceNotFoundError(BaseCustomException):
    """资源未找到错误"""
    pass


class DuplicateResourceError(BaseCustomException):
    """重复资源错误"""
    pass


class SystemError(BaseCustomException):
    """系统错误"""
    pass


# 错误响应模板
class ErrorResponse:
    """标准化错误响应"""

    def __init__(
        self,
        error_code: str,
        message: str,
        details: Dict[str, Any] = None,
        request_id: str = None,
        timestamp: float = None
    ):
        self.error_code = error_code
        self.message = message
        self.details = details or {}
        self.request_id = request_id
        self.timestamp = timestamp

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "error": {
                "code": self.error_code,
                "message": self.message,
                "details": self.details,
                "request_id": self.request_id,
                "timestamp": self.timestamp
            }
        }


def create_error_response(
    error_code: str,
    message: str,
    status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR,
    details: Dict[str, Any] = None,
    request_id: str = None
) -> JSONResponse:
    """创建错误响应"""
    import time

    error_response = ErrorResponse(
        error_code=error_code,
        message=message,
        details=details,
        request_id=request_id,
        timestamp=time.time()
    )

    return JSONResponse(
        status_code=status_code,
        content=error_response.to_dict()
    )


def setup_exception_handlers(app: FastAPI) -> None:
    """设置全局异常处理器"""

    @app.exception_handler(BaseCustomException)
    async def custom_exception_handler(request: Request, exc: BaseCustomException):
        """自定义异常处理器"""
        logger.error(
            f"Custom exception: {exc.error_code}",
            message=exc.message,
            details=exc.details,
            request_id=getattr(request.state, "request_id", None),
            path=request.url.path,
            method=request.method,
            traceback=traceback.format_exc()
        )

        status_code = status.HTTP_500_INTERNAL_SERVER_ERROR

        # 根据异常类型设置状态码
        if isinstance(exc, ValidationError):
            status_code = status.HTTP_422_UNPROCESSABLE_ENTITY
        elif isinstance(exc, (AuthenticationError, AuthorizationError)):
            status_code = status.HTTP_401_UNAUTHORIZED
        elif isinstance(exc, ResourceNotFoundError):
            status_code = status.HTTP_404_NOT_FOUND
        elif isinstance(exc, DuplicateResourceError):
            status_code = status.HTTP_409_CONFLICT
        elif isinstance(exc, RateLimitExceededError):
            status_code = status.HTTP_429_TOO_MANY_REQUESTS
        elif isinstance(exc, (ValidationError, BusinessLogicError)):
            status_code = status.HTTP_400_BAD_REQUEST

        return create_error_response(
            error_code=exc.error_code,
            message=exc.message,
            status_code=status_code,
            details=exc.details,
            request_id=getattr(request.state, "request_id", None)
        )

    @app.exception_handler(StarletteHTTPException)
    async def http_exception_handler(request: Request, exc: StarletteHTTPException):
        """HTTP异常处理器"""
        logger.warning(
            f"HTTP exception: {exc.status_code}",
            message=exc.detail,
            request_id=getattr(request.state, "request_id", None),
            path=request.url.path,
            method=request.method
        )

        return create_error_response(
            error_code="HTTP_ERROR",
            message=exc.detail,
            status_code=exc.status_code,
            request_id=getattr(request.state, "request_id", None)
        )

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        """请求验证异常处理器"""
        logger.warning(
            "Validation error",
            errors=exc.errors(),
            request_id=getattr(request.state, "request_id", None),
            path=request.url.path,
            method=request.method
        )

        return create_error_response(
            error_code="VALIDATION_ERROR",
            message="请求参数验证失败",
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            details={"validation_errors": exc.errors()},
            request_id=getattr(request.state, "request_id", None)
        )

    @app.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception):
        """通用异常处理器"""
        logger.error(
            f"Unhandled exception: {type(exc).__name__}",
            message=str(exc),
            request_id=getattr(request.state, "request_id", None),
            path=request.url.path,
            method=request.method,
            traceback=traceback.format_exc()
        )

        # 在生产环境中，不暴露详细错误信息
        if settings.is_production():
            message = "服务器内部错误"
            details = {}
        else:
            message = str(exc)
            details = {"exception_type": type(exc).__name__}

        return create_error_response(
            error_code="INTERNAL_SERVER_ERROR",
            message=message,
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            details=details,
            request_id=getattr(request.state, "request_id", None)
        )


def safe_execute(func, default=None, error_message: str = None):
    """安全执行函数，捕获异常"""
    try:
        return func()
    except Exception as e:
        logger.error(f"Safe execute failed: {error_message or str(e)}")
        if default is not None:
            return default
        raise


async def safe_execute_async(func, default=None, error_message: str = None):
    """异步安全执行函数，捕获异常"""
    try:
        return await func()
    except Exception as e:
        logger.error(f"Safe execute async failed: {error_message or str(e)}")
        if default is not None:
            return default
        raise


class ErrorContext:
    """错误上下文管理器"""

    def __init__(self, operation: str, **context):
        self.operation = operation
        self.context = context
        self.logger = logging.getLogger(f"error_context.{operation}")

    def __enter__(self):
        self.logger.info(f"Starting {self.operation}", **self.context)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            self.logger.error(
                f"Error in {self.operation}",
                operation=self.operation,
                error_type=exc_type.__name__ if exc_type else None,
                error_message=str(exc_val) if exc_val else None,
                traceback=traceback.format_tb(exc_tb) if exc_tb else None,
                **self.context
            )
            return False  # 不抑制异常
        else:
            self.logger.info(f"Completed {self.operation}", **self.context)
            return True


def handle_database_error(func):
    """数据库错误装饰器"""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Database error in {func.__name__}: {str(e)}")
            raise DatabaseError(
                message=f"数据库操作失败: {str(e)}",
                error_code="DATABASE_ERROR",
                details={"function": func.__name__},
                cause=e
            )
    return wrapper


def handle_external_service_error(service_name: str):
    """外部服务错误装饰器"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.error(f"{service_name} error in {func.__name__}: {str(e)}")
                raise ExternalServiceError(
                    message=f"{service_name}服务调用失败: {str(e)}",
                    error_code=f"{service_name.upper()}_ERROR",
                    details={"service": service_name, "function": func.__name__},
                    cause=e
                )
        return wrapper
    return decorator