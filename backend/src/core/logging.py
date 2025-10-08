"""
日志配置模块

提供结构化日志配置和管理功能。
"""

import logging
import logging.config
import sys
from pathlib import Path
from typing import Any, Dict
import structlog
from pythonjsonlogger import jsonlogger

from .config import settings


class ColoredFormatter(logging.Formatter):
    """彩色控制台日志格式化器"""

    # ANSI颜色代码
    COLORS = {
        'DEBUG': '\033[36m',      # 青色
        'INFO': '\033[32m',       # 绿色
        'WARNING': '\033[33m',    # 黄色
        'ERROR': '\033[31m',      # 红色
        'CRITICAL': '\033[35m',   # 紫色
        'RESET': '\033[0m'        # 重置
    }

    def format(self, record):
        # 添加颜色
        if hasattr(record, 'levelname'):
            record.levelname = (
                f"{self.COLORS.get(record.levelname, '')}"
                f"{record.levelname}"
                f"{self.COLORS['RESET']}"
            )

        return super().format(record)


def setup_logging():
    """设置日志配置"""

    # 日志配置字典
    config: Dict[str, Any] = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "console": {
                "()": ColoredFormatter,
                "format": (
                    "%(asctime)s - %(name)s - %(levelname)s - "
                    "%(message)s [%(pathname)s:%(lineno)d]"
                ),
                "datefmt": "%Y-%m-%d %H:%M:%S"
            },
            "json": {
                "()": jsonlogger.JsonFormatter,
                "format": (
                    "%(asctime)s %(name)s %(levelname)s %(message)s "
                    "%(pathname)s %(lineno)d %(process)d %(thread)d"
                ),
                "datefmt": "%Y-%m-%dT%H:%M:%S"
            },
            "detailed": {
                "()": ColoredFormatter,
                "format": (
                    "%(asctime)s - %(name)s - %(levelname)s - "
                    "%(module)s:%(funcName)s:%(lineno)d - %(message)s"
                ),
                "datefmt": "%Y-%m-%d %H:%M:%S"
            }
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "level": settings.LOG_LEVEL,
                "formatter": "console",
                "stream": sys.stdout
            },
            "file": {
                "class": "logging.handlers.RotatingFileHandler",
                "level": "INFO",
                "formatter": "json",
                "filename": "logs/app.log",
                "maxBytes": 10485760,  # 10MB
                "backupCount": 5,
                "encoding": "utf8"
            },
            "error_file": {
                "class": "logging.handlers.RotatingFileHandler",
                "level": "ERROR",
                "formatter": "json",
                "filename": "logs/error.log",
                "maxBytes": 10485760,  # 10MB
                "backupCount": 5,
                "encoding": "utf8"
            }
        },
        "loggers": {
            "": {  # root logger
                "level": settings.LOG_LEVEL,
                "handlers": ["console", "file", "error_file"],
                "propagate": False
            },
            "uvicorn": {
                "level": "INFO",
                "handlers": ["console", "file"],
                "propagate": False
            },
            "uvicorn.access": {
                "level": "INFO",
                "handlers": ["console", "file"],
                "propagate": False
            },
            "sqlalchemy.engine": {
                "level": "WARNING",
                "handlers": ["console", "file"],
                "propagate": False
            },
            "sqlalchemy.pool": {
                "level": "WARNING",
                "handlers": ["console", "file"],
                "propagate": False
            },
            "celery": {
                "level": "INFO",
                "handlers": ["console", "file"],
                "propagate": False
            },
            "redis": {
                "level": "WARNING",
                "handlers": ["console", "file"],
                "propagate": False
            }
        }
    }

    # 创建日志目录
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    # 应用配置
    logging.config.dictConfig(config)

    # 配置structlog
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer()
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """获取structlog日志器"""
    return structlog.get_logger(name)


# 为方便使用，创建一个默认日志器
logger = get_logger(__name__)


class RequestLogger:
    """请求日志记录器"""

    def __init__(self, logger_name: str = "request"):
        self.logger = get_logger(logger_name)

    async def log_request(self, request_id: str, method: str, path: str,
                         client_ip: str, user_id: str = None):
        """记录请求日志"""
        self.logger.info(
            "Request received",
            request_id=request_id,
            method=method,
            path=path,
            client_ip=client_ip,
            user_id=user_id
        )

    async def log_response(self, request_id: str, status_code: int,
                          process_time: float, response_size: int = None):
        """记录响应日志"""
        self.logger.info(
            "Response sent",
            request_id=request_id,
            status_code=status_code,
            process_time=process_time,
            response_size=response_size
        )

    async def log_error(self, request_id: str, error: Exception,
                       extra: dict = None):
        """记录错误日志"""
        self.logger.error(
            "Request error",
            request_id=request_id,
            error_type=type(error).__name__,
            error_message=str(error),
            extra=extra or {}
        )


class BusinessLogger:
    """业务日志记录器"""

    def __init__(self, component: str):
        self.logger = get_logger(f"business.{component}")

    def log_trading_action(self, action: str, symbol: str, order_id: str = None,
                          user_id: str = None, **kwargs):
        """记录交易操作日志"""
        self.logger.info(
            "Trading action",
            action=action,
            symbol=symbol,
            order_id=order_id,
            user_id=user_id,
            **kwargs
        )

    def log_analysis_result(self, analysis_type: str, symbol: str, result: dict,
                           confidence: float = None, **kwargs):
        """记录分析结果日志"""
        self.logger.info(
            "Analysis completed",
            analysis_type=analysis_type,
            symbol=symbol,
            result=result,
            confidence=confidence,
            **kwargs
        )

    def log_strategy_execution(self, strategy_id: str, status: str,
                              details: dict = None, **kwargs):
        """记录策略执行日志"""
        self.logger.info(
            "Strategy execution",
            strategy_id=strategy_id,
            status=status,
            details=details,
            **kwargs
        )

    def log_system_event(self, event_type: str, severity: str,
                         message: str, **kwargs):
        """记录系统事件日志"""
        log_method = getattr(self.logger, severity.lower(), self.logger.info)
        log_method(
            "System event",
            event_type=event_type,
            message=message,
            **kwargs
        )


# 预定义的日志器实例
request_logger = RequestLogger()
trading_logger = BusinessLogger("trading")
analysis_logger = BusinessLogger("analysis")
news_logger = BusinessLogger("news")
monitoring_logger = BusinessLogger("monitoring")