"""
做空策略分析专用日志记录模块

提供专门的做空策略分析日志记录功能，包括结构化日志、性能监控和审计日志。
"""

import logging
import json
import time
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, asdict
from enum import Enum

from .logging import BusinessLogger
from .cache import get_cache, CacheKeys

logger = logging.getLogger(__name__)


class ShortStrategyLogLevel(Enum):
    """做空策略日志级别"""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class ShortStrategyEventType(Enum):
    """做空策略事件类型"""
    ANALYSIS_STARTED = "short_analysis_started"
    ANALYSIS_COMPLETED = "short_analysis_completed"
    ANALYSIS_FAILED = "short_analysis_failed"
    VALIDATION_STARTED = "short_validation_started"
    VALIDATION_COMPLETED = "short_validation_completed"
    VALIDATION_FAILED = "short_validation_failed"
    LLM_CALL_STARTED = "short_llm_call_started"
    LLM_CALL_COMPLETED = "short_llm_call_completed"
    LLM_CALL_FAILED = "short_llm_call_failed"
    DATA_COLLECTION_STARTED = "short_data_collection_started"
    DATA_COLLECTION_COMPLETED = "short_data_collection_completed"
    DATA_COLLECTION_FAILED = "short_data_collection_failed"
    TECHNICAL_ANALYSIS_STARTED = "short_technical_analysis_started"
    TECHNICAL_ANALYSIS_COMPLETED = "short_technical_analysis_completed"
    STRATEGY_GENERATED = "short_strategy_generated"
    STRATEGY_ADJUSTED = "short_strategy_adjusted"
    RISK_ASSESSMENT = "short_risk_assessment"
    PERFORMANCE_EVALUATION = "short_performance_evaluation"


@dataclass
class ShortStrategyLogEntry:
    """做空策略日志条目"""
    timestamp: datetime
    event_type: ShortStrategyEventType
    symbol: str
    level: ShortStrategyLogLevel
    message: str
    details: Dict[str, Any]
    duration_ms: Optional[float] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    request_id: Optional[str] = None
    exchange: Optional[str] = None
    timeframe: Optional[str] = None


@dataclass
class ShortStrategyPerformanceMetrics:
    """做空策略性能指标"""
    symbol: str
    analysis_duration_ms: float
    validation_duration_ms: float
    llm_call_duration_ms: float
    data_collection_duration_ms: float
    technical_analysis_duration_ms: float
    total_duration_ms: float
    memory_usage_mb: Optional[float] = None
    cache_hit_rate: Optional[float] = None
    success: bool
    error_type: Optional[str] = None
    confidence_score: Optional[float] = None


class ShortStrategyLogger:
    """做空策略专用日志记录器"""

    def __init__(self):
        self.logger = logger
        self.business_logger = BusinessLogger("short_strategy_analyzer")
        self.performance_cache = {}
        self.active_sessions = {}

    def log_analysis_start(
        self,
        symbol: str,
        timeframe: str,
        analysis_period_days: int,
        confidence_threshold: float,
        max_position_size: float,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        request_id: Optional[str] = None
    ) -> str:
        """记录分析开始"""
        timestamp = datetime.utcnow()
        session_id = session_id or f"session_{timestamp.strftime('%Y%m%d_%H%M%S')}_{symbol}"

        # 记录会话开始
        self.active_sessions[session_id] = {
            "start_time": timestamp,
            "symbol": symbol,
            "timeframe": timeframe,
            "user_id": user_id,
            "request_id": request_id
        }

        log_entry = ShortStrategyLogEntry(
            timestamp=timestamp,
            event_type=ShortStrategyEventType.ANALYSIS_STARTED,
            symbol=symbol,
            level=ShortStrategyLogLevel.INFO,
            message=f"开始做空策略分析: {symbol} {timeframe}",
            details={
                "timeframe": timeframe,
                "analysis_period_days": analysis_period_days,
                "confidence_threshold": confidence_threshold,
                "max_position_size": max_position_size,
                "session_id": session_id
            },
            user_id=user_id,
            session_id=session_id,
            request_id=request_id
        )

        self._write_log(log_entry)

        # 记录到业务日志
        self.business_logger.log_system_event(
            event_type="short_strategy_analysis_started",
            severity="info",
            message=log_entry.message,
            details=log_entry.details
        )

        return session_id

    def log_analysis_complete(
        self,
        symbol: str,
        recommendation: str,
        confidence_score: float,
        entry_price: float,
        stop_loss_price: float,
        take_profit_price: float,
        position_size_percent: float,
        risk_level: str,
        session_id: Optional[str] = None,
        additional_details: Optional[Dict[str, Any]] = None
    ):
        """记录分析完成"""
        timestamp = datetime.utcnow()

        # 计算分析持续时间
        duration_ms = None
        if session_id and session_id in self.active_sessions:
            start_time = self.active_sessions[session_id]["start_time"]
            duration_ms = (timestamp - start_time).total_seconds() * 1000
            del self.active_sessions[session_id]

        details = {
            "recommendation": recommendation,
            "confidence_score": confidence_score,
            "entry_price": entry_price,
            "stop_loss_price": stop_loss_price,
            "take_profit_price": take_profit_price,
            "position_size_percent": position_size_percent,
            "risk_level": risk_level
        }

        if additional_details:
            details.update(additional_details)

        if duration_ms is not None:
            details["analysis_duration_ms"] = duration_ms

        log_entry = ShortStrategyLogEntry(
            timestamp=timestamp,
            event_type=ShortStrategyEventType.ANALYSIS_COMPLETED,
            symbol=symbol,
            level=ShortStrategyLogLevel.INFO,
            message=f"做空策略分析完成: {symbol} - {recommendation}",
            details=details,
            duration_ms=duration_ms,
            session_id=session_id
        )

        self._write_log(log_entry)

        # 记录到业务日志
        self.business_logger.log_system_event(
            event_type="short_strategy_analysis_completed",
            severity="info",
            message=log_entry.message,
            details=details
        )

    def log_analysis_error(
        self,
        symbol: str,
        error: Exception,
        stage: str = "analysis",
        session_id: Optional[str] = None,
        additional_context: Optional[Dict[str, Any]] = None
    ):
        """记录分析错误"""
        timestamp = datetime.utcnow()

        # 计算失败前的持续时间
        duration_ms = None
        if session_id and session_id in self.active_sessions:
            start_time = self.active_sessions[session_id]["start_time"]
            duration_ms = (timestamp - start_time).total_seconds() * 1000
            del self.active_sessions[session_id]

        details = {
            "error_type": type(error).__name__,
            "error_message": str(error),
            "stage": stage,
            "session_id": session_id
        }

        if additional_context:
            details.update(additional_context)

        if duration_ms is not None:
            details["failed_duration_ms"] = duration_ms

        log_entry = ShortStrategyLogEntry(
            timestamp=timestamp,
            event_type=ShortStrategyEventType.ANALYSIS_FAILED,
            symbol=symbol,
            level=ShortStrategyLogLevel.ERROR,
            message=f"做空策略分析失败: {symbol} - {stage}",
            details=details,
            duration_ms=duration_ms,
            session_id=session_id
        )

        self._write_log(log_entry)

        # 记录到业务日志
        self.business_logger.log_system_event(
            event_type="short_strategy_analysis_failed",
            severity="error",
            message=log_entry.message,
            details=details
        )

    def log_validation_result(
        self,
        symbol: str,
        validation_result: Dict[str, Any],
        validation_type: str = "request",
        session_id: Optional[str] = None
    ):
        """记录验证结果"""
        timestamp = datetime.utcnow()

        is_valid = validation_result.get("is_valid", False)
        errors = validation_result.get("errors", [])
        warnings = validation_result.get("warnings", [])
        risk_level = validation_result.get("risk_level", "unknown")

        details = {
            "validation_type": validation_type,
            "is_valid": is_valid,
            "errors_count": len(errors),
            "warnings_count": len(warnings),
            "risk_level": risk_level,
            "errors": errors,
            "warnings": warnings
        }

        if "confidence_adjustment" in validation_result:
            details["confidence_adjustment"] = validation_result["confidence_adjustment"]

        level = ShortStrategyLogLevel.WARNING if errors else ShortStrategyLogLevel.INFO
        event_type = (ShortStrategyEventType.VALIDATION_COMPLETED if is_valid
                     else ShortStrategyEventType.VALIDATION_FAILED)

        message = f"做空策略{validation_type}验证: {symbol} - {'通过' if is_valid else '失败'}"

        log_entry = ShortStrategyLogEntry(
            timestamp=timestamp,
            event_type=event_type,
            symbol=symbol,
            level=level,
            message=message,
            details=details,
            session_id=session_id
        )

        self._write_log(log_entry)

    def log_llm_call(
        self,
        symbol: str,
        model: str,
        provider: str,
        prompt_length: int,
        response_length: int,
        duration_ms: float,
        success: bool,
        error_message: Optional[str] = None,
        session_id: Optional[str] = None
    ):
        """记录LLM调用"""
        timestamp = datetime.utcnow()

        details = {
            "model": model,
            "provider": provider,
            "prompt_length": prompt_length,
            "response_length": response_length,
            "duration_ms": duration_ms,
            "success": success
        }

        if error_message:
            details["error_message"] = error_message

        event_type = (ShortStrategyEventType.LLM_CALL_COMPLETED if success
                     else ShortStrategyEventType.LLM_CALL_FAILED)
        level = ShortStrategyLogLevel.INFO if success else ShortStrategyLogLevel.ERROR

        message = f"LLM调用: {symbol} - {model} ({'成功' if success else '失败'})"

        log_entry = ShortStrategyLogEntry(
            timestamp=timestamp,
            event_type=event_type,
            symbol=symbol,
            level=level,
            message=message,
            details=details,
            duration_ms=duration_ms,
            session_id=session_id
        )

        self._write_log(log_entry)

    def log_performance_metrics(
        self,
        metrics: ShortStrategyPerformanceMetrics
    ):
        """记录性能指标"""
        timestamp = datetime.utcnow()

        details = asdict(metrics)

        log_entry = ShortStrategyLogEntry(
            timestamp=timestamp,
            event_type=ShortStrategyEventType.PERFORMANCE_EVALUATION,
            symbol=metrics.symbol,
            level=ShortStrategyLogLevel.INFO,
            message=f"做空策略性能指标: {metrics.symbol}",
            details=details,
            duration_ms=metrics.total_duration_ms
        )

        self._write_log(log_entry)

        # 缓存性能指标用于监控
        try:
            cache = get_cache()
            cache_key = f"{CacheKeys.performance_metrics()}:short:{metrics.symbol}"
            cache.set(cache_key, details, ttl=3600)  # 1小时缓存
        except Exception as e:
            logger.warning(f"缓存性能指标失败: {e}")

    def log_risk_assessment(
        self,
        symbol: str,
        risk_factors: List[str],
        overall_risk_level: str,
        risk_score: float,
        mitigation_strategies: List[str],
        session_id: Optional[str] = None
    ):
        """记录风险评估"""
        timestamp = datetime.utcnow()

        details = {
            "risk_factors": risk_factors,
            "overall_risk_level": overall_risk_level,
            "risk_score": risk_score,
            "mitigation_strategies": mitigation_strategies
        }

        log_entry = ShortStrategyLogEntry(
            timestamp=timestamp,
            event_type=ShortStrategyEventType.RISK_ASSESSMENT,
            symbol=symbol,
            level=ShortStrategyLogLevel.INFO,
            message=f"做空策略风险评估: {symbol} - {overall_risk_level}",
            details=details,
            session_id=session_id
        )

        self._write_log(log_entry)

    def log_strategy_adjustment(
        self,
        symbol: str,
        original_recommendation: str,
        adjusted_recommendation: str,
        adjustment_reason: str,
        confidence_change: float,
        session_id: Optional[str] = None
    ):
        """记录策略调整"""
        timestamp = datetime.utcnow()

        details = {
            "original_recommendation": original_recommendation,
            "adjusted_recommendation": adjusted_recommendation,
            "adjustment_reason": adjustment_reason,
            "confidence_change": confidence_change
        }

        log_entry = ShortStrategyLogEntry(
            timestamp=timestamp,
            event_type=ShortStrategyEventType.STRATEGY_ADJUSTED,
            symbol=symbol,
            level=ShortStrategyLogLevel.WARNING,
            message=f"做空策略调整: {symbol} - {original_recommendation} -> {adjusted_recommendation}",
            details=details,
            session_id=session_id
        )

        self._write_log(log_entry)

    def _write_log(self, log_entry: ShortStrategyLogEntry):
        """写入日志"""
        try:
            # 结构化日志格式
            log_data = {
                "timestamp": log_entry.timestamp.isoformat(),
                "event_type": log_entry.event_type.value,
                "symbol": log_entry.symbol,
                "level": log_entry.level.value,
                "message": log_entry.message,
                "details": log_entry.details
            }

            if log_entry.duration_ms is not None:
                log_data["duration_ms"] = log_entry.duration_ms

            if log_entry.user_id:
                log_data["user_id"] = log_entry.user_id

            if log_entry.session_id:
                log_data["session_id"] = log_entry.session_id

            if log_entry.request_id:
                log_data["request_id"] = log_entry.request_id

            # 根据日志级别写入不同的日志
            log_message = json.dumps(log_data, ensure_ascii=False)

            if log_entry.level == ShortStrategyLogLevel.DEBUG:
                self.logger.debug(log_message)
            elif log_entry.level == ShortStrategyLogLevel.INFO:
                self.logger.info(log_message)
            elif log_entry.level == ShortStrategyLogLevel.WARNING:
                self.logger.warning(log_message)
            elif log_entry.level == ShortStrategyLogLevel.ERROR:
                self.logger.error(log_message)
            elif log_entry.level == ShortStrategyLogLevel.CRITICAL:
                self.logger.critical(log_message)

        except Exception as e:
            self.logger.error(f"写入做空策略日志失败: {e}")

    def get_recent_logs(
        self,
        symbol: Optional[str] = None,
        event_type: Optional[ShortStrategyEventType] = None,
        level: Optional[ShortStrategyLogLevel] = None,
        hours: int = 24,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """获取最近的日志记录"""
        # TODO: 实现从数据库或日志文件查询日志的逻辑
        # 这里返回空列表作为占位符
        return []

    def get_performance_summary(
        self,
        symbol: Optional[str] = None,
        hours: int = 24
    ) -> Dict[str, Any]:
        """获取性能摘要"""
        try:
            cache = get_cache()

            # 尝试从缓存获取
            cache_key = f"{CacheKeys.performance_summary()}:short:{hours}h"
            if symbol:
                cache_key += f":{symbol}"

            cached_summary = cache.get(cache_key)
            if cached_summary:
                return cached_summary

            # TODO: 实现性能统计逻辑
            summary = {
                "total_analyses": 0,
                "successful_analyses": 0,
                "failed_analyses": 0,
                "average_duration_ms": 0,
                "success_rate": 0.0,
                "common_errors": [],
                "performance_by_symbol": {},
                "period_hours": hours
            }

            # 缓存结果
            cache.set(cache_key, summary, ttl=300)  # 5分钟缓存

            return summary

        except Exception as e:
            logger.error(f"获取性能摘要失败: {e}")
            return {
                "error": str(e),
                "period_hours": hours
            }


# 全局实例
short_strategy_logger = ShortStrategyLogger()


# 便捷函数
def log_short_strategy_start(
    symbol: str,
    timeframe: str,
    analysis_period_days: int,
    confidence_threshold: float,
    max_position_size: float,
    **kwargs
) -> str:
    """记录做空策略开始的便捷函数"""
    return short_strategy_logger.log_analysis_start(
        symbol, timeframe, analysis_period_days,
        confidence_threshold, max_position_size, **kwargs
    )


def log_short_strategy_complete(
    symbol: str,
    recommendation: str,
    confidence_score: float,
    entry_price: float,
    stop_loss_price: float,
    take_profit_price: float,
    position_size_percent: float,
    risk_level: str,
    **kwargs
):
    """记录做空策略完成的便捷函数"""
    short_strategy_logger.log_analysis_complete(
        symbol, recommendation, confidence_score,
        entry_price, stop_loss_price, take_profit_price,
        position_size_percent, risk_level, **kwargs
    )


def log_short_strategy_error(
    symbol: str,
    error: Exception,
    stage: str = "analysis",
    **kwargs
):
    """记录做空策略错误的便捷函数"""
    short_strategy_logger.log_analysis_error(symbol, error, stage, **kwargs)