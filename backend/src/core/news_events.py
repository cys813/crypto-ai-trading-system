"""
新闻事件追踪模块

提供详细的新闻收集、处理和摘要生成事件追踪功能。
"""

import logging
import json
from datetime import datetime
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
from enum import Enum

from .logging import BusinessLogger
from .cache import get_cache, CacheKeys


class NewsEventType(Enum):
    """新闻事件类型"""
    COLLECTION_STARTED = "news_collection_started"
    COLLECTION_COMPLETED = "news_collection_completed"
    COLLECTION_FAILED = "news_collection_failed"
    VALIDATION_PASSED = "news_validation_passed"
    VALIDATION_FAILED = "news_validation_failed"
    FILTERING_STARTED = "news_filtering_started"
    FILTERING_COMPLETED = "news_filtering_completed"
    SUMMARY_GENERATION_STARTED = "summary_generation_started"
    SUMMARY_GENERATION_COMPLETED = "summary_generation_completed"
    SUMMARY_GENERATION_FAILED = "summary_generation_failed"
    CACHE_HIT = "news_cache_hit"
    CACHE_MISS = "news_cache_miss"
    DUPLICATE_DETECTED = "news_duplicate_detected"
    SENTIMENT_ANALYSIS_COMPLETED = "sentiment_analysis_completed"
    RELEVANCE_SCORING_COMPLETED = "relevance_scoring_completed"
    MARKET_OVERVIEW_GENERATED = "market_overview_generated"


class EventSeverity(Enum):
    """事件严重程度"""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class NewsEvent:
    """新闻事件"""
    event_type: NewsEventType
    severity: EventSeverity
    message: str
    timestamp: datetime
    details: Optional[Dict[str, Any]] = None
    news_id: Optional[str] = None
    source: Optional[str] = None
    task_id: Optional[str] = None
    duration_ms: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        data = asdict(self)
        data['event_type'] = self.event_type.value
        data['severity'] = self.severity.value
        data['timestamp'] = self.timestamp.isoformat()
        return data


class NewsEventTracker:
    """新闻事件追踪器"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.business_logger = BusinessLogger("news_events")
        self.cache = get_cache()

        # 事件缓存键前缀
        self.event_cache_prefix = "news_events:"

        # 性能统计
        self.performance_stats = {}

    def track_event(
        self,
        event_type: NewsEventType,
        message: str,
        severity: EventSeverity = EventSeverity.INFO,
        details: Optional[Dict[str, Any]] = None,
        news_id: Optional[str] = None,
        source: Optional[str] = None,
        task_id: Optional[str] = None,
        duration_ms: Optional[int] = None
    ) -> NewsEvent:
        """追踪新闻事件"""
        event = NewsEvent(
            event_type=event_type,
            severity=severity,
            message=message,
            timestamp=datetime.utcnow(),
            details=details or {},
            news_id=news_id,
            source=source,
            task_id=task_id,
            duration_ms=duration_ms
        )

        # 记录到业务日志
        self.business_logger.log_system_event(
            event_type=event_type.value,
            severity=severity.value,
            message=message,
            details=details or {}
        )

        # 缓存事件（最近1000个事件）
        self._cache_event(event)

        # 更新性能统计
        self._update_performance_stats(event)

        return event

    def track_collection_start(
        self,
        task_id: str,
        sources: List[str],
        days_back: int,
        max_items: int
    ) -> NewsEvent:
        """追踪新闻收集开始"""
        return self.track_event(
            event_type=NewsEventType.COLLECTION_STARTED,
            message=f"开始收集新闻，源: {sources}, 天数: {days_back}, 最大数量: {max_items}",
            severity=EventSeverity.INFO,
            details={
                "sources": sources,
                "days_back": days_back,
                "max_items": max_items
            },
            task_id=task_id
        )

    def track_collection_complete(
        self,
        task_id: str,
        collected_count: int,
        filtered_count: int,
        duration_ms: int,
        sources_summary: Dict[str, int]
    ) -> NewsEvent:
        """追踪新闻收集完成"""
        return self.track_event(
            event_type=NewsEventType.COLLECTION_COMPLETED,
            message=f"新闻收集完成，收集: {collected_count}, 过滤: {filtered_count}",
            severity=EventSeverity.INFO,
            details={
                "collected_count": collected_count,
                "filtered_count": filtered_count,
                "sources_summary": sources_summary,
                "collection_rate": collected_count / max(duration_ms / 1000, 1)  # 每秒收集数
            },
            task_id=task_id,
            duration_ms=duration_ms
        )

    def track_validation_result(
        self,
        news_id: str,
        title: str,
        is_valid: bool,
        errors: List[str] = None
    ) -> NewsEvent:
        """追踪新闻验证结果"""
        event_type = NewsEventType.VALIDATION_PASSED if is_valid else NewsEventType.VALIDATION_FAILED
        severity = EventSeverity.INFO if is_valid else EventSeverity.WARNING

        message = f"新闻验证{'通过' if is_valid else '失败'}: {title[:50]}..."

        return self.track_event(
            event_type=event_type,
            message=message,
            severity=severity,
            details={
                "news_id": news_id,
                "title": title,
                "validation_errors": errors or []
            },
            news_id=news_id
        )

    def track_summary_generation(
        self,
        news_id: str,
        title: str,
        success: bool,
        sentiment: str = None,
        confidence: float = None,
        error_message: str = None,
        duration_ms: int = None
    ) -> NewsEvent:
        """追踪摘要生成"""
        if success:
            event_type = NewsEventType.SUMMARY_GENERATION_COMPLETED
            severity = EventSeverity.INFO
            message = f"摘要生成成功: {title[:50]}..."
            details = {
                "sentiment": sentiment,
                "confidence_score": confidence,
                "processing_duration": duration_ms
            }
        else:
            event_type = NewsEventType.SUMMARY_GENERATION_FAILED
            severity = EventSeverity.ERROR
            message = f"摘要生成失败: {title[:50]}..."
            details = {
                "error_message": error_message,
                "processing_duration": duration_ms
            }

        return self.track_event(
            event_type=event_type,
            message=message,
            severity=severity,
            details=details,
            news_id=news_id,
            duration_ms=duration_ms
        )

    def track_market_overview(
        self,
        news_count: int,
        overall_sentiment: str,
        risk_assessment: str,
        key_themes: List[str],
        duration_ms: int = None
    ) -> NewsEvent:
        """追踪市场概览生成"""
        return self.track_event(
            event_type=NewsEventType.MARKET_OVERVIEW_GENERATED,
            message=f"市场概览生成完成，基于 {news_count} 条新闻",
            severity=EventSeverity.INFO,
            details={
                "news_count": news_count,
                "overall_sentiment": overall_sentiment,
                "risk_assessment": risk_assessment,
                "key_themes": key_themes,
                "processing_duration": duration_ms
            },
            duration_ms=duration_ms
        )

    def track_duplicate_detection(
        self,
        news_id: str,
        title: str,
        source: str,
        is_duplicate: bool,
        original_news_id: str = None
    ) -> NewsEvent:
        """追踪重复新闻检测"""
        if is_duplicate:
            message = f"检测到重复新闻: {title[:50]}..."
            details = {
                "original_news_id": original_news_id,
                "duplicate_source": source
            }
        else:
            message = f"新闻通过去重检查: {title[:50]}..."
            details = {}

        return self.track_event(
            event_type=NewsEventType.DUPLICATE_DETECTED,
            message=message,
            severity=EventSeverity.DEBUG,
            details=details,
            news_id=news_id,
            source=source
        )

    def get_recent_events(
        self,
        limit: int = 100,
        event_type: Optional[NewsEventType] = None,
        severity: Optional[EventSeverity] = None,
        hours: int = 24
    ) -> List[NewsEvent]:
        """获取最近的事件"""
        try:
            # 构建缓存键
            cache_key = f"{self.event_cache_prefix}recent"
            cached_events = self.cache.get(cache_key)

            if cached_events:
                events = [self._dict_to_event(event_data) for event_data in cached_events]
            else:
                events = []

            # 过滤事件
            filtered_events = []
            cutoff_time = datetime.utcnow() - timedelta(hours=hours)

            for event in events:
                # 时间过滤
                if event.timestamp < cutoff_time:
                    continue

                # 类型过滤
                if event_type and event.event_type != event_type:
                    continue

                # 严重程度过滤
                if severity and event.severity != severity:
                    continue

                filtered_events.append(event)

            # 按时间倒序排列并限制数量
            filtered_events.sort(key=lambda x: x.timestamp, reverse=True)
            return filtered_events[:limit]

        except Exception as e:
            self.logger.error(f"获取最近事件失败: {e}")
            return []

    def get_performance_stats(self) -> Dict[str, Any]:
        """获取性能统计"""
        return self.performance_stats.copy()

    def get_error_summary(self, hours: int = 24) -> Dict[str, Any]:
        """获取错误摘要"""
        error_events = self.get_recent_events(
            limit=1000,
            severity=EventSeverity.ERROR,
            hours=hours
        )

        error_counts = {}
        error_sources = {}

        for event in error_events:
            event_name = event.event_type.value
            error_counts[event_name] = error_counts.get(event_name, 0) + 1

            if event.source:
                error_sources[event.source] = error_sources.get(event.source, 0) + 1

        return {
            "total_errors": len(error_events),
            "error_types": error_counts,
            "error_sources": error_sources,
            "time_range_hours": hours
        }

    def _cache_event(self, event: NewsEvent) -> None:
        """缓存事件"""
        try:
            cache_key = f"{self.event_cache_prefix}recent"
            cached_events = self.cache.get(cache_key) or []

            # 添加新事件
            cached_events.append(event.to_dict())

            # 保持最近1000个事件
            if len(cached_events) > 1000:
                cached_events = cached_events[-1000:]

            # 更新缓存
            self.cache.set(cache_key, cached_events, ttl=3600)  # 1小时缓存

        except Exception as e:
            self.logger.error(f"缓存事件失败: {e}")

    def _update_performance_stats(self, event: NewsEvent) -> None:
        """更新性能统计"""
        if event.duration_ms is None:
            return

        event_type = event.event_type.value
        if event_type not in self.performance_stats:
            self.performance_stats[event_type] = {
                "count": 0,
                "total_duration_ms": 0,
                "avg_duration_ms": 0,
                "min_duration_ms": float('inf'),
                "max_duration_ms": 0
            }

        stats = self.performance_stats[event_type]
        stats["count"] += 1
        stats["total_duration_ms"] += event.duration_ms
        stats["avg_duration_ms"] = stats["total_duration_ms"] / stats["count"]
        stats["min_duration_ms"] = min(stats["min_duration_ms"], event.duration_ms)
        stats["max_duration_ms"] = max(stats["max_duration_ms"], event.duration_ms)

    def _dict_to_event(self, data: Dict[str, Any]) -> NewsEvent:
        """从字典创建事件对象"""
        return NewsEvent(
            event_type=NewsEventType(data["event_type"]),
            severity=EventSeverity(data["severity"]),
            message=data["message"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            details=data.get("details"),
            news_id=data.get("news_id"),
            source=data.get("source"),
            task_id=data.get("task_id"),
            duration_ms=data.get("duration_ms")
        )


# 全局事件追踪器实例
_event_tracker = None


def get_event_tracker() -> NewsEventTracker:
    """获取全局事件追踪器实例"""
    global _event_tracker
    if _event_tracker is None:
        _event_tracker = NewsEventTracker()
    return _event_tracker


# 便捷函数
def track_news_event(
    event_type: NewsEventType,
    message: str,
    severity: EventSeverity = EventSeverity.INFO,
    **kwargs
) -> NewsEvent:
    """追踪新闻事件的便捷函数"""
    tracker = get_event_tracker()
    return tracker.track_event(event_type, message, severity, **kwargs)