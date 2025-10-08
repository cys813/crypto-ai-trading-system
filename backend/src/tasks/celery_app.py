"""
Celery应用配置

配置Celery任务队列和后台任务处理。
"""

import os
from celery import Celery
from celery.schedules import crontab
from .tasks import news_tasks, trading_tasks, analysis_tasks

# Celery配置
CELERY_BROKER_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
CELERY_RESULT_BACKEND = os.getenv("REDIS_URL", "redis://localhost:6379/0")

# 创建Celery应用
celery_app = Celery(
    "crypto_trading",
    broker=CELERY_BROKER_URL,
    backend=CELERY_RESULT_BACKEND,
    include=[
        "src.tasks.news_tasks",
        "src.tasks.trading_tasks",
        "src.tasks.analysis_tasks",
        "src.tasks.monitoring_tasks"
    ]
)

# Celery配置
celery_app.conf.update(
    # 任务序列化
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,

    # 结果过期时间
    result_expires=3600,

    # 任务路由
    task_routes={
        "src.tasks.news_tasks.*": {"queue": "news"},
        "src.tasks.analysis_tasks.*": {"queue": "analysis"},
        "src.tasks.trading_tasks.*": {"queue": "trading"},
        "src.tasks.monitoring_tasks.*": {"queue": "monitoring"},
    },

    # 任务队列配置
    task_default_queue="default",
    task_queues={
        "default": {
            "exchange": "direct",
            "routing_key": "default",
        },
        "news": {
            "exchange": "direct",
            "routing_key": "news",
        },
        "analysis": {
            "exchange": "direct",
            "routing_key": "analysis",
        },
        "trading": {
            "exchange": "direct",
            "routing_key": "trading",
        },
        "monitoring": {
            "exchange": "direct",
            "routing_key": "monitoring",
        },
    },

    # Worker配置
    worker_prefetch_multiplier=1,
    worker_max_tasks_per_child=1000,
    worker_disable_rate_limits=False,

    # 任务优先级
    task_inherit_parent_priority=True,
    task_default_priority=5,
    worker_direct=True,

    # 重试配置
    task_acks_late=True,
    worker_prefetch_multiplier=1,
    task_reject_on_worker_lost=True,

    # 监控
    worker_send_task_events=True,
    task_send_sent_event=True,
)

# 定时任务配置 (Beat Schedule)
celery_app.conf.beat_schedule = {
    # 新闻收集任务 - 每15分钟执行一次
    "collect-news": {
        "task": "src.tasks.news_tasks.collect_news_periodically",
        "schedule": crontab(minute="*/15"),
        "options": {"queue": "news"},
    },

    # 技术分析任务 - 每5分钟执行一次
    "technical-analysis": {
        "task": "src.tasks.analysis_tasks.run_technical_analysis",
        "schedule": crontab(minute="*/5"),
        "options": {"queue": "analysis"},
    },

    # 策略生成任务 - 每10分钟执行一次
    "generate-strategies": {
        "task": "src.tasks.analysis_tasks.generate_trading_strategies",
        "schedule": crontab(minute="*/10"),
        "options": {"queue": "analysis"},
    },

    # 订单监控任务 - 每1分钟执行一次
    "monitor-orders": {
        "task": "src.tasks.trading_tasks.monitor_orders",
        "schedule": crontab(minute="*"),
        "options": {"queue": "trading"},
    },

    # 仓位监控任务 - 每30秒执行一次
    "monitor-positions": {
        "task": "src.tasks.trading_tasks.monitor_positions",
        "schedule": 30.0,
        "options": {"queue": "trading"},
    },

    # 系统健康检查 - 每5分钟执行一次
    "health-check": {
        "task": "src.tasks.monitoring_tasks.system_health_check",
        "schedule": crontab(minute="*/5"),
        "options": {"queue": "monitoring"},
    },

    # 清理过期数据 - 每天凌晨2点执行
    "cleanup-expired-data": {
        "task": "src.tasks.monitoring_tasks.cleanup_expired_data",
        "schedule": crontab(hour=2, minute=0),
        "options": {"queue": "monitoring"},
    },
}

# 导入任务模块以确保任务被注册
from . import news_tasks
from . import trading_tasks
from . import analysis_tasks
from . import monitoring_tasks

if __name__ == "__main__":
    celery_app.start()