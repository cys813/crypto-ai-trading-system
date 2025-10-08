"""
后台任务模块

包含所有Celery后台任务的定义和配置。
"""

from .celery_app import celery_app

__all__ = ["celery_app"]