"""
新闻收集相关任务

处理新闻数据的收集、处理和摘要生成。
"""

import logging
import asyncio
from datetime import datetime, timedelta
from celery import current_task
from typing import List, Dict, Any

from .celery_app import celery_app
from ..core.cache import get_cache, CacheKeys
from ..core.database import SessionLocal
from ..core.logging import BusinessLogger
from ..core.exceptions import ExternalAPIError, LLMError
from ..services.news_collector import NewsCollector
from ..services.news_filter import NewsFilter
from ..services.llm_news_summarizer import LLMNewsSummarizer
from ..models.news import NewsData, NewsSummary

logger = logging.getLogger(__name__)
business_logger = BusinessLogger("news_tasks")


@celery_app.task(bind=True, name="src.tasks.news_tasks.collect_news_periodically")
def collect_news_periodically(self):
    """定期收集新闻任务"""
    task_id = self.request.id
    try:
        logger.info(f"开始执行定期新闻收集任务: {task_id}")

        # 获取缓存以检查上一次收集时间
        cache = get_cache()
        last_collection_key = CacheKeys.news_last_collection()
        last_collection = cache.get(last_collection_key)

        # 避免频繁收集（至少间隔30分钟）
        if last_collection:
            last_time = datetime.fromisoformat(last_collection)
            if datetime.utcnow() - last_time < timedelta(minutes=30):
                logger.info("距离上次收集时间不足30分钟，跳过本次收集")
                return {"status": "skipped", "reason": "too_frequent"}

        # 运行异步收集任务
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            result = loop.run_until_complete(_collect_news_async(task_id))
        finally:
            loop.close()

        # 更新最后收集时间
        cache.set(last_collection_key, datetime.utcnow().isoformat(), ttl=3600)

        # 记录业务日志
        business_logger.log_system_event(
            event_type="periodic_news_collection_completed",
            severity="info",
            message=f"定期新闻收集完成: {task_id}",
            details=result
        )

        logger.info(f"新闻收集任务完成: {task_id}")
        return {"status": "success", "task_id": task_id, **result}

    except ExternalAPIError as e:
        logger.error(f"外部API错误，新闻收集失败: {e}")
        business_logger.log_system_event(
            event_type="news_collection_api_error",
            severity="error",
            message=f"外部API错误: {str(e)}",
            details={"task_id": task_id}
        )
        raise self.retry(exc=e, countdown=300, max_retries=3)  # 5分钟后重试

    except Exception as e:
        logger.error(f"新闻收集任务失败: {e}")
        business_logger.log_system_event(
            event_type="periodic_news_collection_failed",
            severity="error",
            message=f"定期新闻收集失败: {str(e)}",
            details={"task_id": task_id, "error": str(e)}
        )
        raise self.retry(exc=e, countdown=60, max_retries=3)


async def _collect_news_async(task_id: str) -> Dict[str, Any]:
    """异步新闻收集逻辑"""
    db = SessionLocal()

    try:
        # 初始化服务
        collector = NewsCollector()
        news_filter = NewsFilter()
        summarizer = LLMNewsSummarizer()

        # 收集新闻（默认回溯1天，最多100条）
        raw_news = await collector.collect_news(days_back=1, max_items=100)
        logger.info(f"收集到原始新闻: {len(raw_news)} 条")

        # 过滤新闻
        filtered_news = news_filter.filter_news(
            raw_news,
            relevance_threshold=0.7,
            max_count=100,
            max_age_days=1
        )
        logger.info(f"过滤后新闻: {len(filtered_news)} 条")

        # 保存到数据库
        saved_news = []
        for news in filtered_news:
            # 检查是否已存在
            existing = db.query(NewsData).filter_by(hash=news.hash).first()
            if not existing:
                db.add(news)
                saved_news.append(news)

        db.commit()
        logger.info(f"新增新闻: {len(saved_news)} 条")

        # 生成摘要
        summary_count = 0
        if saved_news:
            summaries = await summarizer.generate_batch_summaries(saved_news)
            for summary in summaries:
                db.add(summary)
            summary_count = len(summaries)
            db.commit()
            logger.info(f"生成摘要: {summary_count} 条")

        return {
            "collected_count": len(raw_news),
            "filtered_count": len(filtered_news),
            "saved_count": len(saved_news),
            "summary_count": summary_count,
            "processing_time": datetime.utcnow().isoformat()
        }

    except Exception as e:
        db.rollback()
        raise
    finally:
        db.close()


@celery_app.task(bind=True, name="src.tasks.news_tasks.process_news_item")
def process_news_item(self, news_id: str):
    """处理单条新闻"""
    try:
        logger.info(f"开始处理新闻: {news_id}")

        db = SessionLocal()

        try:
            # 获取新闻数据
            news = db.query(NewsData).filter(NewsData.id == news_id).first()
            if not news:
                logger.warning(f"新闻不存在: {news_id}")
                return {"status": "error", "message": "新闻不存在"}

            # 检查是否已处理
            if news.sentiment and news.relevance_score:
                logger.info(f"新闻已处理: {news_id}")
                return {"status": "skipped", "message": "新闻已处理"}

            # 初始化服务
            news_filter = NewsFilter()
            summarizer = LLMNewsSummarizer()

            # 运行异步处理
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            try:
                # 生成摘要
                summary = await summarizer.generate_single_summary(news)
                if summary:
                    db.add(summary)

                    # 更新新闻数据
                    news.sentiment = summary.sentiment
                    news.relevance_score = summary.confidence_score

                    db.commit()

                    logger.info(f"新闻处理完成: {news_id}")

                    business_logger.log_system_event(
                        event_type="news_item_processed",
                        severity="info",
                        message=f"单条新闻处理完成: {news.title[:50]}...",
                        details={
                            "news_id": news_id,
                            "sentiment": summary.sentiment,
                            "confidence": summary.confidence_score
                        }
                    )

                    return {
                        "status": "success",
                        "news_id": news_id,
                        "sentiment": summary.sentiment,
                        "confidence": summary.confidence_score
                    }
                else:
                    logger.warning(f"摘要生成失败: {news_id}")
                    return {"status": "error", "message": "摘要生成失败"}

            finally:
                loop.close()

        finally:
            db.close()

    except Exception as e:
        logger.error(f"新闻处理失败: {e}")
        business_logger.log_system_event(
            event_type="news_item_processing_failed",
            severity="error",
            message=f"单条新闻处理失败: {str(e)}",
            details={"news_id": news_id, "error": str(e)}
        )
        raise self.retry(exc=e, countdown=30, max_retries=3)


@celery_app.task(bind=True, name="src.tasks.news_tasks.generate_news_summary")
def generate_news_summary(self, news_ids: List[str] = None, time_period_hours: int = 24):
    """生成新闻摘要和市场概览"""
    try:
        task_id = self.request.id
        logger.info(f"开始生成新闻摘要，任务ID: {task_id}")

        db = SessionLocal()

        try:
            # 如果没有指定新闻ID，则根据时间范围获取新闻
            if not news_ids:
                cutoff_time = datetime.utcnow() - timedelta(hours=time_period_hours)
                news_list = db.query(NewsData)\
                    .filter(NewsData.published_at >= cutoff_time)\
                    .filter(NewsData.relevance_score >= 0.7)\
                    .order_by(NewsData.relevance_score.desc())\
                    .limit(50)\
                    .all()
            else:
                # 根据ID列表获取新闻
                news_list = db.query(NewsData)\
                    .filter(NewsData.id.in_(news_ids))\
                    .all()

            if not news_list:
                logger.warning("没有找到相关新闻生成摘要")
                return {"status": "error", "message": "没有找到相关新闻"}

            logger.info(f"基于 {len(news_list)} 条新闻生成摘要")

            # 运行异步摘要生成
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            try:
                summarizer = LLMNewsSummarizer()

                # 生成批量摘要
                summaries = await summarizer.generate_batch_summaries(news_list)
                logger.info(f"生成 {len(summaries)} 条摘要")

                # 保存摘要到数据库
                for summary in summaries:
                    db.add(summary)
                db.commit()

                # 生成市场概览
                market_overview = await summarizer.generate_market_overview(news_list)

                # 缓存市场概览
                cache = get_cache()
                overview_key = CacheKeys.news_market_overview()
                if market_overview:
                    cache.set(overview_key, market_overview, ttl=1800)  # 30分钟缓存

                result = {
                    "status": "success",
                    "task_id": task_id,
                    "news_count": len(news_list),
                    "summary_count": len(summaries),
                    "market_overview": market_overview,
                    "processing_time": datetime.utcnow().isoformat()
                }

                # 记录业务日志
                business_logger.log_system_event(
                    event_type="news_summary_generation_completed",
                    severity="info",
                    message=f"新闻摘要生成完成，处理 {len(news_list)} 条新闻",
                    details=result
                )

                return result

            finally:
                loop.close()

        finally:
            db.close()

    except LLMError as e:
        logger.error(f"LLM API错误，摘要生成失败: {e}")
        business_logger.log_system_event(
            event_type="summary_generation_llm_error",
            severity="error",
            message=f"LLM API错误: {str(e)}",
            details={"task_id": task_id}
        )
        raise self.retry(exc=e, countdown=300, max_retries=2)  # 5分钟后重试

    except Exception as e:
        logger.error(f"新闻摘要生成失败: {e}")
        business_logger.log_system_event(
            event_type="news_summary_generation_failed",
            severity="error",
            message=f"新闻摘要生成失败: {str(e)}",
            details={"task_id": task_id, "error": str(e)}
        )
        raise self.retry(exc=e, countdown=60, max_retries=3)


@celery_app.task(bind=True, name="src.tasks.news_tasks.cleanup_old_news")
def cleanup_old_news(self, days_to_keep: int = 30):
    """清理旧新闻数据"""
    try:
        logger.info(f"开始清理 {days_to_keep} 天前的新闻数据")

        db = SessionLocal()

        try:
            cutoff_date = datetime.utcnow() - timedelta(days=days_to_keep)

            # 删除旧的新闻摘要
            deleted_summaries = db.query(NewsSummary)\
                .filter(NewsSummary.processing_time < cutoff_date)\
                .delete()

            # 删除旧新闻数据
            deleted_news = db.query(NewsData)\
                .filter(NewsData.published_at < cutoff_date)\
                .delete()

            db.commit()

            logger.info(f"清理完成，删除 {deleted_news} 条新闻，{deleted_summaries} 条摘要")

            business_logger.log_system_event(
                event_type="news_cleanup_completed",
                severity="info",
                message=f"旧新闻数据清理完成",
                details={
                    "deleted_news": deleted_news,
                    "deleted_summaries": deleted_summaries,
                    "cutoff_date": cutoff_date.isoformat()
                }
            )

            return {
                "status": "success",
                "deleted_news": deleted_news,
                "deleted_summaries": deleted_summaries,
                "cutoff_date": cutoff_date.isoformat()
            }

        finally:
            db.close()

    except Exception as e:
        logger.error(f"清理旧新闻数据失败: {e}")
        raise self.retry(exc=e, countdown=300, max_retries=1)  # 只重试一次