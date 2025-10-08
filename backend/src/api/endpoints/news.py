"""
新闻API端点

处理新闻收集、处理和摘要相关的API。
"""

from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
from fastapi import APIRouter, Depends, HTTPException, Query, BackgroundTasks
from pydantic import BaseModel
import asyncio

from ...models.news import NewsData, NewsSummary
from ...services.news_collector import NewsCollector
from ...services.news_filter import NewsFilter
from ...services.llm_news_summarizer import LLMNewsSummarizer
from ...core.database import get_db_session
from ...core.exceptions import ValidationError, ExternalAPIError
from ...core.logging import get_logger

logger = get_logger(__name__)
router = APIRouter()

# API响应模型
class NewsResponse(BaseModel):
    id: str
    title: str
    content: str
    source: str
    url: Optional[str]
    published_at: datetime
    relevance_score: Optional[float]
    sentiment: Optional[str]
    hash: Optional[str]

    class Config:
        from_attributes = True

class NewsSummaryResponse(BaseModel):
    news_id: str
    core_summary: str
    key_impact: str
    sentiment: str
    mentioned_cryptos: List[str]
    importance_level: str
    confidence_score: float
    processing_time: datetime
    llm_model: str

    class Config:
        from_attributes = True

class MarketOverviewResponse(BaseModel):
    overall_sentiment: str
    key_themes: List[str]
    market_impact: str
    top_mentioned_cryptos: List[str]
    risk_assessment: str
    news_count: int
    generated_at: datetime
    time_range: Dict[str, str]

class NewsCollectionResponse(BaseModel):
    task_id: str
    status: str
    message: str
    collected_count: Optional[int] = None
    summary_count: Optional[int] = None

@router.get("/news", response_model=List[NewsResponse])
async def get_news(
    limit: int = Query(50, le=100, description="最大返回数量"),
    days: int = Query(15, le=30, description="回溯天数"),
    source: Optional[str] = Query(None, description="新闻源过滤"),
    min_relevance: float = Query(0.7, ge=0.0, le=1.0, description="最小相关性分数"),
    sort_by: str = Query("relevance", regex="^(relevance|date|source)$", description="排序方式")
):
    """获取新闻列表"""
    try:
        # 获取数据库会话
        db_session = next(get_db_session())

        # 查询新闻数据
        query = db_session.query(NewsData)

        # 应用过滤条件
        if days:
            cutoff_date = datetime.utcnow() - timedelta(days=days)
            query = query.filter(NewsData.published_at >= cutoff_date)

        if source:
            query = query.filter(NewsData.source.ilike(f"%{source}%"))

        if min_relevance:
            query = query.filter(NewsData.relevance_score >= min_relevance)

        # 执行查询
        news_list = query.all()

        # 使用新闻过滤器进行排序和后处理
        news_filter = NewsFilter()

        # 排序
        if sort_by:
            news_list = news_filter.sort_news(news_list, sort_by=sort_by)

        # 限制数量
        news_list = news_list[:limit]

        # 转换为响应模型
        response_data = [NewsResponse.from_orm(news) for news in news_list]

        logger.info(f"获取新闻列表成功，返回 {len(response_data)} 条记录")

        return response_data

    except Exception as e:
        logger.error(f"获取新闻列表失败: {e}")
        raise HTTPException(status_code=500, detail="获取新闻列表失败")

@router.get("/news/summary", response_model=List[NewsSummaryResponse])
async def get_news_summaries(
    hours: int = Query(24, le=168, description="时间范围（小时）"),
    limit: int = Query(20, le=100, description="最大返回数量")
):
    """获取新闻摘要列表"""
    try:
        # 获取数据库会话
        db_session = next(get_db_session())

        # 计算时间范围
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)

        # 查询新闻摘要
        summaries = db_session.query(NewsSummary)\
            .filter(NewsSummary.processing_time >= cutoff_time)\
            .order_by(NewsSummary.processing_time.desc())\
            .limit(limit)\
            .all()

        # 转换为响应模型
        response_data = [NewsSummaryResponse.from_orm(summary) for summary in summaries]

        logger.info(f"获取新闻摘要成功，返回 {len(response_data)} 条记录")

        return response_data

    except Exception as e:
        logger.error(f"获取新闻摘要失败: {e}")
        raise HTTPException(status_code=500, detail="获取新闻摘要失败")


@router.get("/news/market-overview", response_model=MarketOverviewResponse)
async def get_market_overview(
    hours: int = Query(24, le=168, description="时间范围（小时）")
):
    """获取市场整体概览"""
    try:
        # 获取数据库会话
        db_session = next(get_db_session())

        # 计算时间范围
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)

        # 查询相关新闻
        news_list = db_session.query(NewsData)\
            .filter(NewsData.published_at >= cutoff_time)\
            .filter(NewsData.relevance_score >= 0.7)\
            .order_by(NewsData.relevance_score.desc())\
            .limit(50)\
            .all()

        if not news_list:
            raise HTTPException(status_code=404, detail="指定时间范围内没有相关新闻")

        # 生成市场概览
        summarizer = LLMNewsSummarizer()
        overview = await summarizer.generate_market_overview(news_list)

        if not overview:
            raise HTTPException(status_code=500, detail="生成市场概览失败")

        # 转换为响应模型
        response_data = MarketOverviewResponse(**overview)

        logger.info(f"生成市场概览成功，基于 {len(news_list)} 条新闻")

        return response_data

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"生成市场概览失败: {e}")
        raise HTTPException(status_code=500, detail="生成市场概览失败")

@router.post("/news/collect", response_model=NewsCollectionResponse)
async def trigger_news_collection(
    background_tasks: BackgroundTasks,
    days_back: int = Query(15, le=30, description="回溯天数"),
    max_items: int = Query(100, le=500, description="最大收集数量"),
    generate_summaries: bool = Query(True, description="是否生成摘要")
):
    """触发新闻收集任务"""
    try:
        import uuid
        task_id = str(uuid.uuid4())

        # 验证参数
        if days_back < 1 or max_items < 1:
            raise ValidationError("参数值必须大于0")

        # 启动后台任务
        background_tasks.add_task(
            _collect_news_background,
            task_id,
            days_back,
            max_items,
            generate_summaries
        )

        logger.info(f"新闻收集任务已启动: {task_id}")

        return NewsCollectionResponse(
            task_id=task_id,
            status="started",
            message="新闻收集任务已启动，正在后台处理"
        )

    except ValidationError as e:
        logger.warning(f"参数验证失败: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"启动新闻收集任务失败: {e}")
        raise HTTPException(status_code=500, detail="启动任务失败")


async def _collect_news_background(
    task_id: str,
    days_back: int,
    max_items: int,
    generate_summaries: bool
):
    """后台新闻收集任务"""
    try:
        logger.info(f"开始执行新闻收集任务: {task_id}")

        # 初始化服务
        collector = NewsCollector()
        news_filter = NewsFilter()
        summarizer = LLMNewsSummarizer() if generate_summaries else None

        # 收集新闻
        raw_news = await collector.collect_news(
            days_back=days_back,
            max_items=max_items
        )

        # 过滤新闻
        filtered_news = news_filter.filter_news(
            raw_news,
            relevance_threshold=0.7,
            max_count=max_items,
            max_age_days=days_back
        )

        # 保存到数据库
        db_session = next(get_db_session())
        saved_news = []

        for news in filtered_news:
            # 检查是否已存在
            existing = db_session.query(NewsData).filter_by(hash=news.hash).first()
            if not existing:
                db_session.add(news)
                saved_news.append(news)

        db_session.commit()

        # 生成摘要
        summary_count = 0
        if summarizer and saved_news:
            summaries = await summarizer.generate_batch_summaries(saved_news)
            for summary in summaries:
                db_session.add(summary)
            summary_count = len(summaries)

            db_session.commit()

        logger.info(
            f"新闻收集任务完成: {task_id}, "
            f"收集: {len(raw_news)}, "
            f"过滤: {len(filtered_news)}, "
            f"新增: {len(saved_news)}, "
            f"摘要: {summary_count}"
        )

    except Exception as e:
        logger.error(f"新闻收集任务失败 {task_id}: {e}")
        if 'db_session' in locals():
            db_session.rollback()


@router.get("/news/statistics")
async def get_news_statistics(
    hours: int = Query(24, le=168, description="时间范围（小时）")
):
    """获取新闻统计信息"""
    try:
        # 获取数据库会话
        db_session = next(get_db_session())

        # 计算时间范围
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)

        # 查询新闻数据
        news_list = db_session.query(NewsData)\
            .filter(NewsData.published_at >= cutoff_time)\
            .all()

        # 获取统计信息
        news_filter = NewsFilter()
        stats = news_filter.get_news_statistics(news_list)

        # 添加时间范围信息
        stats.update({
            "time_range_hours": hours,
            "query_time": datetime.utcnow().isoformat()
        })

        logger.info(f"获取新闻统计信息成功，基于 {len(news_list)} 条新闻")

        return stats

    except Exception as e:
        logger.error(f"获取新闻统计信息失败: {e}")
        raise HTTPException(status_code=500, detail="获取统计信息失败")