"""
新闻收集服务

负责从各种新闻源收集加密货币相关新闻数据。
"""

import asyncio
import logging
import hashlib
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import aiohttp
import feedparser

from ..core.cache import get_cache, CacheKeys
from ..core.config import settings
from ..core.exceptions import ExternalServiceError, BusinessLogicError, NewsAPIError, NewsValidationError
from ..models.news import NewsData
from ..core.logging import BusinessLogger
from ..core.news_validation import NewsValidator
from ..core.news_events import get_event_tracker, NewsEventType, EventSeverity

logger = logging.getLogger(__name__)
business_logger = BusinessLogger("news_collector")


class NewsCollector:
    """新闻收集器"""

    def __init__(self):
        self.logger = logger
        self.business_logger = business_logger
        self.cache = get_cache()

        # 配置新闻源
        self.news_sources = self._load_news_sources()

        # 初始化验证器
        self.validator = NewsValidator()

        # 初始化事件追踪器
        self.event_tracker = get_event_tracker()

    def _load_news_sources(self) -> List[Dict[str, Any]]:
        """加载新闻源配置"""
        # 默认新闻源配置
        default_sources = [
            {
                "name": "CoinDesk",
                "type": "rss",
                "url": "https://www.coindesk.com/arc/outbound-feeds/rss",
                "priority": 1,
                "category": "crypto"
            },
            {
                "name": "Cointelegraph",
                "type": "rss",
                "url": "https://cointelegraph.com/rss",
                "priority": 2,
                "category": "crypto"
            },
            {
                "name": "The Block",
                "type": "rss",
                "url": "https://www.theblock.co/rss/",
                "priority": 3,
                "category": "crypto"
            },
            {
                "name": "Decrypt",
                "type": "rss",
                "url": "https://decrypt.co/feed",
                "priority": 4,
                "category": "crypto"
            },
            {
                "name": "CryptoPanic",
                "type": "api",
                "url": "https://cryptopanic.com/api/v1/posts/",
                "priority": 5,
                "category": "crypto",
                "api_key": getattr(settings, "CRYPTOPANIC_API_KEY", None)
            }
        ]

        return default_sources

    async def collect_news(self, days_back: int = 15, max_items: int = 50) -> List[NewsData]:
        """收集新闻数据"""
        import uuid
        task_id = str(uuid.uuid4())
        start_time = datetime.utcnow()

        # 追踪收集开始事件
        source_names = [source["name"] for source in self.news_sources]
        self.event_tracker.track_collection_start(
            task_id=task_id,
            sources=source_names,
            days_back=days_back,
            max_items=max_items
        )

        try:
            # 检查缓存
            cache_key = f"{CacheKeys.NEWS_DATA}:latest:{days_back}:{max_items}"
            cached_news = await self.cache.get(cache_key)
            if cached_news:
                self.logger.info(f"从缓存获取到 {len(cached_news)} 条新闻")
                # 追踪缓存命中事件
                self.event_tracker.track_event(
                    event_type=NewsEventType.CACHE_HIT,
                    message=f"新闻缓存命中，获取 {len(cached_news)} 条新闻",
                    severity=EventSeverity.DEBUG,
                    details={"cache_key": cache_key, "news_count": len(cached_news)},
                    task_id=task_id
                )
                return [NewsData(**news) for news in cached_news]

            # 追踪缓存未命中事件
            self.event_tracker.track_event(
                event_type=NewsEventType.CACHE_MISS,
                message="新闻缓存未命中，开始实时收集",
                severity=EventSeverity.DEBUG,
                details={"cache_key": cache_key},
                task_id=task_id
            )

            # 并行收集新闻
            all_news = []
            tasks = []

            for source in self.news_sources:
                task = self._collect_from_source(source, days_back, max_items // len(self.news_sources))
                tasks.append(task)

            # 并行执行所有收集任务
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # 处理结果
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    self.logger.error(f"从 {self.news_sources[i]['name']} 收集新闻失败: {result}")
                else:
                    all_news.extend(result)

            # 过滤和排序新闻
            filtered_news = await self._filter_news(all_news, days_back)
            sorted_news = sorted(filtered_news, key=lambda x: (x.relevance_score, x.published_at), reverse=True)[:max_items]

            # 缓存结果
            news_dicts = [self._news_to_dict(news) for news in sorted_news]
            await self.cache.set(cache_key, news_dicts, ttl=1800)  # 30分钟缓存

            self.business_logger.log_system_event(
                event_type="news_collection_completed",
                severity="info",
                message=f"成功收集 {len(sorted_news)} 条新闻",
                details={"total_collected": len(all_news), "filtered_count": len(sorted_news)}
            )

            return sorted_news

        except Exception as e:
            self.logger.error(f"新闻收集失败: {e}")
            raise BusinessLogicError(
                message=f"新闻收集过程中发生错误: {str(e)}",
                error_code="NEWS_COLLECTION_FAILED",
                cause=e
            )

    async def _collect_from_source(self, source: Dict[str, Any], days_back: int, max_items: int) -> List[NewsData]:
        """从单个新闻源收集新闻"""
        try:
            source_name = source["name"]
            source_type = source["type"]

            self.logger.info(f"开始从 {source_name} 收集新闻 (类型: {source_type})")

            if source_type == "rss":
                news_items = await self._collect_from_rss(source, days_back, max_items)
            elif source_type == "api":
                news_items = await self._collect_from_api(source, days_back, max_items)
            else:
                self.logger.warning(f"不支持的新闻源类型: {source_type}")
                return []

            self.business_logger.log_system_event(
                event_type="source_collection_completed",
                severity="info",
                message=f"从 {source_name} 收集到 {len(news_items)} 条新闻"
            )

            return news_items

        except Exception as e:
            self.logger.error(f"从 {source['name']} 收集新闻失败: {e}")
            return []

    async def _collect_from_rss(self, source: Dict[str, Any], days_back: int, max_items: int) -> List[NewsData]:
        """从RSS源收集新闻"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(source["url"], timeout=30) as response:
                    response.raise_for_status()

                    # 解析RSS feed
                    feed = feedparser.parse(response.content)
                    news_items = []

                    cutoff_date = datetime.now() - timedelta(days=days_back)

                    for entry in feed.entries:
                        # 检查日期范围
                        if hasattr(entry, 'published_parsed'):
                            pub_date = entry.published_parsed
                        elif hasattr(entry, 'updated_parsed'):
                            pub_date = entry.updated_parsed
                        else:
                            continue

                        if pub_date < cutoff_date:
                            continue

                        # 准备新闻数据
                        news_data = {
                            "title": entry.title or "",
                            "content": self._extract_content(entry),
                            "url": entry.get("link", ""),
                            "source": source["name"],
                            "author": entry.get("author", ""),
                            "published_at": pub_date
                        }

                        # 验证新闻数据
                        is_valid, errors = self.validator.validate_news_data(news_data)

                        # 追踪验证事件
                        news_hash = self.validator.generate_content_hash(
                            news_data["title"], news_data["content"], news_data["source"]
                        )
                        self.event_tracker.track_validation_result(
                            news_id=news_hash,
                            title=news_data["title"],
                            is_valid=is_valid,
                            errors=errors
                        )

                        if not is_valid:
                            self.logger.warning(f"新闻验证失败: {entry.title[:50]}..., 错误: {errors}")
                            continue

                        # 清洗和标准化数据
                        sanitized_news = self.validator.sanitize_news_data(news_data)

                        # 创建新闻项
                        news_item = NewsData(
                            title=sanitized_news["title"],
                            content=sanitized_news["content"],
                            url=sanitized_news.get("url", ""),
                            source=sanitized_news["source"],
                            author=sanitized_news.get("author", ""),
                            published_at=sanitized_news["published_at"],
                            relevance_score=0.8,  # 默认相关性，后续会调整
                            sentiment="neutral",
                            hash=sanitized_news.get("hash", "")
                        )

                        # 计算相关性分数
                        news_item.relevance_score = await self._calculate_relevance_score(news_item)

                        # 分析情感
                        news_item.sentiment = await self._analyze_sentiment(news_item.content)

                        # 如果验证后的相关性分数太低，跳过
                        if news_item.relevance_score < 0.5:
                            self.logger.debug(f"新闻相关性分数过低: {news_item.title[:50]}...")
                            continue

                        news_items.append(news_item)

                        if len(news_items) >= max_items:
                            break

                    return news_items

        except Exception as e:
            raise ExternalServiceError(
                message=f"从RSS源 {source['name']} 收集新闻失败: {str(e)}",
                error_code="RSS_COLLECTION_FAILED",
                cause=e
            )

    async def _collect_from_api(self, source: Dict[str, Any], days_back: int, max_items: int) -> List[NewsData]:
        """从API源收集新闻"""
        # 这里可以扩展支持更多API源
        # 目前实现为空，可以添加具体的API调用逻辑
        return []

    def _extract_content(self, entry) -> str:
        """提取新闻内容"""
        content_parts = []

        if hasattr(entry, 'summary'):
            content_parts.append(entry.summary)
        if hasattr(entry, 'description'):
            content_parts.append(entry.description)
        if hasattr(entry, 'content'):
            content_parts.append(entry.content[0].value if entry.content else "")

        return " ".join(content_parts).strip()

    async def _calculate_relevance_score(self, news_item: NewsData) -> float:
        """计算新闻相关性分数"""
        score = 0.0

        # 关键词匹配
        crypto_keywords = [
            "bitcoin", "ethereum", "cryptocurrency", "blockchain", "crypto",
            "btc", "eth", "defi", "nft", "altcoin", "digital asset",
            "mining", "trading", "exchange", "wallet", "smart contract"
        ]

        title_lower = news_item.title.lower()
        content_lower = news_item.content.lower()

        for keyword in crypto_keywords:
            if keyword in title_lower:
                score += 0.1
            if keyword in content_lower:
                score += 0.05

        # 权威源加分
        authoritative_sources = ["coindesk", "cointelegraph", "theblock", "decrypt"]
        if any(source in news_item.source.lower() for source in authoritative_sources):
            score += 0.2

        # 内容长度加分
        content_length = len(news_item.content)
        if content_length > 100:
            score += 0.1
        elif content_length > 500:
            score += 0.2

        return min(score, 1.0)

    async def _analyze_sentiment(self, content: str) -> str:
        """分析新闻情感"""
        # 简单的情感分析实现
        # 实际项目中会使用更复杂的NLP模型

        positive_words = [
            "bullish", "surge", "rally", "gain", "profit", "growth", "positive",
            "optimistic", "breakthrough", "success", "strong", "improve"
        ]

        negative_words = [
            "bearish", "crash", "drop", "fall", "decline", "loss", "negative",
            "concern", "weak", "struggle", "risk", "volatility", "fear"
        ]

        content_lower = content.lower()
        positive_count = sum(1 for word in positive_words if word in content_lower)
        negative_count = sum(1 for word in negative_words if word in content_lower)

        if positive_count > negative_count * 1.5:
            return "positive"
        elif negative_count > positive_count * 1.5:
            return "negative"
        else:
            return "neutral"

    def _generate_hash(self, title: str, content: str, source: str) -> str:
        """生成内容哈希用于去重"""
        combined_content = f"{title}{content}{source}"
        return hashlib.md5(combined_content.encode('utf-8')).hexdigest()

    def _news_to_dict(self, news: NewsData) -> Dict[str, Any]:
        """将NewsData对象转换为字典"""
        return {
            "id": str(news.id),
            "title": news.title,
            "content": news.content,
            "url": news.url,
            "source": news.source,
            "author": news.author,
            "relevance_score": float(news.relevance_score) if news.relevance_score else 0.0,
            "sentiment": news.sentiment,
            "published_at": news.published_at.isoformat() if news.published_at else None,
            "collected_at": news.collected_at.isoformat() if news.collected_at else None,
            "hash": news.hash
        }

    async def _filter_news(self, news_list: List[NewsData], days_back: int) -> List[NewsData]:
        """过滤新闻数据"""
        cutoff_date = datetime.now() - timedelta(days=days_back)

        filtered_news = []
        seen_hashes = set()

        for news in news_list:
            # 检查日期范围
            if news.published_at and news.published_at < cutoff_date:
                continue

            # 检查相关性
            if news.relevance_score and news.relevance_score < 0.3:
                continue

            # 去重
            if news.hash and news.hash in seen_hashes:
                continue

            seen_hashes.add(news.hash)
            filtered_news.append(news)

        return filtered_news


# 便捷函数
async def collect_news(days_back: int = 15, max_items: int = 50) -> List[NewsData]:
    """收集新闻的便捷函数"""
    collector = NewsCollector()
    return await collector.collect_news(days_back, max_items)