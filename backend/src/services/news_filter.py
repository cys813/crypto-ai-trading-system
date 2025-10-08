"""
新闻过滤服务

负责过滤、排序和优化新闻数据。
"""

import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any, Set
import re

from ..models.news import NewsData
from ..core.logging import BusinessLogger

logger = logging.getLogger(__name__)
business_logger = BusinessLogger("news_filter")


class NewsFilter:
    """新闻过滤器"""

    def __init__(self):
        self.logger = logger
        self.business_logger = business_logger

        # 配置过滤参数
        self.min_relevance_score = 0.7
        self.max_news_age_days = 15
        self.max_news_count = 50

        # 可信新闻源
        self.credible_sources = {
            "coindesk", "cointelegraph", "theblock", "decrypt", "cryptonews",
            "reuters", "bloomberg", "coindesk markets"
        }

        # 垃圾新闻模式
        self.spam_patterns = [
            r".*click here.*",
            r".*buy now.*",
            r".*limited time.*offer.*",
            r".*guaranteed.*profit.*",
            r".*100%.*return.*"
        ]

    def filter_news(self, news_list: List[NewsData],
                     relevance_threshold: float = None,
                     max_count: int = None,
                     max_age_days: int = None) -> List[NewsData]:
        """过滤新闻数据"""
        threshold = relevance_threshold or self.min_relevance_score
        max_count = max_count or self.max_news_count
        max_age = max_age_days or self.max_news_age_days

        self.business_logger.log_system_event(
            event_type="news_filtering_started",
            severity="info",
            message=f"开始过滤新闻，参数: 相关性={threshold}, 最大数量={max_count}, 最大天数={max_age}"
        )

        try:
            filtered_news = []
            seen_hashes = set()
            cutoff_date = datetime.now() - timedelta(days=max_age)

            for news in news_list:
                # 1. 日期过滤
                if news.published_at and news.published_at < cutoff_date:
                    continue

                # 2. 相关性过滤
                if news.relevance_score and news.relevance_score < threshold:
                    continue

                # 3. 去重过滤
                if news.hash and news.hash in seen_hashes:
                    continue
                if news.hash:
                    seen_hashes.add(news.hash)

                # 4. 质量过滤
                if not self._is_high_quality(news):
                    continue

                # 5. 垃圾内容过滤
                if self._is_spam(news):
                    continue

                filtered_news.append(news)

            # 限制数量
            filtered_news = filtered_news[:max_count]

            self.business_logger.log_system_event(
                event_type="news_filtering_completed",
                severity="info",
                message=f"过滤完成，剩余 {len(filtered_news)} 条新闻",
                details={
                    "original_count": len(news_list),
                    "filtered_count": len(filtered_news),
                    "removed_duplicates": len(seen_hashes),
                    "removed_by_relevance": len(news_list) - len(filtered_news) - len(seen_hashes)
                }
            )

            return filtered_news

        except Exception as e:
            self.logger.error(f"新闻过滤失败: {e}")
            raise

    def sort_news(self, news_list: List[NewsData],
                  sort_by: str = "relevance") -> List[NewsData]:
        """排序新闻数据"""
        try:
            if sort_by == "relevance":
                return sorted(news_list, key=lambda x: (x.relevance_score or 0, x.published_at or datetime.min), reverse=True)
            elif sort_by == "date":
                return sorted(news_list, key=lambda x: x.published_at or datetime.min, reverse=True)
            elif sort_by == "source":
                # 按可信度排序
                return sorted(news_list, key=lambda x: (
                    1.0 if x.source.lower() in self.credible_sources else 0.5,
                    x.relevance_score or 0
                ), reverse=True)
            else:
                return news_list

        except Exception as e:
            self.logger.error(f"新闻排序失败: {e}")
            return news_list

    def group_news_by_source(self, news_list: List[NewsData]) -> Dict[str, List[NewsData]]:
        """按新闻源分组"""
        grouped = {}

        for news in news_list:
            source = news.source or "unknown"
            if source not in grouped:
                grouped[source] = []
            grouped[source].append(news)

        # 对每个源内的新闻进行排序
        for source in grouped:
            grouped[source] = self.sort_news(grouped[source])

        return grouped

    def get_top_news_by_source(self, news_list: List[NewsData],
                                top_per_source: int = 5) -> Dict[str, List[NewsData]]:
        """获取每个来源的热门新闻"""
        grouped = self.group_news_by_source(news_list)
        top_news = {}

        for source, news in grouped.items():
            top_news[source] = news[:top_per_source]

        return top_news

    def filter_by_keywords(self, news_list: List[NewsData],
                          keywords: List[str],
                          require_all: bool = False) -> List[NewsData]:
        """按关键词过滤新闻"""
        if not keywords:
            return news_list

        keywords_lower = [kw.lower() for kw in keywords]
        filtered_news = []

        for news in news_list:
            content = f"{news.title or ''} {news.content or ''}".lower()

            if require_all:
                # 要求包含所有关键词
                if all(kw in content for kw in keywords_lower):
                    filtered_news.append(news)
            else:
                # 要求包含任意关键词
                if any(kw in content for kw in keywords_lower):
                    filtered_news.append(news)

        return filtered_news

    def filter_by_sentiment(self, news_list: List[NewsData],
                           sentiment: str = None) -> List[NewsData]:
        """按情感过滤新闻"""
        if sentiment:
            return [news for news in news_list if news.sentiment == sentiment]
        return news_list

    def filter_by_source(self, news_list: List[NewsData],
                       sources: List[str] = None) -> List[NewsData]:
        """按新闻源过滤新闻"""
        if not sources:
            return news_list

        sources_lower = [source.lower() for source in sources]
        return [news for news in news_list if news.source.lower() in sources_lower]

    def get_news_statistics(self, news_list: List[News]) -> Dict[str, Any]:
        """获取新闻统计信息"""
        if not news_list:
            return {}

        stats = {
            "total_count": len(news_list),
            "date_range": {
                "earliest": min(news.published_at for news in news_list if news.published_at),
                "latest": max(news.published_at for news in news_list if news.published_at)
            },
            "sources": {},
            "sentiments": {},
            "average_relevance": 0.0,
            "high_relevance_count": 0,
            "quality_score": 0.0
        }

        # 统计新闻源
        source_counts = {}
        for news in news_list:
            source = news.source or "unknown"
            source_counts[source] = source_counts.get(source, 0) + 1
        stats["sources"] = source_counts

        # 统计情感分布
        sentiment_counts = {}
        for news in news_list:
            sentiment = news.sentiment or "unknown"
            sentiment_counts[sentiment] = sentiment_counts.get(sentiment, 0) + 1
        stats["sentiments"] = sentiment_counts

        # 计算平均相关性
        relevance_scores = [news.relevance_score for news in news_list if news.relevance_score]
        if relevance_scores:
            stats["average_relevance"] = sum(relevance_scores) / len(relevance_scores)
            stats["high_relevance_count"] = sum(1 for score in relevance_scores if score >= 0.8)

        # 计算质量分数
        quality_scores = []
        for news in news_list:
            score = 0.0
            if news.source and news.source.lower() in self.credible_sources:
                score += 0.3
            if news.content and len(news.content) > 100:
                score += 0.2
            if news.relevance_score and news.relevance_score >= 0.8:
                score += 0.3
            quality_scores.append(score)

        if quality_scores:
            stats["quality_score"] = sum(quality_scores) / len(quality_scores)

        return stats

    def _is_high_quality(self, news: NewsData) -> bool:
        """判断新闻是否为高质量"""
        # 检查来源可信度
        if news.source and news.source.lower() in self.credible_sources:
            source_score = 1.0
        else:
            source_score = 0.5

        # 检查内容质量
        if news.content:
            content_length = len(news.content)
            if content_length > 200:
                content_score = 1.0
            elif content_length > 50:
                content_score = 0.7
            else:
                content_score = 0.3
        else:
            content_score = 0.0

        # 检查标题质量
        if news.title:
            title_length = len(news.title)
            if 10 <= title_length <= 100:
                title_score = 1.0
            else:
                title_score = 0.5
        else:
            title_score = 0.0

        # 综合质量评分
        quality_score = (source_score + content_score + title_score) / 3

        return quality_score >= 0.6

    def _is_spam(self, news: NewsData) -> bool:
        """判断是否为垃圾内容"""
        content = f"{news.title or ''} {news.content or ''}"

        for pattern in self.spam_patterns:
            if re.search(pattern, content, re.IGNORECASE):
                return True

        # 检查可疑的URL模式
        if news.url:
            suspicious_patterns = [
                r".*\.bit\.ly/.*",
                r".*\.xyz/.*",
                r".*short\.link/.*"
            ]
            for pattern in suspicious_patterns:
                if re.search(pattern, news.url, re.IGNORECASE):
                    return True

        return False


# 便捷函数
def filter_news(news_list: List[NewsData], **kwargs) -> List[NewsData]:
    """过滤新闻的便捷函数"""
    filter_instance = NewsFilter()
    return filter_instance.filter_news(news_list, **kwargs)


def sort_news(news_list: List[NewsData], sort_by: str = "relevance") -> List[NewsData]:
    """排序新闻的便捷函数"""
    filter_instance = NewsFilter()
    return filter_instance.sort_news(news_list, sort_by=sort_by)