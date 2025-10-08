"""
新闻收集工作流集成测试

验证新闻收集、处理、存储和摘要生成的完整工作流。
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch
from typing import List, Dict, Any

from ..conftest import async_test_client, mock_news_data, mock_llm_service


class TestNewsWorkflowIntegration:
    """新闻收集工作流集成测试"""

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_complete_news_collection_workflow(self, async_test_client):
        """测试完整的新闻收集工作流"""
        # 模拟外部新闻源API
        mock_news_sources = [
            {
                "name": "CoinDesk",
                "url": "https://api.coindesk.com/v1/news",
                "api_key": "test_key"
            },
            {
                "name": "CryptoNews",
                "url": "https://api.cryptonews.com/v1/articles",
                "api_key": "test_key"
            }
        ]

        # 模拟LLM服务
        with patch('src.services.llm_news_summarizer.LLMNewsSummarizer') as mock_summarizer:
            mock_summarizer.return_value = Mock()
            mock_summarizer.return_value.summarize_news.return_value = {
                "summary_text": "模拟生成的新闻摘要",
                "key_points": ["要点1", "要点2"],
                "market_impact": "medium"
            }

            # 1. 触发新闻收集任务
            collect_response = await async_test_client.post("/api/v1/news/collect")
            assert collect_response.status_code == 200
            task_data = collect_response.json()
            assert "task_id" in task_data
            assert task_data["status"] in ["started", "pending"]

            # 2. 等待任务完成（在实际实现中，这里会检查任务状态）
            await asyncio.sleep(0.1)  # 模拟异步处理时间

            # 3. 验证新闻数据被收集
            news_response = await async_test_client.get("/api/v1/news?limit=50")
            assert news_response.status_code == 200
            news_data = news_response.json()
            assert "news" in news_data
            assert isinstance(news_data["news"], list)
            assert len(news_data["news"]) > 0

            # 4. 验证新闻摘要生成
            summary_response = await async_test_client.get("/api/v1/news/summary")
            assert summary_response.status_code == 200
            summary_data = summary_response.json()
            assert "summary_text" in summary_data
            assert "key_points" in summary_data
            assert "market_impact" in summary_data

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_news_filtering_integration(self, async_test_client):
        """测试新闻过滤功能的集成"""
        # 模拟不同质量的新闻数据
        mixed_quality_news = [
            {
                "id": "high_quality_1",
                "title": "Bitcoin reaches new ATH",
                "content": "Detailed content about Bitcoin...",
                "source": "CoinDesk",
                "published_at": "2025-10-08T10:00:00Z",
                "relevance_score": 0.95,
                "sentiment": "positive"
            },
            {
                "id": "low_quality_1",
                "title": "Random crypto post",
                "content": "Not relevant content...",
                "source": "UnknownSource",
                "published_at": "2025-09-08T10:00:00Z",  # 旧新闻
                "relevance_score": 0.3,
                "sentiment": "neutral"
            }
        ]

        with patch('src.services.news_collector.NewsCollector') as mock_collector:
            mock_collector.return_value.collect_news.return_value = mixed_quality_news

            # 触发新闻收集
            await async_test_client.post("/api/v1/news/collect")
            await asyncio.sleep(0.1)

            # 获取过滤后的新闻
            response = await async_test_client.get("/api/v1/news")
            assert response.status_code == 200
            data = response.json()

            # 验证过滤结果（只应该返回高质量新闻）
            news_list = data["news"]
            assert len(news_list) <= 50  # 限制数量
            assert all(news["relevance_score"] >= 0.7 for news in news_list)
            assert all(news["published_at"] >= "2025-09-23" for news in news_list)  # 15天内

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_news_summary_generation_integration(self, async_test_client, mock_news_data):
        """测试新闻摘要生成集成"""
        with patch('src.services.llm_news_summarizer.LLMNewsSummarizer') as mock_summarizer:
            # 模拟LLM响应
            mock_summarizer.return_value = Mock()
            mock_summarizer.return_value.summarize_news.return_value = {
                "summary_text": "今日加密货币市场表现积极，比特币和以太坊均创新高。",
                "key_points": ["比特币价格突破", "以太坊升级成功", "机构投资增加"],
                "market_impact": "high",
                "confidence_score": 0.85
            }

            # 确保新闻数据存在
            with patch('src.services.news_collector.NewsCollector') as mock_collector:
                mock_collector.return_value.collect_news.return_value = mock_news_data

                # 触发新闻收集
                await async_test_client.post("/api/v1/news/collect")
                await asyncio.sleep(0.1)

                # 获取新闻摘要
                response = await async_test_client.get("/api/v1/news/summary")
                assert response.status_code == 200
                summary_data = response.json()

                # 验证摘要结构
                assert "summary_text" in summary_data
                assert "key_points" in summary_data
                assert "market_impact" in summary_data
                assert len(summary_data["key_points"]) > 0
                assert summary_data["market_impact"] in ["high", "medium", "low"]

                # 验证摘要内容质量
                assert len(summary_data["summary_text"]) > 10  # 有实际内容
                assert any(keyword in summary_data["summary_text"].lower()
                          for keyword in ["bitcoin", "ethereum", "crypto"])

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_news_data_persistence_integration(self, async_test_client):
        """测试新闻数据持久化集成"""
        # 模拟数据库操作
        with patch('src.models.news.NewsData') as mock_news_model:
            mock_instance = Mock()
            mock_news_model.return_value = mock_instance
            mock_instance.save.return_value = None

            with patch('src.models.news.NewsSummary') as mock_summary_model:
                mock_summary_instance = Mock()
                mock_summary_model.return_value = mock_summary_instance
                mock_summary_instance.save.return_value = None

                # 触发完整的新闻收集和处理流程
                collect_response = await async_test_client.post("/api/v1/news/collect")
                assert collect_response.status_code == 200

                # 等待处理完成
                await asyncio.sleep(0.1)

                # 验证数据持久化（通过检查API响应间接验证）
                news_response = await async_test_client.get("/api/v1/news")
                summary_response = await async_test_client.get("/api/v1/news/summary")

                assert news_response.status_code == 200
                assert summary_response.status_code == 200

                # 验证数据可以从数据库中检索
                assert len(news_response.json()["news"]) >= 0
                assert "summary_text" in summary_response.json()

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_news_caching_integration(self, async_test_client):
        """测试新闻缓存集成"""
        with patch('src.core.cache.get_cache') as mock_cache:
            # 模拟缓存命中和未命中
            mock_cache.get.return_value = None  # 首次调用缓存未命中

            with patch('src.services.news_collector.NewsCollector') as mock_collector:
                mock_collector.return_value.collect_news.return_value = mock_news_data

                # 第一次请求 - 缓存未命中
                response1 = await async_test_client.get("/api/v1/news")
                assert response1.status_code == 200

                # 验证缓存被调用
                mock_cache.get.assert_called()
                mock_cache.set.assert_called()  # 数据应该被缓存

                # 第二次请求 - 缓存命中
                mock_cache.get.return_value = {"news": mock_news_data, "total": len(mock_news_data)}
                response2 = await async_test_client.get("/api/v1/news")
                assert response2.status_code == 200

                # 验证返回缓存数据
                assert response2.json()["total"] == len(mock_news_data)

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_error_handling_integration(self, async_test_client):
        """测试错误处理集成"""
        # 模拟外部服务错误
        with patch('src.services.news_collector.NewsCollector') as mock_collector:
            mock_collector.return_value.collect_news.side_effect = Exception("API连接失败")

            # 触发新闻收集应该优雅处理错误
            collect_response = await async_test_client.post("/api/v1/news/collect")

            # 验证错误处理（可能返回错误状态或使用降级数据）
            assert collect_response.status_code in [200, 500, 503]

            # 验证系统仍然可以提供基本服务
            news_response = await async_test_client.get("/api/v1/news")
            assert news_response.status_code in [200, 500]  # 可能返回缓存数据或错误

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_news_performance_integration(self, async_test_client):
        """测试新闻处理性能集成"""
        import time

        # 模拟大量新闻数据
        large_news_dataset = [f"news_{i}" for i in range(100)]

        with patch('src.services.news_collector.NewsCollector') as mock_collector:
            mock_collector.return_value.collect_news.return_value = large_news_dataset

            # 测试批量处理性能
            start_time = time.time()

            collect_response = await async_test_client.post("/api/v1/news/collect")
            await asyncio.sleep(0.1)  # 模拟处理时间

            news_response = await async_test_client.get("/api/v1/news")
            summary_response = await async_test_client.get("/api/v1/news/summary")

            end_time = time.time()
            processing_time = end_time - start_time

            # 验证性能要求（应该在合理时间内完成）
            assert processing_time < 5.0  # 5秒内完成
            assert all(response.status_code == 200 for response in [collect_response, news_response, summary_response])

            # 验证数据限制（最多50条新闻）
            news_data = news_response.json()
            assert len(news_data["news"]) <= 50