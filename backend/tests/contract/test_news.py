"""
新闻API合约测试

验证新闻收集API端点的外部行为和响应格式。
"""

import pytest
import httpx
from datetime import datetime, timedelta
from typing import Dict, Any

from ..conftest import test_client, async_test_client


class TestNewsContracts:
    """新闻API合约测试类"""

    @pytest.mark.asyncio
    async def test_get_news_endpoint_contract(self, async_test_client):
        """测试获取新闻列表端点的契约"""
        # 测试端点存在性
        response = await async_test_client.get("/api/v1/news")

        # 验证基本响应格式
        assert response.status_code == 200

        data = response.json()

        # 验证响应结构
        assert "news" in data
        assert "total" in data
        assert "filters" in data

        # 验证数据类型
        assert isinstance(data["news"], list)
        assert isinstance(data["total"], int)
        assert isinstance(data["filters"], dict)

        # 验证新闻项结构（如果有新闻）
        if data["news"]:
            news_item = data["news"][0]
            required_fields = ["id", "title", "content", "source", "published_at", "relevance_score", "sentiment"]
            for field in required_fields:
                assert field in news_item, f"Missing required field: {field}"

    @pytest.mark.asyncio
    async def test_get_news_summary_endpoint_contract(self, async_test_client):
        """测试获取新闻摘要端点的契约"""
        response = await async_test_client.get("/api/v1/news/summary")

        # 验证基本响应格式
        assert response.status_code == 200

        data = response.json()

        # 验证响应结构
        required_fields = ["id", "summary_text", "key_points", "market_impact", "news_count", "time_period_hours"]
        for field in required_fields:
            assert field in data, f"Missing required field: {field}"

        # 验证数据类型
        assert isinstance(data["id"], str)
        assert isinstance(data["summary_text"], str)
        assert isinstance(data["key_points"], list)
        assert isinstance(data["market_impact"], str)
        assert isinstance(data["news_count"], int)
        assert isinstance(data["time_period_hours"], int)

    @pytest.mark.asyncio
    async def test_news_collection_trigger_endpoint_contract(self, async_test_client):
        """测试触发新闻收集任务的契约"""
        response = await async_test_client.post("/api/v1/news/collect")

        # 验证响应格式
        assert response.status_code == 200

        data = response.json()

        # 验证响应结构
        required_fields = ["task_id", "status", "message"]
        for field in required_fields:
            assert field in data, f"Missing required field: {field}"

        # 验证数据类型
        assert isinstance(data["task_id"], str)
        assert isinstance(data["status"], str)
        assert isinstance(data["message"], str)

        # 验证状态值
        assert data["status"] in ["started", "pending", "running", "completed", "failed"]

    @pytest.mark.asyncio
    async def test_news_query_parameters_contract(self, async_test_client):
        """测试新闻查询参数的契约"""
        # 测试limit参数
        response = await async_test_client.get("/api/v1/news?limit=10")
        assert response.status_code == 200

        data = response.json()
        assert "filters" in data
        assert data["filters"]["limit"] == 10

        # 测试days参数
        response = await async_test_client.get("/api/v1/news?days=7")
        assert response.status_code == 200

        data = response.json()
        assert "filters" in data
        assert data["filters"]["days"] == 7

        # 测试source参数
        response = await async_test_client.get("/api/v1/news?source=CoinDesk")
        assert response.status_code == 200

        data = response.json()
        assert "filters" in data
        assert data["filters"]["source"] == "CoinDesk"

    @pytest.mark.asyncio
    async def test_news_summary_hours_parameter_contract(self, async_test_client):
        """测试新闻摘要hours参数的契约"""
        response = await async_test_client.get("/api/v1/news/summary?hours=48")
        assert response.status_code == 200

        data = response.json()
        assert data["time_period_hours"] == 48

    @pytest.mark.asyncio
    async def test_news_data_validation_contract(self, async_test_client):
        """测试新闻数据验证的契约"""
        response = await async_test_client.get("/api/v1/news")
        assert response.status_code == 200

        data = response.json()

        # 如果有新闻数据，验证数据格式
        if data["news"]:
            for news_item in data["news"]:
                # 验证必需字段
                assert news_item["id"], "News item missing ID"
                assert news_item["title"], "News item missing title"
                assert news_item["content"], "News item missing content"
                assert news_item["source"], "News item missing source"

                # 验证数据类型和格式
                assert isinstance(news_item["title"], str)
                assert isinstance(news_item["content"], str)
                assert isinstance(news_item["source"], str)
                assert isinstance(news_item["relevance_score"], (int, float))
                assert 0 <= news_item["relevance_score"] <= 1
                assert news_item["sentiment"] in ["positive", "negative", "neutral"]

    @pytest.mark.asyncio
    async def test_error_handling_contract(self, async_test_client):
        """测试错误处理的契约"""
        # 测试无效的limit参数
        response = await async_test_client.get("/api/v1/news?limit=invalid")
        # 应该返回400或422错误，或者使用默认值
        assert response.status_code in [200, 400, 422]

        # 测试无效的hours参数
        response = await async_test_client.get("/api/v1/news/summary?hours=invalid")
        assert response.status_code in [200, 400, 422]

    @pytest.mark.asyncio
    async def test_response_headers_contract(self, async_test_client):
        """测试响应头契约"""
        response = await async_test_client.get("/api/v1/news")
        assert response.status_code == 200

        # 验证基本响应头
        assert "content-type" in response.headers
        assert "application/json" in response.headers["content-type"]