"""
pytest配置和共享fixtures

提供测试配置和共享测试工具。
"""

import pytest
import asyncio
from pathlib import Path
import sys
from typing import AsyncGenerator

# 添加src目录到Python路径
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from fastapi.testclient import TestClient
from httpx import AsyncClient
import tempfile
import os

# 导入应用实例
try:
    from main import app
except ImportError:
    # 如果导入失败，创建一个简单的应用实例用于测试
    from fastapi import FastAPI
    app = FastAPI()


@pytest.fixture
def test_client():
    """同步测试客户端"""
    return TestClient(app)


@pytest.fixture
async def async_test_client():
    """异步测试客户端"""
    async with AsyncClient(app=app, base_url="http://test") as client:
        yield client


@pytest.fixture
def temp_env_vars():
    """临时环境变量"""
    original_env = os.environ.copy()

    # 设置测试环境变量
    os.environ.update({
        "ENVIRONMENT": "testing",
        "LOG_LEVEL": "DEBUG",
        "SECRET_KEY": "test_secret_key_for_testing_only",
        "DATABASE_URL": "sqlite:///test.db",
        "REDIS_URL": "redis://localhost:6379/15",
        "OPENAI_API_KEY": "test_openai_key",
        "ANTHROPIC_API_KEY": "test_anthropic_key"
    })

    yield

    # 恢复原始环境变量
    os.environ.clear()
    os.environ.update(original_env)


@pytest.fixture
def mock_news_data():
    """模拟新闻数据"""
    return [
        {
            "id": "news_1",
            "title": "Bitcoin reaches new all-time high",
            "content": "Bitcoin price surged past previous all-time high as institutional adoption increases...",
            "source": "CoinDesk",
            "published_at": "2025-10-08T10:00:00Z",
            "relevance_score": 0.95,
            "sentiment": "positive"
        },
        {
            "id": "news_2",
            "title": "Ethereum network upgrade successful",
            "content": "The latest Ethereum network upgrade has been completed successfully...",
            "source": "CryptoNews",
            "published_at": "2025-10-08T09:30:00Z",
            "relevance_score": 0.88,
            "sentiment": "positive"
        }
    ]


@pytest.fixture
def mock_news_summary():
    """模拟新闻摘要"""
    return {
        "id": "summary_1",
        "summary_text": "今日加密货币市场表现积极，比特币和以太坊均创新高，主要受机构资金流入和技术发展推动。",
        "key_points": [
            "比特币价格突破历史高点",
            "以太坊网络升级成功完成",
            "机构投资者持续买入加密资产",
            "市场情绪整体乐观"
        ],
        "market_impact": "high",
        "news_count": 25,
        "time_period_hours": 24
    }


class MockLLMService:
    """模拟LLM服务"""

    @staticmethod
    async def summarize_news(news_articles):
        """模拟新闻摘要生成"""
        return "模拟生成的新闻摘要"

    @staticmethod
    async def analyze_sentiment(text):
        """模拟情感分析"""
        return "positive"


@pytest.fixture
def mock_llm_service():
    """模拟LLM服务fixture"""
    return MockLLMService()


# 测试标记
pytest_plugins = [
    "pytest_asyncio",
]

# 异步测试配置
@pytest.fixture(scope="session")
def event_loop():
    """创建事件循环"""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


# 测试数据库配置
@pytest.fixture(scope="session")
def test_db():
    """测试数据库配置"""
    # 这里可以配置测试数据库
    # 例如，使用内存SQLite数据库
    pass


# 测试标记定义
def pytest_configure(config):
    """配置pytest标记"""
    config.addinivalue_mark(
        "integration", "标记为集成测试"
    )
    config.addinivalue_mark(
        "contract", "标记为合约测试"
    )
    config.addinivalue_mark(
        "unit", "标记为单元测试"
    )
    config.addinivalue_mark(
        "slow", "标记为慢速测试"
    )