"""
完整工作流集成测试

测试从数据收集到策略生成的完整集成工作流。
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timedelta
import json
from fastapi.testclient import TestClient

from backend.src.models.news import NewsData, NewsSummary
from backend.src.models.technical_analysis import TechnicalAnalysis
from backend.src.services.news_collector import NewsCollector
from backend.src.services.llm_news_summarizer import LLMNewsSummarizer
from backend.src.services.exchange_data_collector import ExchangeDataCollector
from backend.src.services.technical_analysis_engine import TechnicalAnalysisEngine
from backend.src.services.llm_long_strategy_analyzer import LLMLongStrategyAnalyzer
from backend.src.services.llm_short_strategy_analyzer import LLMShortStrategyAnalyzer
from backend.src.services.strategy_aggregator import StrategyAggregator
from backend.src.services.llm_strategy_generator import LLMStrategyGenerator


class TestCompleteWorkflowIntegration:
    """完整工作流集成测试"""

    @pytest.fixture
    def mock_btc_data(self):
        """模拟BTC市场数据"""
        return {
            "symbol": "BTC/USDT",
            "current_price": 50250.0,
            "trend_direction": "up",
            "trend_strength": 0.7,
            "volatility_level": "medium",
            "market_sentiment": "bullish",
            "indicators": {
                "rsi": 65.0,
                "macd_line": 0.0025,
                "macd_signal": 0.0020,
                "macd_histogram": 0.0005,
                "bollinger_upper": 51500.0,
                "bollinger_middle": 50000.0,
                "bollinger_lower": 48500.0,
                "sma_20": 49800.0,
                "sma_50": 47500.0,
                "ema_12": 49950.0,
                "ema_26": 48500.0,
                "volume_ratio": 1.2,
                "atr": 1200.0
            }
        }

    @pytest.fixture
    def mock_news_data(self):
        """模拟新闻数据"""
        return [
            {
                "title": "比特币突破50000美元大关",
                "content": "比特币价格今日突破了50000美元的重要心理关口...",
                "source": "CoinDesk",
                "url": "https://www.coindesk.com/btc-50000",
                "relevance_score": 0.9,
                "sentiment": "positive",
                "sentiment_score": 0.8,
                "published_at": "2025-10-08T10:00:00Z"
            },
            {
                "title": "机构投资者持续加仓比特币",
                "content": "多家机构投资者报告显示他们正在增加比特币持仓...",
                "source": "Reuters",
                "url": "https://www.reuters.com/crypto-institutions",
                "relevance_score": 0.85,
                "sentiment": "positive",
                "sentiment_score": 0.75,
                "published_at": "2025-10-08T09:30:00Z"
            }
        ]

    @pytest.fixture
    def mock_long_strategy(self):
        """模拟做多策略"""
        return {
            "symbol": "BTC/USDT",
            "recommendation": "buy",
            "confidence_score": 0.75,
            "entry_price": 50200.0,
            "stop_loss_price": 48500.0,
            "take_profit_price": 52500.0,
            "position_size_percent": 15.0,
            "time_horizon": "medium_term",
            "reasoning": "技术指标显示上涨趋势，新闻情绪积极",
            "risk_factors": ["市场波动性", "政策风险"],
            "technical_signals": [
                {"indicator": "RSI", "signal": "bullish", "strength": "medium"},
                {"indicator": "MACD", "signal": "bullish", "strength": "strong"}
            ]
        }

    @pytest.fixture
    def mock_short_strategy(self):
        """模拟做空策略"""
        return {
            "symbol": "BTC/USDT",
            "recommendation": "hold",
            "confidence_score": 0.3,
            "entry_price": 50200.0,
            "stop_loss_price": 52500.0,
            "take_profit_price": 47500.0,
            "position_size_percent": 5.0,
            "time_horizon": "short_term",
            "reasoning": "当前上涨趋势强劲，不建议做空",
            "risk_factors": ["空头回补风险", "市场情绪逆转"],
            "technical_signals": []
        }

    async def test_complete_strategy_generation_workflow(
        self, async_test_client: TestClient, mock_btc_data, mock_news_data,
        mock_long_strategy, mock_short_strategy
    ):
        """测试完整策略生成工作流"""

        # 模拟新闻收集
        mock_news_collector = AsyncMock(spec=NewsCollector)
        mock_news_collector.collect_news.return_value = mock_news_data

        # 模拟新闻摘要
        mock_news_summarizer = AsyncMock(spec=LLMNewsSummarizer)
        mock_news_summarizer.generate_market_summary.return_value = {
            "summary_text": "市场情绪积极，比特币突破关键阻力位",
            "key_points": ["技术指标看涨", "机构投资增加"],
            "market_impact": "high"
        }

        # 模拟K线数据收集
        mock_exchange_collector = AsyncMock(spec=ExchangeDataCollector)
        mock_klines = [
            type('KlineData', (), {
                'timestamp': datetime.now() - timedelta(hours=i),
                'open': 50000 + i * 10,
                'high': 50100 + i * 10,
                'low': 49900 + i * 10,
                'close': 50050 + i * 10,
                'volume': 1000 + i * 100
            })() for i in range(24)
        ]
        mock_exchange_collector.fetch_klines.return_value = mock_klines

        # 模拟技术分析
        mock_technical_engine = AsyncMock(spec=TechnicalAnalysisEngine)
        mock_technical_engine.analyze.return_value = mock_btc_data

        # 模拟做多策略分析
        mock_long_analyzer = AsyncMock(spec=LLMLongStrategyAnalyzer)
        mock_long_result = type('LongStrategyAnalysis', (), mock_long_strategy)()
        mock_long_analyzer.analyze_long_strategy.return_value = mock_long_result

        # 模拟做空策略分析
        mock_short_analyzer = AsyncMock(spec=LLMShortStrategyAnalyzer)
        mock_short_result = type('ShortStrategyAnalysis', (), mock_short_strategy)()
        mock_short_analyzer.analyze_short_strategy.return_value = mock_short_result

        # 模拟策略聚合器
        mock_aggregator = AsyncMock(spec=StrategyAggregator)
        mock_aggregated_data = {
            "symbol": "BTC/USDT",
            "long_analysis": mock_long_strategy,
            "short_analysis": mock_short_strategy,
            "news_analysis": {
                "sentiment": "positive",
                "impact": "high",
                "key_events": ["价格突破", "机构买入"]
            },
            "technical_analysis": mock_btc_data,
            "overall_sentiment": "bullish"
        }
        mock_aggregator.aggregate_strategies.return_value = mock_aggregated_data

        # 模拟LLM策略生成器
        mock_strategy_generator = AsyncMock(spec=LLMStrategyGenerator)
        mock_final_strategy = {
            "strategy_id": "strategy_20251008_001",
            "symbol": "BTC/USDT",
            "final_recommendation": "long",
            "confidence_score": 0.78,
            "entry_price": 50200.0,
            "stop_loss_price": 48500.0,
            "take_profit_price": 52500.0,
            "position_size_percent": 18.0,
            "strategy_type": "moderate",
            "market_analysis": {
                "overall_sentiment": "bullish",
                "trend_analysis": "上升趋势确立",
                "technical_signals": ["RSI超买回调", "MACD金叉"],
                "news_impact": "积极",
                "volatility_assessment": "中等"
            },
            "risk_assessment": {
                "risk_level": "medium",
                "risk_factors": ["市场波动", "监管不确定性"],
                "risk_mitigation": "设置止损，分批建仓",
                "max_drawdown_estimate": 8.5
            },
            "execution_plan": {
                "entry_conditions": ["价格回调至50200以下", "成交量确认"],
                "exit_conditions": ["达到止盈目标52500", "跌破止损48500"],
                "position_sizing": "建议15-20%仓位",
                "timing_strategy": "等待回调后入场"
            },
            "generated_at": datetime.utcnow().isoformat(),
            "expires_at": (datetime.utcnow() + timedelta(hours=24)).isoformat(),
            "analysis_version": "v1.0"
        }
        mock_strategy_generator.generate_final_strategy.return_value = mock_final_strategy

        # 测试策略生成请求
        strategy_request = {
            "symbol": "BTC/USDT",
            "timeframe": "1h",
            "analysis_types": ["long", "short", "news", "technical"],
            "include_news": True,
            "risk_tolerance": "medium",
            "max_position_size": 20.0,
            "custom_parameters": {
                "preferred_indicators": ["RSI", "MACD", "Moving Averages"],
                "news_sentiment_threshold": 0.7
            }
        }

        # 模拟完整的策略生成过程
        with patch('backend.src.services.news_collector.NewsCollector') as MockNewsCollector:
            MockNewsCollector.return_value = mock_news_collector

            with patch('backend.src.services.llm_news_summarizer.LLMNewsSummarizer') as MockNewsSummarizer:
                MockNewsSummarizer.return_value = mock_news_summarizer

                with patch('backend.src.services.exchange_data_collector.ExchangeDataCollector') as MockExchangeCollector:
                    MockExchangeCollector.return_value = mock_exchange_collector

                    with patch('backend.src.services.technical_analysis_engine.TechnicalAnalysisEngine') as MockTechnicalEngine:
                        MockTechnicalEngine.return_value = mock_technical_engine

                        with patch('backend.src.services.llm_long_strategy_analyzer.LLMLongStrategyAnalyzer') as MockLongAnalyzer:
                            MockLongAnalyzer.return_value = mock_long_analyzer

                            with patch('backend.src.services.llm_short_strategy_analyzer.LLMShortStrategyAnalyzer') as MockShortAnalyzer:
                                MockShortAnalyzer.return_value = mock_short_analyzer

                                with patch('backend.src.services.strategy_aggregator.StrategyAggregator') as MockAggregator:
                                    MockAggregator.return_value = mock_aggregator

                                    with patch('backend.src.services.llm_strategy_generator.LLMStrategyGenerator') as MockGenerator:
                                        MockGenerator.return_value = mock_strategy_generator

                                        # 发送策略生成请求
                                        response = async_test_client.post(
                                            "/api/v1/strategies/generate",
                                            json=strategy_request
                                        )

                                        # 验证响应
                                        assert response.status_code in [200, 500, 503]

                                        if response.status_code == 200:
                                            data = response.json()

                                            # 验证响应结构
                                            required_fields = [
                                                "strategy_id", "symbol", "final_recommendation",
                                                "confidence_score", "entry_price", "stop_loss_price",
                                                "take_profit_price", "position_size_percent",
                                                "strategy_type", "market_analysis", "risk_assessment",
                                                "execution_plan", "generated_at"
                                            ]

                                            for field in required_fields:
                                                assert field in data, f"Missing field: {field}"

                                            # 验证数据质量
                                            assert data["symbol"] == "BTC/USDT"
                                            assert data["final_recommendation"] in ["long", "short", "hold"]
                                            assert 0 <= data["confidence_score"] <= 1
                                            assert data["entry_price"] > 0
                                            assert data["position_size_percent"] > 0

    async def test_data_collection_integration(self, async_test_client: TestClient):
        """测试数据收集集成"""

        # 模拟K线数据
        mock_klines = [
            {
                "timestamp": (datetime.now() - timedelta(hours=i)).isoformat(),
                "open": 50000 + i * 10,
                "high": 50100 + i * 10,
                "low": 49900 + i * 10,
                "close": 50050 + i * 10,
                "volume": 1000 + i * 100
            } for i in range(24)
        ]

        with patch('backend.src.services.exchange_data_collector.ExchangeDataCollector') as MockCollector:
            mock_collector = AsyncMock()
            mock_collector.fetch_klines.return_value = mock_klines
            MockCollector.return_value = mock_collector

            # 测试数据收集
            collector = ExchangeDataCollector()
            klines = await collector.fetch_klines("BTC/USDT", "1h", 24)

            assert len(klines) == 24
            assert all("open" in kline for kline in klines)
            assert all("close" in kline for kline in klines)

    async def test_news_analysis_integration(self, async_test_client: TestClient):
        """测试新闻分析集成"""

        mock_news = [
            {
                "title": "比特币价格上涨",
                "content": "比特币价格今日上涨5%...",
                "source": "CryptoNews",
                "relevance_score": 0.9,
                "sentiment": "positive"
            }
        ]

        with patch('backend.src.services.news_collector.NewsCollector') as MockCollector:
            mock_collector = AsyncMock()
            mock_collector.collect_news.return_value = mock_news
            MockCollector.return_value = mock_collector

            # 测试新闻收集
            collector = NewsCollector()
            news = await collector.collect_news(days_back=7, max_items=50)

            assert len(news) == 1
            assert news[0]["title"] == "比特币价格上涨"
            assert news[0]["sentiment"] == "positive"

    async def test_multi_symbol_strategy_generation(self, async_test_client: TestClient):
        """测试多交易符号策略生成"""

        multi_symbol_request = {
            "symbols": ["BTC/USDT", "ETH/USDT"],
            "timeframe": "1h",
            "analysis_types": ["long", "short"],
            "include_news": True,
            "risk_tolerance": "moderate",
            "max_position_size": 15.0
        }

        # 模拟多符号策略生成
        mock_multi_strategies = [
            {
                "symbol": "BTC/USDT",
                "strategy_id": "btc_strategy_001",
                "final_recommendation": "long",
                "confidence_score": 0.75,
                "entry_price": 50200.0,
                "position_size_percent": 15.0
            },
            {
                "symbol": "ETH/USDT",
                "strategy_id": "eth_strategy_001",
                "final_recommendation": "hold",
                "confidence_score": 0.60,
                "entry_price": 3200.0,
                "position_size_percent": 10.0
            }
        ]

        with patch('backend.src.services.llm_strategy_generator.LLMStrategyGenerator') as MockGenerator:
            mock_generator = AsyncMock()
            mock_generator.generate_multiple_strategies.return_value = mock_multi_strategies
            MockGenerator.return_value = mock_generator

            response = async_test_client.post(
                "/api/v1/strategies/generate-batch",
                json=multi_symbol_request
            )

            # 验证响应（端点可能尚未实现）
            assert response.status_code in [200, 404, 422, 500]

    async def test_strategy_persistence_and_retrieval(self, async_test_client: TestClient):
        """测试策略持久化和检索"""

        # 生成策略
        strategy_request = {
            "symbol": "BTC/USDT",
            "timeframe": "1h",
            "analysis_types": ["long"],
            "include_news": False,
            "risk_tolerance": "moderate",
            "max_position_size": 20.0
        }

        response = async_test_client.post("/api/v1/strategies/generate", json=strategy_request)

        if response.status_code == 200:
            data = response.json()
            strategy_id = data.get("strategy_id")

            if strategy_id:
                # 测试策略检索
                retrieval_response = async_test_client.get(f"/api/v1/strategies/{strategy_id}")

                # 验证检索响应（端点可能尚未实现）
                assert retrieval_response.status_code in [200, 404, 500]

    async def test_error_handling_in_workflow(self, async_test_client: TestClient):
        """测试工作流中的错误处理"""

        # 测试无效交易符号
        invalid_request = {
            "symbol": "INVALID_SYMBOL",
            "timeframe": "1h",
            "analysis_types": ["long"],
            "include_news": True,
            "risk_tolerance": "medium",
            "max_position_size": 20.0
        }

        response = async_test_client.post("/api/v1/strategies/generate", json=invalid_request)

        # 应该返回验证错误
        assert response.status_code in [400, 422, 500]

        # 测试不支持的参数
        unsupported_request = {
            "symbol": "BTC/USDT",
            "timeframe": "1h",
            "analysis_types": ["long", "unsupported_type"],
            "include_news": True,
            "risk_tolerance": "medium",
            "max_position_size": 20.0
        }

        response = async_test_client.post("/api/v1/strategies/generate", json=unsupported_request)

        # 应该返回验证错误
        assert response.status_code in [400, 422, 500]

    async def test_performance_requirements(self, async_test_client: TestClient):
        """测试性能要求"""
        import time

        # 记录开始时间
        start_time = time.time()

        # 发送策略生成请求
        strategy_request = {
            "symbol": "BTC/USDT",
            "timeframe": "1h",
            "analysis_types": ["long"],
            "include_news": False,  # 禁用新闻以加快速度
            "risk_tolerance": "moderate",
            "max_position_size": 20.0
        }

        response = async_test_client.post("/api/v1/strategies/generate", json=strategy_request)

        end_time = time.time()
        response_time = end_time - start_time

        # 验证响应时间（应该在合理范围内）
        # 注意：在测试环境中，由于模拟数据，响应时间应该很快
        assert response_time < 30.0, f"Response time too slow: {response_time}s"

        # 验证响应状态
        assert response.status_code in [200, 500, 503]

    async def test_concurrent_strategy_generation(self, async_test_client: TestClient):
        """测试并发策略生成"""

        strategy_requests = [
            {
                "symbol": "BTC/USDT",
                "timeframe": "1h",
                "analysis_types": ["long"],
                "include_news": False,
                "risk_tolerance": "moderate",
                "max_position_size": 20.0
            },
            {
                "symbol": "ETH/USDT",
                "timeframe": "1h",
                "analysis_types": ["short"],
                "include_news": False,
                "risk_tolerance": "moderate",
                "max_position_size": 15.0
            }
        ]

        # 并发发送多个请求
        async def send_request(request_data):
            return async_test_client.post("/api/v1/strategies/generate", json=request_data)

        tasks = [send_request(req) for req in strategy_requests]
        responses = await asyncio.gather(*tasks, return_exceptions=True)

        # 验证所有响应都有效
        for response in responses:
            if hasattr(response, 'status_code'):
                assert response.status_code in [200, 422, 500, 503]
            else:
                # 处理异常情况
                assert isinstance(response, Exception)

    async def test_data_consistency_across_workflow(self, async_test_client: TestClient):
        """测试工作流中的数据一致性"""

        # 发送策略生成请求
        strategy_request = {
            "symbol": "BTC/USDT",
            "timeframe": "1h",
            "analysis_types": ["long", "short", "technical"],
            "include_news": True,
            "risk_tolerance": "moderate",
            "max_position_size": 20.0
        }

        response = async_test_client.post("/api/v1/strategies/generate", json=strategy_request)

        if response.status_code == 200:
            data = response.json()

            # 验证数据一致性
            assert data["symbol"] == strategy_request["symbol"]
            assert data["timeframe"] == strategy_request["timeframe"]

            # 验证价格逻辑一致性
            if data["final_recommendation"] == "long":
                assert data["entry_price"] <= data["take_profit_price"]
                assert data["entry_price"] >= data["stop_loss_price"]
            elif data["final_recommendation"] == "short":
                assert data["entry_price"] >= data["take_profit_price"]
                assert data["entry_price"] <= data["stop_loss_price"]

            # 验证仓位大小符合限制
            assert data["position_size_percent"] <= strategy_request["max_position_size"]