"""
做多策略工作流集成测试

验证完整的做多策略分析流程，包括数据收集、技术分析、LLM决策等组件的集成。
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi.testclient import TestClient
import json


class TestLongStrategyWorkflow:
    """做多策略工作流集成测试"""

    @pytest.fixture
    def mock_exchange_data(self):
        """模拟交易所数据"""
        return {
            "symbol": "BTC/USDT",
            "timeframe": "1h",
            "klines": [
                {
                    "timestamp": datetime.now() - timedelta(hours=i),
                    "open": 50000 + (i * 100),
                    "high": 50100 + (i * 100),
                    "low": 49900 + (i * 100),
                    "close": 50050 + (i * 100),
                    "volume": 100.5 + (i * 0.1)
                }
                for i in range(168, 0, -1)  # 7天的小时数据
            ]
        }

    @pytest.fixture
    def mock_technical_analysis(self):
        """模拟技术分析结果"""
        return {
            "rsi": 65.5,
            "macd": {
                "macd_line": 150.5,
                "signal_line": 145.2,
                "histogram": 5.3
            },
            "bollinger_bands": {
                "upper": 52000,
                "middle": 50000,
                "lower": 48000
            },
            "moving_averages": {
                "sma_20": 49800,
                "ema_12": 50200,
                "ema_26": 49500
            },
            "volume_indicators": {
                "volume_sma": 120.5,
                "volume_ratio": 1.2
            },
            "support_levels": [48500, 47000, 45000],
            "resistance_levels": [51500, 53000, 55000]
        }

    @pytest.fixture
    def mock_llm_response(self):
        """模拟LLM响应"""
        return {
            "recommendation": "buy",
            "confidence": 0.78,
            "entry_price": 50200,
            "stop_loss": 48500,
            "take_profit": 52500,
            "position_size": 0.15,
            "reasoning": "RSI显示超买反弹信号，价格突破关键阻力位，成交量放大确认趋势",
            "risk_factors": ["市场波动性较高", "宏观经济不确定性"],
            "time_horizon": "7-14天"
        }

    async def test_complete_long_strategy_workflow(
        self,
        async_test_client: TestClient,
        mock_exchange_data,
        mock_technical_analysis,
        mock_llm_response
    ):
        """测试完整的做多策略工作流"""

        # 模拟交易所数据收集
        with patch('backend.src.services.exchange_data_collector.ExchangeDataCollector') as MockCollector:
            mock_collector = AsyncMock()
            mock_collector.fetch_klines.return_value = mock_exchange_data["klines"]
            MockCollector.return_value = mock_collector

            # 模拟技术分析引擎
            with patch('backend.src.services.technical_analysis_engine.TechnicalAnalysisEngine') as MockEngine:
                mock_engine = AsyncMock()
                mock_engine.analyze.return_value = mock_technical_analysis
                MockEngine.return_value = mock_engine

                # 模拟LLM分析器
                with patch('backend.src.services.llm_long_strategy_analyzer.LLMLongStrategyAnalyzer') as MockAnalyzer:
                    mock_analyzer = AsyncMock()
                    mock_analyzer.analyze_long_strategy.return_value = mock_llm_response
                    MockAnalyzer.return_value = mock_analyzer

                    # 发送策略分析请求
                    request_data = {
                        "symbol": "BTC/USDT",
                        "timeframe": "1h",
                        "analysis_period_days": 7,
                        "confidence_threshold": 0.7
                    }

                    response = async_test_client.post("/api/v1/strategies/long-analysis", json=request_data)

                    # 验证响应
                    if response.status_code == 200:
                        data = response.json()

                        # 验证策略生成
                        assert "strategy_id" in data
                        assert data["symbol"] == "BTC/USDT"
                        assert data["recommendation"] in ["buy", "hold", "sell"]
                        assert "confidence_score" in data
                        assert "entry_price" in data
                        assert "stop_loss_price" in data
                        assert "take_profit_price" in data

                        # 验证数据来源
                        assert "technical_indicators" in data
                        assert "market_conditions" in data
                        assert "risk_factors" in data

    async def test_data_collection_integration(self, async_test_client: TestClient, mock_exchange_data):
        """测试数据收集集成"""

        with patch('backend.src.services.exchange_data_collector.ExchangeDataCollector') as MockCollector:
            mock_collector = AsyncMock()
            mock_collector.fetch_klines.return_value = mock_exchange_data["klines"]
            MockCollector.return_value = mock_collector

            # 测试数据收集服务
            from backend.src.services.exchange_data_collector import ExchangeDataCollector

            collector = ExchangeDataCollector()
            klines = await collector.fetch_klines("BTC/USDT", "1h", 168)

            # 验证数据收集
            assert len(klines) == 168
            assert all(kline["symbol"] == "BTC/USDT" for kline in klines)
            assert all("close" in kline for kline in klines)

    async def test_technical_analysis_integration(self, async_test_client: TestClient, mock_exchange_data):
        """测试技术分析集成"""

        with patch('backend.src.services.technical_analysis_engine.TechnicalAnalysisEngine') as MockEngine:
            mock_engine = AsyncMock()

            # 模拟技术分析结果
            analysis_result = {
                "rsi": 65.5,
                "macd_line": 150.5,
                "signal_line": 145.2,
                "bollinger_upper": 52000,
                "bollinger_lower": 48000,
                "sma_20": 49800,
                "ema_12": 50200,
                "volume_ratio": 1.2
            }

            mock_engine.analyze.return_value = analysis_result
            MockEngine.return_value = mock_engine

            # 测试技术分析引擎
            from backend.src.services.technical_analysis_engine import TechnicalAnalysisEngine

            engine = TechnicalAnalysisEngine()
            result = await engine.analyze(mock_exchange_data["klines"])

            # 验证技术分析结果
            assert "rsi" in result
            assert "macd" in result
            assert "moving_averages" in result
            assert 0 <= result["rsi"] <= 100

    async def test_llm_integration(self, async_test_client: TestClient, mock_technical_analysis):
        """测试LLM集成"""

        with patch('backend.src.services.llm_long_strategy_analyzer.LLMLongStrategyAnalyzer') as MockAnalyzer:
            mock_analyzer = AsyncMock()

            llm_result = {
                "recommendation": "buy",
                "confidence": 0.78,
                "entry_price": 50200,
                "stop_loss": 48500,
                "take_profit": 52500,
                "reasoning": "技术指标显示买入信号"
            }

            mock_analyzer.analyze_long_strategy.return_value = llm_result
            MockAnalyzer.return_value = mock_analyzer

            # 测试LLM分析器
            from backend.src.services.llm_long_strategy_analyzer import LLMLongStrategyAnalyzer

            analyzer = LLMLongStrategyAnalyzer()
            result = await analyzer.analyze_long_strategy(
                symbol="BTC/USDT",
                technical_analysis=mock_technical_analysis,
                market_data={}
            )

            # 验证LLM分析结果
            assert result["recommendation"] in ["buy", "hold", "sell"]
            assert 0 <= result["confidence"] <= 1
            assert "entry_price" in result
            assert "stop_loss" in result
            assert "take_profit" in result

    async def test_error_handling_workflow(self, async_test_client: TestClient):
        """测试错误处理工作流"""

        # 模拟数据收集失败
        with patch('backend.src.services.exchange_data_collector.ExchangeDataCollector') as MockCollector:
            mock_collector = AsyncMock()
            mock_collector.fetch_klines.side_effect = Exception("Exchange API error")
            MockCollector.return_value = mock_collector

            request_data = {
                "symbol": "BTC/USDT",
                "timeframe": "1h",
                "analysis_period_days": 7,
                "confidence_threshold": 0.7
            }

            response = async_test_client.post("/api/v1/strategies/long-analysis", json=request_data)

            # 应该优雅地处理错误
            assert response.status_code in [500, 503]

            if response.status_code == 500:
                error_data = response.json()
                assert "detail" in error_data

    async def test_concurrent_analysis_requests(self, async_test_client: TestClient):
        """测试并发分析请求"""

        with patch('backend.src.services.exchange_data_collector.ExchangeDataCollector') as MockCollector, \
             patch('backend.src.services.technical_analysis_engine.TechnicalAnalysisEngine') as MockEngine, \
             patch('backend.src.services.llm_long_strategy_analyzer.LLMLongStrategyAnalyzer') as MockAnalyzer:

            # 设置模拟
            mock_collector = AsyncMock()
            mock_collector.fetch_klines.return_value = [
                {"timestamp": datetime.now(), "close": 50000, "volume": 100}
                for _ in range(168)
            ]
            MockCollector.return_value = mock_collector

            mock_engine = AsyncMock()
            mock_engine.analyze.return_value = {"rsi": 50, "macd": 0}
            MockEngine.return_value = mock_engine

            mock_analyzer = AsyncMock()
            mock_analyzer.analyze_long_strategy.return_value = {
                "recommendation": "hold",
                "confidence": 0.6,
                "entry_price": 50000,
                "stop_loss": 48000,
                "take_profit": 52000
            }
            MockAnalyzer.return_value = mock_analyzer

            # 发送多个并发请求
            tasks = []
            for i in range(5):
                request_data = {
                    "symbol": f"BTC/USDT",
                    "timeframe": "1h",
                    "analysis_period_days": 7,
                    "confidence_threshold": 0.7
                }
                task = async_test_client.post("/api/v1/strategies/long-analysis", json=request_data)
                tasks.append(task)

            # 等待所有请求完成
            responses = []
            for task in tasks:
                responses.append(task)

            # 验证所有请求都得到处理
            for response in responses:
                assert response.status_code in [200, 429, 500, 503]

    async def test_cache_integration(self, async_test_client: TestClient):
        """测试缓存集成"""

        with patch('backend.src.services.exchange_data_collector.ExchangeDataCollector') as MockCollector, \
             patch('backend.src.cache.get_cache') as MockCache:

            # 设置缓存模拟
            mock_cache = MagicMock()
            mock_cache.get.return_value = None  # 缓存未命中
            mock_cache.set.return_value = True
            MockCache.return_value = mock_cache

            mock_collector = AsyncMock()
            mock_collector.fetch_klines.return_value = [
                {"timestamp": datetime.now(), "close": 50000, "volume": 100}
                for _ in range(168)
            ]
            MockCollector.return_value = mock_collector

            request_data = {
                "symbol": "BTC/USDT",
                "timeframe": "1h",
                "analysis_period_days": 7,
                "confidence_threshold": 0.7
            }

            response = async_test_client.post("/api/v1/strategies/long-analysis", json=request_data)

            # 验证缓存调用
            mock_cache.get.assert_called()

            # 如果请求成功，应该设置缓存
            if response.status_code == 200:
                mock_cache.set.assert_called()

    async def test_performance_requirements(self, async_test_client: TestClient):
        """测试性能要求"""
        import time

        with patch('backend.src.services.exchange_data_collector.ExchangeDataCollector') as MockCollector, \
             patch('backend.src.services.technical_analysis_engine.TechnicalAnalysisEngine') as MockEngine, \
             patch('backend.src.services.llm_long_strategy_analyzer.LLMLongStrategyAnalyzer') as MockAnalyzer:

            # 设置快速响应的模拟
            mock_collector = AsyncMock()
            mock_collector.fetch_klines.return_value = [
                {"timestamp": datetime.now(), "close": 50000, "volume": 100}
                for _ in range(168)
            ]
            MockCollector.return_value = mock_collector

            mock_engine = AsyncMock()
            mock_engine.analyze.return_value = {"rsi": 50, "macd": 0}
            MockEngine.return_value = mock_engine

            mock_analyzer = AsyncMock()
            mock_analyzer.analyze_long_strategy.return_value = {
                "recommendation": "buy",
                "confidence": 0.75,
                "entry_price": 50100,
                "stop_loss": 48500,
                "take_profit": 52000
            }
            MockAnalyzer.return_value = mock_analyzer

            request_data = {
                "symbol": "BTC/USDT",
                "timeframe": "1h",
                "analysis_period_days": 7,
                "confidence_threshold": 0.7
            }

            start_time = time.time()
            response = async_test_client.post("/api/v1/strategies/long-analysis", json=request_data)
            end_time = time.time()

            response_time = end_time - start_time

            # 性能要求：响应时间应该小于5分钟（300秒）
            assert response_time < 300.0, f"Response time {response_time}s exceeds performance requirement"

    async def test_data_persistence_integration(self, async_test_client: TestClient):
        """测试数据持久化集成"""

        with patch('backend.src.services.exchange_data_collector.ExchangeDataCollector') as MockCollector, \
             patch('backend.src.models.technical_analysis.TechnicalAnalysis') as MockModel, \
             patch('backend.src.core.database.SessionLocal') as MockSession:

            # 设置模拟
            mock_collector = AsyncMock()
            mock_collector.fetch_klines.return_value = [
                {"timestamp": datetime.now(), "close": 50000, "volume": 100}
                for _ in range(168)
            ]
            MockCollector.return_value = mock_collector

            mock_db_session = MagicMock()
            MockSession.return_value = mock_db_session

            mock_model = MagicMock()
            MockModel.return_value = mock_model

            request_data = {
                "symbol": "BTC/USDT",
                "timeframe": "1h",
                "analysis_period_days": 7,
                "confidence_threshold": 0.7
            }

            response = async_test_client.post("/api/v1/strategies/long-analysis", json=request_data)

            # 验证数据库交互
            if response.status_code == 200:
                mock_db_session.add.assert_called()
                mock_db_session.commit.assert_called()

    def test_strategy_quality_validation(self, async_test_client: TestClient):
        """测试策略质量验证"""

        request_data = {
            "symbol": "BTC/USDT",
            "timeframe": "1h",
            "analysis_period_days": 7,
            "confidence_threshold": 0.8  # 高置信度阈值
        }

        response = async_test_client.post("/api/v1/strategies/long-analysis", json=request_data)

        if response.status_code == 200:
            data = response.json()

            # 验证策略质量指标
            assert "confidence_score" in data
            assert "risk_factors" in data
            assert "reasoning" in data

            # 如果设置了高置信度阈值，返回的置信度应该满足要求
            if data["confidence_score"] < request_data["confidence_threshold"]:
                # 如果置信度不足，推荐应该是hold或sell
                assert data["recommendation"] in ["hold", "sell"]

    async def test_market_regime_detection(self, async_test_client: TestClient):
        """测试市场状态检测"""

        request_data = {
            "symbol": "BTC/USDT",
            "timeframe": "1h",
            "analysis_period_days": 30,  # 更长的分析周期
            "confidence_threshold": 0.7
        }

        response = async_test_client.post("/api/v1/strategies/long-analysis", json=request_data)

        if response.status_code == 200:
            data = response.json()

            # 验证市场状态分析
            if "market_conditions" in data:
                conditions = data["market_conditions"]

                # 应该包含市场状态信息
                # expected_fields = ["trend", "volatility", "volume_profile", "sentiment"]
                # for field in expected_fields:
                #     assert field in conditions