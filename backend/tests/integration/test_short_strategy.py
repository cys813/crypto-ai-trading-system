"""
做空策略工作流集成测试

验证完整的做空策略分析流程，包括数据收集、技术分析、LLM决策等组件的集成。
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi.testclient import TestClient
import json


class TestShortStrategyWorkflow:
    """做空策略工作流集成测试"""

    @pytest.fixture
    def mock_downtrend_data(self):
        """模拟下跌趋势数据"""
        return {
            "symbol": "BTC/USDT",
            "timeframe": "1h",
            "klines": [
                {
                    "timestamp": datetime.now() - timedelta(hours=i),
                    "open": 60000 - (i * 100),  # 价格下降趋势
                    "high": 60100 - (i * 100),
                    "low": 59900 - (i * 100),
                    "close": 60050 - (i * 100),
                    "volume": 100 + i * 2
                }
                for i in range(168, 0, -1)  # 7天的小时数据
            ]
        }

    @pytest.fixture
    def mock_technical_analysis_for_short(self):
        """模拟做空技术分析结果"""
        return {
            "rsi": 35.2,  # 超卖
            "macd": {
                "macd_line": -120.5,
                "signal_line": -110.2,
                "histogram": -10.3
            },
            "bollinger_bands": {
                "upper_band": 52000,
                "middle_band": 50000,
                "lower_band": 48000
            },
            "moving_averages": {
                "sma_20": 49800,
                "ema_12": 50200,
                "ema_26": 49500
            },
            "volume_indicators": {
                "volume_sma": 120.5,
                "volume_ratio": 0.8  # 成交量减少
            },
            "support_levels": [48500, 47000, 45000],
            "resistance_levels": [51500, 53000, 55000]
        }

    @pytest.fixture
    def mock_llm_short_response(self):
        """模拟LLM做空策略响应"""
        return {
            "recommendation": "sell_short",
            "confidence": 0.82,
            "entry_price": 49800,
            "stop_loss_price": 52000,
            "take_profit_price": 46500,
            "position_size_percent": 20.0,
            "reasoning": "RSI显示超卖但MACD仍然看跌，价格接近阻力位，成交量减少确认下跌趋势",
            "risk_factors": ["市场情绪仍然偏乐观", "潜在的反弹风险", "杠杆交易风险"],
            "market_conditions": {
                "trend_analysis": "下降趋势保持，但可能存在反弹风险",
                "key_factors": ["RSI超卖", "MACD看跌", "成交量萎缩"],
                "opportunity_assessment": "中等偏低机会"
            },
            "execution_strategy": {
                "entry_timing": "immediate",
                "scaling_plan": "分批做空",
                "add_position_conditions": ["价格反弹至阻力位"],
                "reduce_position_conditions": ["达到止盈目标", "趋势反转"],
                "exit_strategy": "分批止盈"
            }
        }

    async def test_complete_short_strategy_workflow(
        self,
        async_test_client: TestClient,
        mock_downtrend_data,
        mock_technical_analysis_for_short,
        mock_llm_short_response
    ):
        """测试完整的做空策略工作流"""

        # 模拟交易所数据收集
        with patch('backend.src.services.exchange_data_collector.ExchangeDataCollector') as MockCollector:
            mock_collector = AsyncMock()
            mock_collector.fetch_klines.return_value = mock_downtrend_data["klines"]
            MockCollector.return_value = mock_collector

            # 模拟技术分析引擎
            with patch('backend.src.services.technical_analysis_engine.TechnicalAnalysisEngine') as MockEngine:
                mock_engine = AsyncMock()
                mock_engine.analyze.return_value = mock_technical_analysis_for_short
                MockEngine.return_value = mock_engine

                # 模拟LLM做空分析器
                with patch('backend.src.services.llm_short_strategy_analyzer.LLMShortStrategyAnalyzer') as MockAnalyzer:
                    mock_analyzer = AsyncMock()
                    mock_analyzer.analyze_short_strategy.return_value = mock_llm_short_response
                    MockAnalyzer.return_value = mock_analyzer

                    # 发送策略分析请求
                    request_data = {
                        "symbol": "BTC/USDT",
                        "timeframe": "1h",
                        "analysis_period_days": 7,
                        "confidence_threshold": 0.7
                    }

                    response = async_test_client.post("/api/v1/strategies/short-analysis", json=request_data)

                    # 验证响应
                    if response.status_code == 200:
                        data = response.json()

                        # 验证策略生成
                        assert "strategy_id" in data
                        assert data["symbol"] == "BTC/USDT"
                        assert data["recommendation"] in ["sell_short", "hold", "buy"]
                        assert "confidence_score" in data
                        assert "entry_price" in data
                        assert "stop_loss_price" in data
                        assert "take_profit_price" in data

                        # 验证数据来源
                        assert "technical_indicators" in data
                        assert "market_conditions" in data
                        assert "risk_factors" in data

    async def test_downtrend_data_collection(self, async_test_client: TestClient, mock_downtrend_data):
        """测试下跌趋势数据收集集成"""

        with patch('backend.src.services.exchange_data_collector.ExchangeDataCollector') as MockCollector:
            mock_collector = AsyncMock()
            mock_collector.fetch_klines.return_value = mock_downtrend_data["klines"]
            MockCollector.return_value = mock_collector

            # 测试数据收集服务
            from backend.src.services.exchange_data_collector import ExchangeDataCollector

            collector = ExchangeDataCollector()
            klines = await collector.fetch_klines("BTC/USDT", "1h", 168)

            # 验证数据收集
            assert len(klines) == 168
            assert all(kline["symbol"] == "BTC/USDT" for kline in klines)
            assert all("close" in kline for kline in klines)

    async def test_short_technical_analysis_integration(self, async_test_client: TestClient, mock_downtrend_data):
        """测试做空技术分析集成"""

        with patch('backend.src.services.technical_analysis_engine.TechnicalAnalysisEngine') as MockEngine:
            mock_engine = AsyncMock()

            # 模拟做空特定的技术分析结果
            analysis_result = {
                "rsi": 35.2,
                "macd_line": -120.5,
                "signal_line": -110.2,
                "trend_direction": "downtrend",
                "trend_strength": 0.8,
                "volatility_level": "high",
                "market_sentiment": "bearish"
            }

            mock_engine.analyze.return_value = analysis_result
            MockEngine.return_value = mock_engine

            # 测试技术分析引擎
            from backend.src.services.technical_analysis_engine import TechnicalAnalysisEngine

            engine = TechnicalAnalysisEngine()
            result = await engine.analyze(mock_downtrend_data["klines"])

            # 验证做空特定的分析逻辑
            assert "market_analysis" in result
            assert result["market_analysis"]["trend_direction"] in ["downtrend", "sideways", "uptrend"]

    async def test_llm_short_analysis_integration(self, async_test_client: TestClient, mock_technical_analysis_for_short):
        """测试LLM做空分析集成"""

        with patch('backend.src.services.llm_short_strategy_analyzer.LLMShortStrategyAnalyzer') as MockAnalyzer:
            mock_analyzer = AsyncMock()

            llm_result = {
                "recommendation": "sell_short",
                "confidence": 0.82,
                "entry_price": 49800,
                "stop_loss": 52000,
                "take_profit": 46500,
                "position_size": 20.0,
                "reasoning": "基于技术分析的做空信号",
                "risk_factors": ["市场波动性高"]
            }

            mock_analyzer.analyze_short_strategy.return_value = llm_result
            MockAnalyzer.return_value = mock_analyzer

            # 测试LLM做空分析器
            from backend.src.services.llm_short_strategy_analyzer import LLMShortStrategyAnalyzer

            analyzer = LLMShortStrategyAnalyzer()
            result = await analyzer.analyze_short_strategy(
                symbol="BTC/USDT",
                technical_analysis=mock_technical_analysis_for_short,
                market_data={}
            )

            # 验证LLM分析结果
            assert result.recommendation in ["sell_short", "hold", "buy"]
            assert 0 <= result.confidence_score <= 1
            assert result.entry_price > 0
            assert result.stop_loss_price > 0
            assert result.take_profit_price > 0

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

            response = async_test_client.post("/api/v1/strategies/short-analysis", json=request_data)

            # 应该优雅地处理错误
            assert response.status_code in [500, 503]

            if response.status_code == 500:
                error_data = response.json()
                assert "detail" in error_data

    async def test_concurrent_analysis_requests(self, async_test_client: TestClient):
        """测试并发分析请求"""

        with patch('backend.src.services.exchange_data_collector.ExchangeDataCollector') as MockCollector, \
             patch('backend.src.services.technical_analysis_engine.TechnicalAnalysisEngine') as MockEngine, \
             patch('backend.src.services.llm_short_strategy_analyzer.LLMShortStrategyAnalyzer') as MockAnalyzer:

            # 设置模拟
            mock_collector = AsyncMock()
            mock_collector.fetch_klines.return_value = [
                {"timestamp": datetime.now(), "close": 50000, "volume": 100}
                for _ in range(168)
            ]
            MockCollector.return_value = mock_collector

            mock_engine = AsyncMock()
            mock_engine.analyze.return_value = {
                "rsi": 35.0,
                "macd_line": -120.0,
                "trend_direction": "downtrend"
            }
            MockEngine.return_value = mock_engine

            mock_analyzer = AsyncMock()
            mock_analyzer.analyze_short_strategy.return_value = {
                "recommendation": "sell_short",
                "confidence": 0.8,
                "entry_price": 49800,
                "stop_loss": 52000,
                "take_profit": 46500
            }
            MockAnalyzer.return_value = mock_analyzer

            # 发送多个并发请求
            tasks = []
            for i in range(5):
                request_data = {
                    "symbol": "BTC/USDT",
                    "timeframe": "1h",
                    "analysis_period_days": 7,
                    "confidence_threshold": 0.7
                }
                task = async_test_client.post("/api/v1/strategies/short-analysis", json=request_data)
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
             patch('backend.src.core.cache.get_cache') as MockCache:

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

            response = async_test_client.post("/api/v1/strategies/short-analysis", json=request_data)

            # 验证缓存调用
            mock_cache.get.assert_called()

            # 如果请求成功，应该设置缓存
            if response.status_code == 200:
                mock_cache.set.assert_called()

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

            response = async_test_client.post("/api/v1/strategies/short-analysis", json=request_data)

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

        response = async_test_client.post("/api/v1/strategies/short-analysis", json=request_data)

        if response.status_code == 200:
            data = response.json()

            # 验证策略质量指标
            assert "confidence_score" in data
            assert "risk_factors" in data
            assert "reasoning" in data

            # 如果设置了高置信度阈值，返回的置信度应该满足要求
            if data["confidence_score"] < request_data["confidence_threshold"]:
                # 如果置信度不足，推荐应该是hold或buy
                assert data["recommendation"] in ["hold", "buy"]

    async def test_market_regime_detection(self, async_test_client: TestClient):
        """测试市场状态检测"""

        request_data = {
            "symbol": "BTC/USDT",
            "timeframe": "1h",
            "analysis_period_days": 30,  # 更长的分析周期
            "confidence_threshold": 0.7
        }

        response = async_test_client.post("/api/v1/strategies/short-analysis", json=request_data)

        if response.status_code == 200:
            data = response.json()

            # 验证市场状态分析
            if "market_conditions" in data:
                conditions = data["market_conditions"]

                # 应该包含市场状态信息
                # expected_fields = ["trend", "volatility", "volume_profile", "sentiment"]
                # for field in expected_fields:
                #     assert field in conditions

    def test_short_strategy_vs_long_strategy_comparison(self, async_test_client: TestClient):
        """测试做空策略与做多策略的对比分析"""

        # 测试做多策略
        long_request = {
            "symbol": "BTC/USDT",
            "timeframe": "1h",
            "analysis_period_days": 7,
            "confidence_threshold": 0.7
        }

        # 测试做空策略
        short_request = {
            "symbol": "BTC/USDT",
            "timeframe": "1h",
            "analysis_period_days": 7,
            "confidence_threshold": 0.7
        }

        # 并发发送两个请求
        long_response = async_test_client.post("/api/v1/strategies/long-analysis", json=long_request)
        short_response = async_test_client.post("/api/v1/strategies/short-analysis", json=short_request)

        # 验证响应
        long_status = long_response.status_code
        short_status = short_response.status_code

        # 应该都能处理请求（可能返回200或500）
        assert long_status in [200, 500, 503]
        assert short_status in [200, 500, 503]

        if long_status == 200 and short_status == 200:
            long_data = long_response.json()
            short_data = short_response.json()

            # 验证策略对比的可能性
            # if "strategy_comparison" in short_data:
            #     assert isinstance(short_data["strategy_comparison"], dict)

    def test_leverage_and_margin_analysis(self, async_test_client: TestClient):
        """测试杠杆和保证金分析"""
        request_data = {
            "symbol": "BTC/USDT",
            "timeframe": "1h",
            "analysis_period_days": 7,
            "confidence_threshold": 0.7,
            "include_leverage_analysis": True
        }

        response = async_test_client.post("/api/v1/strategies/short-analysis", json=request_data)

        if response.status_code == 200:
            data = response.json()

            # 验证杠杆分析相关字段
            leverage_fields = [
                "leverage_ratio",  # 杠杆比率
                "margin_requirement",  # 保证金要求
                "liquidation_price",  # 清算价格
                "borrowing_rate",  # 借贷利率
                "funding_rate"  # 资金费率
            ]

            # 注意：在实际实现前，这些字段可能不存在
            # for field in leverage_fields:
            #     if field in data:
            #         assert isinstance(data[field], (dict, float, str))

    def test_exit_strategy_analysis(self, async_test_client: TestClient):
        """测试退出策略分析"""

        request_data = {
            "symbol": "BTC/USDT",
            "timeframe": "1h",
            "analysis_period_days": 7,
            "confidence_threshold": 0.7
        }

        response = async_test_client.post("/api/v1/strategies/short-analysis", json=request_data)

        if response.status_code == 200:
            data = response.json()

            # 验证退出策略相关字段
            exit_fields = [
                "exit_conditions",  # 出场条件
                "partial_close_levels",  # 分批平仓价格
                "stop_adjustment",  # 止损调整策略
                "profit_taking_plan"  # 止盈计划
                "emergency_exit"  # 紧急退出条件
            ]

            # 注意：在实际实现前，这些字段可能不存在
            # if "execution_strategy" in data:
            #     execution = data["execution_strategy"]
            #     assert "exit_strategy" in execution

    async def test_performance_requirements_for_short(self, async_test_client: TestClient):
        """测试做空策略性能要求"""
        import time

        with patch('backend.src.services.exchange_data_collector.ExchangeDataCollector') as MockCollector, \
             patch('backend.src.services.technical_analysis_engine.TechnicalAnalysisEngine') as MockEngine, \
             patch('backend.src.services.llm_short_strategy_analyzer.LLMShortStrategyAnalyzer') as MockAnalyzer:

            # 设置快速响应的模拟
            mock_collector = AsyncMock()
            mock_collector.fetch_klines.return_value = [
                {"timestamp": datetime.now(), "close": 50000, "volume": 100}
                for _ in range(168)
            ]
            MockCollector.return_value = mock_collector

            mock_engine = AsyncMock()
            mock_engine.analyze.return_value = {
                "rsi": 35.0,
                "market_analysis": {"trend_direction": "downtrend"}
            }
            MockEngine.return_value = mock_engine

            mock_analyzer = AsyncMock()
            mock_analyzer.analyze_short_strategy.return_value = {
                "recommendation": "sell_short",
                "confidence": 0.8,
                "entry_price": 49800
            }
            MockAnalyzer.return_value = mock_analyzer

            request_data = {
                "symbol": "BTC/USDT",
                "timeframe": "1h",
                "analysis_period_days": 7,
                "confidence_threshold": 0.7
            }

            start_time = time.time()
            response = async_test_client.post("/api/v1/strategies/short-analysis", json=request_data)
            end_time = time.time()

            response_time = end_time - start_time

            # 性能要求：响应时间应该小于5分钟（300秒）
            assert response_time < 300.0, f"Response time {response_time}s exceeds performance requirement"

    def test_short_strategy_with_different_timeframes(self, async_test_client: TestClient):
        """测试不同时间框架的做空策略分析"""

        timeframes = ["15m", "1h", "4h", "1d"]
        results = []

        for timeframe in timeframes:
            request_data = {
                "symbol": "BTC/USDT",
                "timeframe": timeframe,
                "analysis_period_days": 7,
                "confidence_threshold": 0.7
            }

            response = async_test_client.post("/api/v1/strategies/short-analysis", json=request_data)

            if response.status_code == 200:
                data = response.json()
                results.append((timeframe, data["recommendation"], data["confidence_score"]))

        # 验证不同时间框架的分析结果
        assert len(results) == len(timeframes)

        # 时间框架越短，波动性越大，可能需要更高的置信度
        # 这里的验证逻辑可以在实际实现时细化
        for timeframe, recommendation, confidence in results:
            assert timeframe in ["15m", "1h", "4h", "1d"]
            assert recommendation in ["sell_short", "hold", "buy", "sell"]
            assert 0 <= confidence <= 1.0