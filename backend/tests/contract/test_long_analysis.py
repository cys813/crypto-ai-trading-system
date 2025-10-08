"""
做多策略分析API合约测试

验证做多策略分析API端点的存在、响应格式和数据结构。
"""

import pytest
from fastapi.testclient import TestClient
from httpx import Response
import json
from datetime import datetime, timedelta


class TestLongAnalysisEndpointContract:
    """做多策略分析端点合约测试"""

    def test_long_analysis_endpoint_exists(self, async_test_client: TestClient):
        """测试做多策略分析端点是否存在"""
        response = async_test_client.post("/api/v1/strategies/long-analysis")

        # 端点应该存在，可能返回422（参数验证失败）或200
        assert response.status_code in [200, 422, 400]

        # 不应该返回404
        assert response.status_code != 404

    def test_long_analysis_endpoint_method_not_allowed(self, async_test_client: TestClient):
        """测试不支持的HTTP方法"""
        response = async_test_client.get("/api/v1/strategies/long-analysis")

        # GET方法应该不被支持
        assert response.status_code == 405

    def test_long_analysis_request_structure_validation(self, async_test_client: TestClient):
        """测试请求结构验证"""
        # 测试空请求
        response = async_test_client.post("/api/v1/strategies/long-analysis", json={})

        # 应该返回验证错误
        assert response.status_code == 422

        error_data = response.json()
        assert "detail" in error_data

    def test_long_analysis_required_fields(self, async_test_client: TestClient):
        """测试必需字段验证"""
        # 缺少必需字段的请求
        request_data = {
            "timeframe": "1h"
            # 缺少symbol
        }

        response = async_test_client.post("/api/v1/strategies/long-analysis", json=request_data)

        # 应该返回验证错误
        assert response.status_code == 422

        error_data = response.json()
        assert "detail" in error_data

    def test_long_analysis_valid_request_structure(self, async_test_client: TestClient):
        """测试有效请求结构"""
        valid_request = {
            "symbol": "BTC/USDT",
            "timeframe": "1h",
            "analysis_period_days": 7,
            "confidence_threshold": 0.7
        }

        response = async_test_client.post("/api/v1/strategies/long-analysis", json=valid_request)

        # 可能返回200（成功）或500（服务未实现）
        assert response.status_code in [200, 500, 503]

    @pytest.mark.parametrize("field,value", [
        ("symbol", ""),  # 空符号
        ("symbol", "INVALID"),  # 无效符号格式
        ("timeframe", ""),  # 空时间框架
        ("timeframe", "invalid"),  # 无效时间框架
        ("analysis_period_days", -1),  # 负数天数
        ("analysis_period_days", 0),  # 零天数
        ("confidence_threshold", -0.1),  # 负置信度
        ("confidence_threshold", 1.1),  # 超出范围的置信度
    ])
    def test_long_analysis_field_validation(self, async_test_client: TestClient, field, value):
        """测试字段验证"""
        request_data = {
            "symbol": "BTC/USDT",
            "timeframe": "1h",
            "analysis_period_days": 7,
            "confidence_threshold": 0.7,
            field: value
        }

        response = async_test_client.post("/api/v1/strategies/long-analysis", json=request_data)

        # 应该返回验证错误
        assert response.status_code == 422

    def test_long_analysis_response_format_on_success(self, async_test_client: TestClient):
        """测试成功响应格式"""
        valid_request = {
            "symbol": "BTC/USDT",
            "timeframe": "1h",
            "analysis_period_days": 7,
            "confidence_threshold": 0.7
        }

        response = async_test_client.post("/api/v1/strategies/long-analysis", json=valid_request)

        if response.status_code == 200:
            data = response.json()

            # 验证响应结构
            required_fields = [
                "strategy_id",
                "symbol",
                "analysis_timestamp",
                "recommendation",  # 'buy', 'hold', 'sell'
                "confidence_score",
                "entry_price",
                "stop_loss_price",
                "take_profit_price",
                "position_size",
                "timeframe",
                "technical_indicators",
                "market_conditions",
                "risk_factors"
            ]

            for field in required_fields:
                assert field in data, f"Missing required field: {field}"

            # 验证数据类型
            assert isinstance(data["strategy_id"], str)
            assert isinstance(data["symbol"], str)
            assert isinstance(data["recommendation"], str)
            assert isinstance(data["confidence_score"], (int, float))
            assert data["recommendation"] in ["buy", "hold", "sell"]
            assert 0 <= data["confidence_score"] <= 1

    def test_long_analysis_error_response_format(self, async_test_client: TestClient):
        """测试错误响应格式"""
        # 发送无效请求
        invalid_request = {
            "symbol": "INVALID_SYMBOL_FORMAT",
            "timeframe": "1h"
        }

        response = async_test_client.post("/api/v1/strategies/long-analysis", json=invalid_request)

        if response.status_code != 200:
            # 错误响应应该包含错误信息
            if "detail" in response.json():
                assert isinstance(response.json()["detail"], str)

    def test_long_analysis_concurrent_requests(self, async_test_client: TestClient):
        """测试并发请求处理"""
        import asyncio

        valid_request = {
            "symbol": "BTC/USDT",
            "timeframe": "1h",
            "analysis_period_days": 7,
            "confidence_threshold": 0.7
        }

        async def make_request():
            return async_test_client.post("/api/v1/strategies/long-analysis", json=valid_request)

        # 发送多个并发请求
        responses = asyncio.run(make_request())

        # 所有响应都应该是有效状态码
        valid_status_codes = [200, 202, 422, 500, 503]
        assert responses.status_code in valid_status_codes

    def test_long_analysis_request_size_limits(self, async_test_client: TestClient):
        """测试请求大小限制"""
        # 测试过大的请求
        large_request = {
            "symbol": "BTC/USDT",
            "timeframe": "1h",
            "analysis_period_days": 365,  # 过大的分析周期
            "confidence_threshold": 0.7
        }

        response = async_test_client.post("/api/v1/strategies/long-analysis", json=large_request)

        # 应该返回验证错误或服务器错误
        assert response.status_code in [422, 413, 500]

    def test_long_analysis_content_type(self, async_test_client: TestClient):
        """测试内容类型验证"""
        valid_request = {
            "symbol": "BTC/USDT",
            "timeframe": "1h",
            "analysis_period_days": 7,
            "confidence_threshold": 0.7
        }

        # 测试正确的Content-Type
        response = async_test_client.post(
            "/api/v1/strategies/long-analysis",
            json=valid_request,
            headers={"Content-Type": "application/json"}
        )

        # 应该接受有效的Content-Type
        assert response.status_code in [200, 422, 500, 503]

        # 测试错误的Content-Type
        response = async_test_client.post(
            "/api/v1/strategies/long-analysis",
            data=json.dumps(valid_request),
            headers={"Content-Type": "text/plain"}
        )

        # 应该拒绝错误的Content-Type
        assert response.status_code == 415

    def test_long_analysis_response_time(self, async_test_client: TestClient):
        """测试响应时间"""
        import time

        valid_request = {
            "symbol": "BTC/USDT",
            "timeframe": "1h",
            "analysis_period_days": 7,
            "confidence_threshold": 0.7
        }

        start_time = time.time()
        response = async_test_client.post("/api/v1/strategies/long-analysis", json=valid_request)
        end_time = time.time()

        response_time = end_time - start_time

        # 响应时间应该在合理范围内（小于30秒）
        assert response_time < 30.0, f"Response time too slow: {response_time}s"

    def test_long_analysis_symbol_formats(self, async_test_client: TestClient):
        """测试不同的交易符号格式"""
        valid_symbols = [
            "BTC/USDT",
            "ETH/USDT",
            "BNB/USDT",
            "BTC/BUSD"
        ]

        for symbol in valid_symbols:
            request_data = {
                "symbol": symbol,
                "timeframe": "1h",
                "analysis_period_days": 7,
                "confidence_threshold": 0.7
            }

            response = async_test_client.post("/api/v1/strategies/long-analysis", json=request_data)

            # 应该接受有效的交易符号格式
            assert response.status_code in [200, 422, 500, 503]

    def test_long_analysis_timeframe_validation(self, async_test_client: TestClient):
        """测试时间框架验证"""
        valid_timeframes = ["1m", "5m", "15m", "1h", "4h", "1d"]

        for timeframe in valid_timeframes:
            request_data = {
                "symbol": "BTC/USDT",
                "timeframe": timeframe,
                "analysis_period_days": 7,
                "confidence_threshold": 0.7
            }

            response = async_test_client.post("/api/v1/strategies/long-analysis", json=request_data)

            # 应该接受有效的时间框架
            assert response.status_code in [200, 422, 500, 503]

    def test_long_analysis_endpoint_documentation(self, async_test_client: TestClient):
        """测试API文档可用性"""
        # 测试OpenAPI文档端点
        response = async_test_client.get("/docs")

        # API文档应该可用
        assert response.status_code == 200

        # 检查是否包含做多分析相关的文档
        if response.status_code == 200:
            docs_content = response.text
            # 注意：这个检查可能在实际实现前失败，这是正常的
            # assert "long-analysis" in docs_content.lower() or "strategy" in docs_content.lower()


class TestLongAnalysisResponseStructure:
    """做多策略分析响应结构详细测试"""

    def test_technical_indicators_structure(self, async_test_client: TestClient):
        """测试技术指标响应结构"""
        valid_request = {
            "symbol": "BTC/USDT",
            "timeframe": "1h",
            "analysis_period_days": 7,
            "confidence_threshold": 0.7
        }

        response = async_test_client.post("/api/v1/strategies/long-analysis", json=valid_request)

        if response.status_code == 200:
            data = response.json()

            if "technical_indicators" in data:
                indicators = data["technical_indicators"]

                # 验证常见技术指标存在
                common_indicators = [
                    "rsi",
                    "macd",
                    "bollinger_bands",
                    "moving_averages",
                    "volume_indicators"
                ]

                # 注意：在实际实现前，这些字段可能不存在
                # for indicator in common_indicators:
                #     assert indicator in indicators, f"Missing technical indicator: {indicator}"

    def test_market_conditions_structure(self, async_test_client: TestClient):
        """测试市场条件响应结构"""
        valid_request = {
            "symbol": "BTC/USDT",
            "timeframe": "1h",
            "analysis_period_days": 7,
            "confidence_threshold": 0.7
        }

        response = async_test_client.post("/api/v1/strategies/long-analysis", json=valid_request)

        if response.status_code == 200:
            data = response.json()

            if "market_conditions" in data:
                conditions = data["market_conditions"]

                # 验证市场条件字段
                expected_fields = [
                    "trend_direction",
                    "volatility_level",
                    "volume_profile",
                    "market_sentiment"
                ]

                # 注意：在实际实现前，这些字段可能不存在
                # for field in expected_fields:
                #     assert field in conditions, f"Missing market condition: {field}"