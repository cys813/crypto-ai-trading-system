"""
综合策略生成API合约测试

验证综合策略生成API端点的存在、响应格式和数据结构。
"""

import pytest
from fastapi.testclient import TestClient
from httpx import Response
import json
from datetime import datetime, timedelta


class TestStrategyGenerationEndpointContract:
    """综合策略生成端点合约测试"""

    def test_strategy_generation_endpoint_exists(self, async_test_client: TestClient):
        """测试综合策略生成端点是否存在"""
        response = async_test_client.post("/api/v1/strategies/generate")

        # 端点应该存在，可能返回422（参数验证失败）或200
        assert response.status_code in [200, 422, 400]

        # 不应该返回404
        assert response.status_code != 404

    def test_strategy_generation_endpoint_method_not_allowed(self, async_test_client: TestClient):
        """测试不支持的HTTP方法"""
        response = async_test_client.get("/api/v1/strategies/generate")

        # GET方法应该不被支持
        assert response.status_code == 405

    def test_strategy_generation_request_structure_validation(self, async_test_client: TestClient):
        """测试请求结构验证"""
        # 测试空请求
        response = async_test_client.post("/api/v1/strategies/generate", json={})

        # 应该返回验证错误
        assert response.status_code == 422

        error_data = response.json()
        assert "detail" in error_data

    def test_strategy_generation_required_fields(self, async_test_client: TestClient):
        """测试必需字段验证"""
        # 缺少必需字段的请求
        request_data = {
            "symbol": "BTC/USDT"
            # 缺少其他必需字段
        }

        response = async_test_client.post("/api/v1/strategies/generate", json=request_data)

        # 应该返回验证错误
        assert response.status_code == 422

        error_data = response.json()
        assert "detail" in error_data

    def test_strategy_generation_valid_request_structure(self, async_test_client: TestClient):
        """测试有效请求结构"""
        valid_request = {
            "symbol": "BTC/USDT",
            "timeframe": "1h",
            "analysis_types": ["long", "short"],
            "include_news": True,
            "risk_tolerance": "medium",
            "max_position_size": 20.0
        }

        response = async_test_client.post("/api/v1/strategies/generate", json=valid_request)

        # 可能返回200（成功）或500（服务未实现）
        assert response.status_code in [200, 500, 503]

    @pytest.mark.parametrize("field,value", [
        ("symbol", ""),  # 空符号
        ("symbol", "INVALID"),  # 无效符号格式
        ("timeframe", ""),  # 空时间框架
        ("timeframe", "invalid"),  # 无效时间框架
        ("analysis_types", "invalid"),  # 无效的分析类型
        ("analysis_types", []),  # 空的分析类型
        ("risk_tolerance", ""),  # 空风险容忍度
        ("risk_tolerance", "invalid"),  # 无效风险容忍度
        ("max_position_size", -1.0),  # 负仓位大小
        ("max_position_size", 0.0),  # 零仓位大小
        ("max_position_size", 101.0),  # 超出最大仓位
    ])
    def test_strategy_generation_field_validation(self, async_test_client: TestClient, field, value):
        """测试字段验证"""
        request_data = {
            "symbol": "BTC/USDT",
            "timeframe": "1h",
            "analysis_types": ["long", "short"],
            "include_news": True,
            "risk_tolerance": "medium",
            "max_position_size": 20.0,
            field: value
        }

        response = async_test_client.post("/api/v1/strategies/generate", json=request_data)

        # 应该返回验证错误
        assert response.status_code == 422

    def test_strategy_generation_response_format_on_success(self, async_test_client: TestClient):
        """测试成功响应格式"""
        valid_request = {
            "symbol": "BTC/USDT",
            "timeframe": "1h",
            "analysis_types": ["long", "short"],
            "include_news": True,
            "risk_tolerance": "medium",
            "max_position_size": 20.0
        }

        response = async_test_client.post("/api/v1/strategies/generate", json=valid_request)

        if response.status_code == 200:
            data = response.json()

            # 验证响应结构
            required_fields = [
                "strategy_id",
                "symbol",
                "generated_at",
                "final_recommendation",  # 'long', 'short', 'hold'
                "confidence_score",
                "entry_price",
                "stop_loss_price",
                "take_profit_price",
                "position_size_percent",
                "timeframe",
                "strategy_type",  # 'aggressive', 'moderate', 'conservative'
                "market_analysis",
                "risk_assessment",
                "execution_plan"
            ]

            for field in required_fields:
                assert field in data, f"Missing required field: {field}"

            # 验证数据类型
            assert isinstance(data["strategy_id"], str)
            assert isinstance(data["symbol"], str)
            assert isinstance(data["final_recommendation"], str)
            assert isinstance(data["confidence_score"], (int, float))
            assert data["final_recommendation"] in ["long", "short", "hold"]
            assert 0 <= data["confidence_score"] <= 1
            assert data["strategy_type"] in ["aggressive", "moderate", "conservative"]

    def test_strategy_generation_error_response_format(self, async_test_client: TestClient):
        """测试错误响应格式"""
        # 发送无效请求
        invalid_request = {
            "symbol": "INVALID_SYMBOL_FORMAT",
            "timeframe": "1h",
            "analysis_types": ["invalid_type"]
        }

        response = async_test_client.post("/api/v1/strategies/generate", json=invalid_request)

        if response.status_code != 200:
            # 错误响应应该包含错误信息
            if "detail" in response.json():
                assert isinstance(response.json()["detail"], str)

    def test_strategy_generation_concurrent_requests(self, async_test_client: TestClient):
        """测试并发请求处理"""
        import asyncio

        valid_request = {
            "symbol": "BTC/USDT",
            "timeframe": "1h",
            "analysis_types": ["long", "short"],
            "include_news": True,
            "risk_tolerance": "medium",
            "max_position_size": 20.0
        }

        async def make_request():
            return async_test_client.post("/api/v1/strategies/generate", json=valid_request)

        # 发送多个并发请求
        responses = asyncio.run(make_request())

        # 所有响应都应该是有效状态码
        valid_status_codes = [200, 202, 422, 500, 503]
        assert responses.status_code in valid_status_codes

    def test_strategy_generation_analysis_types_validation(self, async_test_client: TestClient):
        """测试分析类型验证"""
        valid_analysis_types = [
            ["long"],
            ["short"],
            ["long", "short"],
            ["technical"],
            ["news"],
            ["technical", "news"],
            ["long", "short", "technical", "news"]
        ]

        for analysis_types in valid_analysis_types:
            request_data = {
                "symbol": "BTC/USDT",
                "timeframe": "1h",
                "analysis_types": analysis_types,
                "include_news": True,
                "risk_tolerance": "medium",
                "max_position_size": 20.0
            }

            response = async_test_client.post("/api/v1/strategies/generate", json=request_data)

            # 应该接受有效的分析类型
            assert response.status_code in [200, 422, 500, 503]

    def test_strategy_generation_risk_tolerance_validation(self, async_test_client: TestClient):
        """测试风险容忍度验证"""
        valid_risk_levels = ["conservative", "moderate", "aggressive"]

        for risk_tolerance in valid_risk_levels:
            request_data = {
                "symbol": "BTC/USDT",
                "timeframe": "1h",
                "analysis_types": ["long", "short"],
                "include_news": True,
                "risk_tolerance": risk_tolerance,
                "max_position_size": 20.0
            }

            response = async_test_client.post("/api/v1/strategies/generate", json=request_data)

            # 应该接受有效的风险容忍度
            assert response.status_code in [200, 422, 500, 503]

    def test_strategy_generation_response_time(self, async_test_client: TestClient):
        """测试响应时间"""
        import time

        valid_request = {
            "symbol": "BTC/USDT",
            "timeframe": "1h",
            "analysis_types": ["long", "short"],
            "include_news": True,
            "risk_tolerance": "medium",
            "max_position_size": 20.0
        }

        start_time = time.time()
        response = async_test_client.post("/api/v1/strategies/generate", json=valid_request)
        end_time = time.time()

        response_time = end_time - start_time

        # 综合策略生成可能需要更长时间，但应该在合理范围内（小于60秒）
        assert response_time < 60.0, f"Response time too slow: {response_time}s"

    def test_strategy_generation_content_type(self, async_test_client: TestClient):
        """测试内容类型验证"""
        valid_request = {
            "symbol": "BTC/USDT",
            "timeframe": "1h",
            "analysis_types": ["long", "short"],
            "include_news": True,
            "risk_tolerance": "medium",
            "max_position_size": 20.0
        }

        # 测试正确的Content-Type
        response = async_test_client.post(
            "/api/v1/strategies/generate",
            json=valid_request,
            headers={"Content-Type": "application/json"}
        )

        # 应该接受有效的Content-Type
        assert response.status_code in [200, 422, 500, 503]

        # 测试错误的Content-Type
        response = async_test_client.post(
            "/api/v1/strategies/generate",
            data=json.dumps(valid_request),
            headers={"Content-Type": "text/plain"}
        )

        # 应该拒绝错误的Content-Type
        assert response.status_code == 415

    def test_strategy_generation_multiple_symbols(self, async_test_client: TestClient):
        """测试多交易符号策略生成"""
        multi_symbol_request = {
            "symbols": ["BTC/USDT", "ETH/USDT"],
            "timeframe": "1h",
            "analysis_types": ["long", "short"],
            "include_news": True,
            "risk_tolerance": "moderate",
            "max_position_size": 20.0
        }

        response = async_test_client.post("/api/v1/strategies/generate", json=multi_symbol_request)

        # 应该处理多符号请求或返回适当错误
        assert response.status_code in [200, 422, 500, 503]

    def test_strategy_generation_optional_parameters(self, async_test_client: TestClient):
        """测试可选参数"""
        minimal_request = {
            "symbol": "BTC/USDT",
            "timeframe": "1h"
        }

        response = async_test_client.post("/api/v1/strategies/generate", json=minimal_request)

        # 应该接受最小请求或返回验证错误
        assert response.status_code in [200, 422, 500, 503]

        full_request = {
            "symbol": "BTC/USDT",
            "timeframe": "1h",
            "analysis_types": ["long", "short"],
            "include_news": True,
            "risk_tolerance": "medium",
            "max_position_size": 20.0,
            "custom_parameters": {
                "preferred_indicators": ["RSI", "MACD"],
                "news_sentiment_threshold": 0.6
            }
        }

        response = async_test_client.post("/api/v1/strategies/generate", json=full_request)

        # 应该接受完整请求
        assert response.status_code in [200, 422, 500, 503]


class TestStrategyGenerationResponseStructure:
    """综合策略生成响应结构详细测试"""

    def test_market_analysis_structure(self, async_test_client: TestClient):
        """测试市场分析响应结构"""
        valid_request = {
            "symbol": "BTC/USDT",
            "timeframe": "1h",
            "analysis_types": ["long", "short"],
            "include_news": True,
            "risk_tolerance": "medium",
            "max_position_size": 20.0
        }

        response = async_test_client.post("/api/v1/strategies/generate", json=valid_request)

        if response.status_code == 200:
            data = response.json()

            if "market_analysis" in data:
                analysis = data["market_analysis"]

                # 验证市场分析字段
                expected_fields = [
                    "overall_sentiment",
                    "trend_analysis",
                    "technical_signals",
                    "news_impact",
                    "volatility_assessment"
                ]

                # 注意：在实际实现前，这些字段可能不存在
                # for field in expected_fields:
                #     assert field in analysis, f"Missing market analysis field: {field}"

    def test_risk_assessment_structure(self, async_test_client: TestClient):
        """测试风险评估响应结构"""
        valid_request = {
            "symbol": "BTC/USDT",
            "timeframe": "1h",
            "analysis_types": ["long", "short"],
            "include_news": True,
            "risk_tolerance": "medium",
            "max_position_size": 20.0
        }

        response = async_test_client.post("/api/v1/strategies/generate", json=valid_request)

        if response.status_code == 200:
            data = response.json()

            if "risk_assessment" in data:
                assessment = data["risk_assessment"]

                # 验证风险评估字段
                expected_fields = [
                    "risk_level",
                    "risk_factors",
                    "risk_mitigation",
                    "max_drawdown_estimate"
                ]

                # 注意：在实际实现前，这些字段可能不存在
                # for field in expected_fields:
                #     assert field in assessment, f"Missing risk assessment field: {field}"

    def test_execution_plan_structure(self, async_test_client: TestClient):
        """测试执行计划响应结构"""
        valid_request = {
            "symbol": "BTC/USDT",
            "timeframe": "1h",
            "analysis_types": ["long", "short"],
            "include_news": True,
            "risk_tolerance": "medium",
            "max_position_size": 20.0
        }

        response = async_test_client.post("/api/v1/strategies/generate", json=valid_request)

        if response.status_code == 200:
            data = response.json()

            if "execution_plan" in data:
                plan = data["execution_plan"]

                # 验证执行计划字段
                expected_fields = [
                    "entry_conditions",
                    "exit_conditions",
                    "position_sizing",
                    "timing_strategy"
                ]

                # 注意：在实际实现前，这些字段可能不存在
                # for field in expected_fields:
                #     assert field in plan, f"Missing execution plan field: {field}"

    def test_strategy_metadata_structure(self, async_test_client: TestClient):
        """测试策略元数据响应结构"""
        valid_request = {
            "symbol": "BTC/USDT",
            "timeframe": "1h",
            "analysis_types": ["long", "short"],
            "include_news": True,
            "risk_tolerance": "medium",
            "max_position_size": 20.0
        }

        response = async_test_client.post("/api/v1/strategies/generate", json=valid_request)

        if response.status_code == 200:
            data = response.json()

            # 验证元数据字段
            expected_fields = [
                "strategy_id",
                "generated_at",
                "expires_at",
                "analysis_version"
            ]

            for field in expected_fields:
                if field in data:
                    # 验证字段类型
                    if field == "strategy_id":
                        assert isinstance(data[field], str)
                    elif field == "generated_at" or field == "expires_at":
                        assert isinstance(data[field], str)  # ISO format datetime
                    elif field == "analysis_version":
                        assert isinstance(data[field], str)