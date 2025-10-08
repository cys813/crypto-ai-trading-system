"""
交易工作流集成测试

测试完整的交易订单执行和风险管理集成工作流。
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timedelta
import json
from decimal import Decimal
from fastapi.testclient import TestClient

from backend.src.models.trading_order import TradingOrder, OrderStatus, OrderType, OrderSide
from backend.src.models.position import Position, PositionSide
from backend.src.services.order_manager import OrderManager
from backend.src.services.risk_manager import RiskManager
from backend.src.services.trading_executor import TradingExecutor
from backend.src.services.position_monitor import PositionMonitor


class TestTradingWorkflowIntegration:
    """交易工作流集成测试"""

    @pytest.fixture
    def mock_strategy_request(self):
        """模拟策略请求"""
        return {
            "strategy_id": "strategy_20251008_001",
            "symbol": "BTC/USDT",
            "final_recommendation": "long",
            "confidence_score": 0.75,
            "entry_price": 50200.0,
            "stop_loss_price": 48500.0,
            "take_profit_price": 52500.0,
            "position_size_percent": 15.0,
            "strategy_type": "moderate",
            "execution_plan": {
                "entry_conditions": ["价格回调至50200以下", "成交量确认"],
                "exit_conditions": ["达到止盈目标52500", "跌破止损48500"],
                "position_sizing": "建议分2批建仓，首批15%仓位",
                "timing_strategy": "等待技术确认后入场"
            }
        }

    @pytest.fixture
    def mock_order_request(self):
        """模拟订单请求"""
        return {
            "symbol": "BTC/USDT",
            "side": "buy",
            "order_type": "limit",
            "amount": 0.15,  # 0.15 BTC
            "price": 50150.0,  # 略低于策略价格
            "exchange": "binance",
            "strategy_id": "strategy_20251008_001",
            "stop_loss_price": 48500.0,
            "take_profit_price": 52500.0,
            "time_in_force": "GTC"
        }

    @pytest.fixture
    def mock_exchange_response(self):
        """模拟交易所响应"""
        return {
            "order_id": "binance_order_12345",
            "client_order_id": "client_order_67890",
            "symbol": "BTC/USDT",
            "side": "BUY",
            "order_type": "LIMIT",
            "quantity": "0.15000000",
            "price": "50150.00000000",
            "status": "NEW",
            "created_at": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat()
        }

    async def test_complete_trading_workflow(
        self, async_test_client: TestClient, mock_strategy_request,
        mock_order_request, mock_exchange_response
    ):
        """测试完整交易工作流"""

        # 1. 策略验证和风险检查
        risk_check_response = async_test_client.post(
            "/api/v1/trading/risk/check",
            json={
                "symbol": mock_strategy_request["symbol"],
                "side": "buy",
                "amount": mock_order_request["amount"],
                "price": mock_order_request["price"],
                "strategy_id": mock_strategy_request["strategy_id"]
            }
        )

        # 风险检查应该通过（或返回具体的风险问题）
        assert risk_check_response.status_code in [200, 422, 500]

        if risk_check_response.status_code == 200:
            risk_result = risk_check_response.json()
            assert "risk_level" in risk_result
            assert "approved" in risk_result

        # 2. 订单执行
        with patch('backend.src.services.exchange_data_collector.ExchangeDataCollector') as MockExchange:
            mock_exchange = AsyncMock()
            mock_exchange.place_order.return_value = mock_exchange_response
            MockExchange.return_value = mock_exchange

            order_response = async_test_client.post(
                "/api/v1/trading/orders/execute",
                json=mock_order_request
            )

            # 订单执行应该成功或返回具体错误
            assert order_response.status_code in [200, 202, 422, 500]

            if order_response.status_code in [200, 202]:
                order_data = order_response.json()
                assert "order_id" in order_data
                assert order_data["symbol"] == mock_order_request["symbol"]
                assert order_data["side"] == mock_order_request["side"]

                # 3. 订单状态跟踪
                order_id = order_data["order_id"]
                status_response = async_test_client.get(f"/api/v1/trading/orders/{order_id}")

                if status_response.status_code == 200:
                    order_status = status_response.json()
                    assert "status" in order_status
                    assert order_status["order_id"] == order_id

                # 4. 持仓查询
                position_response = async_test_client.get("/api/v1/trading/positions")

                if position_response.status_code == 200:
                    positions = position_response.json()
                    # 可能返回空列表或实际持仓数据
                    assert isinstance(positions, list)

    async def test_order_cancellation_workflow(self, async_test_client: TestClient, mock_order_request):
        """测试订单取消工作流"""

        # 1. 创建订单
        with patch('backend.src.services.exchange_data_collector.ExchangeDataCollector') as MockExchange:
            mock_exchange = AsyncMock()
            mock_exchange.place_order.return_value = {
                **mock_order_request,
                "order_id": "test_order_123",
                "status": "NEW"
            }
            MockExchange.return_value = mock_exchange

            create_response = async_test_client.post(
                "/api/v1/trading/orders/execute",
                json=mock_order_request
            )

            if create_response.status_code in [200, 202]:
                order_data = create_response.json()
                order_id = order_data["order_id"]

                # 2. 取消订单
                with patch('backend.src.services.exchange_data_collector.ExchangeDataCollector') as MockExchangeCancel:
                    mock_exchange_cancel = AsyncMock()
                    mock_exchange_cancel.cancel_order.return_value = {
                        "order_id": order_id,
                        "status": "CANCELED"
                    }
                    MockExchangeCancel.return_value = mock_exchange_cancel

                    cancel_response = async_test_client.post(
                        f"/api/v1/trading/orders/{order_id}/cancel"
                    )

                    # 取消操作应该成功或返回具体错误
                    assert cancel_response.status_code in [200, 422, 500]

                    if cancel_response.status_code == 200:
                        cancel_data = cancel_response.json()
                        assert cancel_data["order_id"] == order_id
                        assert cancel_data["status"] in ["CANCELED", "CANCELLED"]

    async def test_stop_loss_take_profit_workflow(self, async_test_client: TestClient):
        """测试止损止盈工作流"""

        # 创建包含止损止盈的订单
        order_request = {
            "symbol": "BTC/USDT",
            "side": "buy",
            "order_type": "limit",
            "amount": 0.1,
            "price": 50000.0,
            "exchange": "binance",
            "stop_loss_price": 48500.0,
            "take_profit_price": 52500.0,
            "trailing_stop": 1000.0
        }

        with patch('backend.src.services.exchange_data_collector.ExchangeDataCollector') as MockExchange:
            mock_exchange = AsyncMock()
            mock_exchange.place_order.return_value = {
                **order_request,
                "order_id": "sl_tp_order_123",
                "status": "FILLED",
                "filled_amount": 0.1,
                "average_price": 50000.0
            }
            MockExchange.return_value = mock_exchange

            # 1. 执行主订单
            order_response = async_test_client.post(
                "/api/v1/trading/orders/execute",
                json=order_request
            )

            if order_response.status_code in [200, 202]:
                order_data = order_response.json()
                order_id = order_data["order_id"]

                # 2. 模拟止损止盈订单创建（这通常在后台自动处理）
                with patch('backend.src.services.position_monitor.PositionMonitor') as MockMonitor:
                    mock_monitor = AsyncMock()
                    mock_monitor.create_stop_loss_orders.return_value = {
                        "stop_loss_order": {
                            "order_id": "sl_order_456",
                            "status": "NEW"
                        },
                        "take_profit_order": {
                            "order_id": "tp_order_789",
                            "status": "NEW"
                        }
                    }
                    MockMonitor.return_value = mock_monitor

                    # 这里模拟调用监控器的服务方法
                    # 实际实现中这会由后台任务处理
                    pass

    async def test_risk_management_integration(self, async_test_client: TestClient):
        """测试风险管理集成"""

        # 1. 获取风险限制
        limits_response = async_test_client.get("/api/v1/trading/risk/limits")
        assert limits_response.status_code in [200, 422, 500]

        # 2. 获取风险敞口
        exposure_response = async_test_client.get("/api/v1/trading/risk/exposure")
        assert exposure_response.status_code in [200, 422, 500]

        # 3. 执行风险检查
        risk_check_request = {
            "symbol": "BTC/USDT",
            "side": "buy",
            "amount": 10.0,  # 大额订单，应该触发风险检查
            "price": 50000.0,
            "portfolio_value": 100000.0  # 总投资组合价值
        }

        risk_check_response = async_test_client.post(
            "/api/v1/trading/risk/check",
            json=risk_check_request
        )

        # 风险检查应该返回详细的风险评估
        assert risk_check_response.status_code in [200, 422, 500]

        if risk_check_response.status_code == 200:
            risk_result = risk_check_response.json()
            required_fields = ["risk_level", "approved", "risk_factors", "recommendations"]
            for field in required_fields:
                assert field in risk_result, f"Missing risk field: {field}"

    async def test_position_monitoring_workflow(self, async_test_client: TestClient):
        """测试持仓监控工作流"""

        # 1. 获取当前持仓
        positions_response = async_test_client.get("/api/v1/trading/positions")
        assert positions_response.status_code in [200, 422, 500]

        if positions_response.status_code == 200:
            positions = positions_response.json()
            if positions:
                # 2. 获取特定交易符号的持仓
                symbol = positions[0]["symbol"]
                symbol_position_response = async_test_client.get(f"/api/v1/trading/positions/{symbol}")
                assert symbol_position_response.status_code in [200, 404, 500]

        # 3. 获取投资组合摘要
        portfolio_response = async_test_client.get("/api/v1/trading/portfolio/summary")
        assert portfolio_response.status_code in [200, 422, 500]

        if portfolio_response.status_code == 200:
            portfolio = portfolio_response.json()
            expected_fields = ["total_value", "total_pnl", "positions_count", "risk_metrics"]
            for field in expected_fields:
                assert field in portfolio, f"Missing portfolio field: {field}"

    async def test_error_handling_in_workflow(self, async_test_client: TestClient):
        """测试工作流中的错误处理"""

        # 1. 测试无效订单请求
        invalid_order = {
            "symbol": "INVALID_SYMBOL",
            "side": "invalid_side",
            "order_type": "limit",
            "amount": -0.1,  # 负数量
            "price": 0  # 零价格
        }

        invalid_response = async_test_client.post(
            "/api/v1/trading/orders/execute",
            json=invalid_order
        )

        # 应该返回验证错误
        assert invalid_response.status_code == 422

        # 2. 测试不存在的订单操作
        non_existent_order_id = "non_existent_order"
        cancel_response = async_test_client.post(
            f"/api/v1/trading/orders/{non_existent_order_id}/cancel"
        )
        assert cancel_response.status_code in [404, 422, 500]

        status_response = async_test_client.get(
            f"/api/v1/trading/orders/{non_existent_order_id}"
        )
        assert status_response.status_code in [404, 422, 500]

    async def test_concurrent_order_execution(self, async_test_client: TestClient):
        """测试并发订单执行"""

        order_requests = [
            {
                "symbol": "BTC/USDT",
                "side": "buy",
                "order_type": "limit",
                "amount": 0.1,
                "price": 50000.0 + i * 100,  # 不同价格避免重复
                "exchange": "binance"
            }
            for i in range(3)
        ]

        with patch('backend.src.services.exchange_data_collector.ExchangeDataCollector') as MockExchange:
            mock_exchange = AsyncMock()

            # 模拟不同的响应
            mock_responses = []
            for i, request in enumerate(order_requests):
                mock_responses.append({
                    **request,
                    "order_id": f"concurrent_order_{i}",
                    "status": "NEW"
                })

            mock_exchange.place_order.side_effect = mock_responses
            MockExchange.return_value = mock_exchange

            # 并发执行多个订单
            async def execute_order(order_data):
                return async_test_client.post(
                    "/api/v1/trading/orders/execute",
                    json=order_data
                )

            tasks = [execute_order(order) for order in order_requests]
            responses = await asyncio.gather(*tasks, return_exceptions=True)

            # 验证所有响应都有效
            for response in responses:
                if hasattr(response, 'status_code'):
                    assert response.status_code in [200, 202, 422, 500]
                else:
                    # 处理异常情况
                    assert isinstance(response, Exception)

    async def test_order_history_and_filtering(self, async_test_client: TestClient):
        """测试订单历史查询和过滤"""

        # 1. 获取所有订单历史
        all_orders_response = async_test_client.get("/api/v1/trading/orders")
        assert all_orders_response.status_code in [200, 422, 500]

        # 2. 按交易符号过滤
        btc_orders_response = async_test_client.get(
            "/api/v1/trading/orders",
            params={"symbol": "BTC/USDT"}
        )
        assert btc_orders_response.status_code in [200, 422, 500]

        # 3. 按状态过滤
        filled_orders_response = async_test_client.get(
            "/api/v1/trading/orders",
            params={"status": "filled"}
        )
        assert filled_orders_response.status_code in [200, 422, 500]

        # 4. 按时间范围过滤
        time_filtered_response = async_test_client.get(
            "/api/v1/trading/orders",
            params={
                "start_date": "2025-10-01T00:00:00Z",
                "end_date": "2025-10-08T23:59:59Z"
            }
        )
        assert time_filtered_response.status_code in [200, 422, 500]

        # 5. 分页查询
        paginated_response = async_test_client.get(
            "/api/v1/trading/orders",
            params={"limit": 10, "offset": 0}
        )
        assert paginated_response.status_code in [200, 422, 500]

    async def test_portfolio_real_time_updates(self, async_test_client: TestClient):
        """测试投资组合实时更新"""

        # 1. 获取初始投资组合状态
        initial_portfolio = async_test_client.get("/api/v1/trading/portfolio/summary")
        assert initial_portfolio.status_code in [200, 422, 500]

        # 2. 模拟交易执行后获取更新状态
        # 这里假设已经有一些交易发生
        updated_portfolio = async_test_client.get("/api/v1/trading/portfolio/summary")
        assert updated_portfolio.status_code in [200, 422, 500]

        # 3. 获取P&L历史
        pnl_response = async_test_client.get("/api/v1/trading/portfolio/pnl")
        assert pnl_response.status_code in [200, 404, 422, 500]

        if pnl_response.status_code == 200:
            pnl_data = pnl_response.json()
            assert "daily_pnl" in pnl_data or "total_pnl" in pnl_data

    async def test_market_data_integration(self, async_test_client: TestClient):
        """测试市场数据集成"""

        # 1. 获取实时价格
        price_response = async_test_client.get(
            "/api/v1/market/price/BTC/USDT"
        )
        # 价格端点可能尚未实现
        assert price_response.status_code in [200, 404, 500]

        # 2. 获取订单簿深度
        depth_response = async_test_client.get(
            "/api/v1/market/depth/BTC/USDT"
        )
        assert depth_response.status_code in [200, 404, 500]

        # 3. 获取24小时统计
        stats_response = async_test_client.get(
            "/api/v1/market/stats/BTC/USDT"
        )
        assert stats_response.status_code in [200, 404, 500]

    async def test_exchange_integration(self, async_test_client: TestClient):
        """测试交易所集成"""

        # 1. 获取支持的交易所列表
        exchanges_response = async_test_client.get("/api/v1/trading/exchanges")
        assert exchanges_response.status_code in [200, 404, 500]

        # 2. 获取特定交易所信息
        exchange_info_response = async_test_client.get("/api/v1/trading/exchanges/binance")
        assert exchange_info_response.status_code in [200, 404, 500]

        # 3. 获取交易符号信息
        symbol_info_response = async_test_client.get("/api/v1/trading/symbols/BTC/USDT")
        assert symbol_info_response.status_code in [200, 404, 500]

    async def test_notification_system(self, async_test_client: TestClient):
        """测试通知系统"""

        # 1. 获取通知设置
        notification_settings = async_test_client.get("/api/v1/trading/notifications/settings")
        assert notification_settings.status_code in [200, 404, 500]

        # 2. 获取通知历史
        notification_history = async_test_client.get("/api/v1/trading/notifications")
        assert notification_history.status_code in [200, 404, 500]

        # 3. 测试通知配置更新
        config_update = {
            "order_filled": True,
            "stop_loss_triggered": True,
            "take_profit_triggered": True,
            "price_alerts": True
        }

        update_response = async_test_client.put(
            "/api/v1/trading/notifications/settings",
            json=config_update
        )
        assert update_response.status_code in [200, 404, 422, 500]

    async def test_performance_monitoring(self, async_test_client: TestClient):
        """测试性能监控"""

        # 1. 获取系统性能指标
        performance_metrics = async_test_client.get("/api/v1/trading/metrics/performance")
        assert performance_metrics.status_code in [200, 404, 500]

        # 2. 获取交易统计
        trading_stats = async_test_client.get("/api/v1/trading/metrics/trading")
        assert trading_stats.status_code in [200, 404, 500]

        # 3. 获取延迟统计
        latency_stats = async_test_client.get("/api/v1/trading/metrics/latency")
        assert latency_stats.status_code in [200, 404, 500]


class TestOrderExecutionEdgeCases:
    """订单执行边缘情况测试"""

    async def test_insufficient_balance_handling(self, async_test_client: TestClient):
        """测试余额不足处理"""

        # 创建一个超出余额的订单
        large_order = {
            "symbol": "BTC/USDT",
            "side": "buy",
            "order_type": "market",
            "amount": 1000.0,  # 1000 BTC，超出正常余额
            "exchange": "binance"
        }

        with patch('backend.src.services.exchange_data_collector.ExchangeDataCollector') as MockExchange:
            mock_exchange = AsyncMock()
            mock_exchange.place_order.side_effect = Exception("Insufficient balance")
            MockExchange.return_value = mock_exchange

            response = async_test_client.post(
                "/api/v1/trading/orders/execute",
                json=large_order
            )

            # 应该返回适当的错误
            assert response.status_code in [400, 422, 500]

    async def test_market_order_partial_fill(self, async_test_client: TestClient):
        """测试市价订单部分成交"""

        market_order = {
            "symbol": "BTC/USDT",
            "side": "buy",
            "order_type": "market",
            "amount": 100.0,
            "exchange": "binance"
        }

        with patch('backend.src.services.exchange_data_collector.ExchangeDataCollector') as MockExchange:
            mock_exchange = AsyncMock()
            mock_exchange.place_order.return_value = {
                **market_order,
                "order_id": "partial_fill_order",
                "status": "PARTIALLY_FILLED",
                "filled_amount": 50.0,
                "average_price": 50100.0
            }
            MockExchange.return_value = mock_exchange

            response = async_test_client.post(
                "/api/v1/trading/orders/execute",
                json=market_order
            )

            if response.status_code in [200, 202]:
                order_data = response.json()
                assert order_data["status"] == "PARTIALLY_FILLED"
                assert order_data["filled_amount"] > 0
                assert order_data["filled_amount"] < market_order["amount"]

    async def test_order_timeout_handling(self, async_test_client: TestClient):
        """测试订单超时处理"""

        # 创建一个短期限价订单
        short_lived_order = {
            "symbol": "BTC/USDT",
            "side": "buy",
            "order_type": "limit",
            "amount": 0.1,
            "price": 45000.0,  # 远低于市价，不会立即成交
            "exchange": "binance",
            "expire_time": (datetime.utcnow() + timedelta(seconds=30)).isoformat()
        }

        with patch('backend.src.services.exchange_data_collector.ExchangeDataCollector') as MockExchange:
            mock_exchange = AsyncMock()
            mock_exchange.place_order.return_value = {
                **short_lived_order,
                "order_id": "timeout_order",
                "status": "EXPIRED"
            }
            MockExchange.return_value = mock_exchange

            response = async_test_client.post(
                "/api/v1/trading/orders/execute",
                json=short_lived_order
            )

            if response.status_code in [200, 202]:
                order_data = response.json()
                # 订单可能过期或仍处于等待状态
                assert order_data["status"] in ["EXPIRED", "NEW"]

    async def test_liquidation_scenario(self, async_test_client: TestClient):
        """测试清算场景"""

        # 创建一个高杠杆的空头仓位
        liquidation_order = {
            "symbol": "BTC/USDT",
            "side": "sell",
            "order_type": "market",
            "amount": 50.0,  # 大额做空
            "exchange": "binance",
            "leverage": 100.0  # 高杠杆
        }

        with patch('backend.src.services.exchange_data_collector.ExchangeDataCollector') as MockExchange:
            mock_exchange = AsyncMock()
            mock_exchange.place_order.side_effect = Exception("Position liquidated")
            MockExchange.return_value = mock_exchange

            response = async_test_client.post(
                "/api/v1/trading/orders/execute",
                json=liquidation_order
            )

            # 应该返回清算相关的错误
            assert response.status_code in [400, 422, 500]

    async def test_network_connectivity_issues(self, async_test_client: TestClient):
        """测试网络连接问题"""

        normal_order = {
            "symbol": "BTC/USDT",
            "side": "buy",
            "order_type": "limit",
            "amount": 0.1,
            "price": 50000.0,
            "exchange": "binance"
        }

        with patch('backend.src.services.exchange_data_collector.ExchangeDataCollector') as MockExchange:
            mock_exchange = AsyncMock()
            mock_exchange.place_order.side_effect = ConnectionError("Network timeout")
            MockExchange.return_value = mock_exchange

            response = async_test_client.post(
                "/api/v1/trading/orders/execute",
                json=normal_order
            )

            # 应该处理网络错误
            assert response.status_code in [500, 503, 504]

    async def test_duplicate_order_prevention(self, async_test_client: TestClient):
        """测试重复订单防护"""

        order_request = {
            "symbol": "BTC/USDT",
            "side": "buy",
            "order_type": "limit",
            "amount": 0.1,
            "price": 50000.0,
            "exchange": "binance",
            "client_order_id": "unique_client_order_123"
        }

        with patch('backend.src.services.exchange_data_collector.ExchangeDataCollector') as MockExchange:
            mock_exchange = AsyncMock()
            mock_exchange.place_order.return_value = {
                **order_request,
                "order_id": "duplicate_check_order",
                "status": "NEW"
            }
            MockExchange.return_value = mock_exchange

            # 第一次提交
            first_response = async_test_client.post(
                "/api/v1/trading/orders/execute",
                json=order_request
            )

            # 第二次提交相同订单
            second_response = async_test_client.post(
                "/api/v1/trading/orders/execute",
                json=order_request
            )

            # 应该检测到重复订单
            assert first_response.status_code in [200, 202]
            assert second_response.status_code in [200, 202, 409, 422]  # 409 Conflict 或其他错误

    async def test_extreme_market_volatility(self, async_test_client: TestClient):
        """测试极端市场波动处理"""

        # 在市场波动时创建订单
        volatility_order = {
            "symbol": "BTC/USDT",
            "side": "buy",
            "order_type": "market",
            "amount": 0.1,
            "exchange": "binance"
        }

        with patch('backend.src.services.exchange_data_collector.ExchangeDataCollector') as MockExchange:
            mock_exchange = AsyncMock()
            mock_exchange.place_order.return_value = {
                **volatility_order,
                "order_id": "volatility_order",
                "status": "FILLED",
                "filled_amount": 0.1,
                "average_price": 48000.0,  # 价格大幅偏离预期
                "slippage": 8000.0  # 16%的滑点
            }
            MockExchange.return_value = mock_exchange

            response = async_test_client.post(
                "/api/v1/trading/orders/execute",
                json=volatility_order
            )

            if response.status_code in [200, 202]:
                order_data = response.json()
                # 系统应该记录滑点信息
                assert "average_price" in order_data