"""
风险管理逻辑单元测试

测试风险管理服务的核心逻辑和算法。
"""

import pytest
from datetime import datetime, timedelta
from decimal import Decimal
from unittest.mock import Mock, AsyncMock, patch
import asyncio

from backend.src.services.risk_manager import RiskManager, RiskLevel, RiskAssessment
from backend.src.models.trading_order import OrderStatus, OrderType, OrderSide
from backend.src.models.position import PositionSide
from backend.src.models.trading_strategy import TradingStrategy


class TestRiskManager:
    """风险管理器单元测试"""

    @pytest.fixture
    def risk_manager(self):
        """创建风险管理器实例"""
        return RiskManager()

    @pytest.fixture
    def mock_order_request(self):
        """模拟订单请求"""
        return {
            "symbol": "BTC/USDT",
            "side": "buy",
            "order_type": "limit",
            "amount": Decimal("0.1"),
            "price": Decimal("50000.0"),
            "exchange": "binance",
            "portfolio_value": Decimal("100000.0")
        }

    @pytest.fixture
    def mock_positions(self):
        """模拟持仓数据"""
        return [
            {
                "symbol": "BTC/USDT",
                "side": "long",
                "amount": Decimal("0.5"),
                "average_cost": Decimal("45000.0"),
                "current_price": Decimal("50000.0"),
                "unrealized_pnl": Decimal("2500.0"),
                "exchange": "binance"
            },
            {
                "symbol": "ETH/USDT",
                "side": "short",
                "amount": Decimal("10.0"),
                "average_cost": Decimal("3000.0"),
                "current_price": Decimal("2900.0"),
                "unrealized_pnl": Decimal("1000.0"),
                "exchange": "binance"
            }
        ]

    def test_calculate_position_risk(self, risk_manager, mock_positions):
        """测试持仓风险计算"""
        for position in mock_positions:
            position_side = position["side"]
            amount = position["amount"]
            avg_cost = position["average_cost"]
            current_price = position["current_price"]

            if position_side == "long":
                # 做多仓位：价格下跌风险
                risk_amount = (avg_cost - current_price) * amount
                risk_percentage = (avg_cost - current_price) / avg_cost * 100
            else:
                # 做空仓位：价格上涨风险
                risk_amount = (current_price - avg_cost) * amount
                risk_percentage = (current_price - avg_cost) / avg_cost * 100

            risk_result = risk_manager.calculate_position_risk(position)

            assert risk_result["symbol"] == position["symbol"]
            assert risk_result["side"] == position_side
            assert risk_result["risk_amount"] == risk_amount
            assert risk_result["risk_percentage"] == risk_percentage

            # 验证风险等级
            expected_risk_level = RiskLevel.HIGH if abs(risk_percentage) > 10 else (
                RiskLevel.MEDIUM if abs(risk_percentage) > 5 else RiskLevel.LOW
            )
            assert risk_result["risk_level"] == expected_risk_level

    def test_calculate_order_risk_buy(self, risk_manager, mock_order_request):
        """测试买单风险评估"""
        risk_assessment = risk_manager.calculate_order_risk(mock_order_request)

        assert risk_assessment["symbol"] == mock_order_request["symbol"]
        assert risk_assessment["side"] == mock_order_request["side"]
        assert risk_assessment["amount"] == mock_order_request["amount"]
        assert risk_assessment["price"] == mock_order_request["price"]

        # 计算订单价值
        order_value = mock_order_request["amount"] * mock_order_request["price"]
        portfolio_percentage = (order_value / mock_order_request["portfolio_value"]) * 100

        assert risk_assessment["order_value"] == order_value
        assert risk_assessment["portfolio_percentage"] == portfolio_percentage

        # 验证风险等级
        expected_risk_level = RiskLevel.HIGH if portfolio_percentage > 20 else (
            RiskLevel.MEDIUM if portfolio_percentage > 10 else RiskLevel.LOW
        )
        assert risk_assessment["risk_level"] == expected_risk_level

    def test_calculate_order_risk_sell(self, risk_manager, mock_order_request):
        """测试卖单风险评估"""
        mock_order_request["side"] = "sell"
        mock_order_request["portfolio_value"] = Decimal("100000.0")

        risk_assessment = risk_manager.calculate_order_risk(mock_order_request)

        assert risk_assessment["side"] == "sell"

        # 计算订单价值
        order_value = mock_order_request["amount"] * mock_order_request["price"]
        portfolio_percentage = (order_value / mock_order_request["portfolio_value"]) * 100

        assert risk_assessment["portfolio_percentage"] == portfolio_percentage

    def test_calculate_portfolio_risk(self, risk_manager, mock_positions):
        """测试投资组合风险计算"""
        portfolio_risk = risk_manager.calculate_portfolio_risk(mock_positions)

        assert "total_value" in portfolio_risk
        assert "total_unrealized_pnl" in portfolio_risk
        assert "risk_summary" in portfolio_risk

        # 计算预期总值
        expected_total_value = sum(
            pos["amount"] * pos["current_price"] for pos in mock_positions
        )
        expected_total_pnl = sum(pos["unrealized_pnl"] for pos in mock_positions)

        assert portfolio_risk["total_value"] == expected_total_value
        assert portfolio_risk["total_unrealized_pnl"] == expected_total_pnl

        # 验证高风险持仓数量
        high_risk_count = portfolio_risk["risk_summary"]["high_risk_positions"]
        expected_high_risk = sum(1 for pos in mock_positions
                               if abs((pos["current_price"] - pos["average_cost"]) / pos["average_cost"] * 100) > 10)
        assert high_risk_count == expected_high_risk

    def test_check_concentration_risk(self, risk_manager):
        """测试集中度风险检查"""
        positions = [
            {"symbol": "BTC/USDT", "value": Decimal("50000.0")},
            {"symbol": "ETH/USDT", "value": Decimal("30000.0")},
            {"symbol": "SOL/USDT", "value": Decimal("20000.0")}
        ]
        total_value = Decimal("100000.0")

        concentration_risk = risk_manager.check_concentration_risk(positions, total_value)

        # 验证集中度计算
        for pos in positions:
            concentration = (pos["value"] / total_value) * 100
            symbol = pos["symbol"]
            assert symbol in concentration_risk
            assert concentration_risk[symbol]["percentage"] == concentration

        # 验证风险判断
        btc_concentration = concentration_risk["BTC/USDT"]["percentage"]
        expected_risk = RiskLevel.HIGH if btc_concentration > 50 else (
            RiskLevel.MEDIUM if btc_concentration > 30 else RiskLevel.LOW
        )
        assert concentration_risk["BTC/USDT"]["risk_level"] == expected_risk

    def test_check_leverage_risk(self, risk_manager):
        """测试杠杆风险检查"""
        position = {
            "symbol": "BTC/USDT",
            "side": "long",
            "amount": Decimal("1.0"),
            "average_cost": Decimal("50000.0"),
            "current_price": Decimal("50000.0"),
            "leverage": 10.0
        }

        leverage_risk = risk_manager.check_leverage_risk(position)

        assert leverage_risk["symbol"] == position["symbol"]
        assert leverage_risk["leverage"] == position["leverage"]
        assert "margin_ratio" in leverage_risk

        expected_margin_ratio = 100 / position["leverage"]
        assert leverage_risk["margin_ratio"] == expected_margin_ratio

        # 验证风险等级
        expected_risk = RiskLevel.HIGH if position["leverage"] > 20 else (
            RiskLevel.MEDIUM if position["leverage"] > 10 else RiskLevel.LOW
        )
        assert leverage_risk["risk_level"] == expected_risk

    def test_assess_market_risk(self, risk_manager):
        """测试市场风险评估"""
        market_data = {
            "volatility_24h": 0.05,  # 5%波动率
            "volume_24h": 1000000000,  # 10亿交易量
            "price_change_24h": -0.02,  # -2%价格变化
            "rsi": 75.0,  # 超买
            "bollinger_position": "above_upper"  # 布林带上轨上方
        }

        market_risk = risk_manager.assess_market_risk(market_data)

        assert "risk_level" in market_risk
        assert "risk_factors" in market_risk
        assert "recommendation" in market_risk

        # 高波动率应该触发高风险
        if market_data["volatility_24h"] > 0.04:
            assert market_risk["risk_level"] == RiskLevel.HIGH
            assert "high_volatility" in [f["type"] for f in market_risk["risk_factors"]]

        # 超买信号应该包含风险因素
        if market_data["rsi"] > 70:
            assert "overbought" in [f["type"] for f in market_risk["risk_factors"]]

    def test_validate_order_parameters(self, risk_manager, mock_order_request):
        """测试订单参数验证"""
        validation_result = risk_manager.validate_order_parameters(mock_order_request)

        assert validation_result["valid"]
        assert validation_result["errors"] == []

        # 测试无效参数
        invalid_order = mock_order_request.copy()
        invalid_order["amount"] = Decimal("-0.1")  # 负数量

        invalid_validation = risk_manager.validate_order_parameters(invalid_order)
        assert not invalid_validation["valid"]
        assert len(invalid_validation["errors"]) > 0

    def test_check_position_size_limits(self, risk_manager):
        """测试仓位大小限制检查"""
        order_request = {
            "symbol": "BTC/USDT",
            "side": "buy",
            "amount": Decimal("0.1"),
            "portfolio_value": Decimal("100000.0")
        }

        # 测试仓位限制
        limit_check = risk_manager.check_position_size_limits(order_request)

        assert "approved" in limit_check
        assert "max_allowed_amount" in limit_check
        assert "current_amount" in limit_check

        # 测试超出限制的订单
        large_order = order_request.copy()
        large_order["amount"] = Decimal("100.0")  # 50万价值的订单

        large_check = risk_manager.check_position_size_limits(large_order)
        assert not large_check["approved"]
        assert "reason" in large_check

    def test_calculate_stop_loss_distance(self, risk_manager):
        """测试止损距离计算"""
        current_price = Decimal("50000.0")
        stop_loss_price = Decimal("48500.0")

        stop_loss_distance = risk_manager.calculate_stop_loss_distance(
            current_price, stop_loss_price, "long"
        )

        expected_distance = ((current_price - stop_loss_price) / current_price) * 100
        assert stop_loss_distance["percentage"] == expected_distance
        assert stop_loss_distance["is_reasonable"] == (5 <= expected_distance <= 15)

        # 做空止损距离
        short_stop_loss = risk_manager.calculate_stop_loss_distance(
            current_price, Decimal("51500.0"), "short"
        )
        short_distance = ((Decimal("51500.0") - current_price) / current_price) * 100
        assert short_stop_loss["percentage"] == short_distance

    def test_assess_risk_reward_ratio(self, risk_manager):
        """测试风险回报比评估"""
        entry_price = Decimal("50000.0")
        stop_loss_price = Decimal("48500.0")
        take_profit_price = Decimal("52500.0")

        risk_reward = risk_manager.assess_risk_reward_ratio(
            entry_price, stop_loss_price, take_profit_price, "long"
        )

        expected_ratio = (take_profit_price - entry_price) / (entry_price - stop_loss_price)
        assert risk_reward["ratio"] == expected_ratio
        assert risk_reward["is_acceptable"] == (expected_ratio >= 2.0)

        # 测试不佳的风险回报比
        poor_ratio = risk_manager.assess_risk_reward_ratio(
            entry_price, Decimal("49500.0"), take_profit_price, "long"
        )
        assert not poor_ratio["is_acceptable"]

    def test_check_time_risk(self, risk_manager):
        """测试时间风险检查"""
        order_request = {
            "symbol": "BTC/USDT",
            "side": "buy",
            "order_type": "limit",
            "amount": Decimal("0.1"),
            "price": Decimal("50000.0"),
            "created_at": datetime.utcnow(),
            "expire_time": datetime.utcnow() + timedelta(hours=1)
        }

        time_risk = risk_manager.check_time_risk(order_request)

        assert "time_remaining" in time_risk
        assert "risk_level" in time_risk

        # 测试即将过期的订单
        urgent_order = order_request.copy()
        urgent_order["expire_time"] = datetime.utcnow() + timedelta(minutes=5)

        urgent_risk = risk_manager.check_time_risk(urgent_order)
        assert urgent_risk["risk_level"] == RiskLevel.HIGH
        assert urgent_risk["is_expiring_soon"]

    def test_assess_correlation_risk(self, risk_manager):
        """测试相关性风险评估"""
        positions = [
            {"symbol": "BTC/USDT", "value": Decimal("30000.0")},
            {"symbol": "ETH/USDT", "value": Decimal("25000.0")},
            {"symbol": "LTC/USDT", "value": Decimal("10000.0")},
            {"symbol": "XRP/USDT", "value": Decimal("8000.0")}
        ]

        correlation_matrix = {
            "BTC/USDT": {"ETH/USDT": 0.8, "LTC/USDT": 0.9, "XRP/USDT": 0.3},
            "ETH/USDT": {"BTC/USDT": 0.8, "LTC/USDT": 0.7, "XRP/USDT": 0.2},
            "LTC/USDT": {"BTC/USDT": 0.9, "ETH/USDT": 0.7, "XRP/USDT": 0.4},
            "XRP/USDT": {"BTC/USDT": 0.3, "ETH/USDT": 0.2, "LTC/USDT": 0.4}
        }

        correlation_risk = risk_manager.assess_correlation_risk(positions, correlation_matrix)

        assert "average_correlation" in correlation_risk
        assert "risk_level" in correlation_risk
        assert "high_correlation_pairs" in correlation_risk

        # BTC和ETH的高相关性应该被检测到
        assert ("BTC/USDT", "ETH/USDT") in correlation_risk["high_correlation_pairs"]

    def test_generate_risk_report(self, risk_manager, mock_positions, mock_order_request):
        """测试风险报告生成"""
        portfolio_risk = risk_manager.calculate_portfolio_risk(mock_positions)
        order_risk = risk_manager.calculate_order_risk(mock_order_request)

        risk_report = risk_manager.generate_risk_report(
            portfolio_risk, order_risk, mock_positions
        )

        assert "timestamp" in risk_report
        assert "overall_risk_level" in risk_report
        assert "portfolio_risk" in risk_report
        assert "order_risk" in risk_report
        assert "recommendations" in risk_report
        assert "risk_factors" in risk_report

        # 验证建议数量
        assert len(risk_report["recommendations"]) > 0
        assert len(risk_report["risk_factors"]) > 0

    def test_update_risk_limits(self, risk_manager):
        """测试风险限制更新"""
        new_limits = {
            "max_position_size_percent": 25.0,
            "max_portfolio_risk_percent": 15.0,
            "max_leverage": 10.0,
            "min_stop_loss_distance": 5.0,
            "min_risk_reward_ratio": 2.0
        }

        updated_limits = risk_manager.update_risk_limits(new_limits)

        assert updated_limits["max_position_size_percent"] == 25.0
        assert updated_limits["max_portfolio_risk_percent"] == 15.0
        assert updated_limits["max_leverage"] == 10.0

    def test_monitor_risk_alerts(self, risk_manager):
        """测试风险监控警报"""
        # 模拟高风险数据
        high_risk_data = {
            "total_portfolio_risk": 25.0,  # 25%总风险
            "max_single_position_risk": 18.0,  # 18%单仓风险
            "leverage_usage": 15.0,  # 15倍杠杆
            "correlation_risk": 0.85,  # 高相关性
            "liquidity_risk": "medium"
        }

        alerts = risk_manager.monitor_risk_alerts(high_risk_data)

        assert len(alerts) > 0

        # 验证警报类型
        alert_types = [alert["type"] for alert in alerts]
        assert "portfolio_risk" in alert_types
        assert "position_risk" in alert_types

        # 验证警报严重程度
        severe_alerts = [a for a in alerts if a["severity"] == "high"]
        assert len(severe_alerts) > 0

    @pytest.mark.asyncio
    async def test_real_time_risk_monitoring(self, risk_manager):
        """测试实时风险监控"""
        mock_positions = [
            {
                "symbol": "BTC/USDT",
                "side": "long",
                "amount": Decimal("0.1"),
                "current_price": Decimal("50000.0"),
                "average_cost": Decimal("45000.0")
            }
        ]

        # 模拟实时监控循环
        monitoring_count = 0
        risk_alerts = []

        async def risk_callback(risk_assessment):
            nonlocal monitoring_count, risk_alerts
            monitoring_count += 1
            if risk_assessment["risk_level"] == RiskLevel.HIGH:
                risk_alerts.append(risk_assessment)

        # 模拟监控过程
        for i in range(5):
            # 模拟价格变化
            for position in mock_positions:
                position["current_price"] = Decimal("50000.0") + (i * 1000)

            # 计算风险
            portfolio_risk = risk_manager.calculate_portfolio_risk(mock_positions)
            await risk_callback(portfolio_risk)

            # 模拟时间间隔
            await asyncio.sleep(0.01)

        assert monitoring_count == 5
        # 由于价格变化不大，可能不会有高风险警报

    def test_stress_test_risk_calculations(self, risk_manager):
        """测试风险计算压力测试"""
        import time

        # 大量持仓数据
        positions = []
        for i in range(100):
            positions.append({
                "symbol": f"TOKEN{i}/USDT",
                "side": "long" if i % 2 == 0 else "short",
                "amount": Decimal("0.1"),
                "current_price": Decimal(f"{1000 + i}00.0"),
                "average_cost": Decimal(f"{1000 + i*50}.0"),
                "exchange": "binance"
            })

        # 测试计算性能
        start_time = time.time()
        portfolio_risk = risk_manager.calculate_portfolio_risk(positions)
        end_time = time.time()

        calculation_time = end_time - start_time

        # 计算应该在合理时间内完成（<1秒）
        assert calculation_time < 1.0
        assert portfolio_risk["total_value"] > 0
        assert len(portfolio_risk["position_risks"]) == 100

    def test_edge_cases(self, risk_manager):
        """测试边缘情况"""
        # 空持仓列表
        empty_risk = risk_manager.calculate_portfolio_risk([])
        assert empty_risk["total_value"] == 0
        assert empty_risk["total_unrealized_pnl"] == 0

        # 零价值持仓
        zero_value_position = {
            "symbol": "WORTHLESS/USDT",
            "side": "long",
            "amount": Decimal("100.0"),
            "current_price": Decimal("0.0"),
            "average_cost": Decimal("100.0")
        }

        zero_risk = risk_manager.calculate_position_risk(zero_value_position)
        assert zero_risk["risk_amount"] == 0

        # 极小订单
        tiny_order = {
            "symbol": "BTC/USDT",
            "side": "buy",
            "amount": Decimal("0.000001"),
            "price": Decimal("50000.0"),
            "portfolio_value": Decimal("100000.0")
        }

        tiny_risk = risk_manager.calculate_order_risk(tiny_order)
        assert tiny_risk["portfolio_percentage"] < 0.001


class TestRiskAssessment:
    """风险评估类单元测试"""

    def test_risk_assessment_creation(self):
        """测试风险评估创建"""
        assessment = RiskAssessment(
            risk_level=RiskLevel.HIGH,
            risk_score=0.85,
            risk_factors=["high_volatility", "over_leveraged"],
            recommendations=["reduce_position_size", "stop_loss_tightening"],
            details={"volatility": 0.15, "leverage": 20.0}
        )

        assert assessment.risk_level == RiskLevel.HIGH
        assert assessment.risk_score == 0.85
        assert "high_volatility" in assessment.risk_factors
        assert len(assessment.recommendations) == 2

    def test_risk_assessment_comparison(self):
        """测试风险评估比较"""
        high_risk = RiskAssessment(
            risk_level=RiskLevel.HIGH,
            risk_score=0.8
        )

        low_risk = RiskAssessment(
            risk_level=RiskLevel.LOW,
            risk_score=0.3
        )

        assert high_risk.is_higher_risk_than(low_risk)
        assert not low_risk.is_higher_risk_than(high_risk)

    def test_risk_assessment_aggregation(self):
        """测试风险评估聚合"""
        assessments = [
            RiskAssessment(RiskLevel.LOW, 0.3),
            RiskAssessment(RiskLevel.MEDIUM, 0.5),
            RiskAssessment(RiskLevel.HIGH, 0.8)
        ]

        aggregated = RiskAssessment.aggregate_assessments(assessments)

        # 平均风险分数
        expected_score = (0.3 + 0.5 + 0.8) / 3
        assert abs(aggregated.risk_score - expected_score) < 0.01

        # 应该采用最高风险等级
        assert aggregated.risk_level == RiskLevel.HIGH

    def test_risk_threshold_validation(self):
        """测试风险阈值验证"""
        assessment = RiskAssessment(
            risk_level=RiskLevel.HIGH,
            risk_score=0.9
        )

        # 验证高风险阈值
        assert assessment.exceeds_threshold(0.8)
        assert not assessment.exceeds_threshold(0.95)

        # 验证风险等级阈值
        assert assessment.is_at_or_above_level(RiskLevel.MEDIUM)
        assert assessment.is_at_or_above_level(RiskLevel.HIGH)
        assert not assessment.is_at_or_above_level(RiskLevel.CRITICAL)


# 便捷函数测试
class TestRiskManagementUtils:
    """风险管理工具函数测试"""

    def test_price_deviation_calculation(self):
        """测试价格偏差计算"""
        current_price = Decimal("50000.0")
        reference_price = Decimal("45000.0")

        # 计算偏差百分比
        deviation = (current_price - reference_price) / reference_price * 100
        assert abs(deviation - 11.11) < 0.01

    def test_volatility_calculation(self):
        """测试波动率计算"""
        prices = [
            Decimal("49000.0"), Decimal("49500.0"), Decimal("50500.0"),
            Decimal("51000.0"), Decimal("51500.0")
        ]

        # 计算平均价格
        avg_price = sum(prices) / len(prices)
        assert avg_price == Decimal("50300.0")

        # 计算价格标准差
        variance = sum((p - avg_price) ** 2 for p in prices) / len(prices)
        std_dev = variance.sqrt()
        # 验证计算正确性
        assert std_dev > 0

    def test_leverage_margin_calculation(self):
        """测试杠杆保证金计算"""
        position_value = Decimal("50000.0")
        leverage = Decimal("10.0")

        # 所需保证金 = 仓位价值 / 杠杆
        required_margin = position_value / leverage
        assert required_margin == Decimal("5000.0")

        # 保证金比例
        margin_ratio = (required_margin / position_value) * 100
        assert margin_ratio == 10.0

    def test_drawdown_calculation(self):
        """测试回撤计算"""
        peak_value = Decimal("100000.0")
        current_value = Decimal("85000.0")

        # 计算回撤
        drawdown = (peak_value - current_value) / peak_value * 100
        assert abs(drawdown - 15.0) < 0.01

        # 计算恢复
        recovery = (current_value - peak_value) / peak_value * 100
        assert recovery == -drawdown