"""
动态资金管理单元测试

测试动态资金管理服务的核心逻辑和算法。
"""

import pytest
from datetime import datetime, timedelta
from decimal import Decimal
from unittest.mock import Mock, AsyncMock, patch
import asyncio

from backend.src.services.dynamic_fund_manager import (
    DynamicFundManager, FundAllocationStrategy,
    FundAllocation, PerformanceMetrics
)
from backend.src.models.position import PositionSide


class TestDynamicFundManager:
    """动态资金管理器单元测试"""

    @pytest.fixture
    def fund_manager(self):
        """创建资金管理器实例"""
        return DynamicFundManager(
            total_funds=Decimal("100000.0"),
            initial_allocation={
                "conservative": 0.3,  # 30%保守策略
                "moderate": 0.5,      # 50%中等策略
                "aggressive": 0.2   # 20%激进策略
            },
            max_allocation_change=0.1  # 最大调整幅度10%
        )

    @pytest.fixture
    def mock_positions(self):
        """模拟持仓数据"""
        return [
            {
                "symbol": "BTC/USDT",
                "side": PositionSide.LONG,
                "amount": Decimal("0.5"),
                "current_price": Decimal("50000.0"),
                "unrealized_pnl": Decimal("2500.0"),
                "strategy_type": "moderate",
                "entry_date": datetime.utcnow() - timedelta(days=10)
            },
            {
                "symbol": "ETH/USDT",
                "side": PositionSide.SHORT,
                "amount": Decimal("10.0"),
                "current_price": Decimal("3000.0"),
                "unrealized_pnl": Decimal("1000.0"),
                "strategy_type": "aggressive",
                "entry_date": datetime.utcnow() - timedelta(days=5)
            },
            {
                "symbol": "BNB/USDT",
                "side": PositionSide.LONG,
                "amount": Decimal("100.0"),
                "current_price": Decimal("300.0"),
                "unrealized_pnl": Decimal("-500.0"),
                "strategy_type": "conservative",
                "entry_date": datetime.utcnow() - timedelta(days=15)
            }
        ]

    def test_initial_allocation_setup(self, fund_manager):
        """测试初始分配设置"""
        allocation = fund_manager.get_current_allocation()

        assert allocation["conservative"] == Decimal("0.3")
        assert allocation["moderate"] == Decimal("0.5")
        assert allocation["aggressive"] == Decimal("0.2")
        assert sum(allocation.values()) == Decimal("1.0")

        # 验证资金分配金额
        expected_conservative = fund_manager.total_funds * allocation["conservative"]
        expected_moderate = fund_manager.total_funds * allocation["moderate"]
        expected_aggressive = fund_manager.total_funds * allocation["aggressive"]

        assert allocation["conservative_amount"] == expected_conservative
        assert allocation["moderate_amount"] == expected_moderate
        assert allocation["aggressive_amount"] == expected_aggressive

    def test_calculate_current_performance(self, fund_manager, mock_positions):
        """测试当前表现计算"""
        performance = fund_manager.calculate_current_performance(mock_positions)

        # 计算总盈亏
        expected_total_pnl = sum(pos["unrealized_pnl"] for pos in mock_positions)
        expected_pnl_percentage = (expected_total_pnl / fund_manager.total_funds) * 100

        assert performance["total_unrealized_pnl"] == expected_total_pnl
        assert performance["pnl_percentage"] == expected_pnl_percentage
        assert performance["total_value"] == fund_manager.total_funds + expected_total_pnl

        # 验证各策略表现
        assert "strategy_performance" in performance
        strategy_perf = performance["strategy_performance"]

        assert "moderate" in strategy_perf
        assert "aggressive" in strategy_perf
        assert "conservative" in strategy_perf

    def test_rebalance_allocation(self, fund_manager, mock_positions):
        """测试重新平衡分配"""
        # 模拟激进策略表现优异，保守策略表现不佳
        performance_data = fund_manager.calculate_current_performance(mock_positions)

        # 执行重新平衡
        new_allocation = fund_manager.rebalance_allocation(performance_data)

        # 验证重新平衡结果
        assert sum(new_allocation.values()) == Decimal("1.0")

        # 激进策略应该增加分配
        assert new_allocation["aggressive"] > fund_manager.initial_allocation["aggressive"]

        # 保守策略应该减少分配
        assert new_allocation["conservative"] < fund_manager.initial_allocation["conservative"]

        # 验证调整幅度限制
        for strategy in new_allocation:
            initial = fund_manager.initial_allocation[strategy]
            max_change = fund_manager.max_allocation_change
            assert abs(new_allocation[strategy] - initial) <= max_change

    def test_performance_based_adjustment(self, fund_manager):
        """测试基于表现的调整"""
        # 模拟30天的表现数据
        performance_history = [
            {"date": datetime.utcnow() - timedelta(days=i), "pnl_percentage": i * 0.01}
            for i in range(30)
        ]

        # 添加策略表现
        performance_history[0]["strategy_performance"] = {
            "conservative": {"return": 0.01},
            "moderate": {"return": 0.03},
            "aggressive": {"return": 0.05}
        }

        # 计算调整建议
        adjustment = fund_manager.calculate_performance_adjustment(performance_history)

        assert "adjustments" in adjustment
        assert "reasoning" in adjustment

        # 激进策略表现最好，应该增加分配
        assert adjustment["adjustments"]["aggressive"]["direction"] == "increase"
        assert adjustment["adjustments"]["aggressive"]["magnitude"] > 0

    def test_risk_based_adjustment(self, fund_manager, mock_positions):
        """测试基于风险的调整"""
        # 计算当前风险水平
        risk_metrics = fund_manager.calculate_risk_metrics(mock_positions)

        # 基于风险调整分配
        risk_adjustment = fund_manager.calculate_risk_based_adjustment(risk_metrics)

        assert "risk_level" in risk_adjustment
        assert "adjustments" in risk_adjustment

        # 如果风险高，应该减少激进策略分配
        if risk_adjustment["risk_level"] == "high":
            assert risk_adjustment["adjustments"]["aggressive"]["direction"] == "decrease"
            assert risk_adjustment["adjustments"]["conservative"]["direction"] == "increase"

    def test_volatility_based_adjustment(self, fund_manager):
        """测试基于波动率的调整"""
        # 模拟不同波动率
        volatility_data = {
            "conservative": {"volatility": 0.02, "volatility_score": 0.3},
            "moderate": {"volatility": 0.05, "volatility_score": 0.6},
            "aggressive": {"volatility": 0.10, "volatility_score": 0.9}
        }

        volatility_adjustment = fund_manager.calculate_volatility_adjustment(volatility_data)

        assert "adjustments" in volatility_adjustment

        # 高波动率策略应该减少分配
        assert volatility_adjustment["adjustments"]["aggressive"]["direction"] == "decrease"
        assert volatility_adjustment["adjustments"]["conservative"]["direction"] == "increase"

    def test_correlation_based_adjustment(self, fund_manager):
        """测试基于相关性的调整"""
        # 模拟策略间的相关性
        correlation_matrix = {
            "conservative": {"moderate": 0.3, "aggressive": 0.7},
            "moderate": {"conservative": 0.3, "aggressive": 0.6},
            "aggressive": {"conservative": 0.7, "moderate": 0.6}
        }

        correlation_adjustment = fund_manager.calculate_correlation_based_adjustment(correlation_matrix)

        assert "adjustments" in correlation_adjustment
        assert "diversification_score" in correlation_adjustment

        # 高相关性应该减少重复策略分配
        assert correlation_adjustment["diversification_score"] < 1.0

    def test_liquidity_based_adjustment(self, fund_manager):
        """测试基于流动性的调整"""
        # 模拟流动性数据
        liquidity_data = {
            "conservative": {"liquidity_score": 0.8, "average_daily_volume": 5000000},
            "moderate": {"liquidity_score": 0.6, "average_daily_volume": 2000000},
            "aggressive": {"liquidity_score": 0.4, "average_daily_volume": 1000000}
        }

        liquidity_adjustment = fund_manager.calculate_liquidity_based_adjustment(liquidity_data)

        assert "adjustments" in liquidity_adjustment

        # 流动性好的策略可以增加分配
        assert liquidity_adjustment["adjustments"]["conservative"]["direction"] == "increase"
        assert liquidity_adjustment["adjustments"]["aggressive"]["direction"] == "decrease"

    def test_market_cycle_adjustment(self, fund_manager):
        """测试市场周期调整"""
        # 模拟牛市市场
        bull_market_data = {
            "market_regime": "bull",
            "trend_strength": 0.8,
            "volatility_regime": "normal",
            "market_sentiment": "optimistic"
        }

        bull_adjustment = fund_manager.calculate_market_cycle_adjustment(bull_market_data)

        assert "adjustments" in bull_adjustment
        assert "market_regime" in bull_adjustment

        # 牛市应该增加激进分配
        assert bull_adjustment["adjustments"]["aggressive"]["direction"] == "increase"

        # 模拟熊市
        bear_market_data = {
            "market_regime": "bear",
            "trend_strength": 0.7,
            "volatility_regime": "high",
            "market_sentiment": "pessimistic"
        }

        bear_adjustment = fund_manager.calculate_market_cycle_adjustment(bear_market_data)

        # 熊市应该增加保守分配
        assert bear_adjustment["adjustments"]["conservative"]["direction"] == "increase"

    def test_combined_adjustment_strategy(self, fund_manager, mock_positions):
        """测试综合调整策略"""
        # 收集所有调整因子
        performance_data = fund_manager.calculate_current_performance(mock_positions)
        risk_metrics = fund_manager.calculate_risk_metrics(mock_positions)
        volatility_data = {"moderate": {"volatility_score": 0.6}}
        market_data = {"market_regime": "bull", "trend_strength": 0.7}

        # 综合调整
        combined_adjustment = fund_manager.calculate_combined_adjustment(
            performance_data=performance_data,
            risk_metrics=risk_metrics,
            volatility_data=volatility_data,
            market_data=market_data
        )

        assert "final_allocation" in combined_adjustment
        assert "adjustment_factors" in combined_adjustment
        assert "confidence_score" in combined_adjustment

        # 验证最终分配总和
        assert sum(combined_adjustment["final_allocation"].values()) == Decimal("1.0")

    def test_allocation_change_validation(self, fund_manager):
        """测试分配变化验证"""
        # 正常变化
        valid_change = {
            "conservative": 0.25,
            "moderate": 0.55,
            "aggressive": 0.20
        }

        is_valid = fund_manager.validate_allocation_change(valid_change)
        assert is_valid["valid"]
        assert is_valid["reason"] == "Allocation change within limits"

        # 超出限制的变化
        invalid_change = {
            "conservative": 0.5,  # 超出10%调整幅度
            "moderate": 0.3,
            "aggressive": 0.2
        }

        is_invalid = fund_manager.validate_allocation_change(invalid_change)
        assert not is_invalid["valid"]
        assert "conservative" in [error["strategy"] for error in is_invalid["errors"]]

    def test_minimum_allocation_enforcement(self, fund_manager):
        """测试最小分配执行"""
        # 设置最小分配限制
        min_allocations = {
            "conservative": 0.1,
            "moderate": 0.2,
            "aggressive": 0.05
        }

        fund_manager.minimum_allocations = min_allocations

        # 测试违反最小分配的情况
        invalid_allocation = {
            "conservative": 0.05,  # 小于最小值
            "moderate": 0.15,
            "aggressive": 0.8
        }

        corrected = fund_manager.enforce_minimum_allocations(invalid_allocation)

        assert corrected["conservative"] >= min_allocations["conservative"]
        assert corrected["moderate"] >= min_allocations["moderate"]
        assert corrected["aggressive"] >= min_allocations["aggressive"]

        # 确保总和仍然为1
        assert sum(corrected.values()) == Decimal("1.0")

    def test_maximum_allocation_enforcement(self, fund_manager):
        """测试最大分配执行"""
        # 设置最大分配限制
        max_allocations = {
            "conservative": 0.6,
            "moderate": 0.8,
            "aggressive": 0.4
        }

        fund_manager.maximum_allocations = max_allocations

        # 测试超出最大分配的情况
        invalid_allocation = {
            "conservative": 0.7,  # 超过最大值
            "moderate": 0.3,
            "aggressive": 0.0
        }

        corrected = fund_manager.enforce_maximum_allocations(invalid_allocation)

        assert corrected["conservative"] <= max_allocations["conservative"]
        assert corrected["moderate"] <= max_allocations["moderate"]
        assert corrected["aggressive"] <= max_allocations["aggressive"]

    def test_allocation_history_tracking(self, fund_manager):
        """测试分配历史跟踪"""
        initial_allocation = fund_manager.get_current_allocation()

        # 记录分配变更
        new_allocation = {
            "conservative": 0.25,
            "moderate": 0.55,
            "aggressive": 0.20
        }

        fund_manager.record_allocation_change(
            from_allocation=initial_allocation,
            to_allocation=new_allocation,
            reason="performance_based_rebalancing",
            timestamp=datetime.utcnow()
        )

        # 验证历史记录
        history = fund_manager.get_allocation_history()
        assert len(history) == 1

        record = history[0]
        assert record["from_allocation"] == initial_allocation
        assert record["to_allocation"] == new_allocation
        assert record["reason"] == "performance_based_rebalancing"

    def test_performance_tracking(self, fund_manager, mock_positions):
        """测试表现跟踪"""
        # 记录初始表现
        initial_performance = fund_manager.calculate_current_performance(mock_positions)
        fund_manager.record_performance_snapshot(
            allocation=fund_manager.get_current_allocation(),
            performance=initial_performance,
            timestamp=datetime.utcnow()
        )

        # 模拟一段时间后的表现变化
        updated_positions = mock_positions.copy()
        for pos in updated_positions:
            pos["unrealized_pnl"] *= 2  # 假设盈翻倍

        updated_performance = fund_manager.calculate_current_performance(updated_positions)

        fund_manager.record_performance_snapshot(
            allocation=fund_manager.get_current_allocation(),
            performance=updated_performance,
            timestamp=datetime.utcnow()
        )

        # 验证表现历史
        performance_history = fund_manager.get_performance_history()
        assert len(performance_history) == 2

        # 验证计算指标
        metrics = fund_manager.calculate_performance_metrics(performance_history)
        assert "average_return" in metrics
        assert "sharpe_ratio" in metrics
        assert "max_drawdown" in metrics

    def test_rebalancing_triggers(self, fund_manager, mock_positions):
        """测试重新平衡触发器"""
        # 设置触发条件
        triggers = {
            "performance_threshold": 0.15,  # 15%表现差异
            "time_threshold": 30,        # 30天
            "volatility_threshold": 0.3,  # 30%波动率
            "correlation_threshold": 0.8  # 80%相关性
        }

        fund_manager.rebalancing_triggers = triggers

        # 模拟触发条件满足的情况
        performance_data = fund_manager.calculate_current_performance(mock_positions)

        # 假设激进策略表现显著优于保守策略
        performance_data["strategy_performance"] = {
            "conservative": {"return": 0.02},
            "aggressive": {"return": 0.25}
        }

        should_rebalance = fund_manager.should_trigger_rebalancing(performance_data)

        # 表现差异超过阈值，应该触发重新平衡
        assert should_rebalance["should_rebalance"]
        assert "performance_based" in should_rebalance["triggers"]

    def test_automated_rebalancing(self, fund_manager, mock_positions):
        """测试自动重新平衡"""
        # 设置自动重新平衡
        fund_manager.enable_automated_rebalancing = True
        fund_manager.rebalancing_frequency = timedelta(days=7)

        # 模拟重新平衡执行
        performance_data = fund_manager.calculate_current_performance(mock_positions)

        rebalance_result = fund_manager.execute_automated_rebalancing(performance_data)

        assert "success" in rebalance_result
        assert "new_allocation" in rebalance_result
        assert "execution_time" in rebalance_result

        if rebalance_result["success"]:
            # 验证新分配已保存
            current_allocation = fund_manager.get_current_allocation()
            assert current_allocation == rebalance_result["new_allocation"]

    def test_fund_allocation_strategies(self, fund_manager):
        """测试资金分配策略"""
        # 测试保守策略
        conservative_strategy = FundAllocationStrategy.CONSERVATIVE
        conservative_allocation = fund_manager.get_strategy_allocation(conservative_strategy)

        assert conservative_allocation["conservative"] >= 0.6
        assert conservative_allocation["aggressive"] <= 0.1

        # 测试激进策略
        aggressive_strategy = FundAllocationStrategy.AGGRESSIVE
        aggressive_allocation = fund_manager.get_strategy_allocation(aggressive_strategy)

        assert aggressive_allocation["aggressive"] >= 0.6
        assert aggressive_allocation["conservative"] <= 0.1

        # 测试平衡策略
        balanced_strategy = FundAllocationStrategy.BALANCED
        balanced_allocation = fund_manager.get_strategy_allocation(balanced_strategy)

        assert abs(balanced_allocation["conservative"] - 0.33) < 0.05
        assert abs(balanced_allocation["aggressive"] - 0.33) < 0.05

    def test_emergency_adjustments(self, fund_manager):
        """测试紧急调整"""
        # 模拟市场危机情况
        crisis_data = {
            "market_crash": True,
            "volatility_spike": True,
            "liquidity_crisis": False,
            "sentiment_panic": True
        }

        emergency_adjustment = fund_manager.calculate_emergency_adjustment(crisis_data)

        assert "adjustments" in emergency_adjustment
        assert "urgency_level" in emergency_adjustment

        # 危机情况下应该大幅增加保守分配
        assert emergency_adjustment["adjustments"]["conservative"]["direction"] == "increase"
        assert emergency_adjustment["adjustments"]["conservative"]["magnitude"] > 0.2

    def test_allocation_optimization(self, fund_manager, mock_positions):
        """测试分配优化"""
        # 运行分配优化
        optimization_result = fund_manager.optimize_allocation(
            mock_positions,
            lookback_period=30,
            optimization_target="sharpe_ratio"
        )

        assert "optimized_allocation" in optimization_result
        assert "expected_improvement" in optimization_result
        assert "optimization_method" in optimization_result

        # 验证优化结果有效性
        optimized = optimization_result["optimized_allocation"]
        assert sum(optimized.values()) == Decimal("1.0")

    def test_backtesting_allocation_strategy(self, fund_manager):
        """测试分配策略回测"""
        # 模拟历史数据
        historical_data = []
        for i in range(100):
            day_data = {
                "date": datetime.utcnow() - timedelta(days=i),
                "returns": {
                    "conservative": 0.001 + i * 0.0001,
                    "moderate": 0.002 + i * 0.0002,
                    "aggressive": 0.003 + i * 0.0003
                }
            }
            historical_data.append(day_data)

        # 运行回测
        backtest_result = fund_manager.backtest_allocation_strategy(
            historical_data,
            rebalance_frequency=30,
            transaction_cost=0.001
        )

        assert "total_return" in backtest_result
        assert "sharpe_ratio" in backtest_result
        assert "max_drawdown" in backtest_result
        assert "rebalance_count" in backtest_result

        # 验证回测统计数据
        stats = backtest_result["statistics"]
        assert "win_rate" in stats
        assert "average_rebalance_cost" in stats


class TestFundAllocation:
    """资金分配类单元测试"""

    def test_fund_allocation_creation(self):
        """测试资金分配创建"""
        allocation = FundAllocation(
            strategy_type="moderate",
            allocation_percent=Decimal("0.5"),
            allocated_amount=Decimal("50000.0"),
            target_symbols=["BTC/USDT", "ETH/USDT"],
            created_at=datetime.utcnow()
        )

        assert allocation.strategy_type == "moderate"
        assert allocation.allocation_percent == Decimal("0.5")
        assert allocation.allocated_amount == Decimal("50000.0")

    def test_fund_allocation_comparison(self):
        """测试资金分配比较"""
        allocation1 = FundAllocation(
            strategy_type="conservative",
            allocation_percent=Decimal("0.3"),
            allocated_amount=Decimal("30000.0")
        )

        allocation2 = FundAllocation(
            strategy_type="aggressive",
            allocation_percent=Decimal("0.2"),
            allocated_amount=Decimal("20000.0")
        )

        # 比较分配
        assert allocation1.is_larger_than(allocation2)
        assert not allocation2.is_larger_than(allocation1)

    def test_fund_allocation_calculation(self):
        """测试资金分配计算"""
        total_funds = Decimal("100000.0")
        allocation_percent = Decimal("0.25")

        calculated_amount = total_funds * allocation_percent
        assert calculated_amount == Decimal("25000.0")

        # 创建分配对象
        allocation = FundAllocation(
            strategy_type="moderate",
            allocation_percent=allocation_percent,
            calculated_amount=calculated_amount
        )

        # 更新分配百分比
        new_percent = Decimal("0.3")
        allocation.update_allocation_percent(new_percent, total_funds)
        assert allocation.allocation_percent == new_percent
        assert allocation.allocated_amount == total_funds * new_percent


class TestPerformanceMetrics:
    """表现指标类单元测试"""

    def test_performance_metrics_creation(self):
        """测试表现指标创建"""
        metrics = PerformanceMetrics(
            period="30d",
            total_return=Decimal("0.05"),
            sharpe_ratio=Decimal("1.2"),
            max_drawdown=Decimal("-0.03"),
            win_rate=Decimal("0.65"),
            volatility=Decimal("0.15")
        )

        assert metrics.period == "30d"
        assert metrics.total_return == Decimal("0.05")
        assert metrics.sharpe_ratio == Decimal("1.2")

    def test_sharpe_ratio_calculation(self):
        """测试夏普比率计算"""
        annual_return = Decimal("0.15")  # 15%
        risk_free_rate = Decimal("0.02")    # 2%
        volatility = Decimal("0.20")      # 20%

        # Sharpe Ratio = (Portfolio Return - Risk-Free Rate) / Portfolio Volatility
        expected_sharpe = (annual_return - risk_free_rate) / volatility
        calculated_sharpe = PerformanceMetrics.calculate_sharpe_ratio(
            annual_return, risk_free_rate, volatility
        )

        assert abs(calculated_sharpe - expected_sharpe) < Decimal("0.01")

    def test_max_drawdown_calculation(self):
        """测试最大回撤计算"""
        peak_value = Decimal("100000.0")
        trough_value = Decimal("85000.0")

        # Max Drawdown = (Peak - Trough) / Peak
        expected_drawdown = (peak_value - trough_value) / peak_value
        calculated_drawdown = PerformanceMetrics.calculate_max_drawdown(peak_value, trough_value)

        assert abs(calculated_drawdown - expected_drawdown) < Decimal("0.0001")

    def test_calmaratio_calculation(self):
        """测试卡玛比率计算"""
        annual_return = Decimal("0.20")
        max_drawdown = Decimal("-0.10")

        # Calmar Ratio = Annual Return / |Max Drawdown|
        expected_calmar = annual_return / abs(max_drawdown)
        calculated_calmar = PerformanceMetrics.calculate_calmar_ratio(
            annual_return, max_drawdown
        )

        assert abs(calculated_calmar - expected_calmar) < Decimal("0.01")

    def test_information_ratio_calculation(self):
        """测试信息比率计算"""
        annual_return = Decimal("0.15")
        annual_volatility = Decimal("0.25")
        risk_free_rate = Decimal("0.02")

        # Information Ratio = (Portfolio Return - Risk-Free Rate) / Portfolio Volatility
        expected_ir = (annual_return - risk_free_rate) / annual_volatility
        calculated_ir = PerformanceMetrics.calculate_information_ratio(
            annual_return, risk_free_rate, annual_volatility
        )

        assert abs(calculated_ir - expected_ir) < Decimal("0.01")


# 边缘情况测试
class TestDynamicFundManagerEdgeCases:
    """动态资金管理器边缘情况测试"""

    def test_empty_positions(self, fund_manager):
        """测试空持仓情况"""
        empty_positions = []

        performance = fund_manager.calculate_current_performance(empty_positions)
        assert performance["total_unrealized_pnl"] == Decimal("0")
        assert performance["total_value"] == fund_manager.total_funds

        # 重新平衡空持仓
        rebalance_result = fund_manager.rebalance_allocation(performance)
        assert "final_allocation" in rebalance_result

    def test_zero_funds(self):
        """测试零资金情况"""
        zero_funds_manager = DynamicFundManager(total_funds=Decimal("0.0"))

        with pytest.raises(ValueError):
            zero_funds_manager.rebalance_allocation({})

    def test_single_strategy_allocation(self, fund_manager):
        """测试单一策略分配"""
        fund_manager.initial_allocation = {"moderate": Decimal("1.0")}

        adjustment = fund_manager.calculate_performance_adjustment([])
        assert adjustment["adjustments"]["moderate"]["direction"] == "maintain"

    def test_extreme_performance_scenarios(self, fund_manager):
        """测试极端表现情况"""
        # 极端盈利情况
        extreme_profit = {
            "total_unrealized_pnl": Decimal("50000.0"),
            "pnl_percentage": Decimal("50.0")
        }

        profit_adjustment = fund_manager.calculate_performance_adjustment([extreme_profit])
        assert profit_adjustment["adjustments"]["aggressive"]["direction"] == "increase"

        # 极端亏损情况
        extreme_loss = {
            "total_unrealized_pnl": Decimal("-30000.0"),
            "pnl_percentage": Decimal("-30.0")
        }

        loss_adjustment = fund_manager.calculate_performance_adjustment([extreme_loss])
        assert loss_adjustment["adjustments"]["conservative"]["direction"] == "increase"

    def test_rapid_rebalancing(self, fund_manager):
        """测试快速重新平衡"""
        # 在短时间内执行多次重新平衡
        for i in range(5):
            mock_performance = {
                "total_unrealized_pnl": Decimal(f"{i * 1000.0}")
            }

            adjustment = fund_manager.calculate_performance_adjustment([mock_performance])
            fund_manager.record_allocation_change(
                from_allocation=fund_manager.get_current_allocation(),
                to_allocation=adjustment.get("new_allocation", fund_manager.get_current_allocation()),
                reason="rapid_test"
            )

        # 验证历史记录
        history = fund_manager.get_allocation_history()
        assert len(history) == 5

    def test_invalid_allocation_data(self, fund_manager):
        """测试无效分配数据"""
        # 总和不等于1的分配
        invalid_allocation = {
            "conservative": 0.4,
            "moderate": 0.4,
            "aggressive": 0.3  # 总和1.1
        }

        with pytest.raises(ValueError):
            fund_manager.validate_allocation(invalid_allocation)

        # 负数分配
        negative_allocation = {
            "conservative": -0.1,
            "moderate": 0.6,
            "aggressive": 0.5
        }

        with pytest.raises(ValueError):
            fund_manager.validate_allocation(negative_allocation)