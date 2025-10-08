"""
做空策略分析验证和错误处理模块

提供专门的做空策略验证逻辑和错误处理。
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass

from .exceptions import ValidationError, BusinessLogicError, ExternalServiceError
from .logging import get_logger

logger = get_logger(__name__)


@dataclass
class ShortStrategyValidationResult:
    """做空策略验证结果"""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    risk_level: str
    confidence_adjustment: float  # 置信度调整因子


@dataclass
class ShortMarketConditions:
    """做空市场条件"""
    trend_direction: str
    volatility_level: str
    volume_profile: str
    market_sentiment: str
    borrowing_cost_rate: float
    liquidation_risk_level: str
    margin_requirement_percent: float


class ShortStrategyValidator:
    """做空策略验证器"""

    def __init__(self):
        self.logger = logger

        # 做空策略验证规则
        self.validation_rules = {
            # 仓位限制
            "max_position_size": {
                "low_risk": 10.0,      # 低风险最大仓位
                "medium_risk": 20.0,    # 中等风险最大仓位
                "high_risk": 30.0       # 高风险最大仓位
            },

            # 置信度要求
            "min_confidence_threshold": {
                "low_risk": 0.8,        # 低风险最小置信度
                "medium_risk": 0.7,     # 中等风险最小置信度
                "high_risk": 0.6        # 高风险最小置信度
            },

            # 风险回报比要求
            "min_risk_reward_ratio": {
                "low_risk": 2.5,        # 低风险最小风险回报比
                "medium_risk": 2.0,     # 中等风险最小风险回报比
                "high_risk": 1.5        # 高风险最小风险回报比
            },

            # 借贷成本限制
            "max_borrowing_cost_rate": 0.15,  # 最大借贷成本率（年化15%）

            # 清算风险评估
            "liquidation_risk_threshold": {
                "acceptable": "low",
                "caution": "medium",
                "avoid": "high"
            }
        }

    def validate_short_strategy_request(
        self,
        symbol: str,
        timeframe: str,
        analysis_period_days: int,
        confidence_threshold: float,
        max_position_size: float,
        market_data: Optional[Dict[str, Any]] = None
    ) -> ShortStrategyValidationResult:
        """验证做空策略请求"""
        errors = []
        warnings = []

        try:
            # 1. 基础参数验证
            self._validate_basic_parameters(
                symbol, timeframe, analysis_period_days,
                confidence_threshold, max_position_size, errors, warnings
            )

            # 2. 市场条件验证
            if market_data:
                self._validate_market_conditions(market_data, errors, warnings)

            # 3. 风险评估
            risk_level = self._assess_risk_level(errors, warnings)

            # 4. 置信度调整
            confidence_adjustment = self._calculate_confidence_adjustment(warnings, risk_level)

            is_valid = len(errors) == 0

            result = ShortStrategyValidationResult(
                is_valid=is_valid,
                errors=errors,
                warnings=warnings,
                risk_level=risk_level,
                confidence_adjustment=confidence_adjustment
            )

            self.logger.info(f"做空策略请求验证完成: {symbol}, 有效={is_valid}, 风险等级={risk_level}")

            return result

        except Exception as e:
            self.logger.error(f"做空策略请求验证失败: {e}")
            errors.append(f"验证过程发生错误: {str(e)}")

            return ShortStrategyValidationResult(
                is_valid=False,
                errors=errors,
                warnings=warnings,
                risk_level="high",
                confidence_adjustment=-0.2
            )

    def validate_short_strategy_result(
        self,
        strategy_result: Dict[str, Any],
        technical_analysis: Dict[str, Any],
        current_price: float
    ) -> ShortStrategyValidationResult:
        """验证做空策略结果"""
        errors = []
        warnings = []

        try:
            # 1. 价格逻辑验证
            self._validate_price_logic(strategy_result, current_price, errors, warnings)

            # 2. 风险指标验证
            self._validate_risk_metrics(strategy_result, errors, warnings)

            # 3. 技术一致性验证
            self._validate_technical_consistency(
                strategy_result, technical_analysis, errors, warnings
            )

            # 4. 做空特定验证
            self._validate_short_specific_factors(strategy_result, errors, warnings)

            # 5. 市场环境验证
            self._validate_market_environment(
                strategy_result, technical_analysis, errors, warnings
            )

            # 6. 风险评估
            risk_level = self._assess_result_risk_level(errors, warnings, strategy_result)

            # 7. 置信度调整
            confidence_adjustment = self._calculate_result_confidence_adjustment(
                warnings, risk_level, strategy_result
            )

            is_valid = len(errors) == 0

            result = ShortStrategyValidationResult(
                is_valid=is_valid,
                errors=errors,
                warnings=warnings,
                risk_level=risk_level,
                confidence_adjustment=confidence_adjustment
            )

            self.logger.info(f"做空策略结果验证完成，有效={is_valid}, 风险等级={risk_level}")

            return result

        except Exception as e:
            self.logger.error(f"做空策略结果验证失败: {e}")
            errors.append(f"结果验证过程发生错误: {str(e)}")

            return ShortStrategyValidationResult(
                is_valid=False,
                errors=errors,
                warnings=warnings,
                risk_level="high",
                confidence_adjustment=-0.3
            )

    def _validate_basic_parameters(
        self,
        symbol: str,
        timeframe: str,
        analysis_period_days: int,
        confidence_threshold: float,
        max_position_size: float,
        errors: List[str],
        warnings: List[str]
    ):
        """验证基础参数"""
        # 交易符号验证
        if not symbol or not isinstance(symbol, str):
            errors.append("交易符号不能为空且必须是字符串")
        elif "/" not in symbol:
            errors.append("交易符号格式无效，应为 BASE/QUOTE 格式")
        elif len(symbol.split("/")) != 2:
            errors.append("交易符号格式无效，应包含一个斜杠分隔符")

        # 时间框架验证
        valid_timeframes = ["1m", "5m", "15m", "30m", "1h", "4h", "1d"]
        if timeframe not in valid_timeframes:
            errors.append(f"时间框架无效，支持的值: {', '.join(valid_timeframes)}")

        # 分析周期验证
        if not isinstance(analysis_period_days, int) or analysis_period_days < 1 or analysis_period_days > 30:
            errors.append("分析周期必须是1-30之间的整数")

        if analysis_period_days < 3:
            warnings.append("分析周期较短，可能影响策略分析的准确性")

        # 置信度阈值验证
        if not isinstance(confidence_threshold, (int, float)) or confidence_threshold < 0.5 or confidence_threshold > 1.0:
            errors.append("置信度阈值必须是0.5-1.0之间的数值")

        if confidence_threshold > 0.9:
            warnings.append("置信度阈值设置过高，可能导致策略机会减少")

        # 最大仓位验证
        if not isinstance(max_position_size, (int, float)) or max_position_size < 1.0 or max_position_size > 50.0:
            errors.append("最大仓位必须是1.0-50.0之间的数值")

        if max_position_size > 25.0:
            warnings.append("做空仓位设置较大，增加风险敞口")

    def _validate_market_conditions(
        self,
        market_data: Dict[str, Any],
        errors: List[str],
        warnings: List[str]
    ):
        """验证市场条件"""
        # 检查借贷成本
        borrowing_cost = market_data.get("borrowing_cost_rate", 0)
        if borrowing_cost > self.validation_rules["max_borrowing_cost_rate"]:
            errors.append(f"借贷成本过高: {borrowing_cost:.2%} > {self.validation_rules['max_borrowing_cost_rate']:.2%}")
        elif borrowing_cost > 0.1:
            warnings.append(f"借贷成本较高: {borrowing_cost:.2%}")

        # 检查市场波动性
        volatility = market_data.get("volatility_level", "medium")
        if volatility == "extreme":
            errors.append("市场波动性过高，不适合做空策略")
        elif volatility == "high":
            warnings.append("市场波动性较高，做空风险增加")

        # 检查趋势方向
        trend = market_data.get("trend_direction", "neutral")
        if trend == "strong_up":
            errors.append("强劲上涨趋势，做空风险极高")
        elif trend == "up":
            warnings.append("上涨趋势，做空需要谨慎")

    def _validate_price_logic(
        self,
        strategy_result: Dict[str, Any],
        current_price: float,
        errors: List[str],
        warnings: List[str]
    ):
        """验证价格逻辑"""
        entry_price = strategy_result.get("entry_price", 0)
        stop_loss = strategy_result.get("stop_loss_price", 0)
        take_profit = strategy_result.get("take_profit_price", 0)

        # 基础价格验证
        if entry_price <= 0:
            errors.append("入场价格必须大于0")
        if stop_loss <= 0:
            errors.append("止损价格必须大于0")
        if take_profit <= 0:
            errors.append("止盈价格必须大于0")

        # 做空价格逻辑验证
        if entry_price > 0 and stop_loss > 0:
            if stop_loss <= entry_price:
                errors.append("做空策略止损价格必须高于入场价格")
            else:
                stop_loss_distance = (stop_loss - entry_price) / entry_price
                if stop_loss_distance < 0.03:  # 3%
                    warnings.append("止损距离过近，可能容易被触发")
                elif stop_loss_distance > 0.15:  # 15%
                    warnings.append("止损距离过远，风险控制不足")

        if entry_price > 0 and take_profit > 0:
            if take_profit >= entry_price:
                errors.append("做空策略止盈价格必须低于入场价格")
            else:
                profit_distance = (entry_price - take_profit) / entry_price
                if profit_distance < 0.05:  # 5%
                    warnings.append("止盈距离过近，利润空间有限")

        # 当前价格与入场价格偏差检查
        if current_price > 0 and entry_price > 0:
            price_deviation = abs(current_price - entry_price) / current_price
            if price_deviation > 0.05:  # 5%
                warnings.append(f"入场价格与当前价格偏差较大: {price_deviation:.2%}")

    def _validate_risk_metrics(
        self,
        strategy_result: Dict[str, Any],
        errors: List[str],
        warnings: List[str]
    ):
        """验证风险指标"""
        position_size = strategy_result.get("position_size_percent", 0)
        confidence = strategy_result.get("confidence_score", 0)

        # 仓位大小验证
        if position_size <= 0 or position_size > 50:
            errors.append("仓位大小必须在0-50%之间")

        if position_size > 30:
            warnings.append("做空仓位过大，风险敞口较高")

        # 置信度验证
        if confidence < 0 or confidence > 1:
            errors.append("置信度必须在0-1之间")

        if confidence < 0.6:
            warnings.append("策略置信度较低，建议谨慎执行")

        # 风险回报比验证
        entry_price = strategy_result.get("entry_price", 0)
        stop_loss = strategy_result.get("stop_loss_price", 0)
        take_profit = strategy_result.get("take_profit_price", 0)

        if all(x > 0 for x in [entry_price, stop_loss, take_profit]):
            risk_reward_ratio = (entry_price - take_profit) / (stop_loss - entry_price)
            if risk_reward_ratio < 1.0:
                errors.append("风险回报比必须大于1.0")
            elif risk_reward_ratio < 1.5:
                warnings.append("风险回报比较低，建议重新评估")

    def _validate_technical_consistency(
        self,
        strategy_result: Dict[str, Any],
        technical_analysis: Dict[str, Any],
        errors: List[str],
        warnings: List[str]
    ):
        """验证技术一致性"""
        recommendation = strategy_result.get("recommendation", "")
        market_analysis = technical_analysis.get("market_analysis", {})
        trend_direction = market_analysis.get("trend_direction", "neutral")

        # 推荐与趋势一致性检查
        if recommendation in ["sell", "strong_sell"]:
            if trend_direction == "strong_up":
                errors.append("做空推荐与强劲上涨趋势不一致")
            elif trend_direction == "up":
                warnings.append("做空推荐与上涨趋势存在冲突")
        elif recommendation in ["buy", "strong_buy"]:
            if trend_direction in ["down", "strong_down"]:
                errors.append("做多推荐与下跌趋势不一致")

        # 技术指标检查
        indicators = technical_analysis.get("indicators", {})
        rsi = indicators.get("rsi", 50)

        if recommendation in ["sell", "strong_sell"] and rsi < 30:
            warnings.append("推荐做空但RSI显示超卖，可能存在反弹风险")
        elif recommendation in ["buy", "strong_buy"] and rsi > 70:
            warnings.append("推荐做多但RSI显示超买，可能存在回调风险")

    def _validate_short_specific_factors(
        self,
        strategy_result: Dict[str, Any],
        errors: List[str],
        warnings: List[str]
    ):
        """验证做空特定因素"""
        short_factors = strategy_result.get("short_specific_factors", {})

        # 借贷成本检查
        borrowing_cost = short_factors.get("borrowing_cost_rate", 0)
        if borrowing_cost > 0.2:  # 20%
            errors.append(f"借贷成本过高: {borrowing_cost:.2%}")
        elif borrowing_cost > 0.1:  # 10%
            warnings.append(f"借贷成本较高: {borrowing_cost:.2%}")

        # 清算风险检查
        liquidation_risk = short_factors.get("liquidation_risk_level", "medium")
        if liquidation_risk == "high":
            errors.append("清算风险过高，不适合当前做空策略")
        elif liquidation_risk == "medium":
            warnings.append("清算风险中等，需要密切监控")

        # 保证金要求检查
        margin_requirement = short_factors.get("margin_requirement_percent", 0)
        if margin_requirement > 50:
            warnings.append(f"保证金要求较高: {margin_requirement}%")

    def _validate_market_environment(
        self,
        strategy_result: Dict[str, Any],
        technical_analysis: Dict[str, Any],
        errors: List[str],
        warnings: List[str]
    ):
        """验证市场环境"""
        # 检查市场深度
        volume_ratio = technical_analysis.get("indicators", {}).get("volume_ratio", 1.0)
        if volume_ratio < 0.5:
            warnings.append("市场成交量偏低，流动性可能不足")

        # 检查价格波动
        atr = technical_analysis.get("indicators", {}).get("atr", 0)
        current_price = technical_analysis.get("current_price", 0)

        if atr > 0 and current_price > 0:
            volatility_ratio = atr / current_price
            if volatility_ratio > 0.05:  # 5%
                warnings.append("价格波动性较高，做空风险增加")

    def _assess_risk_level(self, errors: List[str], warnings: List[str]) -> str:
        """评估风险等级"""
        if len(errors) > 0:
            return "high"
        elif len(warnings) >= 3:
            return "high"
        elif len(warnings) >= 1:
            return "medium"
        else:
            return "low"

    def _assess_result_risk_level(
        self,
        errors: List[str],
        warnings: List[str],
        strategy_result: Dict[str, Any]
    ) -> str:
        """评估结果风险等级"""
        base_risk = self._assess_risk_level(errors, warnings)

        # 根据策略特征调整风险等级
        confidence = strategy_result.get("confidence_score", 0.5)
        if confidence < 0.6:
            if base_risk == "low":
                return "medium"
            elif base_risk == "medium":
                return "high"

        position_size = strategy_result.get("position_size_percent", 10)
        if position_size > 25:
            if base_risk != "high":
                return "high"

        return base_risk

    def _calculate_confidence_adjustment(
        self,
        warnings: List[str],
        risk_level: str
    ) -> float:
        """计算置信度调整因子"""
        adjustment = 0.0

        # 根据警告数量调整
        adjustment -= len(warnings) * 0.02

        # 根据风险等级调整
        if risk_level == "high":
            adjustment -= 0.1
        elif risk_level == "medium":
            adjustment -= 0.05

        return max(adjustment, -0.3)  # 最大调整-0.3

    def _calculate_result_confidence_adjustment(
        self,
        warnings: List[str],
        risk_level: str,
        strategy_result: Dict[str, Any]
    ) -> float:
        """计算结果置信度调整因子"""
        adjustment = self._calculate_confidence_adjustment(warnings, risk_level)

        # 根据策略特征额外调整
        confidence = strategy_result.get("confidence_score", 0.5)
        if confidence < 0.7:
            adjustment -= 0.05

        # 检查推荐的一致性
        recommendation = strategy_result.get("recommendation", "")
        if recommendation not in ["sell", "strong_sell", "hold"]:
            adjustment -= 0.1

        return max(adjustment, -0.4)  # 最大调整-0.4


class ShortStrategyErrorHandler:
    """做空策略错误处理器"""

    def __init__(self):
        self.logger = logger

    def handle_validation_error(
        self,
        validation_result: ShortStrategyValidationResult,
        symbol: str,
        context: str = "request"
    ) -> Tuple[bool, Optional[str]]:
        """处理验证错误"""
        if validation_result.is_valid:
            return True, None

        # 构建错误消息
        if validation_result.errors:
            error_message = f"做空策略{context}验证失败 ({symbol}): " + "; ".join(validation_result.errors)
        else:
            error_message = f"做空策略{context}验证存在严重问题 ({symbol})"

        self.logger.warning(error_message)

        # 根据风险等级决定是否抛出异常
        if validation_result.risk_level == "high" or len(validation_result.errors) > 0:
            raise ValidationError(error_message)

        return False, error_message

    def handle_analysis_error(
        self,
        error: Exception,
        symbol: str,
        stage: str = "analysis"
    ) -> str:
        """处理分析错误"""
        error_type = type(error).__name__
        error_message = str(error)

        # 记录错误
        self.logger.error(f"做空策略{stage}失败 ({symbol}): {error_type} - {error_message}")

        # 构建用户友好的错误消息
        if isinstance(error, ValidationError):
            return f"做空策略参数验证失败: {error_message}"
        elif isinstance(error, ExternalServiceError):
            return f"外部服务调用失败: {error_message}"
        elif isinstance(error, BusinessLogicError):
            return f"业务逻辑错误: {error_message}"
        else:
            return f"做空策略{stage}过程中发生未知错误，请稍后重试"

    def apply_confidence_adjustment(
        self,
        strategy_result: Dict[str, Any],
        validation_result: ShortStrategyValidationResult
    ) -> Dict[str, Any]:
        """应用置信度调整"""
        if validation_result.confidence_adjustment != 0:
            original_confidence = strategy_result.get("confidence_score", 0.5)
            adjusted_confidence = max(
                0.0,
                min(1.0, original_confidence + validation_result.confidence_adjustment)
            )

            strategy_result["confidence_score"] = adjusted_confidence
            strategy_result["original_confidence_score"] = original_confidence
            strategy_result["confidence_adjustment"] = validation_result.confidence_adjustment

            self.logger.info(
                f"置信度已调整: {original_confidence:.3f} -> {adjusted_confidence:.3f} "
                f"(调整: {validation_result.confidence_adjustment:.3f})"
            )

        return strategy_result


# 全局实例
short_strategy_validator = ShortStrategyValidator()
short_strategy_error_handler = ShortStrategyErrorHandler()


# 便捷函数
def validate_short_strategy_request(
    symbol: str,
    timeframe: str,
    analysis_period_days: int,
    confidence_threshold: float,
    max_position_size: float,
    market_data: Optional[Dict[str, Any]] = None
) -> ShortStrategyValidationResult:
    """验证做空策略请求的便捷函数"""
    return short_strategy_validator.validate_short_strategy_request(
        symbol, timeframe, analysis_period_days,
        confidence_threshold, max_position_size, market_data
    )


def validate_short_strategy_result(
    strategy_result: Dict[str, Any],
    technical_analysis: Dict[str, Any],
    current_price: float
) -> ShortStrategyValidationResult:
    """验证做空策略结果的便捷函数"""
    return short_strategy_validator.validate_short_strategy_result(
        strategy_result, technical_analysis, current_price
    )