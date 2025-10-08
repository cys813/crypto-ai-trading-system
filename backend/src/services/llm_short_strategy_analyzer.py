"""
LLM做空策略分析服务

使用大语言模型分析技术分析结果，生成做空策略建议。
"""

import logging
import time
import asyncio
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass
import json

from ..core.llm_integration import get_llm_service, LLMRequest, LLMProvider, LLMModel
from ..core.logging import BusinessLogger
from ..core.exceptions import LLMServiceError, ValidationError, BusinessLogicError
from ..core.short_strategy_validation import (
    validate_short_strategy_request, validate_short_strategy_result,
    short_strategy_error_handler, ShortStrategyValidationResult
)
from ..core.short_strategy_logging import (
    short_strategy_logger, ShortStrategyPerformanceMetrics,
    log_short_strategy_start, log_short_strategy_complete,
    log_short_strategy_error
)
from ..models.technical_analysis import TechnicalAnalysis
from ..services.exchange_data_collector import KlineData

logger = logging.getLogger(__name__)
business_logger = BusinessLogger("llm_short_strategy_analyzer")


@dataclass
class ShortStrategyAnalysis:
    """做空策略分析结果"""
    symbol: str
    timeframe: str
    recommendation: str  # 'strong_sell', 'sell', 'hold', 'buy', 'strong_buy'
    confidence_score: float  # 0.0 - 1.0
    entry_price: float
    stop_loss_price: float
    take_profit_price: float
    position_size_percent: float  # 建议仓位百分比
    time_horizon: str  # 'short_term', 'medium_term', 'long_term'
    reasoning: str
    risk_factors: List[str]
    market_conditions: Dict[str, Any]
    technical_signals: List[Dict[str, Any]]
    price_targets: Dict[str, float]
    execution_strategy: Dict[str, Any]
    analysis_timestamp: datetime
    short_specific_factors: Dict[str, Any]  # 做空特定因素


class LLMShortStrategyAnalyzer:
    """LLM做空策略分析器"""

    def __init__(self):
        self.logger = logger
        self.business_logger = business_logger
        self.llm_service = get_llm_service()

        # 默认配置
        self.default_model = LLMModel.GPT_4O
        self.default_provider = LLMProvider.OPENAI
        self.temperature = 0.3  # 较低温度确保一致性
        self.max_tokens = 2000

        # 做空策略模板
        self.strategy_prompt_template = """
你是一位专业的加密货币交易分析师，专注于做空策略分析。请基于以下技术分析结果，提供详细的做空策略建议。

## 技术分析数据

**基本信息:**
- 交易符号: {symbol}
- 时间框架: {timeframe}
- 当前价格: ${current_price:.2f}
- 分析时间: {analysis_time}

**技术指标:**
- RSI: {rsi_value}
- MACD: macd_line={macd_line}, signal_line={signal_line}, histogram={macd_histogram}
- 布林带: 上轨={bb_upper}, 中轨={bb_middle}, 下轨={bb_lower}
- 移动平均线: SMA20={sma_20}, SMA50={sma_50}, EMA12={ema_12}, EMA26={ema_26}
- 成交量比率: {volume_ratio}
- ATR: {atr_value}

**市场条件:**
- 趋势方向: {trend_direction}
- 趋势强度: {trend_strength}
- 波动性水平: {volatility_level}
- 市场情绪: {market_sentiment}

**支撑阻力位:**
- 最近支撑位: {support_levels}
- 最近阻力位: {resistance_levels}

**技术信号:**
{technical_signals}

**做空特定指标:**
- 超买信号: {overbought_signals}
- 阻力位突破: {resistance_breakdown}
- 成交量背离: {volume_divergence}
- 负面动能: {negative_momentum}

## 分析要求

请提供以下格式的详细分析：

### 1. 做空策略建议
- **推荐操作**: [strong_sell/sell/hold/buy/strong_buy]
- **置信度**: [0.0-1.0之间的数值]
- **主要理由**: [1-2句话的核心原因]

### 2. 做空入场策略
- **建议入场价**: [具体的入场价格]
- **入场时机**: [立即/等待反弹/等待跌破]
- **仓位大小**: [建议资金百分比，如5-20%]
- **时间框架**: [短期/中期/长期]

### 3. 做空风险管理
- **止损价格**: [具体止损价格]
- **止盈价格**: [具体止盈价格]
- **风险回报比**: [计算的风险回报比]
- **最大持仓时间**: [建议的最长持仓时间]

### 4. 做空市场分析
- **下跌趋势分析**: [对当前下跌趋势的详细分析]
- **关键做空因素**: [支持做空决策的关键技术因素]
- **做空风险**: [2-3个主要做空风险点]
- **做空机会评估**: [做空机会的大小和质量]

### 5. 做空执行策略
- **分批建仓**: [是否建议分批建仓做空]
- **加仓条件**: [什么情况下可以加仓做空]
- **减仓条件**: [什么情况下应该减仓]
- **退出策略**: [完整的做空退出计划]

### 6. 做空特定考虑
- **借贷成本**: [预估的借贷成本]
- **清算风险**: [清算风险评估]
- **保证金要求**: [保证金要求分析]
- **市场深度**: [市场深度对做空的影响]

请以JSON格式回复：
{{
    "recommendation": "sell",
    "confidence_score": 0.75,
    "reasoning": "基于技术分析的综合判断...",
    "entry_price": 50200,
    "stop_loss_price": 52500,
    "take_profit_price": 47500,
    "position_size_percent": 12.0,
    "time_horizon": "short_term",
    "risk_reward_ratio": 2.0,
    "max_hold_time_days": 10,
    "risk_factors": ["空头回补风险", "市场反弹风险"],
    "market_conditions": {{
        "downtrend_analysis": "下降趋势形成...",
        "key_short_factors": ["阻力位有效", "动能减弱"],
        "short_opportunity_assessment": "中等偏高机会"
    }},
    "execution_strategy": {{
        "entry_timing": "immediate",
        "scaling_plan": "分批建仓",
        "add_position_conditions": ["价格反弹至阻力位"],
        "reduce_position_conditions": ["跌破止盈目标", "趋势反转"],
        "exit_strategy": "分批止盈"
    }},
    "short_specific_factors": {{
        "borrowing_cost_rate": 0.05,
        "liquidation_risk_level": "medium",
        "margin_requirement_percent": 15.0,
        "market_depth_impact": "low"
    }}
}}
"""

        # 做空策略验证规则
        self.validation_rules = {
            "position_size_min": 3.0,  # 做空最小仓位百分比（更保守）
            "position_size_max": 30.0,  # 做空最大仓位百分比（更保守）
            "risk_reward_min": 2.0,  # 做空最小风险回报比（更高要求）
            "confidence_min": 0.7  # 做空最小置信度（更高要求）
        }

    async def analyze_short_strategy(
        self,
        symbol: str,
        technical_analysis: Dict[str, Any],
        market_data: Optional[Dict[str, Any]] = None,
        analysis_config: Optional[Dict[str, Any]] = None
    ) -> ShortStrategyAnalysis:
        """分析做空策略"""
        session_id = None
        start_time = time.time()

        try:
            # 记录分析开始
            if analysis_config:
                session_id = log_short_strategy_start(
                    symbol=symbol,
                    timeframe=analysis_config.get("timeframe", "1h"),
                    analysis_period_days=analysis_config.get("analysis_period_days", 7),
                    confidence_threshold=analysis_config.get("confidence_threshold", 0.7),
                    max_position_size=analysis_config.get("max_position_size", 20.0),
                    user_id=analysis_config.get("user_id"),
                    session_id=analysis_config.get("session_id"),
                    request_id=analysis_config.get("request_id")
                )

            self.logger.info(f"开始分析{symbol}的做空策略")

            # 1. 验证输入数据
            self._validate_analysis_input(symbol, technical_analysis)

            # 2. 验证请求参数（如果提供了配置）
            if analysis_config:
                validation_result = validate_short_strategy_request(
                    symbol=symbol,
                    timeframe=analysis_config.get("timeframe", "1h"),
                    analysis_period_days=analysis_config.get("analysis_period_days", 7),
                    confidence_threshold=analysis_config.get("confidence_threshold", 0.7),
                    max_position_size=analysis_config.get("max_position_size", 20.0),
                    market_data=market_data
                )

                # 处理验证错误
                is_valid, error_message = short_strategy_error_handler.handle_validation_error(
                    validation_result, symbol, "request"
                )
                if not is_valid:
                    raise ValidationError(error_message or "请求验证失败")

                if validation_result.warnings:
                    self.logger.warning(f"做空策略请求警告 ({symbol}): {'; '.join(validation_result.warnings)}")

            # 3. 构建LLM提示词
            prompt = self._build_strategy_prompt(symbol, technical_analysis, market_data)

            # 4. 调用LLM（记录性能）
            llm_start_time = time.time()
            llm_response = await self._call_llm(prompt)
            llm_duration_ms = (time.time() - llm_start_time) * 1000

            # 记录LLM调用日志
            short_strategy_logger.log_llm_call(
                symbol=symbol,
                model=self.default_model.value,
                provider=self.default_provider.value,
                prompt_length=len(prompt),
                response_length=len(llm_response),
                duration_ms=llm_duration_ms,
                success=True,
                session_id=session_id
            )

            # 5. 解析LLM响应
            strategy_data = self._parse_llm_response(llm_response)

            # 6. 验证策略结果
            current_price = technical_analysis.get("current_price", 0)
            result_validation = validate_short_strategy_result(strategy_data, technical_analysis, current_price)

            # 处理结果验证错误
            is_valid, error_message = short_strategy_error_handler.handle_validation_error(
                result_validation, symbol, "result"
            )
            if not is_valid:
                # 对于结果验证，不抛出异常，而是记录警告并调整策略
                self.logger.warning(f"做空策略结果验证问题 ({symbol}): {error_message}")

            # 7. 应用验证调整
            validated_strategy = short_strategy_error_handler.apply_confidence_adjustment(
                strategy_data, result_validation
            )

            # 8. 最终策略验证
            final_strategy = self._validate_strategy_result(validated_strategy, technical_analysis)

            # 9. 创建分析结果对象
            analysis = ShortStrategyAnalysis(
                symbol=symbol,
                timeframe=analysis_config.get("timeframe", "1h") if analysis_config else "1h",
                recommendation=final_strategy["recommendation"],
                confidence_score=final_strategy["confidence_score"],
                entry_price=final_strategy["entry_price"],
                stop_loss_price=final_strategy["stop_loss_price"],
                take_profit_price=final_strategy["take_profit_price"],
                position_size_percent=final_strategy["position_size_percent"],
                time_horizon=final_strategy["time_horizon"],
                reasoning=final_strategy["reasoning"],
                risk_factors=final_strategy["risk_factors"],
                market_conditions=final_strategy["market_conditions"],
                technical_signals=technical_analysis.get("technical_signals", []),
                price_targets={
                    "target_1": final_strategy["take_profit_price"],
                    "stop_loss": final_strategy["stop_loss_price"]
                },
                execution_strategy=final_strategy["execution_strategy"],
                analysis_timestamp=datetime.utcnow(),
                short_specific_factors=final_strategy.get("short_specific_factors", {})
            )

            # 10. 记录分析完成和性能指标
            total_duration_ms = (time.time() - start_time) * 1000

            # 记录分析完成日志
            log_short_strategy_complete(
                symbol=symbol,
                recommendation=analysis.recommendation,
                confidence_score=analysis.confidence_score,
                entry_price=analysis.entry_price,
                stop_loss_price=analysis.stop_loss_price,
                take_profit_price=analysis.take_profit_price,
                position_size_percent=analysis.position_size_percent,
                risk_level=result_validation.risk_level,
                session_id=session_id,
                additional_details={
                    "risk_reward_ratio": self._calculate_risk_reward_ratio(analysis),
                    "short_factors": analysis.short_specific_factors,
                    "validation_warnings": len(result_validation.warnings),
                    "validation_errors": len(result_validation.errors),
                    "time_horizon": analysis.time_horizon,
                    "llm_duration_ms": llm_duration_ms,
                    "total_duration_ms": total_duration_ms
                }
            )

            # 记录性能指标
            performance_metrics = ShortStrategyPerformanceMetrics(
                symbol=symbol,
                analysis_duration_ms=total_duration_ms,
                validation_duration_ms=0,  # TODO: 需要分别计时
                llm_call_duration_ms=llm_duration_ms,
                data_collection_duration_ms=0,  # TODO: 需要分别计时
                technical_analysis_duration_ms=0,  # TODO: 需要分别计时
                total_duration_ms=total_duration_ms,
                success=True,
                confidence_score=analysis.confidence_score
            )

            short_strategy_logger.log_performance_metrics(performance_metrics)

            # 记录业务日志（向后兼容）
            log_details = {
                "symbol": symbol,
                "recommendation": analysis.recommendation,
                "confidence": analysis.confidence_score,
                "entry_price": analysis.entry_price,
                "risk_reward_ratio": self._calculate_risk_reward_ratio(analysis),
                "short_factors": analysis.short_specific_factors,
                "validation_warnings": len(result_validation.warnings),
                "validation_errors": len(result_validation.errors),
                "risk_level": result_validation.risk_level,
                "total_duration_ms": total_duration_ms
            }

            if result_validation.confidence_adjustment != 0:
                log_details["confidence_adjustment"] = result_validation.confidence_adjustment

            self.business_logger.log_system_event(
                event_type="short_strategy_analyzed",
                severity="info",
                message=f"做空策略分析完成: {symbol}",
                details=log_details
            )

            return analysis

        except Exception as e:
            # 使用错误处理器处理异常
            error_message = short_strategy_error_handler.handle_analysis_error(e, symbol, "analysis")

            # 记录错误日志
            total_duration_ms = (time.time() - start_time) * 1000
            log_short_strategy_error(
                symbol=symbol,
                error=e,
                stage="analysis",
                session_id=session_id,
                additional_context={
                    "error_message": error_message,
                    "total_duration_ms": total_duration_ms,
                    "error_type": type(e).__name__
                }
            )

            # 记录失败的性能指标
            performance_metrics = ShortStrategyPerformanceMetrics(
                symbol=symbol,
                analysis_duration_ms=total_duration_ms,
                validation_duration_ms=0,
                llm_call_duration_ms=0,
                data_collection_duration_ms=0,
                technical_analysis_duration_ms=0,
                total_duration_ms=total_duration_ms,
                success=False,
                error_type=type(e).__name__
            )

            short_strategy_logger.log_performance_metrics(performance_metrics)

            # 记录业务日志（向后兼容）
            self.business_logger.log_system_event(
                event_type="short_strategy_analysis_failed",
                severity="error",
                message=error_message,
                details={
                    "symbol": symbol,
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "total_duration_ms": total_duration_ms,
                    "session_id": session_id
                }
            )

            # 重新抛出适当的异常
            if isinstance(e, (ValidationError, LLMServiceError, BusinessLogicError)):
                raise
            else:
                raise LLMServiceError(error_message) from e

    async def analyze_multiple_short_strategies(
        self,
        analysis_requests: List[Dict[str, Any]]
    ) -> List[ShortStrategyAnalysis]:
        """批量分析多个交易符号的做空策略"""
        try:
            self.logger.info(f"开始批量分析{len(analysis_requests)}个交易符号的做空策略")

            # 并发执行分析
            tasks = []
            for request in analysis_requests:
                task = self.analyze_short_strategy(
                    symbol=request["symbol"],
                    technical_analysis=request["technical_analysis"],
                    market_data=request.get("market_data"),
                    analysis_config=request.get("config")
                )
                tasks.append(task)

            results = await asyncio.gather(*tasks, return_exceptions=True)

            # 处理结果
            strategies = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    self.logger.error(f"分析{analysis_requests[i]['symbol']}失败: {result}")
                else:
                    strategies.append(result)

            self.business_logger.log_system_event(
                event_type="batch_short_strategy_analysis_completed",
                severity="info",
                message=f"批量做空策略分析完成",
                details={
                    "total_requests": len(analysis_requests),
                    "successful_analyses": len(strategies),
                    "success_rate": len(strategies) / len(analysis_requests) if analysis_requests else 0
                }
            )

            return strategies

        except Exception as e:
            self.logger.error(f"批量做空策略分析失败: {e}")
            raise

    def _validate_analysis_input(
        self,
        symbol: str,
        technical_analysis: Dict[str, Any]
    ):
        """验证分析输入数据"""
        if not symbol:
            raise ValidationError("交易符号不能为空")

        if not technical_analysis:
            raise ValidationError("技术分析数据不能为空")

        required_fields = ["current_price", "indicators", "market_analysis"]
        for field in required_fields:
            if field not in technical_analysis:
                raise ValidationError(f"缺少必需的技术分析字段: {field}")

    def _build_strategy_prompt(
        self,
        symbol: str,
        technical_analysis: Dict[str, Any],
        market_data: Optional[Dict[str, Any]]
    ) -> str:
        """构建做空策略分析提示词"""
        # 提取技术指标
        indicators = technical_analysis.get("indicators", {})
        market_analysis = technical_analysis.get("market_analysis", {})
        technical_signals = technical_analysis.get("technical_signals", [])

        # 支撑阻力位
        support_resistance = technical_analysis.get("support_resistance", {})
        support_levels = support_resistance.get("support_levels", [])
        resistance_levels = support_resistance.get("resistance_levels", [])

        # 做空特定分析
        short_signals = self._analyze_short_signals(technical_analysis)

        # 构建提示词变量
        prompt_vars = {
            "symbol": symbol,
            "timeframe": market_data.get("timeframe", "1h") if market_data else "1h",
            "current_price": technical_analysis.get("current_price", 0),
            "analysis_time": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),

            # 技术指标
            "rsi_value": f"{indicators.get('rsi', 0):.2f}" if indicators.get('rsi') else "N/A",
            "macd_line": f"{indicators.get('macd_line', 0):.4f}" if indicators.get('macd_line') else "N/A",
            "signal_line": f"{indicators.get('macd_signal', 0):.4f}" if indicators.get('macd_signal') else "N/A",
            "macd_histogram": f"{indicators.get('macd_histogram', 0):.4f}" if indicators.get('macd_histogram') else "N/A",
            "bb_upper": f"${indicators.get('bollinger_upper', 0):.2f}" if indicators.get('bollinger_upper') else "N/A",
            "bb_middle": f"${indicators.get('bollinger_middle', 0):.2f}" if indicators.get('bollinger_middle') else "N/A",
            "bb_lower": f"${indicators.get('bollinger_lower', 0):.2f}" if indicators.get('bollinger_lower') else "N/A",
            "sma_20": f"${indicators.get('sma_20', 0):.2f}" if indicators.get('sma_20') else "N/A",
            "sma_50": f"${indicators.get('sma_50', 0):.2f}" if indicators.get('sma_50') else "N/A",
            "ema_12": f"${indicators.get('ema_12', 0):.2f}" if indicators.get('ema_12') else "N/A",
            "ema_26": f"${indicators.get('ema_26', 0):.2f}" if indicators.get('ema_26') else "N/A",
            "volume_ratio": f"{indicators.get('volume_ratio', 1.0):.2f}x",
            "atr_value": f"{indicators.get('atr', 0):.2f}" if indicators.get('atr') else "N/A",

            # 市场条件
            "trend_direction": market_analysis.get("trend_direction", "neutral"),
            "trend_strength": f"{market_analysis.get('trend_strength', 0):.2f}",
            "volatility_level": market_analysis.get("volatility_level", "medium"),
            "market_sentiment": market_analysis.get("market_sentiment", "neutral"),

            # 支撑阻力位
            "support_levels": f"${', '.join(f'{s:.2f}' for s in support_levels[:3])}" if support_levels else "无",
            "resistance_levels": f"${', '.join(f'{r:.2f}' for r in resistance_levels[:3])}" if resistance_levels else "无",

            # 技术信号
            "technical_signals": self._format_technical_signals(technical_signals),

            # 做空特定指标
            "overbought_signals": short_signals["overbought_signals"],
            "resistance_breakdown": short_signals["resistance_breakdown"],
            "volume_divergence": short_signals["volume_divergence"],
            "negative_momentum": short_signals["negative_momentum"]
        }

        return self.strategy_prompt_template.format(**prompt_vars)

    def _analyze_short_signals(self, technical_analysis: Dict[str, Any]) -> Dict[str, str]:
        """分析做空特定信号"""
        indicators = technical_analysis.get("indicators", {})
        market_analysis = technical_analysis.get("market_analysis", {})

        signals = {
            "overbought_signals": "无",
            "resistance_breakdown": "无",
            "volume_divergence": "无",
            "negative_momentum": "无"
        }

        # RSI超买分析
        rsi = indicators.get('rsi', 0)
        if rsi > 70:
            signals["overbought_signals"] = f"RSI超买 ({rsi:.1f})"
        elif rsi > 65:
            signals["overbought_signals"] = f"RSI接近超买 ({rsi:.1f})"

        # 阻力位分析
        current_price = technical_analysis.get("current_price", 0)
        support_resistance = technical_analysis.get("support_resistance", {})
        resistance_levels = support_resistance.get("resistance_levels", [])

        if resistance_levels:
            nearest_resistance = min(resistance_levels, key=lambda x: abs(x - current_price))
            if current_price < nearest_resistance and nearest_resistance - current_price < current_price * 0.05:
                signals["resistance_breakdown"] = f"接近阻力位 ${nearest_resistance:.2f}"

        # 成交量背离分析
        volume_ratio = indicators.get('volume_ratio', 1.0)
        trend_direction = market_analysis.get('trend_direction', 'neutral')

        if trend_direction == 'down' and volume_ratio < 1.0:
            signals["volume_divergence"] = "下跌时成交量萎缩"
        elif trend_direction == 'down' and volume_ratio > 2.0:
            signals["volume_divergence"] = "放量下跌"

        # 负面动能分析
        macd_histogram = indicators.get('macd_histogram', 0)
        if macd_histogram < 0:
            signals["negative_momentum"] = f"MACD柱状图为负 ({macd_histogram:.4f})"

        # 移动平均线分析
        sma_20 = indicators.get('sma_20', 0)
        sma_50 = indicators.get('sma_50', 0)

        if current_price > 0 and sma_20 > 0 and sma_50 > 0:
            if current_price < sma_20 < sma_50:
                if signals["negative_momentum"] != "无":
                    signals["negative_momentum"] += "；价格位于均线下方"
                else:
                    signals["negative_momentum"] = "价格位于均线下方"

        return signals

    def _format_technical_signals(self, signals: List[Dict[str, Any]]) -> str:
        """格式化技术信号"""
        if not signals:
            return "无明确信号"

        signal_texts = []
        for signal in signals[:5]:  # 最多显示5个信号
            signal_type = signal.get("type", "neutral")
            indicator = signal.get("indicator", "Unknown")
            reason = signal.get("reason", "")
            strength = signal.get("strength", "medium")

            signal_texts.append(f"- {indicator}: {signal_type.upper()} ({strength}) - {reason}")

        return "\n".join(signal_texts)

    async def _call_llm(self, prompt: str) -> str:
        """调用LLM服务"""
        try:
            request = LLMRequest(
                prompt=prompt,
                model=self.default_model,
                provider=self.default_provider,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )

            response = await self.llm_service.generate_completion_with_response(request)
            return response.content

        except Exception as e:
            self.logger.error(f"LLM调用失败: {e}")
            raise LLMServiceError(
                message=f"LLM做空策略分析调用失败: {str(e)}",
                error_code="LLM_SHORT_STRATEGY_ANALYSIS_FAILED",
                cause=e
            )

    def _parse_llm_response(self, response: str) -> Dict[str, Any]:
        """解析LLM响应"""
        try:
            # 尝试提取JSON内容
            if "```json" in response:
                start = response.find("```json") + 7
                end = response.find("```", start)
                json_str = response[start:end].strip()
            elif "{" in response and "}" in response:
                start = response.find("{")
                end = response.rfind("}") + 1
                json_str = response[start:end]
            else:
                # 尝试解析整个响应
                json_str = response.strip()

            data = json.loads(json_str)

            # 验证必需字段
            required_fields = [
                "recommendation", "confidence_score", "entry_price",
                "stop_loss_price", "take_profit_price", "position_size_percent"
            ]

            for field in required_fields:
                if field not in data:
                    self.logger.warning(f"LLM响应缺少字段: {field}")
                    # 设置默认值
                    if field == "recommendation":
                        data[field] = "hold"
                    elif field == "confidence_score":
                        data[field] = 0.5
                    elif field in ["entry_price", "stop_loss_price", "take_profit_price"]:
                        data[field] = 0.0
                    elif field == "position_size_percent":
                        data[field] = 8.0  # 做空默认仓位更小

            return data

        except json.JSONDecodeError as e:
            self.logger.error(f"LLM响应JSON解析失败: {e}")
            self.logger.error(f"原始响应: {response[:500]}...")
            raise ValidationError("LLM响应格式无效，无法解析做空策略数据")

    def _validate_strategy_result(
        self,
        strategy_data: Dict[str, Any],
        technical_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """验证做空策略结果"""
        current_price = technical_analysis.get("current_price", 0)

        # 验证推荐操作
        valid_recommendations = ["strong_sell", "sell", "hold", "buy", "strong_buy"]
        if strategy_data["recommendation"] not in valid_recommendations:
            strategy_data["recommendation"] = "hold"

        # 验证置信度
        if not 0 <= strategy_data["confidence_score"] <= 1:
            strategy_data["confidence_score"] = max(0, min(1, strategy_data["confidence_score"]))

        # 验证价格
        if strategy_data["entry_price"] <= 0:
            strategy_data["entry_price"] = current_price

        # 验证止损价格（做空止损在入场价之上）
        if strategy_data["stop_loss_price"] <= 0:
            if current_price > 0:
                strategy_data["stop_loss_price"] = current_price * 1.05  # 做空默认5%止损
            else:
                strategy_data["stop_loss_price"] = strategy_data["entry_price"] * 1.05

        # 验证止盈价格（做空止盈在入场价之下）
        if strategy_data["take_profit_price"] <= 0:
            if current_price > 0:
                strategy_data["take_profit_price"] = current_price * 0.9  # 做空默认10%止盈
            else:
                strategy_data["take_profit_price"] = strategy_data["entry_price"] * 0.9

        # 验证仓位大小（做空更保守）
        if not self.validation_rules["position_size_min"] <= strategy_data["position_size_percent"] <= self.validation_rules["position_size_max"]:
            strategy_data["position_size_percent"] = max(
                self.validation_rules["position_size_min"],
                min(self.validation_rules["position_size_max"], strategy_data["position_size_percent"])
            )

        # 确保做空价格逻辑正确
        if strategy_data["recommendation"] in ["sell", "strong_sell"]:
            # 做空策略：入场 < 止损，止盈 < 入场
            if strategy_data["stop_loss_price"] <= strategy_data["entry_price"]:
                strategy_data["stop_loss_price"] = strategy_data["entry_price"] * 1.05
            if strategy_data["take_profit_price"] >= strategy_data["entry_price"]:
                strategy_data["take_profit_price"] = strategy_data["entry_price"] * 0.9

        # 添加默认缺失字段
        if "reasoning" not in strategy_data:
            strategy_data["reasoning"] = "基于技术分析的综合判断"
        if "risk_factors" not in strategy_data:
            strategy_data["risk_factors"] = ["空头回补风险", "市场反弹风险"]
        if "market_conditions" not in strategy_data:
            strategy_data["market_conditions"] = {}
        if "execution_strategy" not in strategy_data:
            strategy_data["execution_strategy"] = {}
        if "short_specific_factors" not in strategy_data:
            strategy_data["short_specific_factors"] = {}

        return strategy_data

    def _calculate_risk_reward_ratio(self, analysis: ShortStrategyAnalysis) -> float:
        """计算做空风险回报比"""
        if analysis.recommendation in ["sell", "strong_sell"]:
            profit_potential = analysis.entry_price - analysis.take_profit_price
            risk_amount = analysis.stop_loss_price - analysis.entry_price
        else:
            return 1.0

        return profit_potential / risk_amount if risk_amount > 0 else 1.0

    async def evaluate_short_strategy_performance(
        self,
        symbol: str,
        strategy: ShortStrategyAnalysis,
        klines: List[KlineData],
        days_back: int = 30
    ) -> Dict[str, Any]:
        """评估做空策略历史表现"""
        try:
            # 简化的做空回测逻辑
            current_price = klines[-1].close if klines else strategy.entry_price
            entry_price = strategy.entry_price
            stop_loss = strategy.stop_loss_price
            take_profit = strategy.take_profit_price

            # 计算做空收益
            unrealized_pnl = (entry_price - current_price) / entry_price
            unrealized_pnl_percent = unrealized_pnl * 100

            # 检查是否触发止损或止盈
            status = "open"
            exit_price = None
            exit_reason = None

            # 简化的止损止盈检查（实际应该基于完整的价格历史）
            for kline in klines[-days_back:]:  # 检查最近几天
                if kline.high >= stop_loss:
                    status = "stopped_out"
                    exit_price = stop_loss
                    exit_reason = "stop_loss_triggered"
                    break
                elif kline.low <= take_profit:
                    status = "profit_taken"
                    exit_price = take_profit
                    exit_reason = "take_profit_triggered"
                    break

            # 计算最终收益
            if status != "open" and exit_price:
                final_pnl = (entry_price - exit_price) / entry_price
                final_pnl_percent = final_pnl * 100
            else:
                final_pnl = unrealized_pnl
                final_pnl_percent = unrealized_pnl_percent

            return {
                "symbol": symbol,
                "strategy_id": f"{symbol}_{strategy.analysis_timestamp.strftime('%Y%m%d_%H%M%S')}",
                "status": status,
                "entry_price": entry_price,
                "exit_price": exit_price,
                "current_price": current_price,
                "unrealized_pnl_percent": unrealized_pnl_percent,
                "final_pnl_percent": final_pnl_percent,
                "exit_reason": exit_reason,
                "evaluation_date": datetime.utcnow(),
                "performance_days": days_back,
                "strategy_type": "short"
            }

        except Exception as e:
            self.logger.error(f"做空策略表现评估失败: {e}")
            return {
                "symbol": symbol,
                "status": "evaluation_failed",
                "error": str(e),
                "strategy_type": "short"
            }


# 便捷函数
async def analyze_short_strategy(
    symbol: str,
    technical_analysis: Dict[str, Any],
    market_data: Optional[Dict[str, Any]] = None
) -> ShortStrategyAnalysis:
    """分析做空策略的便捷函数"""
    analyzer = LLMShortStrategyAnalyzer()
    return await analyzer.analyze_short_strategy(symbol, technical_analysis, market_data)


async def evaluate_short_strategy_performance(
    symbol: str,
    strategy: ShortStrategyAnalysis,
    klines: List[KlineData]
) -> Dict[str, Any]:
    """评估做空策略表现的便捷函数"""
    analyzer = LLMShortStrategyAnalyzer()
    return await analyzer.evaluate_short_strategy_performance(symbol, strategy, klines)