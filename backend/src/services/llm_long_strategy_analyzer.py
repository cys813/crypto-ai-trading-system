"""
LLM做多策略分析服务

使用大语言模型分析技术分析结果，生成做多策略建议。
"""

import logging
import asyncio
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass
import json

from ..core.llm_integration import get_llm_service, LLMRequest, LLMProvider, LLMModel
from ..core.logging import BusinessLogger
from ..core.exceptions import LLMServiceError, ValidationError, BusinessLogicError
from ..models.technical_analysis import TechnicalAnalysis
from ..services.exchange_data_collector import KlineData

logger = logging.getLogger(__name__)
business_logger = BusinessLogger("llm_long_strategy_analyzer")


@dataclass
class LongStrategyAnalysis:
    """做多策略分析结果"""
    symbol: str
    timeframe: str
    recommendation: str  # 'strong_buy', 'buy', 'hold', 'sell', 'strong_sell'
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


class LLMLongStrategyAnalyzer:
    """LLM做多策略分析器"""

    def __init__(self):
        self.logger = logger
        self.business_logger = business_logger
        self.llm_service = get_llm_service()

        # 默认配置
        self.default_model = LLMModel.GPT_4O
        self.default_provider = LLMProvider.OPENAI
        self.temperature = 0.3  # 较低温度确保一致性
        self.max_tokens = 2000

        # 策略模板
        self.strategy_prompt_template = """
你是一位专业的加密货币交易分析师，专注于做多策略分析。请基于以下技术分析结果，提供详细的做多策略建议。

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

## 分析要求

请提供以下格式的详细分析：

### 1. 策略建议
- **推荐操作**: [strong_buy/buy/hold/sell/strong_sell]
- **置信度**: [0.0-1.0之间的数值]
- **主要理由**: [1-2句话的核心原因]

### 2. 入场策略
- **建议入场价**: [具体的入场价格]
- **入场时机**: [立即/等待回调/等待突破]
- **仓位大小**: [建议资金百分比，如10-25%]
- **时间框架**: [短期/中期/长期]

### 3. 风险管理
- **止损价格**: [具体止损价格]
- **止盈价格**: [具体止盈价格]
- **风险回报比**: [计算的风险回报比]
- **最大持仓时间**: [建议的最长持仓时间]

### 4. 市场分析
- **趋势分析**: [对当前趋势的详细分析]
- **关键因素**: [影响决策的关键技术因素]
- **潜在风险**: [2-3个主要风险点]
- **机会评估**: [做多机会的大小和质量]

### 5. 执行策略
- **分批建仓**: [是否建议分批建仓]
- **加仓条件**: [什么情况下可以加仓]
- **减仓条件**: [什么情况下应该减仓]
- **退出策略**: [完整的退出计划]

请以JSON格式回复：
{{
    "recommendation": "buy",
    "confidence_score": 0.75,
    "reasoning": "基于技术分析的综合判断...",
    "entry_price": 50200,
    "stop_loss_price": 48500,
    "take_profit_price": 52500,
    "position_size_percent": 15.0,
    "time_horizon": "medium_term",
    "risk_reward_ratio": 2.5,
    "max_hold_time_days": 14,
    "risk_factors": ["市场波动性", "宏观经济风险"],
    "market_conditions": {{
        "trend_analysis": "上升趋势保持良好...",
        "key_factors": ["RSI显示超卖反弹", "MACD金叉形成"],
        "opportunity_assessment": "中等偏高机会"
    }},
    "execution_strategy": {{
        "entry_timing": "immediate",
        "scaling_plan": "分批建仓",
        "add_position_conditions": ["价格回调至支撑位"],
        "reduce_position_conditions": ["达到止盈目标", "趋势反转"],
        "exit_strategy": "分批止盈"
    }}
}}
"""

        # 策略验证规则
        self.validation_rules = {
            "position_size_min": 5.0,  # 最小仓位百分比
            "position_size_max": 50.0,  # 最大仓位百分比
            "risk_reward_min": 1.5,  # 最小风险回报比
            "confidence_min": 0.6  # 最小置信度
        }

    async def analyze_long_strategy(
        self,
        symbol: str,
        technical_analysis: Dict[str, Any],
        market_data: Optional[Dict[str, Any]] = None,
        analysis_config: Optional[Dict[str, Any]] = None
    ) -> LongStrategyAnalysis:
        """分析做多策略"""
        try:
            self.logger.info(f"开始分析{symbol}的做多策略")

            # 验证输入数据
            self._validate_analysis_input(symbol, technical_analysis)

            # 构建LLM提示词
            prompt = self._build_strategy_prompt(symbol, technical_analysis, market_data)

            # 调用LLM
            llm_response = await self._call_llm(prompt)

            # 解析LLM响应
            strategy_data = self._parse_llm_response(llm_response)

            # 验证策略结果
            validated_strategy = self._validate_strategy_result(strategy_data, technical_analysis)

            # 创建分析结果对象
            analysis = LongStrategyAnalysis(
                symbol=symbol,
                timeframe=analysis_config.get("timeframe", "1h") if analysis_config else "1h",
                recommendation=validated_strategy["recommendation"],
                confidence_score=validated_strategy["confidence_score"],
                entry_price=validated_strategy["entry_price"],
                stop_loss_price=validated_strategy["stop_loss_price"],
                take_profit_price=validated_strategy["take_profit_price"],
                position_size_percent=validated_strategy["position_size_percent"],
                time_horizon=validated_strategy["time_horizon"],
                reasoning=validated_strategy["reasoning"],
                risk_factors=validated_strategy["risk_factors"],
                market_conditions=validated_strategy["market_conditions"],
                technical_signals=technical_analysis.get("technical_signals", []),
                price_targets={
                    "target_1": validated_strategy["take_profit_price"],
                    "stop_loss": validated_strategy["stop_loss_price"]
                },
                execution_strategy=validated_strategy["execution_strategy"],
                analysis_timestamp=datetime.utcnow()
            )

            # 记录业务日志
            self.business_logger.log_system_event(
                event_type="long_strategy_analyzed",
                severity="info",
                message=f"做多策略分析完成: {symbol}",
                details={
                    "symbol": symbol,
                    "recommendation": analysis.recommendation,
                    "confidence": analysis.confidence_score,
                    "entry_price": analysis.entry_price,
                    "risk_reward_ratio": self._calculate_risk_reward_ratio(analysis)
                }
            )

            return analysis

        except Exception as e:
            self.logger.error(f"做多策略分析失败: {e}")
            self.business_logger.log_system_event(
                event_type="long_strategy_analysis_failed",
                severity="error",
                message=f"做多策略分析失败: {str(e)}",
                details={"symbol": symbol, "error": str(e)}
            )
            raise

    async def analyze_multiple_strategies(
        self,
        analysis_requests: List[Dict[str, Any]]
    ) -> List[LongStrategyAnalysis]:
        """批量分析多个交易符号的做多策略"""
        try:
            self.logger.info(f"开始批量分析{len(analysis_requests)}个交易符号的做多策略")

            # 并发执行分析
            tasks = []
            for request in analysis_requests:
                task = self.analyze_long_strategy(
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
                event_type="batch_long_strategy_analysis_completed",
                severity="info",
                message=f"批量做多策略分析完成",
                details={
                    "total_requests": len(analysis_requests),
                    "successful_analyses": len(strategies),
                    "success_rate": len(strategies) / len(analysis_requests) if analysis_requests else 0
                }
            )

            return strategies

        except Exception as e:
            self.logger.error(f"批量做多策略分析失败: {e}")
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
        """构建策略分析提示词"""
        # 提取技术指标
        indicators = technical_analysis.get("indicators", {})
        market_analysis = technical_analysis.get("market_analysis", {})
        technical_signals = technical_analysis.get("technical_signals", [])

        # 支撑阻力位
        support_resistance = technical_analysis.get("support_resistance", {})
        support_levels = support_resistance.get("support_levels", [])
        resistance_levels = support_resistance.get("resistance_levels", [])

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
            "technical_signals": self._format_technical_signals(technical_signals)
        }

        return self.strategy_prompt_template.format(**prompt_vars)

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
                message=f"LLM策略分析调用失败: {str(e)}",
                error_code="LLM_STRATEGY_ANALYSIS_FAILED",
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
                        data[field] = 10.0

            return data

        except json.JSONDecodeError as e:
            self.logger.error(f"LLM响应JSON解析失败: {e}")
            self.logger.error(f"原始响应: {response[:500]}...")
            raise ValidationError("LLM响应格式无效，无法解析策略数据")

    def _validate_strategy_result(
        self,
        strategy_data: Dict[str, Any],
        technical_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """验证策略结果"""
        current_price = technical_analysis.get("current_price", 0)

        # 验证推荐操作
        valid_recommendations = ["strong_buy", "buy", "hold", "sell", "strong_sell"]
        if strategy_data["recommendation"] not in valid_recommendations:
            strategy_data["recommendation"] = "hold"

        # 验证置信度
        if not 0 <= strategy_data["confidence_score"] <= 1:
            strategy_data["confidence_score"] = max(0, min(1, strategy_data["confidence_score"]))

        # 验证价格
        if strategy_data["entry_price"] <= 0:
            strategy_data["entry_price"] = current_price

        # 验证止损价格
        if strategy_data["stop_loss_price"] <= 0:
            if current_price > 0:
                strategy_data["stop_loss_price"] = current_price * 0.95  # 默认5%止损
            else:
                strategy_data["stop_loss_price"] = strategy_data["entry_price"] * 0.95

        # 验证止盈价格
        if strategy_data["take_profit_price"] <= 0:
            if current_price > 0:
                strategy_data["take_profit_price"] = current_price * 1.1  # 默认10%止盈
            else:
                strategy_data["take_profit_price"] = strategy_data["entry_price"] * 1.1

        # 验证仓位大小
        if not self.validation_rules["position_size_min"] <= strategy_data["position_size_percent"] <= self.validation_rules["position_size_max"]:
            strategy_data["position_size_percent"] = max(
                self.validation_rules["position_size_min"],
                min(self.validation_rules["position_size_max"], strategy_data["position_size_percent"])
            )

        # 确保价格逻辑正确
        if strategy_data["recommendation"] in ["buy", "strong_buy"]:
            # 做多策略：止损 < 入场 < 止盈
            if strategy_data["stop_loss_price"] >= strategy_data["entry_price"]:
                strategy_data["stop_loss_price"] = strategy_data["entry_price"] * 0.95
            if strategy_data["take_profit_price"] <= strategy_data["entry_price"]:
                strategy_data["take_profit_price"] = strategy_data["entry_price"] * 1.1

        # 添加默认缺失字段
        if "reasoning" not in strategy_data:
            strategy_data["reasoning"] = "基于技术分析的综合判断"
        if "risk_factors" not in strategy_data:
            strategy_data["risk_factors"] = ["市场波动性"]
        if "market_conditions" not in strategy_data:
            strategy_data["market_conditions"] = {}
        if "execution_strategy" not in strategy_data:
            strategy_data["execution_strategy"] = {}

        return strategy_data

    def _calculate_risk_reward_ratio(self, analysis: LongStrategyAnalysis) -> float:
        """计算风险回报比"""
        if analysis.recommendation in ["buy", "strong_buy"]:
            profit_potential = analysis.take_profit_price - analysis.entry_price
            risk_amount = analysis.entry_price - analysis.stop_loss_price
        else:
            return 1.0

        return profit_potential / risk_amount if risk_amount > 0 else 1.0

    async def evaluate_strategy_performance(
        self,
        symbol: str,
        strategy: LongStrategyAnalysis,
        klines: List[KlineData],
        days_back: int = 30
    ) -> Dict[str, Any]:
        """评估策略历史表现"""
        try:
            # 简化的回测逻辑
            # 在实际应用中，这里可以实现更复杂的回测算法

            current_price = klines[-1].close if klines else strategy.entry_price
            entry_price = strategy.entry_price
            stop_loss = strategy.stop_loss_price
            take_profit = strategy.take_profit_price

            # 计算收益
            unrealized_pnl = (current_price - entry_price) / entry_price
            unrealized_pnl_percent = unrealized_pnl * 100

            # 检查是否触发止损或止盈
            status = "open"
            exit_price = None
            exit_reason = None

            # 简化的止损止盈检查（实际应该基于完整的价格历史）
            for kline in klines[-days_back:]:  # 检查最近几天
                if kline.low <= stop_loss:
                    status = "stopped_out"
                    exit_price = stop_loss
                    exit_reason = "stop_loss_triggered"
                    break
                elif kline.high >= take_profit:
                    status = "profit_taken"
                    exit_price = take_profit
                    exit_reason = "take_profit_triggered"
                    break

            # 计算最终收益
            if status != "open" and exit_price:
                final_pnl = (exit_price - entry_price) / entry_price
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
                "performance_days": days_back
            }

        except Exception as e:
            self.logger.error(f"策略表现评估失败: {e}")
            return {
                "symbol": symbol,
                "status": "evaluation_failed",
                "error": str(e)
            }


# 便捷函数
async def analyze_long_strategy(
    symbol: str,
    technical_analysis: Dict[str, Any],
    market_data: Optional[Dict[str, Any]] = None
) -> LongStrategyAnalysis:
    """分析做多策略的便捷函数"""
    analyzer = LLMLongStrategyAnalyzer()
    return await analyzer.analyze_long_strategy(symbol, technical_analysis, market_data)


async def evaluate_strategy_performance(
    symbol: str,
    strategy: LongStrategyAnalysis,
    klines: List[KlineData]
) -> Dict[str, Any]:
    """评估策略表现的便捷函数"""
    analyzer = LLMLongStrategyAnalyzer()
    return await analyzer.evaluate_strategy_performance(symbol, strategy, klines)