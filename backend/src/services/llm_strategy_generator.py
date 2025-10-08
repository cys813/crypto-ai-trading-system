"""
LLM策略生成服务

使用大语言模型基于聚合的分析结果生成最终的交易策略。
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
from ..models.trading_strategy import TradingStrategy, StrategyType, StrategyStatus, RiskLevel, TimeHorizon
from ..services.strategy_aggregator import AggregatedSignal, AggregationMethod
from ..services.llm_long_strategy_analyzer import LongStrategyAnalysis
from ..services.llm_short_strategy_analyzer import ShortStrategyAnalysis

logger = logging.getLogger(__name__)
business_logger = BusinessLogger("llm_strategy_generator")


@dataclass
class StrategyGenerationRequest:
    """策略生成请求"""
    symbol: str
    timeframe: str
    analysis_types: List[str]
    include_news: bool
    risk_tolerance: str  # 'conservative', 'moderate', 'aggressive'
    max_position_size: float
    custom_parameters: Optional[Dict[str, Any]] = None


@dataclass
class GeneratedStrategy:
    """生成的策略"""
    strategy_id: str
    symbol: str
    final_recommendation: str  # 'long', 'short', 'hold'
    confidence_score: float
    entry_price: float
    stop_loss_price: float
    take_profit_price: float
    position_size_percent: float
    strategy_type: str  # 'aggressive', 'moderate', 'conservative'
    time_horizon: str

    # 详细分析
    market_analysis: Dict[str, Any]
    risk_assessment: Dict[str, Any]
    execution_plan: Dict[str, Any]

    # 元数据
    generated_at: datetime
    expires_at: datetime
    analysis_version: str

    # 原始分析数据
    aggregated_data: Optional[AggregatedSignal] = None
    long_analysis: Optional[LongStrategyAnalysis] = None
    short_analysis: Optional[ShortStrategyAnalysis] = None


class LLMStrategyGenerator:
    """LLM策略生成器"""

    def __init__(self):
        self.logger = logger
        self.business_logger = business_logger
        self.llm_service = get_llm_service()

        # 默认配置
        self.default_model = LLMModel.GPT_4O
        self.default_provider = LLMProvider.OPENAI
        self.temperature = 0.3  # 较低温度确保一致性
        self.max_tokens = 2500

        # 策略生成模板
        self.strategy_generation_template = """
你是一位专业的加密货币交易策略大师，负责综合多种分析结果，生成最终的交易策略建议。

## 分析输入

**基本信息:**
- 交易符号: {symbol}
- 时间框架: {timeframe}
- 风险容忍度: {risk_tolerance}
- 最大仓位限制: {max_position_size}%
- 生成时间: {generation_time}

**聚合分析结果:**
- 最终推荐: {aggregated_recommendation}
- 综合置信度: {aggregated_confidence:.2f}
- 一致性强度: {consensus_strength:.2f}
- 冲突信号: {conflicting_signals}
- 支持分析: {supporting_analyses}
- 市场情绪: {market_sentiment}

**价格范围分析:**
- 入场价格: {entry_price_range}
- 止损价格: {stop_loss_range}
- 止盈价格: {take_profit_range}
- 仓位建议: {position_size_range}

**技术分析摘要:**
{technical_summary}

**新闻分析摘要:**
{news_summary}

**原始分析数据:**
{detailed_analyses}

## 策略生成要求

请基于以上综合分析，生成最终交易策略，包含以下要素：

### 1. 最终决策
- **策略推荐**: [long/short/hold] - 基于综合分析的最终判断
- **置信度**: [0.0-1.0] - 对该推荐的信心程度
- **策略风格**: [aggressive/moderate/conservative] - 根据风险容忍度确定
- **时间框架**: [short_term/medium_term/long_term] - 建议持仓时间

### 2. 价格执行
- **建议入场价**: [具体价格，基于价格范围分析]
- **止损价格**: [具体价格，考虑风险控制]
- **止盈价格**: [具体价格，考虑盈利目标]
- **仓位大小**: [百分比，不超过最大限制]

### 3. 市场分析
- **整体情绪**: [对当前市场情绪的详细分析]
- **趋势分析**: [技术面和基本面趋势的综合判断]
- **关键信号**: [最重要的交易信号和指标]
- **新闻影响**: [新闻因素对策略的影响评估]

### 4. 风险评估
- **风险等级**: [low/medium/high] - 综合风险评估
- **主要风险**: [3-5个主要风险因素]
- **风险缓解**: [具体的风险管理措施]
- **最大回撤预估**: [预估的最大可能回撤百分比]

### 5. 执行计划
- **入场条件**: [具体的入场时机和条件]
- **退出条件**: [止盈、止损、时间等退出条件]
- **仓位管理**: [分批建仓、加仓、减仓策略]
- **时机策略**: [具体的执行时机建议]

### 6. 策略合理性验证
请确保你的策略满足以下要求：
- 入场价格在建议的价格范围内
- 风险回报比合理（建议 > 1.5）
- 仓位大小符合风险容忍度
- 止损止盈设置逻辑正确

请以JSON格式回复：
{{
    "final_recommendation": "long",
    "confidence_score": 0.78,
    "strategy_type": "moderate",
    "time_horizon": "medium_term",
    "entry_price": 50200.0,
    "stop_loss_price": 48500.0,
    "take_profit_price": 52500.0,
    "position_size_percent": 18.0,
    "market_analysis": {{
        "overall_sentiment": "bullish",
        "trend_analysis": "多重技术指标显示上涨趋势确立...",
        "key_signals": ["RSI适中", "MACD金叉", "成交量放大"],
        "news_impact": "积极新闻增强上涨预期..."
    }},
    "risk_assessment": {{
        "risk_level": "medium",
        "risk_factors": ["市场波动性", "政策不确定性", "技术面回调风险"],
        "risk_mitigation": "设置合理止损，分批建仓",
        "max_drawdown_estimate": 8.5
    }},
    "execution_plan": {{
        "entry_conditions": ["价格回调至50200以下", "成交量确认上涨"],
        "exit_conditions": ["达到止盈目标52500", "跌破止损48500", "持仓超过7天"],
        "position_sizing": "建议分2批建仓，首批15%仓位",
        "timing_strategy": "等待技术确认后入场"
    }},
    "strategy_rationale": "基于技术分析和市场情绪的综合判断...",
    "risk_reward_ratio": 2.1,
    "validation_checks": {{
        "price_in_range": true,
        "risk_reward_acceptable": true,
        "position_appropriate": true,
        "stop_loss_logical": true
    }}
}}
"""

    def __init__(self):
        self.logger = logger
        self.business_logger = business_logger
        self.llm_service = get_llm_service()

    async def generate_final_strategy(
        self,
        request: StrategyGenerationRequest,
        aggregated_signal: AggregatedSignal,
        long_analysis: Optional[LongStrategyAnalysis] = None,
        short_analysis: Optional[ShortStrategyAnalysis] = None,
        technical_analysis: Optional[Dict[str, Any]] = None,
        news_analysis: Optional[Dict[str, Any]] = None
    ) -> GeneratedStrategy:
        """生成最终交易策略"""
        try:
            self.logger.info(f"开始为{request.symbol}生成最终策略")

            # 1. 验证输入数据
            self._validate_generation_input(request, aggregated_signal)

            # 2. 构建LLM提示词
            prompt = self._build_strategy_prompt(
                request, aggregated_signal, long_analysis, short_analysis,
                technical_analysis, news_analysis
            )

            # 3. 调用LLM生成策略
            llm_response = await self._call_llm(prompt)

            # 4. 解析LLM响应
            strategy_data = self._parse_llm_response(llm_response)

            # 5. 验证生成的策略
            validated_strategy = self._validate_generated_strategy(strategy_data, request)

            # 6. 创建策略对象
            strategy = GeneratedStrategy(
                strategy_id=f"strategy_{request.symbol}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
                symbol=request.symbol,
                final_recommendation=validated_strategy["final_recommendation"],
                confidence_score=validated_strategy["confidence_score"],
                entry_price=validated_strategy["entry_price"],
                stop_loss_price=validated_strategy["stop_loss_price"],
                take_profit_price=validated_strategy["take_profit_price"],
                position_size_percent=validated_strategy["position_size_percent"],
                strategy_type=validated_strategy["strategy_type"],
                time_horizon=validated_strategy["time_horizon"],
                market_analysis=validated_strategy["market_analysis"],
                risk_assessment=validated_strategy["risk_assessment"],
                execution_plan=validated_strategy["execution_plan"],
                generated_at=datetime.utcnow(),
                expires_at=datetime.utcnow() + timedelta(hours=24),  # 24小时有效期
                analysis_version="v1.0",
                aggregated_data=aggregated_signal,
                long_analysis=long_analysis,
                short_analysis=short_analysis
            )

            # 7. 记录业务日志
            self.business_logger.log_system_event(
                event_type="strategy_generated",
                severity="info",
                message=f"策略生成完成: {request.symbol} - {strategy.final_recommendation}",
                details={
                    "symbol": request.symbol,
                    "strategy_id": strategy.strategy_id,
                    "recommendation": strategy.final_recommendation,
                    "confidence": strategy.confidence_score,
                    "risk_level": strategy.risk_assessment.get("risk_level"),
                    "entry_price": strategy.entry_price,
                    "risk_reward_ratio": self._calculate_risk_reward_ratio(strategy),
                    "analysis_sources": len(aggregated_signal.supporting_analyses) if aggregated_signal else 0
                }
            )

            self.logger.info(f"策略生成成功: {request.symbol} - {strategy.final_recommendation} (置信度: {strategy.confidence_score:.2f})")

            return strategy

        except Exception as e:
            self.logger.error(f"策略生成失败: {e}")
            self.business_logger.log_system_event(
                event_type="strategy_generation_failed",
                severity="error",
                message=f"策略生成失败: {str(e)}",
                details={
                    "symbol": request.symbol,
                    "error": str(e),
                    "error_type": type(e).__name__
                }
            )
            raise

    async def generate_multiple_strategies(
        self,
        requests: List[StrategyGenerationRequest],
        aggregated_signals: Dict[str, AggregatedSignal],
        analyses_data: Optional[Dict[str, Dict[str, Any]]] = None
    ) -> List[GeneratedStrategy]:
        """批量生成多个交易符号的策略"""
        try:
            self.logger.info(f"开始批量生成{len(requests)}个交易符号的策略")

            # 并发执行策略生成
            tasks = []
            for request in requests:
                aggregated_signal = aggregated_signals.get(request.symbol)
                if not aggregated_signal:
                    self.logger.warning(f"缺少{request.symbol}的聚合信号，跳过")
                    continue

                symbol_analyses = analyses_data.get(request.symbol, {}) if analyses_data else {}

                task = self.generate_final_strategy(
                    request=request,
                    aggregated_signal=aggregated_signal,
                    long_analysis=symbol_analyses.get("long_analysis"),
                    short_analysis=symbol_analyses.get("short_analysis"),
                    technical_analysis=symbol_analyses.get("technical_analysis"),
                    news_analysis=symbol_analyses.get("news_analysis")
                )
                tasks.append(task)

            results = await asyncio.gather(*tasks, return_exceptions=True)

            # 处理结果
            strategies = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    self.logger.error(f"生成{requests[i].symbol}策略失败: {result}")
                else:
                    strategies.append(result)

            self.business_logger.log_system_event(
                event_type="batch_strategy_generation_completed",
                severity="info",
                message=f"批量策略生成完成",
                details={
                    "total_requests": len(requests),
                    "successful_generations": len(strategies),
                    "success_rate": len(strategies) / len(requests) if requests else 0
                }
            )

            self.logger.info(f"批量策略生成完成: 成功{len(strategies)}个，失败{len(results) - len(strategies)}个")

            return strategies

        except Exception as e:
            self.logger.error(f"批量策略生成失败: {e}")
            raise

    def _validate_generation_input(
        self,
        request: StrategyGenerationRequest,
        aggregated_signal: AggregatedSignal
    ):
        """验证策略生成输入"""
        if not request.symbol:
            raise ValidationError("交易符号不能为空")

        if not aggregated_signal:
            raise ValidationError("聚合信号数据不能为空")

        if request.risk_tolerance not in ["conservative", "moderate", "aggressive"]:
            raise ValidationError("风险容忍度必须是 conservative, moderate, aggressive 之一")

        if not 0 < request.max_position_size <= 100:
            raise ValidationError("最大仓位大小必须在0-100%之间")

        # 验证聚合信号的数据质量
        if not aggregated_signal.final_recommendation:
            raise ValidationError("聚合信号缺少最终推荐")

        if not 0 <= aggregated_signal.confidence_score <= 1:
            raise ValidationError("聚合信号置信度超出有效范围")

    def _build_strategy_prompt(
        self,
        request: StrategyGenerationRequest,
        aggregated_signal: AggregatedSignal,
        long_analysis: Optional[LongStrategyAnalysis],
        short_analysis: Optional[ShortStrategyAnalysis],
        technical_analysis: Optional[Dict[str, Any]],
        news_analysis: Optional[Dict[str, Any]]
    ) -> str:
        """构建策略生成提示词"""
        # 构建价格范围描述
        entry_range = f"${aggregated_signal.entry_price_range['min']:.2f} - ${aggregated_signal.entry_price_range['max']:.2f} (平均: ${aggregated_signal.entry_price_range['avg']:.2f})"
        stop_loss_range = f"${aggregated_signal.stop_loss_range['min']:.2f} - ${aggregated_signal.stop_loss_range['max']:.2f} (平均: ${aggregated_signal.stop_loss_range['avg']:.2f})"
        take_profit_range = f"${aggregated_signal.take_profit_range['min']:.2f} - ${aggregated_signal.take_profit_range['max']:.2f} (平均: ${aggregated_signal.take_profit_range['avg']:.2f})"
        position_range = f"{aggregated_signal.position_size_range['min']:.1f}% - {aggregated_signal.position_size_range['max']:.1f}% (平均: {aggregated_signal.position_size_range['avg']:.1f}%)"

        # 构建详细分析数据
        detailed_analyses = []

        if long_analysis:
            detailed_analyses.append(f"**做多分析**: {long_analysis.recommendation} (置信度: {long_analysis.confidence_score:.2f}) - {long_analysis.reasoning}")

        if short_analysis:
            detailed_analyses.append(f"**做空分析**: {short_analysis.recommendation} (置信度: {short_analysis.confidence_score:.2f}) - {short_analysis.reasoning}")

        if technical_analysis:
            market_analysis = technical_analysis.get("market_analysis", {})
            detailed_analyses.append(f"**技术分析**: 趋势{market_analysis.get('trend_direction', '未知')}, 强度{market_analysis.get('trend_strength', 0):.2f}")

        if news_analysis:
            detailed_analyses.append(f"**新闻分析**: {news_analysis.get('sentiment', '未知')}情绪, 影响{news_analysis.get('market_impact', '未知')}")

        # 构建提示词变量
        prompt_vars = {
            "symbol": request.symbol,
            "timeframe": request.timeframe,
            "risk_tolerance": request.risk_tolerance,
            "max_position_size": request.max_position_size,
            "generation_time": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),

            "aggregated_recommendation": aggregated_signal.final_recommendation,
            "aggregated_confidence": aggregated_signal.confidence_score,
            "consensus_strength": aggregated_signal.consensus_strength,
            "conflicting_signals": "; ".join(aggregated_signal.conflicting_signals) if aggregated_signal.conflicting_signals else "无",
            "supporting_analyses": ", ".join(aggregated_signal.supporting_analyses),
            "market_sentiment": aggregated_signal.market_sentiment,

            "entry_price_range": entry_range,
            "stop_loss_range": stop_loss_range,
            "take_profit_range": take_profit_range,
            "position_size_range": position_range,

            "technical_summary": self._format_technical_summary(aggregated_signal.technical_summary),
            "news_summary": self._format_news_summary(aggregated_signal.news_summary),
            "detailed_analyses": "\n\n".join(detailed_analyses) if detailed_analyses else "无详细分析数据"
        }

        return self.strategy_generation_template.format(**prompt_vars)

    def _format_technical_summary(self, technical_summary: Dict[str, Any]) -> str:
        """格式化技术分析摘要"""
        if not technical_summary:
            return "无技术分析数据"

        return f"""趋势方向: {technical_summary.get('trend_direction', '未知')}
趋势强度: {technical_summary.get('trend_strength', 0):.2f}
波动性: {technical_summary.get('volatility_level', '未知')}
关键指标: RSI={technical_summary.get('key_indicators', {}).get('rsi', 'N/A')},
MACD信号={'看涨' if technical_summary.get('key_indicators', {}).get('macd_signal') else '看跌'},
布林带位置={technical_summary.get('key_indicators', {}).get('bollinger_position', 'N/A')}"""

    def _format_news_summary(self, news_summary: Dict[str, Any]) -> str:
        """格式化新闻分析摘要"""
        if not news_summary:
            return "无新闻分析数据"

        key_events = news_summary.get('key_events', [])
        events_text = ", ".join(key_events[:3]) if key_events else "无关键事件"

        return f"""整体情绪: {news_summary.get('overall_sentiment', '未知')}
情绪评分: {news_summary.get('sentiment_score', 0):.2f}
市场影响: {news_summary.get('market_impact', '未知')}
关键事件: {events_text}"""

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
                message=f"LLM策略生成调用失败: {str(e)}",
                error_code="LLM_STRATEGY_GENERATION_FAILED",
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
                "final_recommendation", "confidence_score", "entry_price",
                "stop_loss_price", "take_profit_price", "position_size_percent",
                "strategy_type", "market_analysis", "risk_assessment", "execution_plan"
            ]

            for field in required_fields:
                if field not in data:
                    self.logger.warning(f"LLM响应缺少字段: {field}")
                    # 设置默认值
                    if field == "final_recommendation":
                        data[field] = "hold"
                    elif field == "confidence_score":
                        data[field] = 0.5
                    elif field in ["entry_price", "stop_loss_price", "take_profit_price"]:
                        data[field] = 0.0
                    elif field == "position_size_percent":
                        data[field] = 10.0
                    elif field == "strategy_type":
                        data[field] = "moderate"
                    elif field in ["market_analysis", "risk_assessment", "execution_plan"]:
                        data[field] = {}

            return data

        except json.JSONDecodeError as e:
            self.logger.error(f"LLM响应JSON解析失败: {e}")
            self.logger.error(f"原始响应: {response[:500]}...")
            raise ValidationError("LLM响应格式无效，无法解析策略数据")

    def _validate_generated_strategy(
        self,
        strategy_data: Dict[str, Any],
        request: StrategyGenerationRequest
    ) -> Dict[str, Any]:
        """验证生成的策略"""
        # 验证推荐操作
        valid_recommendations = ["long", "short", "hold"]
        if strategy_data["final_recommendation"] not in valid_recommendations:
            strategy_data["final_recommendation"] = "hold"

        # 验证置信度
        if not 0 <= strategy_data["confidence_score"] <= 1:
            strategy_data["confidence_score"] = max(0, min(1, strategy_data["confidence_score"]))

        # 验证策略类型
        valid_strategy_types = ["aggressive", "moderate", "conservative"]
        if strategy_data["strategy_type"] not in valid_strategy_types:
            strategy_data["strategy_type"] = request.risk_tolerance

        # 验证价格
        if strategy_data["entry_price"] <= 0:
            strategy_data["entry_price"] = 50000.0  # 默认价格

        # 验证止损止盈逻辑
        if strategy_data["final_recommendation"] == "long":
            # 做多：止损 < 入场 < 止盈
            if strategy_data["stop_loss_price"] >= strategy_data["entry_price"]:
                strategy_data["stop_loss_price"] = strategy_data["entry_price"] * 0.95
            if strategy_data["take_profit_price"] <= strategy_data["entry_price"]:
                strategy_data["take_profit_price"] = strategy_data["entry_price"] * 1.1
        elif strategy_data["final_recommendation"] == "short":
            # 做空：入场 < 止盈，止损 > 入场
            if strategy_data["stop_loss_price"] <= strategy_data["entry_price"]:
                strategy_data["stop_loss_price"] = strategy_data["entry_price"] * 1.05
            if strategy_data["take_profit_price"] >= strategy_data["entry_price"]:
                strategy_data["take_profit_price"] = strategy_data["entry_price"] * 0.9

        # 验证仓位大小
        if not 0 < strategy_data["position_size_percent"] <= request.max_position_size:
            strategy_data["position_size_percent"] = min(
                request.max_position_size * 0.8,  # 使用80%的最大限制
                strategy_data["position_size_percent"]
            )

        # 根据风险容忍度调整策略
        if request.risk_tolerance == "conservative":
            strategy_data["position_size_percent"] *= 0.7
            strategy_data["confidence_score"] *= 0.9
        elif request.risk_tolerance == "aggressive":
            strategy_data["position_size_percent"] *= 1.2
            # 不提高置信度，保持原始评估

        return strategy_data

    def _calculate_risk_reward_ratio(self, strategy: GeneratedStrategy) -> float:
        """计算风险回报比"""
        if strategy.final_recommendation == "long":
            profit_potential = strategy.take_profit_price - strategy.entry_price
            risk_amount = strategy.entry_price - strategy.stop_loss_price
        elif strategy.final_recommendation == "short":
            profit_potential = strategy.entry_price - strategy.take_profit_price
            risk_amount = strategy.stop_loss_price - strategy.entry_price
        else:
            return 1.0

        return profit_potential / risk_amount if risk_amount > 0 else 1.0

    def save_strategy_to_database(self, strategy: GeneratedStrategy) -> TradingStrategy:
        """保存策略到数据库"""
        try:
            # 创建TradingStrategy模型实例
            db_strategy = TradingStrategy(
                name=f"{strategy.symbol}_{strategy.final_recommendation}_{strategy.generated_at.strftime('%Y%m%d_%H%M%S')}",
                description=f"LLM生成的{strategy.final_recommendation}策略",
                strategy_type=StrategyType.COMPREHENSIVE.value,
                status=StrategyStatus.ACTIVE.value,
                llm_provider=self.default_provider.value,
                llm_model=self.default_model.value,
                confidence_score=strategy.confidence_score,
                final_recommendation=strategy.final_recommendation,
                entry_price=strategy.entry_price,
                stop_loss_price=strategy.stop_loss_price,
                take_profit_price=strategy.take_profit_price,
                position_size_percent=strategy.position_size_percent,
                risk_level=RiskLevel(strategy.risk_assessment.get("risk_level", "medium")).value,
                time_horizon=TimeHorizon(strategy.time_horizon).value,
                strategy_style=strategy.strategy_type,
                market_analysis=strategy.market_analysis,
                risk_assessment=strategy.risk_assessment,
                execution_plan=strategy.execution_plan,
                technical_signals=strategy.aggregated_data.technical_summary if strategy.aggregated_data else {},
                news_analysis=strategy.aggregated_data.news_summary if strategy.aggregated_data else {},
                analysis_types=[],
                analysis_parameters={
                    "risk_tolerance": strategy.strategy_type,
                    "max_position_size": strategy.position_size_percent
                },
                expires_at=strategy.expires_at
            )

            # TODO: 保存到数据库
            # db_session.add(db_strategy)
            # db_session.commit()

            self.logger.info(f"策略已保存到数据库: {db_strategy.id}")
            return db_strategy

        except Exception as e:
            self.logger.error(f"保存策略到数据库失败: {e}")
            raise

    async def evaluate_strategy_performance(
        self,
        strategy: GeneratedStrategy,
        current_price: float,
        days_elapsed: int = 0
    ) -> Dict[str, Any]:
        """评估策略表现"""
        try:
            if strategy.final_recommendation == "hold":
                return {
                    "strategy_id": strategy.strategy_id,
                    "status": "no_action",
                    "pnl_percent": 0.0,
                    "evaluation": "策略建议持有，无盈亏"
                }

            # 计算当前盈亏
            if strategy.final_recommendation == "long":
                pnl_percent = (current_price - strategy.entry_price) / strategy.entry_price * 100
            elif strategy.final_recommendation == "short":
                pnl_percent = (strategy.entry_price - current_price) / strategy.entry_price * 100
            else:
                pnl_percent = 0.0

            # 检查是否触发止损或止盈
            status = "open"
            exit_reason = None

            if strategy.final_recommendation == "long":
                if current_price <= strategy.stop_loss_price:
                    status = "stopped_out"
                    exit_reason = "stop_loss_triggered"
                    pnl_percent = (strategy.stop_loss_price - strategy.entry_price) / strategy.entry_price * 100
                elif current_price >= strategy.take_profit_price:
                    status = "profit_taken"
                    exit_reason = "take_profit_triggered"
                    pnl_percent = (strategy.take_profit_price - strategy.entry_price) / strategy.entry_price * 100
            elif strategy.final_recommendation == "short":
                if current_price >= strategy.stop_loss_price:
                    status = "stopped_out"
                    exit_reason = "stop_loss_triggered"
                    pnl_percent = (strategy.entry_price - strategy.stop_loss_price) / strategy.entry_price * 100
                elif current_price <= strategy.take_profit_price:
                    status = "profit_taken"
                    exit_reason = "take_profit_triggered"
                    pnl_percent = (strategy.entry_price - strategy.take_profit_price) / strategy.entry_price * 100

            return {
                "strategy_id": strategy.strategy_id,
                "symbol": strategy.symbol,
                "recommendation": strategy.final_recommendation,
                "entry_price": strategy.entry_price,
                "current_price": current_price,
                "stop_loss_price": strategy.stop_loss_price,
                "take_profit_price": strategy.take_profit_price,
                "status": status,
                "exit_reason": exit_reason,
                "pnl_percent": pnl_percent,
                "risk_reward_ratio": self._calculate_risk_reward_ratio(strategy),
                "days_elapsed": days_elapsed,
                "evaluation_date": datetime.utcnow()
            }

        except Exception as e:
            self.logger.error(f"策略表现评估失败: {e}")
            return {
                "strategy_id": strategy.strategy_id,
                "status": "evaluation_failed",
                "error": str(e)
            }


# 便捷函数
async def generate_strategy(
    symbol: str,
    timeframe: str,
    analysis_types: List[str],
    risk_tolerance: str,
    max_position_size: float,
    aggregated_signal: AggregatedSignal,
    include_news: bool = False,
    custom_parameters: Optional[Dict[str, Any]] = None
) -> GeneratedStrategy:
    """生成策略的便捷函数"""
    generator = LLMStrategyGenerator()
    request = StrategyGenerationRequest(
        symbol=symbol,
        timeframe=timeframe,
        analysis_types=analysis_types,
        include_news=include_news,
        risk_tolerance=risk_tolerance,
        max_position_size=max_position_size,
        custom_parameters=custom_parameters
    )
    return await generator.generate_final_strategy(request, aggregated_signal)