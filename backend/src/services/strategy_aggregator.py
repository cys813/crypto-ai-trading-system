"""
策略聚合服务

负责聚合来自不同分析源的结果，为最终的策略生成提供统一的数据输入。
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import numpy as np

from ..core.logging import get_logger
from ..core.exceptions import ValidationError, BusinessLogicError
from ..models.trading_strategy import TradingStrategy, StrategyType, RiskLevel
from ..services.llm_long_strategy_analyzer import LongStrategyAnalysis
from ..services.llm_short_strategy_analyzer import ShortStrategyAnalysis
from ..services.news_collector import NewsData
from ..services.llm_news_summarizer import NewsSummary

logger = get_logger(__name__)


class AggregationMethod(Enum):
    """聚合方法"""
    WEIGHTED_AVERAGE = "weighted_average"
    MAJORITY_VOTE = "majority_vote"
    CONFIDENCE_BASED = "confidence_based"
    RISK_ADJUSTED = "risk_adjusted"


@dataclass
class AggregatedSignal:
    """聚合后的信号"""
    symbol: str
    final_recommendation: str  # 'long', 'short', 'hold'
    confidence_score: float
    consensus_strength: float  # 0.0-1.0，各分析源的一致性强度
    conflicting_signals: List[str]  # 冲突的信号
    supporting_analyses: List[str]  # 支持该推荐的分析类型
    risk_level: RiskLevel
    aggregation_method: AggregationMethod
    aggregated_at: datetime

    # 价格相关
    entry_price_range: Dict[str, float]  # {'min': x, 'max': y, 'avg': z}
    stop_loss_range: Dict[str, float]
    take_profit_range: Dict[str, float]
    position_size_range: Dict[str, float]

    # 综合分析数据
    technical_summary: Dict[str, Any]
    news_summary: Dict[str, Any]
    market_sentiment: str


@dataclass
class AnalysisSource:
    """分析源数据"""
    source_type: str  # 'long_analysis', 'short_analysis', 'technical', 'news'
    data: Any
    confidence_score: float
    weight: float
    timestamp: datetime


class StrategyAggregator:
    """策略聚合器"""

    def __init__(self):
        self.logger = logger

        # 默认权重配置
        self.default_weights = {
            "long_analysis": 0.4,
            "short_analysis": 0.4,
            "technical_analysis": 0.15,
            "news_analysis": 0.05
        }

        # 分析类型映射
        self.analysis_mapping = {
            "long": "long_analysis",
            "short": "short_analysis",
            "technical": "technical_analysis",
            "news": "news_analysis"
        }

    async def aggregate_strategies(
        self,
        symbol: str,
        long_analysis: Optional[LongStrategyAnalysis] = None,
        short_analysis: Optional[ShortStrategyAnalysis] = None,
        technical_analysis: Optional[Dict[str, Any]] = None,
        news_analysis: Optional[Dict[str, Any]] = None,
        analysis_weights: Optional[Dict[str, float]] = None,
        method: AggregationMethod = AggregationMethod.WEIGHTED_AVERAGE
    ) -> AggregatedSignal:
        """聚合多个策略分析结果"""
        try:
            self.logger.info(f"开始聚合{symbol}的策略分析结果")

            # 1. 收集所有可用的分析源
            analysis_sources = self._collect_analysis_sources(
                symbol, long_analysis, short_analysis, technical_analysis, news_analysis
            )

            if not analysis_sources:
                raise ValidationError("没有可用的分析数据进行聚合")

            # 2. 应用权重配置
            weights = analysis_weights or self.default_weights
            weighted_sources = self._apply_weights(analysis_sources, weights)

            # 3. 根据聚合方法进行聚合
            if method == AggregationMethod.WEIGHTED_AVERAGE:
                aggregated = self._weighted_average_aggregation(weighted_sources)
            elif method == AggregationMethod.MAJORITY_VOTE:
                aggregated = self._majority_vote_aggregation(weighted_sources)
            elif method == AggregationMethod.CONFIDENCE_BASED:
                aggregated = self._confidence_based_aggregation(weighted_sources)
            elif method == AggregationMethod.RISK_ADJUSTED:
                aggregated = self._risk_adjusted_aggregation(weighted_sources)
            else:
                raise ValidationError(f"不支持的聚合方法: {method}")

            # 4. 生成综合分析摘要
            aggregated.technical_summary = self._generate_technical_summary(technical_analysis)
            aggregated.news_summary = self._generate_news_summary(news_analysis)
            aggregated.market_sentiment = self._assess_market_sentiment(aggregated, technical_analysis, news_analysis)

            # 5. 验证聚合结果
            self._validate_aggregated_result(aggregated)

            self.logger.info(f"策略聚合完成: {symbol} -> {aggregated.final_recommendation} (置信度: {aggregated.confidence_score:.2f})")

            return aggregated

        except Exception as e:
            self.logger.error(f"策略聚合失败: {e}")
            raise BusinessLogicError(f"策略聚合失败: {str(e)}")

    async def aggregate_multiple_symbols(
        self,
        symbol_analyses: Dict[str, Dict[str, Any]],
        method: AggregationMethod = AggregationMethod.WEIGHTED_AVERAGE
    ) -> Dict[str, AggregatedSignal]:
        """批量聚合多个交易符号的策略"""
        try:
            self.logger.info(f"开始批量聚合{len(symbol_analyses)}个交易符号的策略")

            results = {}
            errors = {}

            for symbol, analyses in symbol_analyses.items():
                try:
                    # 提取各类型分析数据
                    long_analysis = analyses.get("long_analysis")
                    short_analysis = analyses.get("short_analysis")
                    technical_analysis = analyses.get("technical_analysis")
                    news_analysis = analyses.get("news_analysis")

                    # 聚合单个符号的策略
                    aggregated = await self.aggregate_strategies(
                        symbol=symbol,
                        long_analysis=long_analysis,
                        short_analysis=short_analysis,
                        technical_analysis=technical_analysis,
                        news_analysis=news_analysis,
                        method=method
                    )

                    results[symbol] = aggregated

                except Exception as e:
                    self.logger.error(f"聚合{symbol}策略失败: {e}")
                    errors[symbol] = str(e)

            if errors:
                self.logger.warning(f"部分符号聚合失败: {errors}")

            self.logger.info(f"批量聚合完成: 成功{len(results)}个，失败{len(errors)}个")

            return results

        except Exception as e:
            self.logger.error(f"批量策略聚合失败: {e}")
            raise

    def _collect_analysis_sources(
        self,
        symbol: str,
        long_analysis: Optional[LongStrategyAnalysis],
        short_analysis: Optional[ShortStrategyAnalysis],
        technical_analysis: Optional[Dict[str, Any]],
        news_analysis: Optional[Dict[str, Any]]
    ) -> List[AnalysisSource]:
        """收集所有可用的分析源"""
        sources = []

        if long_analysis:
            sources.append(AnalysisSource(
                source_type="long_analysis",
                data=long_analysis,
                confidence_score=long_analysis.confidence_score,
                weight=self.default_weights.get("long_analysis", 0.4),
                timestamp=datetime.utcnow()
            ))

        if short_analysis:
            sources.append(AnalysisSource(
                source_type="short_analysis",
                data=short_analysis,
                confidence_score=short_analysis.confidence_score,
                weight=self.default_weights.get("short_analysis", 0.4),
                timestamp=datetime.utcnow()
            ))

        if technical_analysis:
            # 从技术分析中提取推荐和置信度
            tech_recommendation = technical_analysis.get("market_analysis", {}).get("trend_direction", "neutral")
            tech_confidence = technical_analysis.get("market_analysis", {}).get("trend_strength", 0.5)

            sources.append(AnalysisSource(
                source_type="technical_analysis",
                data=technical_analysis,
                confidence_score=tech_confidence,
                weight=self.default_weights.get("technical_analysis", 0.15),
                timestamp=datetime.utcnow()
            ))

        if news_analysis:
            # 从新闻分析中提取情绪和置信度
            news_sentiment = news_analysis.get("sentiment", "neutral")
            news_confidence = news_analysis.get("confidence_score", 0.5)

            sources.append(AnalysisSource(
                source_type="news_analysis",
                data=news_analysis,
                confidence_score=news_confidence,
                weight=self.default_weights.get("news_analysis", 0.05),
                timestamp=datetime.utcnow()
            ))

        return sources

    def _apply_weights(
        self,
        sources: List[AnalysisSource],
        weights: Dict[str, float]
    ) -> List[AnalysisSource]:
        """应用权重到分析源"""
        weighted_sources = []
        total_weight = sum(weights.values())

        for source in sources:
            if source.source_type in weights:
                normalized_weight = weights[source.source_type] / total_weight
                weighted_sources.append(AnalysisSource(
                    source_type=source.source_type,
                    data=source.data,
                    confidence_score=source.confidence_score,
                    weight=normalized_weight,
                    timestamp=source.timestamp
                ))

        return weighted_sources

    def _weighted_average_aggregation(self, sources: List[AnalysisSource]) -> AggregatedSignal:
        """加权平均聚合"""
        symbol = sources[0].data.symbol if hasattr(sources[0].data, 'symbol') else "UNKNOWN"

        # 提取推荐和权重
        recommendations = []
        confidences = []
        total_weight = 0

        for source in sources:
            if hasattr(source.data, 'recommendation'):
                recommendations.append(source.data.recommendation)
                confidences.append(source.confidence_score * source.weight)
                total_weight += source.weight

        # 计算加权平均置信度
        avg_confidence = sum(confidences) / total_weight if total_weight > 0 else 0

        # 确定最终推荐（基于权重）
        recommendation_scores = {"long": 0, "short": 0, "hold": 0}
        for i, source in enumerate(sources):
            if hasattr(source.data, 'recommendation'):
                rec = source.data.recommendation
                if rec in ["buy", "strong_buy"]:
                    recommendation_scores["long"] += source.weight * source.confidence_score
                elif rec in ["sell", "strong_sell"]:
                    recommendation_scores["short"] += source.weight * source.confidence_score
                else:
                    recommendation_scores["hold"] += source.weight * source.confidence_score

        final_recommendation = max(recommendation_scores, key=recommendation_scores.get)

        # 计算一致性强度
        consensus_strength = self._calculate_consensus_strength(sources)

        # 聚合价格范围
        price_ranges = self._aggregate_price_ranges(sources)

        return AggregatedSignal(
            symbol=symbol,
            final_recommendation=final_recommendation,
            confidence_score=avg_confidence,
            consensus_strength=consensus_strength,
            conflicting_signals=self._identify_conflicting_signals(sources),
            supporting_analyses=[s.source_type for s in sources],
            risk_level=self._assess_risk_level(sources),
            aggregation_method=AggregationMethod.WEIGHTED_AVERAGE,
            aggregated_at=datetime.utcnow(),
            entry_price_range=price_ranges["entry"],
            stop_loss_range=price_ranges["stop_loss"],
            take_profit_range=price_ranges["take_profit"],
            position_size_range=price_ranges["position_size"],
            technical_summary={},
            news_summary={},
            market_sentiment=""
        )

    def _majority_vote_aggregation(self, sources: List[AnalysisSource]) -> AggregatedSignal:
        """多数投票聚合"""
        symbol = sources[0].data.symbol if hasattr(sources[0].data, 'symbol') else "UNKNOWN"

        # 统计推荐投票
        vote_counts = {"long": 0, "short": 0, "hold": 0}
        vote_confidences = {"long": [], "short": [], "hold": []}

        for source in sources:
            if hasattr(source.data, 'recommendation'):
                rec = source.data.recommendation
                if rec in ["buy", "strong_buy"]:
                    vote_counts["long"] += 1
                    vote_confidences["long"].append(source.confidence_score)
                elif rec in ["sell", "strong_sell"]:
                    vote_counts["short"] += 1
                    vote_confidences["short"].append(source.confidence_score)
                else:
                    vote_counts["hold"] += 1
                    vote_confidences["hold"].append(source.confidence_score)

        # 确定多数投票结果
        final_recommendation = max(vote_counts, key=vote_counts.get)

        # 计算平均置信度
        if vote_confidences[final_recommendation]:
            avg_confidence = np.mean(vote_confidences[final_recommendation])
        else:
            avg_confidence = 0.5

        # 计算一致性强度
        total_votes = sum(vote_counts.values())
        consensus_strength = vote_counts[final_recommendation] / total_votes if total_votes > 0 else 0

        # 聚合价格范围
        price_ranges = self._aggregate_price_ranges(sources)

        return AggregatedSignal(
            symbol=symbol,
            final_recommendation=final_recommendation,
            confidence_score=avg_confidence,
            consensus_strength=consensus_strength,
            conflicting_signals=self._identify_conflicting_signals(sources),
            supporting_analyses=[s.source_type for s in sources],
            risk_level=self._assess_risk_level(sources),
            aggregation_method=AggregationMethod.MAJORITY_VOTE,
            aggregated_at=datetime.utcnow(),
            entry_price_range=price_ranges["entry"],
            stop_loss_range=price_ranges["stop_loss"],
            take_profit_range=price_ranges["take_profit"],
            position_size_range=price_ranges["position_size"],
            technical_summary={},
            news_summary={},
            market_sentiment=""
        )

    def _confidence_based_aggregation(self, sources: List[AnalysisSource]) -> AggregatedSignal:
        """基于置信度的聚合"""
        # 按置信度排序
        sorted_sources = sorted(sources, key=lambda x: x.confidence_score, reverse=True)

        # 使用最高置信度的分析作为主要参考
        primary_source = sorted_sources[0]
        symbol = primary_source.data.symbol if hasattr(primary_source.data, 'symbol') else "UNKNOWN"

        # 获取主要推荐
        if hasattr(primary_source.data, 'recommendation'):
            primary_rec = primary_source.data.recommendation
            if primary_rec in ["buy", "strong_buy"]:
                final_recommendation = "long"
            elif primary_rec in ["sell", "strong_sell"]:
                final_recommendation = "short"
            else:
                final_recommendation = "hold"
        else:
            final_recommendation = "hold"

        # 计算加权置信度（考虑高置信度源的权重）
        total_confidence_weight = sum(s.confidence_score * s.weight for s in sources)
        avg_confidence = total_confidence_weight / sum(s.weight for s in sources)

        # 聚合价格范围
        price_ranges = self._aggregate_price_ranges(sources)

        return AggregatedSignal(
            symbol=symbol,
            final_recommendation=final_recommendation,
            confidence_score=avg_confidence,
            consensus_strength=self._calculate_consensus_strength(sources),
            conflicting_signals=self._identify_conflicting_signals(sources),
            supporting_analyses=[s.source_type for s in sources],
            risk_level=self._assess_risk_level(sources),
            aggregation_method=AggregationMethod.CONFIDENCE_BASED,
            aggregated_at=datetime.utcnow(),
            entry_price_range=price_ranges["entry"],
            stop_loss_range=price_ranges["stop_loss"],
            take_profit_range=price_ranges["take_profit"],
            position_size_range=price_ranges["position_size"],
            technical_summary={},
            news_summary={},
            market_sentiment=""
        )

    def _risk_adjusted_aggregation(self, sources: List[AnalysisSource]) -> AggregatedSignal:
        """风险调整聚合"""
        # 基于风险水平调整聚合策略
        risk_levels = []
        for source in sources:
            if hasattr(source.data, 'risk_factors'):
                risk_count = len(source.data.risk_factors)
                risk_levels.append(risk_count)

        avg_risk = np.mean(risk_levels) if risk_levels else 1

        # 高风险时更保守的聚合策略
        if avg_risk > 3:
            # 倾向于保守的推荐
            conservative_sources = [s for s in sources if
                                  hasattr(s.data, 'recommendation') and
                                  s.data.recommendation in ['hold', 'sell']]
            if conservative_sources:
                sources = conservative_sources

        symbol = sources[0].data.symbol if hasattr(sources[0].data, 'symbol') else "UNKNOWN"

        # 使用加权平均作为基础
        base_result = self._weighted_average_aggregation(sources)
        base_result.aggregation_method = AggregationMethod.RISK_ADJUSTED

        # 根据风险水平调整置信度
        risk_adjusted_confidence = base_result.confidence_score * max(0.7, 1 - avg_risk * 0.1)
        base_result.confidence_score = risk_adjusted_confidence

        # 调整风险等级
        if avg_risk > 4:
            base_result.risk_level = RiskLevel.HIGH
        elif avg_risk > 2:
            base_result.risk_level = RiskLevel.MEDIUM
        else:
            base_result.risk_level = RiskLevel.LOW

        return base_result

    def _calculate_consensus_strength(self, sources: List[AnalysisSource]) -> float:
        """计算分析源之间的一致性强度"""
        if len(sources) <= 1:
            return 1.0

        recommendations = []
        for source in sources:
            if hasattr(source.data, 'recommendation'):
                rec = source.data.recommendation
                if rec in ["buy", "strong_buy"]:
                    recommendations.append("long")
                elif rec in ["sell", "strong_sell"]:
                    recommendations.append("short")
                else:
                    recommendations.append("hold")

        if not recommendations:
            return 0.0

        # 计算推荐的一致性
        rec_counts = {"long": recommendations.count("long"),
                     "short": recommendations.count("short"),
                     "hold": recommendations.count("hold")}

        max_count = max(rec_counts.values())
        consensus_strength = max_count / len(recommendations)

        return consensus_strength

    def _identify_conflicting_signals(self, sources: List[AnalysisSource]) -> List[str]:
        """识别冲突的信号"""
        conflicts = []
        recommendations = []

        for source in sources:
            if hasattr(source.data, 'recommendation'):
                rec = source.data.recommendation
                if rec in ["buy", "strong_buy"]:
                    recommendations.append(("long", source.source_type))
                elif rec in ["sell", "strong_sell"]:
                    recommendations.append(("short", source.source_type))
                else:
                    recommendations.append(("hold", source.source_type))

        # 检查是否存在相反的推荐
        long_sources = [src for rec, src in recommendations if rec == "long"]
        short_sources = [src for rec, src in recommendations if rec == "short"]

        if long_sources and short_sources:
            conflicts.append(f"做多({','.join(long_sources)}) vs 做空({','.join(short_sources)})")

        return conflicts

    def _aggregate_price_ranges(self, sources: List[AnalysisSource]) -> Dict[str, Dict[str, float]]:
        """聚合价格范围"""
        ranges = {
            "entry": {"values": [], "min": 0, "max": 0, "avg": 0},
            "stop_loss": {"values": [], "min": 0, "max": 0, "avg": 0},
            "take_profit": {"values": [], "min": 0, "max": 0, "avg": 0},
            "position_size": {"values": [], "min": 0, "max": 0, "avg": 0}
        }

        for source in sources:
            data = source.data

            # 入场价格
            if hasattr(data, 'entry_price') and data.entry_price:
                ranges["entry"]["values"].append(float(data.entry_price))

            # 止损价格
            if hasattr(data, 'stop_loss_price') and data.stop_loss_price:
                ranges["stop_loss"]["values"].append(float(data.stop_loss_price))

            # 止盈价格
            if hasattr(data, 'take_profit_price') and data.take_profit_price:
                ranges["take_profit"]["values"].append(float(data.take_profit_price))

            # 仓位大小
            if hasattr(data, 'position_size_percent') and data.position_size_percent:
                ranges["position_size"]["values"].append(float(data.position_size_percent))

        # 计算统计值
        for key, range_data in ranges.items():
            if range_data["values"]:
                range_data["min"] = min(range_data["values"])
                range_data["max"] = max(range_data["values"])
                range_data["avg"] = np.mean(range_data["values"])

        return ranges

    def _assess_risk_level(self, sources: List[AnalysisSource]) -> RiskLevel:
        """评估风险等级"""
        risk_factors = []
        confidence_scores = []

        for source in sources:
            # 收集风险因子
            if hasattr(source.data, 'risk_factors'):
                risk_factors.extend(source.data.risk_factors)

            # 收集置信度
            confidence_scores.append(source.confidence_score)

        # 基于风险因子数量评估
        risk_count = len(risk_factors)

        # 基于置信度评估
        avg_confidence = np.mean(confidence_scores) if confidence_scores else 0.5

        # 综合评估
        if risk_count > 5 or avg_confidence < 0.6:
            return RiskLevel.HIGH
        elif risk_count > 2 or avg_confidence < 0.8:
            return RiskLevel.MEDIUM
        else:
            return RiskLevel.LOW

    def _generate_technical_summary(self, technical_analysis: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """生成技术分析摘要"""
        if not technical_analysis:
            return {}

        market_analysis = technical_analysis.get("market_analysis", {})
        indicators = technical_analysis.get("indicators", {})

        return {
            "trend_direction": market_analysis.get("trend_direction", "neutral"),
            "trend_strength": market_analysis.get("trend_strength", 0),
            "volatility_level": market_analysis.get("volatility_level", "medium"),
            "key_indicators": {
                "rsi": indicators.get("rsi"),
                "macd_signal": indicators.get("macd_histogram", 0) > 0,
                "bollinger_position": self._get_bollinger_position(indicators)
            }
        }

    def _generate_news_summary(self, news_analysis: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """生成新闻分析摘要"""
        if not news_analysis:
            return {}

        return {
            "overall_sentiment": news_analysis.get("sentiment", "neutral"),
            "sentiment_score": news_analysis.get("sentiment_score", 0.5),
            "key_events": news_analysis.get("key_events", []),
            "market_impact": news_analysis.get("market_impact", "medium")
        }

    def _assess_market_sentiment(
        self,
        aggregated: AggregatedSignal,
        technical_analysis: Optional[Dict[str, Any]] = None,
        news_analysis: Optional[Dict[str, Any]] = None
    ) -> str:
        """评估市场情绪"""
        sentiments = []
        weights = []

        # 聚合结果情绪
        if aggregated.final_recommendation == "long":
            sentiments.append(1.0)  # 看涨
        elif aggregated.final_recommendation == "short":
            sentiments.append(-1.0)  # 看跌
        else:
            sentiments.append(0.0)  # 中性
        weights.append(0.4)

        # 技术分析情绪
        if technical_analysis:
            trend = technical_analysis.get("market_analysis", {}).get("trend_direction", "neutral")
            if trend == "up":
                sentiments.append(0.8)
            elif trend == "down":
                sentiments.append(-0.8)
            else:
                sentiments.append(0.0)
            weights.append(0.4)

        # 新闻情绪
        if news_analysis:
            news_sentiment = news_analysis.get("sentiment", "neutral")
            if news_sentiment == "positive":
                sentiments.append(0.6)
            elif news_sentiment == "negative":
                sentiments.append(-0.6)
            else:
                sentiments.append(0.0)
            weights.append(0.2)

        # 计算加权平均情绪
        if sentiments and weights:
            weighted_sentiment = sum(s * w for s, w in zip(sentiments, weights)) / sum(weights)
        else:
            weighted_sentiment = 0.0

        # 转换为文字描述
        if weighted_sentiment > 0.3:
            return "bullish"
        elif weighted_sentiment < -0.3:
            return "bearish"
        else:
            return "neutral"

    def _get_bollinger_position(self, indicators: Dict[str, Any]) -> str:
        """获取布林带位置"""
        current_price = indicators.get("current_price", 0)
        upper = indicators.get("bollinger_upper", 0)
        middle = indicators.get("bollinger_middle", 0)
        lower = indicators.get("bollinger_lower", 0)

        if current_price >= upper:
            return "above_upper"
        elif current_price <= lower:
            return "below_lower"
        elif current_price >= middle:
            return "upper_half"
        else:
            return "lower_half"

    def _validate_aggregated_result(self, result: AggregatedSignal) -> None:
        """验证聚合结果"""
        if not result.final_recommendation:
            raise ValidationError("聚合结果缺少最终推荐")

        if not 0 <= result.confidence_score <= 1:
            raise ValidationError("聚合结果置信度超出有效范围")

        if not result.entry_price_range["avg"] or result.entry_price_range["avg"] <= 0:
            raise ValidationError("聚合结果入场价格无效")

        # 验证价格逻辑
        if result.final_recommendation == "long":
            if (result.stop_loss_range["avg"] and result.take_profit_range["avg"] and
                result.entry_price_range["avg"]):
                if result.stop_loss_range["avg"] >= result.entry_price_range["avg"]:
                    self.logger.warning("做多策略止损价格异常")
                if result.take_profit_range["avg"] <= result.entry_price_range["avg"]:
                    self.logger.warning("做多策略止盈价格异常")
        elif result.final_recommendation == "short":
            if (result.stop_loss_range["avg"] and result.take_profit_range["avg"] and
                result.entry_price_range["avg"]):
                if result.stop_loss_range["avg"] <= result.entry_price_range["avg"]:
                    self.logger.warning("做空策略止损价格异常")
                if result.take_profit_range["avg"] >= result.entry_price_range["avg"]:
                    self.logger.warning("做空策略止盈价格异常")


# 便捷函数
async def aggregate_strategies(
    symbol: str,
    long_analysis: Optional[LongStrategyAnalysis] = None,
    short_analysis: Optional[ShortStrategyAnalysis] = None,
    technical_analysis: Optional[Dict[str, Any]] = None,
    news_analysis: Optional[Dict[str, Any]] = None,
    method: AggregationMethod = AggregationMethod.WEIGHTED_AVERAGE
) -> AggregatedSignal:
    """聚合策略的便捷函数"""
    aggregator = StrategyAggregator()
    return await aggregator.aggregate_strategies(
        symbol, long_analysis, short_analysis, technical_analysis, news_analysis, method=method
    )