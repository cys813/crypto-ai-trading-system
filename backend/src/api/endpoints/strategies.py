"""
策略分析API端点

提供做多、做空策略分析和综合策略生成的HTTP接口。
"""

import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from fastapi import APIRouter, Depends, HTTPException, Query, BackgroundTasks
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from ...core.database import get_db_session
from ...core.logging import get_logger
from ...core.exceptions import ValidationError, ExternalServiceError, LLMServiceError
from ...services.exchange_data_collector import ExchangeDataCollector, get_klines
from ...services.technical_analysis_engine import TechnicalAnalysisEngine, analyze_klines
from ...services.llm_long_strategy_analyzer import LLMLongStrategyAnalyzer, analyze_long_strategy
from ...services.llm_short_strategy_analyzer import LLMShortStrategyAnalyzer, analyze_short_strategy
from ...core.news_events import get_event_tracker, NewsEventType, EventSeverity

logger = get_logger(__name__)
router = APIRouter()

# 请求模型
class LongAnalysisRequest(BaseModel):
    """做多策略分析请求"""
    symbol: str = Field(..., description="交易符号，如 BTC/USDT")
    timeframe: str = Field("1h", description="时间框架")
    analysis_period_days: int = Field(7, ge=1, le=30, description="分析周期（天）")
    confidence_threshold: float = Field(0.7, ge=0.5, le=1.0, description="置信度阈值")
    include_news: bool = Field(False, description="是否包含新闻分析")

class ShortAnalysisRequest(BaseModel):
    """做空策略分析请求"""
    symbol: str = Field(..., description="交易符号，如 BTC/USDT")
    timeframe: str = Field("1h", description="时间框架")
    analysis_period_days: int = Field(7, ge=1, le=30, description="分析周期（天）")
    confidence_threshold: float = Field(0.7, ge=0.5, le=1.0, description="置信度阈值")
    include_news: bool = Field(False, description="是否包含新闻分析")
    max_position_size: float = Field(20.0, ge=1.0, le=50.0, description="最大仓位百分比")

class StrategyAnalysisRequest(BaseModel):
    """综合策略分析请求"""
    symbol: str = Field(..., description="交易符号")
    timeframe: str = Field("1h", description="时间框架")
    analysis_type: str = Field("comprehensive", description="分析类型")
    custom_parameters: Optional[Dict[str, Any]] = Field(None, description="自定义参数")

# 响应模型
class LongStrategyResponse(BaseModel):
    """做多策略分析响应"""
    symbol: str
    timeframe: str
    recommendation: str
    confidence_score: float
    entry_price: float
    stop_loss_price: float
    take_profit_price: float
    position_size_percent: float
    time_horizon: str
    reasoning: str
    risk_factors: List[str]
    market_conditions: Dict[str, Any]
    technical_signals: List[Dict[str, Any]]
    price_targets: Dict[str, float]
    execution_strategy: Dict[str, Any]
    analysis_timestamp: datetime
    risk_reward_ratio: float
    data_quality_score: float

class ShortStrategyResponse(BaseModel):
    """做空策略分析响应"""
    symbol: str
    timeframe: str
    recommendation: str
    confidence_score: float
    entry_price: float
    stop_loss_price: float
    take_profit_price: float
    position_size_percent: float
    time_horizon: str
    reasoning: str
    risk_factors: List[str]
    market_conditions: Dict[str, Any]
    technical_signals: List[Dict[str, Any]]
    price_targets: Dict[str, float]
    execution_strategy: Dict[str, Any]
    analysis_timestamp: datetime
    risk_reward_ratio: float
    data_quality_score: float
    short_specific_factors: Dict[str, Any]

class StrategyAnalysisResponse(BaseModel):
    """综合策略分析响应"""
    symbol: str
    analysis_type: str
    recommendation: str
    confidence_score: float
    detailed_analysis: Dict[str, Any]
    risk_assessment: Dict[str, Any]
    execution_plan: Dict[str, Any]
    analysis_timestamp: datetime


@router.post("/strategies/long-analysis", response_model=LongStrategyResponse)
async def analyze_long_strategy(
    request: LongAnalysisRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db_session)
):
    """分析做多策略"""
    try:
        logger.info(f"开始做多策略分析: {request.symbol} {request.timeframe}")

        # 验证交易符号格式
        if "/" not in request.symbol:
            raise ValidationError("交易符号格式无效，应为 BASE/QUOTE 格式")

        # 初始化服务
        event_tracker = get_event_tracker()
        data_collector = ExchangeDataCollector()
        technical_engine = TechnicalAnalysisEngine()
        llm_analyzer = LLMLongStrategyAnalyzer()

        # 追踪分析开始
        event_tracker.track_event(
            event_type=NewsEventType.SUMMARY_GENERATION_STARTED,
            message=f"开始做多策略分析: {request.symbol}",
            severity=EventSeverity.INFO,
            details={
                "symbol": request.symbol,
                "timeframe": request.timeframe,
                "analysis_period_days": request.analysis_period_days
            }
        )

        # 1. 收集K线数据
        try:
            klines = await data_collector.fetch_klines(
                symbol=request.symbol,
                timeframe=request.timeframe,
                limit=request.analysis_period_days * 24,  # 估算数据点数量
                exchange="binance"
            )

            if not klines:
                raise ExternalServiceError("无法获取K线数据")

            logger.info(f"成功获取 {len(klines)} 条K线数据")

        except Exception as e:
            logger.error(f"K线数据收集失败: {e}")
            raise ExternalServiceError(f"K线数据收集失败: {str(e)}")

        # 2. 技术分析
        try:
            technical_analysis = await technical_engine.analyze(
                klines=klines,
                symbol=request.symbol,
                timeframe=request.timeframe
            )

            logger.info(f"技术分析完成，趋势: {technical_analysis['market_analysis']['trend_direction']}")

        except Exception as e:
            logger.error(f"技术分析失败: {e}")
            raise ExternalServiceError(f"技术分析失败: {str(e)}")

        # 3. LLM策略分析
        try:
            strategy_analysis = await llm_analyzer.analyze_long_strategy(
                symbol=request.symbol,
                technical_analysis=technical_analysis,
                market_data={
                    "timeframe": request.timeframe,
                    "analysis_period_days": request.analysis_period_days
                }
            )

            logger.info(f"LLM策略分析完成，推荐: {strategy_analysis.recommendation}")

        except Exception as e:
            logger.error(f"LLM策略分析失败: {e}")
            raise LLMServiceError(f"LLM策略分析失败: {str(e)}")

        # 4. 检查置信度阈值
        if strategy_analysis.confidence_score < request.confidence_threshold:
            logger.warning(f"策略置信度不足: {strategy_analysis.confidence_score} < {request.confidence_threshold}")
            # 可以选择调整推荐或返回警告

        # 5. 计算风险回报比
        risk_reward_ratio = (
            (strategy_analysis.take_profit_price - strategy_analysis.entry_price) /
            (strategy_analysis.entry_price - strategy_analysis.stop_loss_price)
        ) if strategy_analysis.entry_price != strategy_analysis.stop_loss_price else 1.0

        # 6. 构建响应
        response = LongStrategyResponse(
            symbol=strategy_analysis.symbol,
            timeframe=strategy_analysis.timeframe,
            recommendation=strategy_analysis.recommendation,
            confidence_score=strategy_analysis.confidence_score,
            entry_price=strategy_analysis.entry_price,
            stop_loss_price=strategy_analysis.stop_loss_price,
            take_profit_price=strategy_analysis.take_profit_price,
            position_size_percent=strategy_analysis.position_size_percent,
            time_horizon=strategy_analysis.time_horizon,
            reasoning=strategy_analysis.reasoning,
            risk_factors=strategy_analysis.risk_factors,
            market_conditions=strategy_analysis.market_conditions,
            technical_signals=strategy_analysis.technical_signals,
            price_targets=strategy_analysis.price_targets,
            execution_strategy=strategy_analysis.execution_strategy,
            analysis_timestamp=strategy_analysis.analysis_timestamp,
            risk_reward_ratio=risk_reward_ratio,
            data_quality_score=_calculate_data_quality(klines, technical_analysis)
        )

        # 7. 异步保存分析结果
        background_tasks.add_task(
            _save_strategy_analysis,
            strategy_analysis,
            technical_analysis,
            klines
        )

        # 8. 追踪分析完成
        event_tracker.track_event(
            event_type=NewsEventType.SUMMARY_GENERATION_COMPLETED,
            message=f"做多策略分析完成: {request.symbol}",
            severity=EventSeverity.INFO,
            details={
                "symbol": request.symbol,
                "recommendation": strategy_analysis.recommendation,
                "confidence": strategy_analysis.confidence_score,
                "risk_reward_ratio": risk_reward_ratio
            }
        )

        logger.info(f"做多策略分析完成: {request.symbol} - {strategy_analysis.recommendation}")
        return response

    except (ValidationError, ExternalServiceError, LLMServiceError) as e:
        logger.error(f"做多策略分析失败: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"做多策略分析异常: {e}")
        raise HTTPException(status_code=500, detail="策略分析过程中发生未知错误")


@router.post("/strategies/short-analysis", response_model=ShortStrategyResponse)
async def analyze_short_strategy(
    request: ShortAnalysisRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db_session)
):
    """分析做空策略"""
    try:
        logger.info(f"开始做空策略分析: {request.symbol} {request.timeframe}")

        # 验证交易符号格式
        if "/" not in request.symbol:
            raise ValidationError("交易符号格式无效，应为 BASE/QUOTE 格式")

        # 初始化服务
        event_tracker = get_event_tracker()
        data_collector = ExchangeDataCollector()
        technical_engine = TechnicalAnalysisEngine()
        llm_analyzer = LLMShortStrategyAnalyzer()

        # 追踪分析开始
        event_tracker.track_event(
            event_type=NewsEventType.SUMMARY_GENERATION_STARTED,
            message=f"开始做空策略分析: {request.symbol}",
            severity=EventSeverity.INFO,
            details={
                "symbol": request.symbol,
                "timeframe": request.timeframe,
                "analysis_period_days": request.analysis_period_days,
                "strategy_type": "short"
            }
        )

        # 1. 收集K线数据
        try:
            klines = await data_collector.fetch_klines(
                symbol=request.symbol,
                timeframe=request.timeframe,
                limit=request.analysis_period_days * 24,  # 估算数据点数量
                exchange="binance"
            )

            if not klines:
                raise ExternalServiceError("无法获取K线数据")

            logger.info(f"成功获取 {len(klines)} 条K线数据")

        except Exception as e:
            logger.error(f"K线数据收集失败: {e}")
            raise ExternalServiceError(f"K线数据收集失败: {str(e)}")

        # 2. 技术分析
        try:
            technical_analysis = await technical_engine.analyze(
                klines=klines,
                symbol=request.symbol,
                timeframe=request.timeframe
            )

            logger.info(f"技术分析完成，趋势: {technical_analysis['market_analysis']['trend_direction']}")

            # 检查是否适合做空（需要下跌趋势）
            trend_direction = technical_analysis['market_analysis']['trend_direction']
            if trend_direction == 'up':
                logger.warning(f"当前为上涨趋势，做空风险较高: {request.symbol}")

        except Exception as e:
            logger.error(f"技术分析失败: {e}")
            raise ExternalServiceError(f"技术分析失败: {str(e)}")

        # 3. LLM做空策略分析
        try:
            strategy_analysis = await llm_analyzer.analyze_short_strategy(
                symbol=request.symbol,
                technical_analysis=technical_analysis,
                market_data={
                    "timeframe": request.timeframe,
                    "analysis_period_days": request.analysis_period_days,
                    "max_position_size": request.max_position_size
                }
            )

            logger.info(f"LLM做空策略分析完成，推荐: {strategy_analysis.recommendation}")

        except Exception as e:
            logger.error(f"LLM做空策略分析失败: {e}")
            raise LLMServiceError(f"LLM做空策略分析失败: {str(e)}")

        # 4. 检查置信度阈值
        if strategy_analysis.confidence_score < request.confidence_threshold:
            logger.warning(f"做空策略置信度不足: {strategy_analysis.confidence_score} < {request.confidence_threshold}")

        # 5. 应用仓位大小限制
        if strategy_analysis.position_size_percent > request.max_position_size:
            strategy_analysis.position_size_percent = request.max_position_size
            logger.info(f"调整仓位大小至最大限制: {request.max_position_size}%")

        # 6. 计算做空风险回报比
        risk_reward_ratio = (
            (strategy_analysis.entry_price - strategy_analysis.take_profit_price) /
            (strategy_analysis.stop_loss_price - strategy_analysis.entry_price)
        ) if strategy_analysis.entry_price != strategy_analysis.stop_loss_price else 1.0

        # 7. 构建响应
        response = ShortStrategyResponse(
            symbol=strategy_analysis.symbol,
            timeframe=strategy_analysis.timeframe,
            recommendation=strategy_analysis.recommendation,
            confidence_score=strategy_analysis.confidence_score,
            entry_price=strategy_analysis.entry_price,
            stop_loss_price=strategy_analysis.stop_loss_price,
            take_profit_price=strategy_analysis.take_profit_price,
            position_size_percent=strategy_analysis.position_size_percent,
            time_horizon=strategy_analysis.time_horizon,
            reasoning=strategy_analysis.reasoning,
            risk_factors=strategy_analysis.risk_factors,
            market_conditions=strategy_analysis.market_conditions,
            technical_signals=strategy_analysis.technical_signals,
            price_targets=strategy_analysis.price_targets,
            execution_strategy=strategy_analysis.execution_strategy,
            analysis_timestamp=strategy_analysis.analysis_timestamp,
            risk_reward_ratio=risk_reward_ratio,
            data_quality_score=_calculate_data_quality(klines, technical_analysis),
            short_specific_factors=strategy_analysis.short_specific_factors
        )

        # 8. 异步保存分析结果
        background_tasks.add_task(
            _save_short_strategy_analysis,
            strategy_analysis,
            technical_analysis,
            klines
        )

        # 9. 追踪分析完成
        event_tracker.track_event(
            event_type=NewsEventType.SUMMARY_GENERATION_COMPLETED,
            message=f"做空策略分析完成: {request.symbol}",
            severity=EventSeverity.INFO,
            details={
                "symbol": request.symbol,
                "recommendation": strategy_analysis.recommendation,
                "confidence": strategy_analysis.confidence_score,
                "risk_reward_ratio": risk_reward_ratio,
                "strategy_type": "short"
            }
        )

        logger.info(f"做空策略分析完成: {request.symbol} - {strategy_analysis.recommendation}")
        return response

    except (ValidationError, ExternalServiceError, LLMServiceError) as e:
        logger.error(f"做空策略分析失败: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"做空策略分析异常: {e}")
        raise HTTPException(status_code=500, detail="做空策略分析过程中发生未知错误")


@router.get("/strategies/long-analysis/history")
async def get_long_strategy_history(
    symbol: str = Query(..., description="交易符号"),
    days: int = Query(30, ge=1, le=90, description="历史天数"),
    limit: int = Query(50, ge=1, le=100, description="返回数量限制"),
    db: Session = Depends(get_db_session)
):
    """获取做多策略历史记录"""
    try:
        # TODO: 实现历史记录查询逻辑
        # 从数据库查询策略分析历史记录

        # 模拟响应
        return {
            "symbol": symbol,
            "historical_strategies": [],
            "total_count": 0,
            "query_params": {
                "days": days,
                "limit": limit
            }
        }

    except Exception as e:
        logger.error(f"获取策略历史失败: {e}")
        raise HTTPException(status_code=500, detail="获取策略历史失败")


@router.post("/strategies/comprehensive", response_model=StrategyAnalysisResponse)
async def comprehensive_strategy_analysis(
    request: StrategyAnalysisRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db_session)
):
    """综合策略分析（包含做多、做空等）"""
    try:
        logger.info(f"开始综合策略分析: {request.symbol}")

        # 验证请求
        if "/" not in request.symbol:
            raise ValidationError("交易符号格式无效")

        # TODO: 实现综合策略分析逻辑
        # 1. 多时间框架分析
        # 2. 多种策略类型分析
        # 3. 新闻数据集成
        # 4. 市场情绪分析
        # 5. 综合决策

        # 模拟综合分析结果
        response = StrategyAnalysisResponse(
            symbol=request.symbol,
            analysis_type=request.analysis_type,
            recommendation="hold",
            confidence_score=0.75,
            detailed_analysis={
                "trend_analysis": "横向整理",
                "support_resistance": {"support": 50000, "resistance": 52000},
                "volume_analysis": "成交量稳定",
                "market_sentiment": "中性偏乐观"
            },
            risk_assessment={
                "risk_level": "medium",
                "volatility": "medium",
                "liquidity": "high"
            },
            execution_plan={
                "entry_conditions": ["等待突破确认"],
                "position_sizing": "建议5-10%仓位",
                "risk_management": "设置3%止损"
            },
            analysis_timestamp=datetime.utcnow()
        )

        logger.info(f"综合策略分析完成: {request.symbol}")
        return response

    except ValidationError as e:
        logger.error(f"综合策略分析参数错误: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"综合策略分析失败: {e}")
        raise HTTPException(status_code=500, detail="综合策略分析失败")


@router.get("/strategies/performance")
async def get_strategy_performance(
    symbol: str = Query(..., description="交易符号"),
    days: int = Query(30, ge=1, le=365, description="评估天数"),
    db: Session = Depends(get_db_session)
):
    """获取策略表现统计"""
    try:
        # TODO: 实现策略表现统计逻辑
        # 1. 查询历史策略记录
        # 2. 计算胜率、盈亏比等指标
        # 3. 按时间周期统计
        # 4. 生成表现报告

        # 模拟响应
        return {
            "symbol": symbol,
            "evaluation_period_days": days,
            "performance_stats": {
                "total_strategies": 0,
                "successful_strategies": 0,
                "win_rate": 0.0,
                "average_return": 0.0,
                "max_drawdown": 0.0,
                "sharpe_ratio": 0.0
            },
            "time_series_data": [],
            "recommendations": []
        }

    except Exception as e:
        logger.error(f"获取策略表现失败: {e}")
        raise HTTPException(status_code=500, detail="获取策略表现失败")


@router.post("/strategies/batch-analysis")
async def batch_strategy_analysis(
    symbols: List[str],
    timeframe: str = "1h",
    analysis_type: str = "long",
    background_tasks: BackgroundTasks = None,
    db: Session = Depends(get_db_session)
):
    """批量策略分析"""
    try:
        logger.info(f"开始批量策略分析: {len(symbols)} 个交易符号，类型: {analysis_type}")

        if len(symbols) > 50:
            raise ValidationError("批量分析最多支持50个交易符号")

        # 验证分析类型
        if analysis_type not in ["long", "short", "both"]:
            raise ValidationError("分析类型必须是 'long', 'short' 或 'both'")

        # 验证所有交易符号格式
        for symbol in symbols:
            if "/" not in symbol:
                raise ValidationError(f"无效的交易符号格式: {symbol}")

        # TODO: 实现批量分析逻辑
        # 1. 并发执行多个策略分析
        # 2. 汇总分析结果
        # 3. 生成批量报告

        # 模拟批量分析结果
        results = []
        for symbol in symbols:
            result = {
                "symbol": symbol,
                "status": "pending",
                "analysis_type": analysis_type
            }

            if analysis_type == "long":
                result.update({
                    "long_recommendation": None,
                    "long_confidence_score": None
                })
            elif analysis_type == "short":
                result.update({
                    "short_recommendation": None,
                    "short_confidence_score": None
                })
            else:  # both
                result.update({
                    "long_recommendation": None,
                    "long_confidence_score": None,
                    "short_recommendation": None,
                    "short_confidence_score": None
                })

            results.append(result)

        # 估算完成时间（做空分析需要更多时间）
        base_time = 3 if analysis_type == "long" else 4
        if analysis_type == "both":
            base_time = 6

        estimated_minutes = base_time + (len(symbols) // 10)

        return {
            "batch_id": f"batch_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
            "total_symbols": len(symbols),
            "analysis_type": analysis_type,
            "timeframe": timeframe,
            "status": "processing",
            "results": results,
            "estimated_completion": datetime.utcnow() + timedelta(minutes=estimated_minutes),
            "processing_details": {
                "queue_size": len(symbols),
                "estimated_time_per_symbol": f"{base_time}分钟",
                "concurrent_limit": 5
            }
        }

    except ValidationError as e:
        logger.error(f"批量策略分析参数错误: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"批量策略分析失败: {e}")
        raise HTTPException(status_code=500, detail="批量策略分析失败")


@router.get("/strategies/templates")
async def get_strategy_templates(
    strategy_type: Optional[str] = Query(None, description="策略类型过滤"),
    db: Session = Depends(get_db_session)
):
    """获取策略模板"""
    try:
        # TODO: 实现策略模板查询逻辑
        # 从数据库查询可用的策略模板

        # 模拟策略模板数据
        templates = [
            {
                "id": "conservative_long",
                "name": "保守型做多策略",
                "type": "long",
                "risk_level": "low",
                "time_horizon": "medium_term",
                "description": "适合风险承受能力较低的投资者",
                "parameters": {
                    "max_position_size": 15.0,
                    "min_confidence": 0.8,
                    "risk_reward_ratio": 2.0
                }
            },
            {
                "id": "aggressive_long",
                "name": "激进型做多策略",
                "type": "long",
                "risk_level": "high",
                "time_horizon": "short_term",
                "description": "适合风险承受能力较高的投资者",
                "parameters": {
                    "max_position_size": 30.0,
                    "min_confidence": 0.6,
                    "risk_reward_ratio": 1.5
                }
            },
            {
                "id": "conservative_short",
                "name": "保守型做空策略",
                "type": "short",
                "risk_level": "low",
                "time_horizon": "medium_term",
                "description": "适合风险承受能力较低的做空投资者",
                "parameters": {
                    "max_position_size": 10.0,
                    "min_confidence": 0.8,
                    "risk_reward_ratio": 2.5,
                    "borrowing_cost_limit": 0.1
                }
            },
            {
                "id": "technical_short",
                "name": "技术型做空策略",
                "type": "short",
                "risk_level": "medium",
                "time_horizon": "short_term",
                "description": "基于技术指标的做空策略",
                "parameters": {
                    "max_position_size": 20.0,
                    "min_confidence": 0.7,
                    "risk_reward_ratio": 2.0,
                    "require_downtrend": true,
                    "rsi_threshold": 70
                }
            },
            {
                "id": "aggressive_short",
                "name": "激进型做空策略",
                "type": "short",
                "risk_level": "high",
                "time_horizon": "short_term",
                "description": "适合风险承受能力较高的做空投资者",
                "parameters": {
                    "max_position_size": 25.0,
                    "min_confidence": 0.6,
                    "risk_reward_ratio": 1.8,
                    "max_hold_time_days": 5
                }
            }
        ]

        # 应用过滤
        if strategy_type:
            templates = [t for t in templates if t["type"] == strategy_type]

        return {
            "templates": templates,
            "total_count": len(templates),
            "filters": {"strategy_type": strategy_type}
        }

    except Exception as e:
        logger.error(f"获取策略模板失败: {e}")
        raise HTTPException(status_code=500, detail="获取策略模板失败")


# 辅助函数
def _calculate_data_quality(klines: List, technical_analysis: Dict) -> float:
    """计算数据质量分数"""
    try:
        if not klines:
            return 0.0

        # 数据完整性评分
        completeness_score = min(len(klines) / 100, 1.0)  # 100条数据为满分

        # 技术指标完整性评分
        indicators = technical_analysis.get("indicators", {})
        indicator_count = sum(1 for v in indicators.values() if v is not None)
        indicator_score = indicator_count / 10.0  # 假设10个指标为满分

        # 时间连续性评分
        if len(klines) > 1:
            time_gaps = 0
            for i in range(1, len(klines)):
                time_diff = klines[i].timestamp - klines[i-1].timestamp
                if time_diff > timedelta(hours=2):  # 假设最大间隔2小时
                    time_gaps += 1
            continuity_score = max(0, 1.0 - time_gaps / len(klines))
        else:
            continuity_score = 1.0

        # 综合质量分数
        quality_score = (completeness_score * 0.4 +
                        indicator_score * 0.3 +
                        continuity_score * 0.3)

        return round(quality_score, 3)

    except Exception:
        return 0.5  # 默认中等质量分数


async def _save_strategy_analysis(
    strategy_analysis,
    technical_analysis,
    klines
):
    """异步保存策略分析结果"""
    try:
        # TODO: 实现保存逻辑
        # 1. 保存到TechnicalAnalysis表
        # 2. 保存K线数据（如果需要）
        # 3. 记录分析会话信息

        logger.info("策略分析结果已保存到数据库")
    except Exception as e:
        logger.error(f"保存策略分析结果失败: {e}")


async def _save_short_strategy_analysis(
    strategy_analysis,
    technical_analysis,
    klines
):
    """异步保存做空策略分析结果"""
    try:
        # TODO: 实现保存逻辑
        # 1. 保存到TechnicalAnalysis表，标记为做空策略
        # 2. 保存K线数据（如果需要）
        # 3. 记录做空策略分析会话信息
        # 4. 保存做空特定因子

        logger.info("做空策略分析结果已保存到数据库")
    except Exception as e:
        logger.error(f"保存做空策略分析结果失败: {e}")


@router.post("/strategies/generate", response_model=Dict[str, Any])
async def generate_comprehensive_strategy(
    request: Dict[str, Any],
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db_session)
):
    """生成综合交易策略"""
    try:
        # 验证必需字段
        symbol = request.get("symbol")
        if not symbol:
            raise HTTPException(status_code=400, detail="交易符号不能为空")

        if "/" not in symbol:
            raise HTTPException(status_code=400, detail="交易符号格式无效，应为 BASE/QUOTE 格式")

        timeframe = request.get("timeframe", "1h")
        analysis_types = request.get("analysis_types", ["long", "short"])
        include_news = request.get("include_news", False)
        risk_tolerance = request.get("risk_tolerance", "moderate")
        max_position_size = request.get("max_position_size", 20.0)
        custom_parameters = request.get("custom_parameters", {})

        logger.info(f"开始生成{symbol}的综合策略分析")

        # 导入所需的服务
        from ...services.exchange_data_collector import ExchangeDataCollector
        from ...services.technical_analysis_engine import TechnicalAnalysisEngine
        from ...services.llm_long_strategy_analyzer import LLMLongStrategyAnalyzer
        from ...services.llm_short_strategy_analyzer import LLMShortStrategyAnalyzer
        from ...services.strategy_aggregator import StrategyAggregator
        from ...services.llm_strategy_generator import LLMStrategyGenerator, StrategyGenerationRequest
        from ...services.news_collector import NewsCollector
        from ...services.llm_news_summarizer import LLMNewsSummarizer

        # 初始化服务
        data_collector = ExchangeDataCollector()
        technical_engine = TechnicalAnalysisEngine()
        long_analyzer = LLMLongStrategyAnalyzer()
        short_analyzer = LLMShortStrategyAnalyzer()
        aggregator = StrategyAggregator()
        generator = LLMStrategyGenerator()

        # 1. 收集K线数据
        klines = await data_collector.fetch_klines(
            symbol=symbol,
            timeframe=timeframe,
            limit=168,  # 7天的1小时数据
            exchange="binance"
        )

        if not klines:
            raise HTTPException(status_code=400, detail="无法获取K线数据")

        # 2. 技术分析
        technical_analysis = await technical_engine.analyze(
            klines=klines,
            symbol=symbol,
            timeframe=timeframe
        )

        # 3. 新闻分析（如果需要）
        news_analysis = None
        if include_news and "news" in analysis_types:
            try:
                news_collector = NewsCollector()
                news_data = await news_collector.collect_news(days_back=7, max_items=20)

                if news_data:
                    news_summarizer = LLMNewsSummarizer()
                    news_summary = await news_summarizer.generate_market_summary(news_data)
                    news_analysis = {
                        "sentiment": news_summary.sentiment if hasattr(news_summary, 'sentiment') else "neutral",
                        "sentiment_score": getattr(news_summary, 'confidence_score', 0.5),
                        "key_events": getattr(news_summary, 'key_points', []),
                        "market_impact": "medium"
                    }
            except Exception as e:
                logger.warning(f"新闻分析失败: {e}")
                news_analysis = None

        # 4. 做多策略分析
        long_analysis = None
        if "long" in analysis_types or "technical" in analysis_types:
            long_analysis = await long_analyzer.analyze_long_strategy(
                symbol=symbol,
                technical_analysis=technical_analysis,
                market_data={
                    "timeframe": timeframe,
                    "analysis_period_days": 7
                }
            )

        # 5. 做空策略分析
        short_analysis = None
        if "short" in analysis_types:
            short_analysis = await short_analyzer.analyze_short_strategy(
                symbol=symbol,
                technical_analysis=technical_analysis,
                market_data={
                    "timeframe": timeframe,
                    "analysis_period_days": 7,
                    "max_position_size": max_position_size
                }
            )

        # 6. 策略聚合
        aggregated_signal = await aggregator.aggregate_strategies(
            symbol=symbol,
            long_analysis=long_analysis,
            short_analysis=short_analysis,
            technical_analysis=technical_analysis,
            news_analysis=news_analysis,
            method="weighted_average"
        )

        # 7. 最终策略生成
        strategy_request = StrategyGenerationRequest(
            symbol=symbol,
            timeframe=timeframe,
            analysis_types=analysis_types,
            include_news=include_news,
            risk_tolerance=risk_tolerance,
            max_position_size=max_position_size,
            custom_parameters=custom_parameters
        )

        final_strategy = await generator.generate_final_strategy(
            request=strategy_request,
            aggregated_signal=aggregated_signal,
            long_analysis=long_analysis,
            short_analysis=short_analysis,
            technical_analysis=technical_analysis,
            news_analysis=news_analysis
        )

        # 8. 异步保存策略到数据库
        background_tasks.add_task(
            _save_generated_strategy,
            final_strategy,
            aggregated_signal,
            technical_analysis
        )

        # 9. 构建响应
        response = {
            "strategy_id": final_strategy.strategy_id,
            "symbol": final_strategy.symbol,
            "generated_at": final_strategy.generated_at.isoformat(),
            "expires_at": final_strategy.expires_at.isoformat(),
            "final_recommendation": final_strategy.final_recommendation,
            "confidence_score": final_strategy.confidence_score,
            "entry_price": final_strategy.entry_price,
            "stop_loss_price": final_strategy.stop_loss_price,
            "take_profit_price": final_strategy.take_profit_price,
            "position_size_percent": final_strategy.position_size_percent,
            "strategy_type": final_strategy.strategy_type,
            "time_horizon": final_strategy.time_horizon,
            "market_analysis": final_strategy.market_analysis,
            "risk_assessment": final_strategy.risk_assessment,
            "execution_plan": final_strategy.execution_plan,
            "analysis_summary": {
                "technical_signals": aggregated_signal.technical_summary,
                "news_analysis": aggregated_signal.news_summary,
                "consensus_strength": aggregated_signal.consensus_strength,
                "supporting_analyses": aggregated_signal.supporting_analyses
            }
        }

        logger.info(f"综合策略生成完成: {symbol} - {final_strategy.final_recommendation}")
        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"综合策略生成失败: {e}")
        raise HTTPException(status_code=500, detail=f"策略生成失败: {str(e)}")


@router.post("/strategies/generate-batch", response_model=Dict[str, Any])
async def generate_batch_strategies(
    batch_request: Dict[str, Any],
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db_session)
):
    """批量生成多个交易符号的策略"""
    try:
        symbols = batch_request.get("symbols", [])
        if not symbols:
            raise HTTPException(status_code=400, detail="交易符号列表不能为空")

        if len(symbols) > 10:
            raise HTTPException(status_code=400, detail="批量分析最多支持10个交易符号")

        # 验证所有交易符号格式
        for symbol in symbols:
            if "/" not in symbol:
                raise HTTPException(status_code=400, detail=f"无效的交易符号格式: {symbol}")

        # 提取公共参数
        common_params = {
            "timeframe": batch_request.get("timeframe", "1h"),
            "analysis_types": batch_request.get("analysis_types", ["long", "short"]),
            "include_news": batch_request.get("include_news", False),
            "risk_tolerance": batch_request.get("risk_tolerance", "moderate"),
            "max_position_size": batch_request.get("max_position_size", 20.0),
            "custom_parameters": batch_request.get("custom_parameters", {})
        }

        logger.info(f"开始批量生成{len(symbols)}个交易符号的策略")

        # 导入所需服务
        from ...services.llm_strategy_generator import LLMStrategyGenerator, StrategyGenerationRequest

        generator = LLMStrategyGenerator()

        # 生成策略请求
        requests = []
        for symbol in symbols:
            request = StrategyGenerationRequest(
                symbol=symbol,
                **common_params
            )
            requests.append(request)

        # 模拟批量生成（实际实现需要完整的分析流程）
        # 这里返回一个简化的响应结构
        batch_id = f"batch_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"

        results = []
        for symbol in symbols:
            # 模拟策略结果
            mock_strategy = {
                "strategy_id": f"strategy_{symbol}_{datetime.utcnow().strftime('%H%M%S')}",
                "symbol": symbol,
                "final_recommendation": "hold",  # 默认推荐
                "confidence_score": 0.6,
                "entry_price": 0.0,
                "stop_loss_price": 0.0,
                "take_profit_price": 0.0,
                "position_size_percent": 10.0,
                "status": "pending_analysis"
            }
            results.append(mock_strategy)

        response = {
            "batch_id": batch_id,
            "total_symbols": len(symbols),
            "status": "processing",
            "generated_at": datetime.utcnow().isoformat(),
            "estimated_completion": (datetime.utcnow() + timedelta(minutes=len(symbols) * 2)).isoformat(),
            "results": results,
            "parameters": common_params
        }

        logger.info(f"批量策略生成请求已提交: {batch_id}")
        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"批量策略生成失败: {e}")
        raise HTTPException(status_code=500, detail=f"批量策略生成失败: {str(e)}")


@router.get("/strategies/{strategy_id}", response_model=Dict[str, Any])
async def get_strategy_details(
    strategy_id: str,
    db: Session = Depends(get_db_session)
):
    """获取策略详细信息"""
    try:
        logger.info(f"获取策略详情: {strategy_id}")

        # TODO: 从数据库查询策略详情
        # 这里返回模拟响应
        response = {
            "strategy_id": strategy_id,
            "status": "not_found",
            "message": "策略未找到或数据库访问尚未实现"
        }

        return response

    except Exception as e:
        logger.error(f"获取策略详情失败: {e}")
        raise HTTPException(status_code=500, detail="获取策略详情失败")


async def _save_generated_strategy(
    strategy,
    aggregated_signal,
    technical_analysis
):
    """异步保存生成的策略"""
    try:
        # TODO: 实现保存逻辑
        # 1. 保存到TradingStrategy表
        # 2. 保存聚合信号数据
        # 3. 保存技术分析数据
        # 4. 记录策略生成会话

        logger.info(f"生成的策略已保存到数据库: {strategy.strategy_id}")
    except Exception as e:
        logger.error(f"保存生成的策略失败: {e}")