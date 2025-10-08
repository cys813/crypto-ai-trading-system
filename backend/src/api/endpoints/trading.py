"""
交易API端点

处理交易策略、订单和持仓相关的API。
"""

import uuid
from typing import List, Optional, Dict, Any
from datetime import datetime
from decimal import Decimal

from fastapi import APIRouter, Depends, HTTPException, Query, BackgroundTasks
from pydantic import BaseModel, Field, validator
from sqlalchemy.orm import Session

from ...core.database import get_db
from ...core.auth import get_current_user
from ...core.logging import APILogger
from ...models.user import User
from ...models.trading_strategy import TradingStrategy
from ...models.trading_order import TradingOrder, OrderSide, OrderType, OrderStatus
from ...models.position import Position, PositionStatus
from ...services.trading_executor import TradingExecutor, TradingSignal, ExecutionConfig, ExecutionMode
from ...services.order_manager import OrderManager, OrderRequest
from ...services.risk_manager import RiskManager, RiskAssessment
from ...services.dynamic_fund_manager import DynamicFundManager, FundManagementConfig
from ...services.position_monitor import PositionMonitor, MonitorConfig

router = APIRouter()
api_logger = APILogger("trading_api")


# Pydantic模型
class OrderRequestModel(BaseModel):
    """订单请求模型"""
    symbol: str = Field(..., description="交易符号")
    side: str = Field(..., regex="^(buy|sell)$", description="订单方向")
    order_type: str = Field(..., regex="^(market|limit|stop|stop_limit)$", description="订单类型")
    amount: float = Field(..., gt=0, description="订单数量")
    price: Optional[float] = Field(None, gt=0, description="订单价格")
    stop_loss_price: Optional[float] = Field(None, gt=0, description="止损价格")
    take_profit_price: Optional[float] = Field(None, gt=0, description="止盈价格")
    time_in_force: str = Field("GTC", regex="^(GTC|IOC|FOK|GTD)$", description="订单有效期")
    expire_time: Optional[datetime] = Field(None, description="过期时间")
    metadata: Optional[Dict[str, Any]] = Field(None, description="元数据")


class StrategyExecutionRequest(BaseModel):
    """策略执行请求模型"""
    strategy_id: str = Field(..., description="策略ID")
    execution_mode: str = Field("simulation", regex="^(simulation|paper|live)$", description="执行模式")
    force_execution: bool = Field(False, description="强制执行")


class TradingSignalRequest(BaseModel):
    """交易信号请求模型"""
    strategy_id: str = Field(..., description="策略ID")
    symbol: str = Field(..., description="交易符号")
    action: str = Field(..., regex="^(buy|sell|hold)$", description="交易动作")
    confidence: float = Field(..., ge=0, le=1, description="置信度")
    entry_price: Optional[float] = Field(None, gt=0, description="入场价格")
    stop_loss_price: Optional[float] = Field(None, gt=0, description="止损价格")
    take_profit_price: Optional[float] = Field(None, gt=0, description="止盈价格")
    position_size_percent: Optional[float] = Field(None, gt=0, le=100, description="仓位大小百分比")
    expiration_time: Optional[datetime] = Field(None, description="过期时间")
    metadata: Optional[Dict[str, Any]] = Field(None, description="元数据")


class FundAllocationRequest(BaseModel):
    """资金分配请求模型"""
    strategy_id: str = Field(..., description="策略ID")
    amount: float = Field(..., gt=0, description="分配金额")


class RiskControlRequest(BaseModel):
    """风险控制请求模型"""
    enable_emergency_stop: bool = Field(False, description="启用紧急停止")
    max_position_size_percent: Optional[float] = Field(None, gt=0, le=100, description="最大持仓百分比")
    max_daily_trades: Optional[int] = Field(None, gt=0, description="每日最大交易次数")


# 响应模型
class OrderResponse(BaseModel):
    """订单响应模型"""
    success: bool
    order_id: Optional[str] = None
    exchange_order_id: Optional[str] = None
    error_code: Optional[str] = None
    error_message: Optional[str] = None
    filled_amount: Optional[float] = None
    filled_price: Optional[float] = None
    fee: Optional[float] = None


class ExecutionResponse(BaseModel):
    """执行响应模型"""
    success: bool
    strategy_id: str
    orders_created: List[str] = []
    positions_opened: List[str] = []
    error_message: Optional[str] = None
    execution_time: Optional[datetime] = None
    execution_details: Dict[str, Any] = {}


# 依赖注入
def get_trading_executor() -> TradingExecutor:
    """获取交易执行器实例"""
    # 这里应该从应用状态获取配置的执行器
    config = ExecutionConfig(mode=ExecutionMode.SIMULATION)
    return TradingExecutor(config)


def get_order_manager() -> OrderManager:
    """获取订单管理器实例"""
    return OrderManager()


def get_risk_manager() -> RiskManager:
    """获取风险管理器实例"""
    return RiskManager()


# 策略相关端点
@router.get("/strategies", response_model=Dict[str, Any])
async def get_trading_strategies(
    symbol: Optional[str] = Query(None, description="过滤交易符号"),
    status: Optional[str] = Query(None, description="过滤状态"),
    limit: int = Query(100, ge=1, le=1000, description="限制数量"),
    offset: int = Query(0, ge=0, description="偏移量"),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """获取交易策略列表"""
    try:
        query = db.query(TradingStrategy).filter(
            TradingStrategy.created_by == current_user.id
        )

        if symbol:
            query = query.join(TradingStrategy.symbol).filter(
                TradingStrategy.symbol.has(symbol=symbol)
            )

        if status:
            query = query.filter(TradingStrategy.status == status)

        total = query.count()
        strategies = query.order_by(TradingStrategy.created_at.desc()).offset(offset).limit(limit).all()

        return {
            "strategies": [strategy.to_dict() for strategy in strategies],
            "total": total,
            "offset": offset,
            "limit": limit
        }

    except Exception as e:
        api_logger.log_error("get_trading_strategies", str(e))
        raise HTTPException(status_code=500, detail="获取策略列表失败")


@router.post("/strategies/{strategy_id}/execute", response_model=ExecutionResponse)
async def execute_strategy(
    strategy_id: str,
    request: StrategyExecutionRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
    executor: TradingExecutor = Depends(get_trading_executor)
):
    """执行交易策略"""
    try:
        # 配置执行器
        executor.config.mode = ExecutionMode(request.execution_mode)

        # 初始化服务（如果需要）
        if not executor.fund_manager:
            total_funds = Decimal('10000')  # 应该从用户账户获取
            await executor.initialize_services(str(current_user.id), total_funds)

        # 执行策略
        result = await executor.execute_strategy(strategy_id, str(current_user.id), db)

        # 记录API调用
        api_logger.log_info(
            "strategy_executed",
            user_id=str(current_user.id),
            strategy_id=strategy_id,
            mode=request.execution_mode,
            success=result.success
        )

        return ExecutionResponse(
            success=result.success,
            strategy_id=result.strategy_id,
            orders_created=result.orders_created,
            positions_opened=result.positions_opened,
            error_message=result.error_message,
            execution_time=result.execution_time,
            execution_details=result.execution_details
        )

    except Exception as e:
        api_logger.log_error("execute_strategy", str(e))
        raise HTTPException(status_code=500, detail="策略执行失败")


@router.post("/signals/execute", response_model=ExecutionResponse)
async def execute_trading_signal(
    request: TradingSignalRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
    executor: TradingExecutor = Depends(get_trading_executor)
):
    """直接执行交易信号"""
    try:
        # 创建交易信号
        signal = TradingSignal(
            strategy_id=request.strategy_id,
            symbol=request.symbol,
            action=request.action,
            confidence=request.confidence,
            entry_price=Decimal(str(request.entry_price)) if request.entry_price else None,
            stop_loss_price=Decimal(str(request.stop_loss_price)) if request.stop_loss_price else None,
            take_profit_price=Decimal(str(request.take_profit_price)) if request.take_profit_price else None,
            position_size_percent=request.position_size_percent,
            expiration_time=request.expiration_time,
            metadata=request.metadata or {}
        )

        # 执行信号
        result = await executor.execute_signal(signal, str(current_user.id), db)

        # 记录API调用
        api_logger.log_info(
            "signal_executed",
            user_id=str(current_user.id),
            strategy_id=request.strategy_id,
            symbol=request.symbol,
            action=request.action,
            success=result.success
        )

        return ExecutionResponse(
            success=result.success,
            strategy_id=result.strategy_id,
            orders_created=result.orders_created,
            positions_opened=result.positions_opened,
            error_message=result.error_message,
            execution_time=result.execution_time,
            execution_details=result.execution_details
        )

    except Exception as e:
        api_logger.log_error("execute_trading_signal", str(e))
        raise HTTPException(status_code=500, detail="信号执行失败")


# 订单相关端点
@router.post("/orders", response_model=OrderResponse)
async def create_trading_order(
    request: OrderRequestModel,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
    order_manager: OrderManager = Depends(get_order_manager),
    risk_manager: RiskManager = Depends(get_risk_manager)
):
    """创建交易订单"""
    try:
        # 风险检查
        risk_assessment = await risk_manager.assess_order_risk(
            request.dict(),
            str(current_user.id),
            db
        )

        if risk_assessment.risk_level.value == 'critical':
            return OrderResponse(
                success=False,
                error_code="RISK_TOO_HIGH",
                error_message=f"风险等级过高: {risk_assessment.risk_level.value}"
            )

        # 创建订单请求
        order_request = OrderRequest(
            symbol=request.symbol,
            side=request.side,
            order_type=request.order_type,
            amount=Decimal(str(request.amount)),
            price=Decimal(str(request.price)) if request.price else None,
            stop_loss_price=Decimal(str(request.stop_loss_price)) if request.stop_loss_price else None,
            take_profit_price=Decimal(str(request.take_profit_price)) if request.take_profit_price else None,
            time_in_force=request.time_in_force,
            expire_time=request.expire_time,
            user_id=str(current_user.id),
            metadata=request.metadata
        )

        # 执行订单
        result = await order_manager.create_order(order_request, db)

        # 记录API调用
        api_logger.log_info(
            "order_created",
            user_id=str(current_user.id),
            symbol=request.symbol,
            side=request.side,
            amount=request.amount,
            success=result.success
        )

        return OrderResponse(
            success=result.success,
            order_id=result.order_id,
            exchange_order_id=result.exchange_order_id,
            error_code=result.error_code,
            error_message=result.error_message,
            filled_amount=float(result.filled_amount) if result.filled_amount else None,
            filled_price=float(result.filled_price) if result.filled_price else None,
            fee=float(result.fee) if result.fee else None
        )

    except Exception as e:
        api_logger.log_error("create_trading_order", str(e))
        raise HTTPException(status_code=500, detail="订单创建失败")


@router.get("/orders", response_model=Dict[str, Any])
async def get_trading_orders(
    symbol: Optional[str] = Query(None, description="过滤交易符号"),
    status: Optional[str] = Query(None, description="过滤状态"),
    limit: int = Query(100, ge=1, le=1000, description="限制数量"),
    offset: int = Query(0, ge=0, description="偏移量"),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """获取交易订单列表"""
    try:
        query = db.query(TradingOrder).filter(
            TradingOrder.user_id == current_user.id
        )

        if symbol:
            query = query.filter(TradingOrder.symbol == symbol)

        if status:
            query = query.filter(TradingOrder.status == status)

        total = query.count()
        orders = query.order_by(TradingOrder.created_at.desc()).offset(offset).limit(limit).all()

        return {
            "orders": [order.to_dict() for order in orders],
            "total": total,
            "offset": offset,
            "limit": limit
        }

    except Exception as e:
        api_logger.log_error("get_trading_orders", str(e))
        raise HTTPException(status_code=500, detail="获取订单列表失败")


@router.delete("/orders/{order_id}", response_model=Dict[str, Any])
async def cancel_trading_order(
    order_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
    order_manager: OrderManager = Depends(get_order_manager)
):
    """取消交易订单"""
    try:
        result = await order_manager.cancel_order(order_id, str(current_user.id), db)

        # 记录API调用
        api_logger.log_info(
            "order_cancelled",
            user_id=str(current_user.id),
            order_id=order_id,
            success=result.success
        )

        return {
            "success": result.success,
            "order_id": result.order_id,
            "error_code": result.error_code,
            "error_message": result.error_message
        }

    except Exception as e:
        api_logger.log_error("cancel_trading_order", str(e))
        raise HTTPException(status_code=500, detail="订单取消失败")


@router.get("/orders/{order_id}", response_model=Dict[str, Any])
async def get_order_status(
    order_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
    order_manager: OrderManager = Depends(get_order_manager)
):
    """获取订单状态"""
    try:
        order_status = await order_manager.get_order_status(order_id, db)

        if not order_status:
            raise HTTPException(status_code=404, detail="订单不存在")

        # 验证订单所有权
        if order_status.get('user_id') and str(order_status['user_id']) != str(current_user.id):
            raise HTTPException(status_code=403, detail="无权限访问此订单")

        return order_status

    except HTTPException:
        raise
    except Exception as e:
        api_logger.log_error("get_order_status", str(e))
        raise HTTPException(status_code=500, detail="获取订单状态失败")


# 持仓相关端点
@router.get("/positions", response_model=Dict[str, Any])
async def get_positions(
    symbol: Optional[str] = Query(None, description="过滤交易符号"),
    status: Optional[str] = Query(None, description="过滤状态"),
    limit: int = Query(100, ge=1, le=1000, description="限制数量"),
    offset: int = Query(0, ge=0, description="偏移量"),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """获取持仓信息"""
    try:
        query = db.query(Position).filter(
            Position.user_id == current_user.id
        )

        if symbol:
            query = query.filter(Position.symbol == symbol)

        if status:
            query = query.filter(Position.status == status)

        total = query.count()
        positions = query.order_by(Position.opened_at.desc()).offset(offset).limit(limit).all()

        # 计算总价值
        total_value = sum(
            float(pos.current_value or 0) for pos in positions
        )
        total_pnl = sum(
            float(pos.total_pnl or 0) for pos in positions
        )

        return {
            "positions": [position.to_dict() for position in positions],
            "total": total,
            "total_value": total_value,
            "total_pnl": total_pnl,
            "offset": offset,
            "limit": limit
        }

    except Exception as e:
        api_logger.log_error("get_positions", str(e))
        raise HTTPException(status_code=500, detail="获取持仓信息失败")


@router.get("/positions/{position_id}", response_model=Dict[str, Any])
async def get_position_details(
    position_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """获取持仓详情"""
    try:
        position = db.query(Position).filter(
            and_(
                Position.id == uuid.UUID(position_id),
                Position.user_id == current_user.id
            )
        ).first()

        if not position:
            raise HTTPException(status_code=404, detail="持仓不存在")

        return position.to_dict()

    except HTTPException:
        raise
    except Exception as e:
        api_logger.log_error("get_position_details", str(e))
        raise HTTPException(status_code=500, detail="获取持仓详情失败")


# 投资组合相关端点
@router.get("/portfolio", response_model=Dict[str, Any])
async def get_portfolio_summary(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """获取投资组合摘要"""
    try:
        # 获取用户持仓
        positions = db.query(Position).filter(
            and_(
                Position.user_id == current_user.id,
                Position.status == PositionStatus.OPEN.value
            )
        ).all()

        # 计算总价值和盈亏
        total_value = sum(
            float(pos.current_value or 0) for pos in positions
        )
        total_pnl = sum(
            float(pos.total_pnl or 0) for pos in positions
        )

        # 获取今日交易统计
        today = datetime.utcnow().date()
        today_orders = db.query(TradingOrder).filter(
            and_(
                TradingOrder.user_id == current_user.id,
                func.date(TradingOrder.created_at) == today
            )
        ).count()

        # 计算资产配置
        asset_allocation = {}
        for position in positions:
            symbol = position.symbol
            value = float(position.current_value or 0)
            asset_allocation[symbol] = asset_allocation.get(symbol, 0) + value

        # 获取账户总价值（包括现金）
        total_account_value = total_value + 1000  # 假设有1000现金

        return {
            "total_value": total_account_value,
            "total_pnl": total_pnl,
            "pnl_percentage": (total_pnl / total_account_value * 100) if total_account_value > 0 else 0,
            "positions_count": len(positions),
            "today_trades": today_orders,
            "asset_allocation": asset_allocation,
            "last_updated": datetime.utcnow()
        }

    except Exception as e:
        api_logger.log_error("get_portfolio_summary", str(e))
        raise HTTPException(status_code=500, detail="获取投资组合摘要失败")


# 风险管理端点
@router.get("/risk/assessment", response_model=Dict[str, Any])
async def get_risk_assessment(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
    risk_manager: RiskManager = Depends(get_risk_manager)
):
    """获取风险评估"""
    try:
        assessment = await risk_manager.check_risk_limits(str(current_user.id), db)

        return {
            "risk_level": assessment.risk_level.value,
            "risk_score": assessment.risk_score,
            "risk_metrics": {
                "total_exposure_percent": assessment.risk_metrics.total_exposure_percent,
                "leverage_ratio": assessment.risk_metrics.leverage_ratio,
                "daily_pnl_percent": assessment.risk_metrics.daily_pnl_percent,
                "current_drawdown": assessment.risk_metrics.current_drawdown,
                "var_95": assessment.risk_metrics.var_95,
                "position_concentration": assessment.risk_metrics.position_concentration
            },
            "warnings": assessment.warnings,
            "recommendations": assessment.recommendations,
            "required_actions": [action.value for action in assessment.required_actions]
        }

    except Exception as e:
        api_logger.log_error("get_risk_assessment", str(e))
        raise HTTPException(status_code=500, detail="获取风险评估失败")


@router.post("/risk/control", response_model=Dict[str, Any])
async def set_risk_control(
    request: RiskControlRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user)
):
    """设置风险控制"""
    try:
        # 这里应该实现风险控制设置逻辑
        # 包括紧急停止、限制设置等

        if request.enable_emergency_stop:
            # 启动紧急停止
            background_tasks.add_task(
                api_logger.log_warning,
                "emergency_stop_activated",
                user_id=str(current_user.id)
            )

        return {
            "success": True,
            "message": "风险控制设置成功",
            "settings": request.dict()
        }

    except Exception as e:
        api_logger.log_error("set_risk_control", str(e))
        raise HTTPException(status_code=500, detail="设置风险控制失败")


# 监控端点
@router.get("/monitoring/stats", response_model=Dict[str, Any])
async def get_monitoring_stats(
    current_user: User = Depends(get_current_user)
):
    """获取监控统计"""
    try:
        # 这里应该从监控服务获取统计信息
        return {
            "active_monitors": 0,
            "alerts_today": 0,
            "actions_executed": 0,
            "last_update": datetime.utcnow()
        }

    except Exception as e:
        api_logger.log_error("get_monitoring_stats", str(e))
        raise HTTPException(status_code=500, detail="获取监控统计失败")


@router.post("/monitoring/start", response_model=Dict[str, Any])
async def start_monitoring(
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user)
):
    """启动监控"""
    try:
        # 这里应该启动持仓监控
        background_tasks.add_task(
            api_logger.log_info,
            "monitoring_started",
            user_id=str(current_user.id)
        )

        return {
            "success": True,
            "message": "监控已启动"
        }

    except Exception as e:
        api_logger.log_error("start_monitoring", str(e))
        raise HTTPException(status_code=500, detail="启动监控失败")


@router.post("/monitoring/stop", response_model=Dict[str, Any])
async def stop_monitoring(
    current_user: User = Depends(get_current_user)
):
    """停止监控"""
    try:
        # 这里应该停止持仓监控
        await api_logger.log_info(
            "monitoring_stopped",
            user_id=str(current_user.id)
        )

        return {
            "success": True,
            "message": "监控已停止"
        }

    except Exception as e:
        api_logger.log_error("stop_monitoring", str(e))
        raise HTTPException(status_code=500, detail="停止监控失败")