"""
交易所集成服务

提供统一的交易所API接口，支持多个主流交易所的API调用。
"""

import logging
import asyncio
from typing import Dict, Any, List, Optional, Union, Callable
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
import time
from datetime import datetime, timedelta

import ccxt.async_support as ccxt
from ccxt.base.errors import ExchangeError, NetworkError, RateLimitExceeded

from .config import settings
from .exceptions import ExchangeAPIError, ExternalServiceError
from .logging import BusinessLogger

logger = logging.getLogger(__name__)
business_logger = BusinessLogger("exchange_integration")


class ExchangeType(Enum):
    """交易所类型"""
    SPOT = "spot"
    FUTURES = "futures"
    MARGIN = "margin"
    SWAP = "swap"


class OrderSide(Enum):
    """订单方向"""
    BUY = "buy"
    SELL = "sell"


class OrderType(Enum):
    """订单类型"""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"
    STOP_MARKET = "stop_market"


class OrderStatus(Enum):
    """订单状态"""
    OPEN = "open"
    CLOSED = "closed"
    CANCELED = "canceled"
    EXPIRED = "expired"
    FAILED = "failed"


@dataclass
class ExchangeCredentials:
    """交易所凭据"""
    api_key: str
    secret: str
    passphrase: Optional[str] = None
    sandbox: bool = False


@dataclass
class TradingSymbol:
    """交易符号信息"""
    symbol: str
    base: str
    quote: str
    active: bool = True
    spot: bool = True
    margin: bool = False
    futures: bool = False
    contract_size: Optional[float] = None
    precision: Dict[str, int] = None
    limits: Dict[str, Dict[str, float]] = None
    fees: Dict[str, float] = None

    def __post_init__(self):
        if self.precision is None:
            self.precision = {"amount": 8, "price": 8}
        if self.limits is None:
            self.limits = {"amount": {"min": 0.0, "max": float('inf')}, "price": {"min": 0.0, "max": float('inf')}}
        if self.fees is None:
            self.fees = {"maker": 0.001, "taker": 0.001}


@dataclass
class KlineData:
    """K线数据"""
    symbol: str
    timeframe: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    quote_volume: Optional[float] = None
    trades_count: Optional[int] = None


@dataclass
class OrderRequest:
    """订单请求"""
    symbol: str
    side: OrderSide
    type: OrderType
    amount: float
    price: Optional[float] = None
    stop_price: Optional[float] = None
    time_in_force: Optional[str] = None
    client_order_id: Optional[str] = None
    params: Optional[Dict[str, Any]] = None


@dataclass
class OrderResponse:
    """订单响应"""
    id: str
    client_order_id: Optional[str]
    symbol: str
    side: OrderSide
    type: OrderType
    amount: float
    price: Optional[float]
    filled: float
    remaining: float
    status: OrderStatus
    timestamp: datetime
    fee: Optional[float] = None
    fee_currency: Optional[str] = None
    trades: Optional[List[Dict[str, Any]]] = None


class BaseExchangeClient(ABC):
    """交易所客户端基类"""

    def __init__(self, exchange_id: str, credentials: Optional[ExchangeCredentials] = None):
        self.exchange_id = exchange_id
        self.credentials = credentials
        self.client = None
        self.logger = logger
        self.business_logger = business_logger
        self._initialize_client()

    @abstractmethod
    def _initialize_client(self):
        """初始化交易所客户端"""
        pass

    @abstractmethod
    async def get_markets(self) -> Dict[str, TradingSymbol]:
        """获取市场信息"""
        pass

    @abstractmethod
    async def fetch_klines(
        self,
        symbol: str,
        timeframe: str,
        limit: int = 100,
        since: Optional[datetime] = None
    ) -> List[KlineData]:
        """获取K线数据"""
        pass

    @abstractmethod
    async def create_order(self, request: OrderRequest) -> OrderResponse:
        """创建订单"""
        pass

    @abstractmethod
    async def cancel_order(self, order_id: str, symbol: str) -> bool:
        """取消订单"""
        pass

    @abstractmethod
    async def fetch_order(self, order_id: str, symbol: str) -> OrderResponse:
        """获取订单信息"""
        pass

    @abstractmethod
    async def fetch_balance(self) -> Dict[str, Dict[str, float]]:
        """获取账户余额"""
        pass

    async def test_connection(self) -> bool:
        """测试连接"""
        try:
            await self.client.fetch_markets()
            return True
        except Exception as e:
            self.logger.error(f"测试{self.exchange_id}连接失败: {e}")
            return False


class BinanceClient(BaseExchangeClient):
    """币安客户端"""

    def _initialize_client(self):
        """初始化币安客户端"""
        try:
            if self.credentials:
                self.client = ccxt.binance({
                    'apiKey': self.credentials.api_key,
                    'secret': self.credentials.secret,
                    'sandbox': self.credentials.sandbox,
                    'enableRateLimit': True,
                    'options': {
                        'defaultType': 'spot',
                    }
                })
            else:
                self.client = ccxt.binance({
                    'enableRateLimit': True,
                    'options': {
                        'defaultType': 'spot',
                    }
                })

            self.logger.info("Binance客户端初始化成功")
        except Exception as e:
            raise ExchangeAPIError(
                message=f"Binance客户端初始化失败: {str(e)}",
                error_code="BINANCE_INIT_FAILED",
                cause=e
            )

    async def get_markets(self) -> Dict[str, TradingSymbol]:
        """获取市场信息"""
        try:
            markets = await self.client.fetch_markets()
            result = {}

            for market in markets:
                if market['active']:
                    symbol_info = TradingSymbol(
                        symbol=market['symbol'],
                        base=market['base'],
                        quote=market['quote'],
                        active=market['active'],
                        spot=market['type'] == 'spot',
                        margin=market['type'] == 'margin',
                        futures=market['type'] in ['future', 'swap'],
                        precision=market['precision'],
                        limits=market['limits'],
                        fees=market.get('fees', {'maker': 0.001, 'taker': 0.001})
                    )
                    result[market['symbol']] = symbol_info

            return result

        except Exception as e:
            self.logger.error(f"获取Binance市场信息失败: {e}")
            raise ExchangeAPIError(
                message=f"获取市场信息失败: {str(e)}",
                error_code="FETCH_MARKETS_FAILED",
                cause=e
            )

    async def fetch_klines(
        self,
        symbol: str,
        timeframe: str,
        limit: int = 100,
        since: Optional[datetime] = None
    ) -> List[KlineData]:
        """获取K线数据"""
        try:
            since_ms = int(since.timestamp() * 1000) if since else None

            ohlcv = await self.client.fetch_ohlcv(
                symbol,
                timeframe,
                limit=limit,
                since=since_ms
            )

            klines = []
            for candle in ohlcv:
                kline = KlineData(
                    symbol=symbol,
                    timeframe=timeframe,
                    timestamp=datetime.fromtimestamp(candle[0] / 1000),
                    open=float(candle[1]),
                    high=float(candle[2]),
                    low=float(candle[3]),
                    close=float(candle[4]),
                    volume=float(candle[5]),
                    quote_volume=float(candle[6]) if len(candle) > 6 else None
                )
                klines.append(kline)

            return klines

        except Exception as e:
            self.logger.error(f"获取Binance K线数据失败: {e}")
            raise ExchangeAPIError(
                message=f"获取K线数据失败: {str(e)}",
                error_code="FETCH_KLINES_FAILED",
                cause=e
            )

    async def create_order(self, request: OrderRequest) -> OrderResponse:
        """创建订单"""
        try:
            order_params = {
                'symbol': request.symbol,
                'type': request.type.value,
                'side': request.side.value,
                'amount': request.amount,
            }

            if request.price is not None:
                order_params['price'] = request.price

            if request.stop_price is not None:
                order_params['stopPrice'] = request.stop_price

            if request.time_in_force:
                order_params['timeInForce'] = request.time_in_force

            if request.client_order_id:
                order_params['clientOrderId'] = request.client_order_id

            if request.params:
                order_params.update(request.params)

            order = await self.client.create_order(**order_params)

            return OrderResponse(
                id=str(order['id']),
                client_order_id=order.get('clientOrderId'),
                symbol=order['symbol'],
                side=OrderSide(order['side']),
                type=OrderType(order['type']),
                amount=float(order['amount']),
                price=float(order['price']) if order.get('price') else None,
                filled=float(order['filled']),
                remaining=float(order['remaining']),
                status=OrderStatus(order['status']),
                timestamp=datetime.fromtimestamp(order['timestamp'] / 1000),
                fee=float(order['fee']['cost']) if order.get('fee') else None,
                fee_currency=order['fee']['currency'] if order.get('fee') else None,
                trades=order.get('trades', [])
            )

        except Exception as e:
            self.logger.error(f"创建Binance订单失败: {e}")
            raise ExchangeAPIError(
                message=f"创建订单失败: {str(e)}",
                error_code="CREATE_ORDER_FAILED",
                cause=e
            )

    async def cancel_order(self, order_id: str, symbol: str) -> bool:
        """取消订单"""
        try:
            await self.client.cancel_order(order_id, symbol)
            return True
        except Exception as e:
            self.logger.error(f"取消Binance订单失败: {e}")
            return False

    async def fetch_order(self, order_id: str, symbol: str) -> OrderResponse:
        """获取订单信息"""
        try:
            order = await self.client.fetch_order(order_id, symbol)

            return OrderResponse(
                id=str(order['id']),
                client_order_id=order.get('clientOrderId'),
                symbol=order['symbol'],
                side=OrderSide(order['side']),
                type=OrderType(order['type']),
                amount=float(order['amount']),
                price=float(order['price']) if order.get('price') else None,
                filled=float(order['filled']),
                remaining=float(order['remaining']),
                status=OrderStatus(order['status']),
                timestamp=datetime.fromtimestamp(order['timestamp'] / 1000),
                fee=float(order['fee']['cost']) if order.get('fee') else None,
                fee_currency=order['fee']['currency'] if order.get('fee') else None,
                trades=order.get('trades', [])
            )

        except Exception as e:
            self.logger.error(f"获取Binance订单信息失败: {e}")
            raise ExchangeAPIError(
                message=f"获取订单信息失败: {str(e)}",
                error_code="FETCH_ORDER_FAILED",
                cause=e
            )

    async def fetch_balance(self) -> Dict[str, Dict[str, float]]:
        """获取账户余额"""
        try:
            balance = await self.client.fetch_balance()
            return balance
        except Exception as e:
            self.logger.error(f"获取Binance账户余额失败: {e}")
            raise ExchangeAPIError(
                message=f"获取账户余额失败: {str(e)}",
                error_code="FETCH_BALANCE_FAILED",
                cause=e
            )


class ExchangeManager:
    """交易所管理器"""

    def __init__(self):
        self.logger = logger
        self.business_logger = business_logger
        self.clients = {}
        self.supported_exchanges = {
            'binance': BinanceClient,
            # 可以在这里添加更多交易所
        }

    def add_exchange(
        self,
        exchange_id: str,
        credentials: Optional[ExchangeCredentials] = None
    ) -> BaseExchangeClient:
        """添加交易所客户端"""
        if exchange_id not in self.supported_exchanges:
            raise ExchangeAPIError(
                message=f"不支持的交易所: {exchange_id}",
                error_code="UNSUPPORTED_EXCHANGE"
            )

        try:
            client_class = self.supported_exchanges[exchange_id]
            client = client_class(exchange_id, credentials)
            self.clients[exchange_id] = client

            self.logger.info(f"成功添加交易所客户端: {exchange_id}")
            self.business_logger.log_system_event(
                event_type="exchange_client_added",
                severity="info",
                message=f"成功添加交易所客户端: {exchange_id}",
                details={"exchange_id": exchange_id}
            )

            return client

        except Exception as e:
            self.logger.error(f"添加交易所客户端失败: {exchange_id}, 错误: {e}")
            raise

    def get_client(self, exchange_id: str) -> Optional[BaseExchangeClient]:
        """获取交易所客户端"""
        return self.clients.get(exchange_id)

    async def test_all_connections(self) -> Dict[str, bool]:
        """测试所有交易所连接"""
        results = {}

        for exchange_id, client in self.clients.items():
            try:
                results[exchange_id] = await client.test_connection()
            except Exception as e:
                self.logger.error(f"测试{exchange_id}连接失败: {e}")
                results[exchange_id] = False

        return results

    def get_supported_exchanges(self) -> List[str]:
        """获取支持的交易所列表"""
        return list(self.supported_exchanges.keys())

    def get_active_exchanges(self) -> List[str]:
        """获取已激活的交易所列表"""
        return list(self.clients.keys())

    async def get_all_markets(self) -> Dict[str, Dict[str, TradingSymbol]]:
        """获取所有交易所的市场信息"""
        markets = {}

        for exchange_id, client in self.clients.items():
            try:
                exchange_markets = await client.get_markets()
                markets[exchange_id] = exchange_markets
            except Exception as e:
                self.logger.error(f"获取{exchange_id}市场信息失败: {e}")

        return markets

    async def fetch_klines_from_multiple(
        self,
        symbol: str,
        timeframe: str,
        exchanges: Optional[List[str]] = None,
        limit: int = 100
    ) -> Dict[str, List[KlineData]]:
        """从多个交易所获取K线数据"""
        if exchanges is None:
            exchanges = list(self.clients.keys())

        results = {}

        tasks = []
        for exchange_id in exchanges:
            if exchange_id in self.clients:
                task = self._fetch_single_exchange_klines(exchange_id, symbol, timeframe, limit)
                tasks.append(task)

        if tasks:
            responses = await asyncio.gather(*tasks, return_exceptions=True)

            for i, response in enumerate(responses):
                exchange_id = exchanges[i]
                if isinstance(response, Exception):
                    self.logger.error(f"从{exchange_id}获取K线数据失败: {response}")
                else:
                    results[exchange_id] = response

        return results

    async def _fetch_single_exchange_klines(
        self,
        exchange_id: str,
        symbol: str,
        timeframe: str,
        limit: int
    ) -> List[KlineData]:
        """从单个交易所获取K线数据"""
        client = self.clients.get(exchange_id)
        if not client:
            raise ExchangeAPIError(
                message=f"交易所客户端不存在: {exchange_id}",
                error_code="CLIENT_NOT_FOUND"
            )

        return await client.fetch_klines(symbol, timeframe, limit)


# 全局交易所管理器实例
_exchange_manager = None


def get_exchange_manager() -> ExchangeManager:
    """获取交易所管理器实例"""
    global _exchange_manager
    if _exchange_manager is None:
        _exchange_manager = ExchangeManager()
    return _exchange_manager


# 便捷函数
async def get_klines(
    symbol: str,
    timeframe: str,
    exchange: str = "binance",
    limit: int = 100
) -> List[KlineData]:
    """获取K线数据的便捷函数"""
    manager = get_exchange_manager()
    client = manager.get_client(exchange)

    if not client:
        raise ExchangeAPIError(
            message=f"交易所客户端不存在: {exchange}",
            error_code="CLIENT_NOT_FOUND"
        )

    return await client.fetch_klines(symbol, timeframe, limit)


async def create_order(
    symbol: str,
    side: OrderSide,
    order_type: OrderType,
    amount: float,
    price: Optional[float] = None,
    exchange: str = "binance"
) -> OrderResponse:
    """创建订单的便捷函数"""
    manager = get_exchange_manager()
    client = manager.get_client(exchange)

    if not client:
        raise ExchangeAPIError(
            message=f"交易所客户端不存在: {exchange}",
            error_code="CLIENT_NOT_FOUND"
        )

    request = OrderRequest(
        symbol=symbol,
        side=side,
        type=order_type,
        amount=amount,
        price=price
    )

    return await client.create_order(request)