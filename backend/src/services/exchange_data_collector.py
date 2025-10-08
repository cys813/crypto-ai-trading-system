"""
交易所数据收集服务

负责从多个交易所收集K线数据、市场数据和历史数据。
"""

import logging
import asyncio
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass

from ..core.exchange_integration import get_exchange_manager, KlineData, ExchangeCredentials
from ..core.cache import get_cache, CacheKeys
from ..core.database import SessionLocal
from ..core.logging import BusinessLogger
from ..core.exceptions import ExchangeAPIError, ExternalServiceError, ValidationError
from ..models.technical_analysis import KlineData as KlineDataModel
from ..models.market import TradingSymbol

logger = logging.getLogger(__name__)
business_logger = BusinessLogger("exchange_data_collector")


@dataclass
class DataCollectionConfig:
    """数据收集配置"""
    exchange: str
    symbol: str
    timeframe: str
    limit: int = 1000
    since: Optional[datetime] = None
    batch_size: int = 500
    max_retries: int = 3
    retry_delay: int = 5


class ExchangeDataCollector:
    """交易所数据收集器"""

    def __init__(self):
        self.logger = logger
        self.business_logger = business_logger
        self.exchange_manager = get_exchange_manager()
        self.cache = get_cache()

        # 支持的时间框架
        self.supported_timeframes = [
            '1m', '3m', '5m', '15m', '30m',
            '1h', '2h', '4h', '6h', '8h', '12h',
            '1d', '3d', '1w', '1M'
        ]

        # 数据收集限制
        self.max_data_points = 10000
        self.default_limit = 1000

    async def fetch_klines(
        self,
        symbol: str,
        timeframe: str,
        limit: int = None,
        exchange: str = "binance",
        since: Optional[datetime] = None
    ) -> List[KlineData]:
        """获取K线数据"""
        try:
            # 验证参数
            self._validate_kline_request(symbol, timeframe, limit, exchange)

            # 检查缓存
            cache_key = self._get_cache_key(symbol, timeframe, exchange, limit, since)
            cached_data = await self.cache.get(cache_key)
            if cached_data:
                self.logger.info(f"从缓存获取K线数据: {symbol} {timeframe}")
                return [KlineData(**data) for data in cached_data]

            # 从交易所获取数据
            client = self.exchange_manager.get_client(exchange)
            if not client:
                raise ExchangeAPIError(
                    message=f"交易所客户端不存在: {exchange}",
                    error_code="CLIENT_NOT_FOUND"
                )

            # 处理大数据量请求
            actual_limit = min(limit or self.default_limit, self.max_data_points)

            if actual_limit > self.default_limit:
                klines = await self._fetch_large_dataset(
                    client, symbol, timeframe, actual_limit, since
                )
            else:
                klines = await client.fetch_klines(symbol, timeframe, actual_limit, since)

            # 数据验证
            validated_klines = self._validate_klines_data(klines)

            # 缓存数据
            if validated_klines:
                cache_data = [
                    {
                        "symbol": k.symbol,
                        "timeframe": k.timeframe,
                        "timestamp": k.timestamp,
                        "open": k.open,
                        "high": k.high,
                        "low": k.low,
                        "close": k.close,
                        "volume": k.volume,
                        "quote_volume": k.quote_volume,
                        "trades_count": k.trades_count
                    }
                    for k in validated_klines
                ]
                await self.cache.set(cache_key, cache_data, ttl=300)  # 5分钟缓存

            self.business_logger.log_system_event(
                event_type="klines_data_collected",
                severity="info",
                message=f"成功收集K线数据: {symbol} {timeframe} ({len(validated_klines)}条)",
                details={
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "exchange": exchange,
                    "count": len(validated_klines),
                    "data_range": {
                        "start": validated_klines[0].timestamp.isoformat() if validated_klines else None,
                        "end": validated_klines[-1].timestamp.isoformat() if validated_klines else None
                    }
                }
            )

            return validated_klines

        except Exception as e:
            self.logger.error(f"获取K线数据失败: {e}")
            self.business_logger.log_system_event(
                event_type="klines_collection_failed",
                severity="error",
                message=f"获取K线数据失败: {str(e)}",
                details={
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "exchange": exchange,
                    "error": str(e)
                }
            )
            raise

    async def fetch_multiple_symbols(
        self,
        symbols: List[str],
        timeframe: str,
        limit: int = None,
        exchange: str = "binance"
    ) -> Dict[str, List[KlineData]]:
        """获取多个交易符号的K线数据"""
        try:
            self.logger.info(f"开始获取多个符号的K线数据: {symbols}")

            # 并发获取数据
            tasks = []
            for symbol in symbols:
                task = self.fetch_klines(symbol, timeframe, limit, exchange)
                tasks.append(task)

            results = await asyncio.gather(*tasks, return_exceptions=True)

            # 组织结果
            klines_data = {}
            for i, result in enumerate(results):
                symbol = symbols[i]
                if isinstance(result, Exception):
                    self.logger.error(f"获取{symbol}数据失败: {result}")
                    klines_data[symbol] = []
                else:
                    klines_data[symbol] = result

            success_count = sum(1 for data in klines_data.values() if data)
            self.business_logger.log_system_event(
                event_type="multiple_symbols_collection_completed",
                severity="info",
                message=f"多符号数据收集完成: {success_count}/{len(symbols)} 成功",
                details={
                    "symbols": symbols,
                    "timeframe": timeframe,
                    "exchange": exchange,
                    "success_count": success_count,
                    "total_count": len(symbols)
                }
            )

            return klines_data

        except Exception as e:
            self.logger.error(f"获取多个符号K线数据失败: {e}")
            raise

    async def fetch_historical_data(
        self,
        symbol: str,
        timeframe: str,
        start_date: datetime,
        end_date: Optional[datetime] = None,
        exchange: str = "binance"
    ) -> List[KlineData]:
        """获取历史数据"""
        try:
            if not end_date:
                end_date = datetime.utcnow()

            # 计算需要的数据点数量
            timeframe_minutes = self._timeframe_to_minutes(timeframe)
            total_minutes = (end_date - start_date).total_seconds() / 60
            needed_points = int(total_minutes / timeframe_minutes)

            self.logger.info(f"获取历史数据: {symbol} {start_date} 到 {end_date}")

            # 分批获取数据
            all_klines = []
            current_end = end_date

            while current_end > start_date:
                # 计算当前批次的开始时间
                batch_points = min(needed_points, self.default_limit)
                batch_start = current_end - timedelta(minutes=batch_points * timeframe_minutes)

                if batch_start < start_date:
                    batch_start = start_date

                # 获取当前批次数据
                batch_klines = await self.fetch_klines(
                    symbol, timeframe, batch_points, exchange, batch_start
                )

                if batch_klines:
                    # 过滤时间范围
                    filtered_klines = [
                        k for k in batch_klines
                        if start_date <= k.timestamp <= current_end
                    ]
                    all_klines.extend(filtered_klines)

                current_end = batch_start
                needed_points -= batch_points

                # 避免无限循环
                if batch_points <= 0:
                    break

            # 按时间排序并去重
            all_klines.sort(key=lambda x: x.timestamp)
            unique_klines = self._remove_duplicates(all_klines)

            self.business_logger.log_system_event(
                event_type="historical_data_collected",
                severity="info",
                message=f"历史数据收集完成: {symbol} {timeframe}",
                details={
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "start_date": start_date.isoformat(),
                    "end_date": end_date.isoformat(),
                    "total_points": len(unique_klines)
                }
            )

            return unique_klines

        except Exception as e:
            self.logger.error(f"获取历史数据失败: {e}")
            raise

    async def save_klines_to_database(
        self,
        klines: List[KlineData],
        exchange: str = "binance"
    ) -> int:
        """保存K线数据到数据库"""
        try:
            if not klines:
                return 0

            db = SessionLocal()
            saved_count = 0

            try:
                # 获取交易符号ID
                symbol = klines[0].symbol
                trading_symbol = db.query(TradingSymbol).filter(
                    TradingSymbol.symbol == symbol
                ).first()

                if not trading_symbol:
                    self.logger.warning(f"交易符号不存在: {symbol}")
                    return 0

                # 批量保存数据
                batch_size = 1000
                for i in range(0, len(klines), batch_size):
                    batch = klines[i:i + batch_size]

                    for kline in batch:
                        # 检查是否已存在
                        existing = db.query(KlineDataModel).filter(
                            KlineDataModel.timestamp == kline.timestamp,
                            KlineDataModel.symbol_id == trading_symbol.id
                        ).first()

                        if not existing:
                            kline_record = KlineDataModel(
                                timestamp=kline.timestamp,
                                symbol_id=trading_symbol.id,
                                exchange_id=trading_symbol.exchange_id,
                                open_price=kline.open,
                                high_price=kline.high,
                                low_price=kline.low,
                                close_price=kline.close,
                                volume=kline.volume,
                                quote_volume=kline.quote_volume,
                                trades_count=kline.trades_count
                            )
                            db.add(kline_record)
                            saved_count += 1

                    # 提交批次
                    db.commit()
                    self.logger.debug(f"已保存 {i + len(batch)}/{len(klines)} 条K线数据")

                self.business_logger.log_system_event(
                    event_type="klines_data_saved",
                    severity="info",
                    message=f"K线数据保存完成: {symbol} ({saved_count}条)",
                    details={
                        "symbol": symbol,
                        "exchange": exchange,
                        "saved_count": saved_count,
                        "total_count": len(klines)
                    }
                )

                return saved_count

            finally:
                db.close()

        except Exception as e:
            self.logger.error(f"保存K线数据到数据库失败: {e}")
            raise

    async def update_market_data(
        self,
        symbols: List[str],
        timeframes: List[str],
        exchanges: List[str] = None
    ) -> Dict[str, Dict[str, int]]:
        """更新市场数据"""
        try:
            if not exchanges:
                exchanges = ["binance"]  # 默认交易所

            update_stats = {}

            for exchange in exchanges:
                exchange_stats = {}

                for symbol in symbols:
                    symbol_stats = {}

                    for timeframe in timeframes:
                        try:
                            # 获取最新数据
                            klines = await self.fetch_klines(
                                symbol, timeframe, limit=100, exchange=exchange
                            )

                            if klines:
                                # 保存到数据库
                                saved_count = await self.save_klines_to_database(klines, exchange)
                                symbol_stats[timeframe] = saved_count

                        except Exception as e:
                            self.logger.error(f"更新{exchange} {symbol} {timeframe}数据失败: {e}")
                            symbol_stats[timeframe] = 0

                    exchange_stats[symbol] = symbol_stats

                update_stats[exchange] = exchange_stats

            self.business_logger.log_system_event(
                event_type="market_data_update_completed",
                severity="info",
                message="市场数据更新完成",
                details={
                    "symbols": symbols,
                    "timeframes": timeframes,
                    "exchanges": exchanges,
                    "update_stats": update_stats
                }
            )

            return update_stats

        except Exception as e:
            self.logger.error(f"更新市场数据失败: {e}")
            raise

    def _validate_kline_request(
        self,
        symbol: str,
        timeframe: str,
        limit: Optional[int],
        exchange: str
    ):
        """验证K线数据请求参数"""
        if not symbol:
            raise ValidationError("交易符号不能为空")

        if timeframe not in self.supported_timeframes:
            raise ValidationError(f"不支持的时间框架: {timeframe}")

        if limit and limit <= 0:
            raise ValidationError("数据条数必须大于0")

        if limit and limit > self.max_data_points:
            raise ValidationError(f"数据条数不能超过 {self.max_data_points}")

        if not exchange:
            raise ValidationError("交易所不能为空")

    def _get_cache_key(
        self,
        symbol: str,
        timeframe: str,
        exchange: str,
        limit: Optional[int],
        since: Optional[datetime]
    ) -> str:
        """生成缓存键"""
        since_str = since.isoformat() if since else "none"
        limit_str = str(limit) if limit else "default"
        return f"klines:{exchange}:{symbol}:{timeframe}:{limit_str}:{since_str}"

    async def _fetch_large_dataset(
        self,
        client,
        symbol: str,
        timeframe: str,
        limit: int,
        since: Optional[datetime]
    ) -> List[KlineData]:
        """获取大数据集（分批获取）"""
        all_klines = []
        batch_size = self.default_limit
        remaining = limit
        current_since = since

        while remaining > 0:
            current_limit = min(batch_size, remaining)

            batch_klines = await client.fetch_klines(
                symbol, timeframe, current_limit, current_since
            )

            if not batch_klines:
                break

            all_klines.extend(batch_klines)
            remaining -= len(batch_klines)

            # 更新起始时间
            if batch_klines:
                current_since = batch_klines[-1].timestamp + timedelta(minutes=1)

            # 避免API限制
            await asyncio.sleep(0.1)

        return all_klines

    def _validate_klines_data(self, klines: List[KlineData]) -> List[KlineData]:
        """验证K线数据质量"""
        validated_klines = []

        for kline in klines:
            # 基本验证
            if not all([
                kline.open > 0,
                kline.high > 0,
                kline.low > 0,
                kline.close > 0,
                kline.volume >= 0
            ]):
                self.logger.warning(f"跳过无效K线数据: {kline.timestamp}")
                continue

            # 价格逻辑验证
            if not (kline.low <= kline.open <= kline.high and
                   kline.low <= kline.close <= kline.high):
                self.logger.warning(f"跳过价格逻辑错误的K线: {kline.timestamp}")
                continue

            validated_klines.append(kline)

        return validated_klines

    def _timeframe_to_minutes(self, timeframe: str) -> int:
        """将时间框架转换为分钟数"""
        timeframe_map = {
            '1m': 1, '3m': 3, '5m': 5, '15m': 15, '30m': 30,
            '1h': 60, '2h': 120, '4h': 240, '6h': 360, '8h': 480, '12h': 720,
            '1d': 1440, '3d': 4320, '1w': 10080, '1M': 43200
        }
        return timeframe_map.get(timeframe, 60)

    def _remove_duplicates(self, klines: List[KlineData]) -> List[KlineData]:
        """移除重复的K线数据"""
        seen_timestamps = set()
        unique_klines = []

        for kline in klines:
            if kline.timestamp not in seen_timestamps:
                seen_timestamps.add(kline.timestamp)
                unique_klines.append(kline)

        return unique_klines


# 便捷函数
async def get_klines(
    symbol: str,
    timeframe: str,
    limit: int = 100,
    exchange: str = "binance"
) -> List[KlineData]:
    """获取K线数据的便捷函数"""
    collector = ExchangeDataCollector()
    return await collector.fetch_klines(symbol, timeframe, limit, exchange)


async def get_historical_klines(
    symbol: str,
    timeframe: str,
    days: int,
    exchange: str = "binance"
) -> List[KlineData]:
    """获取历史K线数据的便捷函数"""
    collector = ExchangeDataCollector()
    end_date = datetime.utcnow()
    start_date = end_date - timedelta(days=days)
    return await collector.fetch_historical_data(symbol, timeframe, start_date, end_date, exchange)