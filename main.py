#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多交易所API集成限流和并发处理系统 - 主程序入口
整合所有组件，提供完整的解决方案
"""

import asyncio
import logging
import yaml
import os
import sys
from typing import Dict, Any, Optional
from datetime import datetime

from exchange_rate_limiter import ExchangeAPIClient, ExchangeRateLimiter, APIRequest, Priority
from advanced_rate_limiting import ExchangeRateLimitManager
from priority_queue_manager import PriorityQueueManager, ExchangeTaskHandler
from fallback_retry_system import ResilientAPIClient, RetryConfig, FallbackConfig, CircuitBreakerConfig
import redis
import httpx

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MultiExchangeTradingSystem:
    """多交易所交易系统主类"""

    def __init__(self, config_path: str = "config.yaml"):
        self.config = self._load_config(config_path)
        self.redis_client = self._init_redis()

        # 初始化各个组件
        self.rate_limiter = ExchangeRateLimiter(self.redis_client)
        self.advanced_rate_manager = ExchangeRateLimitManager(self.redis_client)
        self.api_client = ExchangeAPIClient(self.rate_limiter, self.redis_client)
        self.resilient_client = ResilientAPIClient(self.redis_client)
        self.queue_manager = PriorityQueueManager(self.redis_client)

        # 任务处理器
        self.task_handler = ExchangeTaskHandler(self.advanced_rate_manager, self.resilient_client)

        # 系统状态
        self.running = False
        self.worker_tasks = []

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """加载配置文件"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)

            # 处理环境变量替换
            config = self._replace_env_vars(config)
            logger.info(f"配置文件加载成功: {config_path}")
            return config

        except Exception as e:
            logger.error(f"加载配置文件失败: {e}")
            sys.exit(1)

    def _replace_env_vars(self, obj: Any) -> Any:
        """递归替换配置中的环境变量"""
        if isinstance(obj, dict):
            return {k: self._replace_env_vars(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._replace_env_vars(item) for item in obj]
        elif isinstance(obj, str) and obj.startswith("${") and obj.endswith("}"):
            env_var = obj[2:-1]
            return os.getenv(env_var, obj)
        else:
            return obj

    def _init_redis(self) -> redis.Redis:
        """初始化Redis客户端"""
        redis_config = self.config.get('redis', {})

        try:
            client = redis.Redis(
                host=redis_config.get('host', 'localhost'),
                port=redis_config.get('port', 6379),
                db=redis_config.get('db', 0),
                password=redis_config.get('password'),
                socket_connect_timeout=5,
                socket_timeout=5,
                retry_on_timeout=True
            )

            # 测试连接
            client.ping()
            logger.info("Redis连接成功")
            return client

        except Exception as e:
            logger.error(f"Redis连接失败: {e}")
            sys.exit(1)

    async def start(self):
        """启动系统"""
        if self.running:
            logger.warning("系统已经在运行中")
            return

        logger.info("启动多交易所交易系统...")
        self.running = True

        try:
            # 启动队列管理器
            await self.queue_manager.start()

            # 注册工作池
            self._register_worker_pools()

            # 启动工作池
            await self._start_worker_pools()

            # 启动API客户端
            worker_tasks = await self.api_client.start()
            self.worker_tasks.extend(worker_tasks)

            logger.info("系统启动成功")

            # 启动监控协程
            monitor_task = asyncio.create_task(self._system_monitor())
            self.worker_tasks.append(monitor_task)

        except Exception as e:
            logger.error(f"系统启动失败: {e}")
            await self.stop()
            raise

    def _register_worker_pools(self):
        """注册工作池"""
        from priority_queue_manager import QueueType

        # 注册优先级队列工作池
        self.queue_manager.register_worker_pool(
            QueueType.PRIORITY,
            "trading_pool",
            self.config['queue_manager']['worker_pools']['priority']['size'],
            self._handle_trading_task
        )

        # 注册延时队列工作池
        self.queue_manager.register_worker_pool(
            QueueType.DELAYED,
            "delayed_pool",
            self.config['queue_manager']['worker_pools']['delayed']['size'],
            self._handle_delayed_task
        )

    async def _start_worker_pools(self):
        """启动工作池"""
        from priority_queue_manager import QueueType

        for queue_type in [QueueType.PRIORITY, QueueType.DELAYED]:
            if queue_type in self.queue_manager.worker_pools:
                await self.queue_manager.worker_pools[queue_type].start()

    async def _handle_trading_task(self, task) -> Dict[str, Any]:
        """处理交易任务"""
        logger.info(f"处理交易任务: {task.id}")

        try:
            payload = task.payload
            exchange = payload.get('exchange')
            endpoint = payload.get('endpoint')
            method = payload.get('method', 'GET')
            params = payload.get('params', {})
            data = payload.get('data', {})

            # 使用弹性客户端执行请求
            result = await self.resilient_client.resilient_request(
                exchange=exchange,
                endpoint=endpoint,
                method=method,
                params=params,
                data=data,
                retry_config=self._get_retry_config(payload.get('priority', 'normal')),
                fallback_config=self._get_fallback_config(payload.get('priority', 'normal')),
                circuit_breaker_config=self._get_circuit_breaker_config(exchange)
            )

            logger.info(f"交易任务完成: {task.id}")
            return result

        except Exception as e:
            logger.error(f"交易任务失败: {task.id} - {e}")
            raise

    async def _handle_delayed_task(self, task) -> Dict[str, Any]:
        """处理延时任务"""
        logger.info(f"处理延时任务: {task.id}")

        # 延时任务通常是一些定时任务，如定期查询价格、更新状态等
        # 这里可以调用相应的处理逻辑
        await asyncio.sleep(1)  # 模拟处理时间

        return {"status": "completed", "task_id": task.id}

    def _get_retry_config(self, priority: str) -> RetryConfig:
        """根据优先级获取重试配置"""
        retry_configs = self.config.get('retry_strategies', {})

        if priority == 'critical':
            return RetryConfig(**retry_configs.get('aggressive', {}))
        elif priority == 'background':
            return RetryConfig(**retry_configs.get('conservative', {}))
        else:
            return RetryConfig(**retry_configs.get('default', {}))

    def _get_fallback_config(self, priority: str) -> FallbackConfig:
        """根据优先级获取降级配置"""
        fallback_configs = self.config.get('fallback_strategies', {})

        if priority == 'critical':
            return FallbackConfig(**fallback_configs.get('critical', {}))
        elif priority == 'important':
            return FallbackConfig(**fallback_configs.get('important', {}))
        elif priority == 'background':
            return FallbackConfig(**fallback_configs.get('background', {}))
        else:
            return FallbackConfig(**fallback_configs.get('normal', {}))

    def _get_circuit_breaker_config(self, exchange: str) -> CircuitBreakerConfig:
        """获取熔断器配置"""
        exchange_config = self.config.get('exchanges', {}).get(exchange, {})
        return CircuitBreakerConfig(**exchange_config.get('circuit_breaker', {}))

    async def submit_trading_request(
        self,
        exchange: str,
        endpoint: str,
        method: str = "GET",
        params: Optional[Dict[str, Any]] = None,
        data: Optional[Dict[str, Any]] = None,
        priority: str = "normal",
        delay_until: Optional[datetime] = None,
        callback: Optional[callable] = None
    ) -> str:
        """提交交易请求"""

        # 转换优先级
        priority_map = {
            'critical': Priority.CRITICAL,
            'high': Priority.HIGH,
            'medium': Priority.MEDIUM,
            'low': Priority.LOW,
            'background': Priority.LOW
        }

        task_priority = priority_map.get(priority, Priority.MEDIUM)

        # 提交任务到队列
        task_id = await self.queue_manager.submit_task(
            priority=task_priority,
            payload={
                'exchange': exchange,
                'endpoint': endpoint,
                'method': method,
                'params': params or {},
                'data': data or {},
                'priority': priority
            },
            delay_until=delay_until,
            callback=callback,
            metadata={'submitted_at': datetime.now().isoformat()}
        )

        logger.info(f"交易请求已提交: {task_id}")
        return task_id

    async def get_system_status(self) -> Dict[str, Any]:
        """获取系统状态"""
        status = {
            'timestamp': datetime.now().isoformat(),
            'system': {
                'running': self.running,
                'worker_tasks': len(self.worker_tasks)
            },
            'redis': {
                'connected': self.redis_client.ping()
            },
            'queue_stats': await self.queue_manager.get_queue_stats(),
            'circuit_breakers': await self.resilient_client.get_circuit_breaker_status(),
            'rate_limits': {},
            'api_metrics': await self.resilient_client.get_request_metrics()
        }

        # 获取各个交易所的限流状态
        for exchange in self.config.get('exchanges', {}).keys():
            try:
                exchange_status = await self.advanced_rate_manager.get_exchange_status(exchange)
                status['rate_limits'][exchange] = exchange_status
            except Exception as e:
                status['rate_limits'][exchange] = {'error': str(e)}

        return status

    async def _system_monitor(self):
        """系统监控协程"""
        logger.info("系统监控协程已启动")

        while self.running:
            try:
                # 获取系统状态
                status = await self.get_system_status()

                # 检查告警条件
                await self._check_alerts(status)

                # 记录状态到Redis
                await self._save_system_status(status)

                await asyncio.sleep(60)  # 每分钟检查一次

            except Exception as e:
                logger.error(f"系统监控错误: {e}")
                await asyncio.sleep(30)

    async def _check_alerts(self, status: Dict[str, Any]):
        """检查告警条件"""
        alerts_config = self.config.get('monitoring', {}).get('alerts', {})

        if not alerts_config.get('enabled', True):
            return

        # 检查错误率
        api_metrics = status.get('api_metrics', {})
        for exchange, metrics in api_metrics.items():
            if isinstance(metrics, dict):
                failed = metrics.get('failed', 0)
                total = failed + metrics.get('success', 0)

                if total > 0:
                    error_rate = failed / total
                    if error_rate > alerts_config.get('thresholds', {}).get('error_rate', 0.1):
                        await self._send_alert(f"高错误率告警", f"{exchange} 错误率: {error_rate:.2%}")

        # 检查熔断器状态
        circuit_breakers = status.get('circuit_breakers', {}).get('breakers', {})
        for name, breaker in circuit_breakers.items():
            if breaker.get('state') == 'open':
                await self._send_alert(f"熔断器开启", f"熔断器 {name} 已开启")

        # 检查队列积压
        queue_stats = status.get('queue_stats', {}).get('queues', {})
        for queue_type, size in queue_stats.items():
            if size > alerts_config.get('thresholds', {}).get('queue_size', 1000):
                await self._send_alert(f"队列积压告警", f"队列 {queue_type} 积压: {size}")

    async def _send_alert(self, title: str, message: str):
        """发送告警"""
        logger.warning(f"告警: {title} - {message}")

        # 这里可以集成具体的告警渠道
        # 如邮件、钉钉、企业微信、Slack等
        alert_data = {
            'title': title,
            'message': message,
            'timestamp': datetime.now().isoformat()
        }

        # 保存告警到Redis
        try:
            alert_key = f"alerts:{int(datetime.now().timestamp())}"
            self.redis_client.setex(alert_key, 86400, str(alert_data))
        except Exception as e:
            logger.error(f"保存告警失败: {e}")

    async def _save_system_status(self, status: Dict[str, Any]):
        """保存系统状态到Redis"""
        try:
            import json
            status_key = f"system_status:{int(datetime.now().timestamp())}"
            self.redis_client.setex(status_key, 3600, json.dumps(status))
        except Exception as e:
            logger.error(f"保存系统状态失败: {e}")

    async def stop(self):
        """停止系统"""
        if not self.running:
            return

        logger.info("正在停止系统...")
        self.running = False

        try:
            # 停止API客户端
            await self.api_client.stop()

            # 停止队列管理器
            await self.queue_manager.stop()

            # 取消所有工作任务
            for task in self.worker_tasks:
                task.cancel()

            # 等待任务完成
            if self.worker_tasks:
                await asyncio.gather(*self.worker_tasks, return_exceptions=True)

            # 关闭Redis连接
            self.redis_client.close()

            logger.info("系统已停止")

        except Exception as e:
            logger.error(f"停止系统时出错: {e}")


# 示例使用
async def main():
    """主函数示例"""

    # 创建系统实例
    system = MultiExchangeTradingSystem("config.yaml")

    try:
        # 启动系统
        await system.start()

        # 提交一些测试请求
        print("=== 提交测试请求 ===")

        # 高优先级请求 - 获取价格
        price_task_id = await system.submit_trading_request(
            exchange="binance",
            endpoint="/api/v3/ticker/price",
            params={"symbol": "BTCUSDT"},
            priority="high",
            callback=lambda result, error: print(f"价格回调: {result or error}")
        )

        # 中等优先级请求 - 获取订单簿
        orderbook_task_id = await system.submit_trading_request(
            exchange="binance",
            endpoint="/api/v3/depth",
            params={"symbol": "BTCUSDT", "limit": 100},
            priority="medium",
            callback=lambda result, error: print(f"订单簿回调: 获取到{len(result.get('bids', []))}条买单" if result else f"错误: {error}")
        )

        # 延时任务 - 统计分析
        from datetime import datetime, timedelta
        analysis_task_id = await system.submit_trading_request(
            exchange="binance",
            endpoint="/api/v3/klines",
            params={"symbol": "BTCUSDT", "interval": "1h", "limit": 24},
            priority="low",
            delay_until=datetime.now() + timedelta(seconds=10),
            callback=lambda result, error: print(f"分析任务完成: {len(result) if result else error}")
        )

        # 多交易所并发请求
        exchanges = ["binance", "coinbase", "kraken", "huobi", "okex"]
        for exchange in exchanges:
            await system.submit_trading_request(
                exchange=exchange,
                endpoint="/api/v3/ticker/price" if exchange != "coinbase" else "/products/BTC-USD/ticker",
                params={"symbol": "BTCUSDT"} if exchange != "coinbase" else {},
                priority="medium",
                callback=lambda result, error, ex=exchange: print(f"{ex} 价格回调: {result or error}")
            )

        # 运行一段时间观察效果
        print("\n系统运行中，观察30秒...")
        await asyncio.sleep(30)

        # 获取系统状态
        print("\n=== 系统状态 ===")
        status = await system.get_system_status()

        # 打印关键信息
        print(f"系统运行状态: {status['system']['running']}")
        print(f"工作任务数: {status['system']['worker_tasks']}")
        print(f"Redis连接: {status['redis']['connected']}")

        print("\n队列统计:")
        for queue_type, size in status['queue_stats']['queues'].items():
            print(f"  {queue_type}: {size}")

        print("\n熔断器状态:")
        for name, breaker in status['circuit_breakers']['breakers'].items():
            print(f"  {name}: {breaker['state']} (失败: {breaker['failure_count']})")

        print("\n请求指标:")
        for exchange, metrics in status['api_metrics'].items():
            if isinstance(metrics, dict) and 'success' in metrics:
                total = metrics.get('success', 0) + metrics.get('failed', 0)
                success_rate = metrics.get('success', 0) / total if total > 0 else 0
                print(f"  {exchange}: 成功率 {success_rate:.2%} ({total} 请求)")

        # 继续运行一段时间
        print("\n继续运行30秒...")
        await asyncio.sleep(30)

    except KeyboardInterrupt:
        print("\n收到中断信号，正在停止...")
    except Exception as e:
        print(f"系统运行错误: {e}")
    finally:
        # 停止系统
        await system.stop()
        print("系统已停止")


if __name__ == "__main__":
    # 设置事件循环策略（Windows兼容性）
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

    # 运行主程序
    asyncio.run(main())