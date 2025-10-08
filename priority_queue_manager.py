#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
API调用优先级和队列管理系统
支持智能调度、负载均衡、故障转移等功能
"""

import asyncio
import heapq
import time
import json
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable
from enum import Enum
from datetime import datetime, timedelta
import redis
import threading
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)


class Priority(Enum):
    """请求优先级"""
    CRITICAL = 1    # 紧急止损、强平、流动性危机
    HIGH = 2        # 市价单、大额交易
    MEDIUM = 3      # 限价单、查询余额
    LOW = 4         # 历史数据、统计分析
    BACKGROUND = 5  # 定时任务、系统维护


class TaskStatus(Enum):
    """任务状态"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    RETRY = "retry"


class QueueType(Enum):
    """队列类型"""
    FIFO = "fifo"           # 先进先出
    PRIORITY = "priority"   # 优先级队列
    DELAYED = "delayed"     # 延时队列
    DEAD_LETTER = "dead_letter"  # 死信队列


@dataclass
class Task:
    """任务数据结构"""
    id: str
    priority: Priority
    payload: Dict[str, Any]
    queue_type: QueueType = QueueType.PRIORITY
    retry_count: int = 0
    max_retries: int = 3
    delay_until: Optional[datetime] = None
    created_at: datetime = field(default_factory=datetime.now)
    scheduled_at: Optional[datetime] = None
    timeout: int = 30
    callback: Optional[Callable] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __lt__(self, other):
        """用于优先级队列排序"""
        if self.delay_until and other.delay_until:
            if self.delay_until != other.delay_until:
                return self.delay_until < other.delay_until

        # 优先级数字越小，优先级越高
        if self.priority.value != other.priority.value:
            return self.priority.value < other.priority.value

        # 相同优先级按创建时间排序
        return self.created_at < other.created_at


class WorkerPool:
    """工作协程池"""

    def __init__(self, name: str, size: int, task_handler: Callable):
        self.name = name
        self.size = size
        self.task_handler = task_handler
        self.workers = []
        self.task_queue = asyncio.Queue()
        self.running = False
        self.active_tasks = set()
        self.completed_tasks = 0
        self.failed_tasks = 0

    async def start(self):
        """启动工作池"""
        self.running = True
        for i in range(self.size):
            worker = asyncio.create_task(self._worker(f"{self.name}-worker-{i}"))
            self.workers.append(worker)
        logger.info(f"工作池 {self.name} 已启动，工作协程数: {self.size}")

    async def stop(self):
        """停止工作池"""
        self.running = False
        for worker in self.workers:
            worker.cancel()

        await asyncio.gather(*self.workers, return_exceptions=True)
        logger.info(f"工作池 {self.name} 已停止")

    async def submit_task(self, task: Task):
        """提交任务"""
        await self.task_queue.put(task)

    async def _worker(self, worker_name: str):
        """工作协程"""
        logger.info(f"工作协程 {worker_name} 已启动")

        while self.running:
            try:
                # 获取任务
                task = await asyncio.wait_for(
                    self.task_queue.get(),
                    timeout=1.0
                )

                # 处理任务
                await self._process_task(task, worker_name)

            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"工作协程 {worker_name} 错误: {e}")

        logger.info(f"工作协程 {worker_name} 已停止")

    async def _process_task(self, task: Task, worker_name: str):
        """处理单个任务"""
        task_id = task.id
        self.active_tasks.add(task_id)

        try:
            logger.debug(f"{worker_name} 开始处理任务: {task_id}")

            # 检查任务是否到期
            if task.delay_until and task.delay_until > datetime.now():
                # 重新放入队列
                await asyncio.sleep((task.delay_until - datetime.now()).total_seconds())
                await self.submit_task(task)
                return

            # 执行任务
            start_time = time.time()

            try:
                result = await asyncio.wait_for(
                    self.task_handler(task),
                    timeout=task.timeout
                )

                execution_time = time.time() - start_time
                logger.info(f"任务 {task_id} 执行成功，耗时: {execution_time:.2f}s")

                # 调用回调
                if task.callback:
                    await task.callback(result, None)

                self.completed_tasks += 1

            except asyncio.TimeoutError:
                logger.error(f"任务 {task_id} 执行超时")
                raise
            except Exception as e:
                logger.error(f"任务 {task_id} 执行失败: {e}")
                raise

        except Exception as e:
            # 任务失败处理
            self.failed_tasks += 1

            if task.retry_count < task.max_retries:
                # 重试
                task.retry_count += 1
                task.delay_until = datetime.now() + timedelta(seconds=2 ** task.retry_count)

                logger.info(f"任务 {task_id} 将在第 {task.retry_count} 次重试")
                await self.submit_task(task)
            else:
                # 超过最大重试次数
                logger.error(f"任务 {task_id} 超过最大重试次数，标记为失败")
                if task.callback:
                    await task.callback(None, e)

        finally:
            self.active_tasks.discard(task_id)
            self.task_queue.task_done()

    def get_stats(self) -> Dict[str, Any]:
        """获取工作池统计信息"""
        return {
            "name": self.name,
            "size": self.size,
            "running": self.running,
            "queue_size": self.task_queue.qsize(),
            "active_tasks": len(self.active_tasks),
            "completed_tasks": self.completed_tasks,
            "failed_tasks": self.failed_tasks,
            "success_rate": self.completed_tasks / max(1, self.completed_tasks + self.failed_tasks)
        }


class PriorityQueueManager:
    """优先级队列管理器"""

    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client
        self.queues = {
            QueueType.PRIORITY: asyncio.PriorityQueue(),
            QueueType.DELAYED: asyncio.PriorityQueue(),
            QueueType.DEAD_LETTER: asyncio.Queue(),
        }
        self.worker_pools = {}
        self.running = False
        self.task_counter = 0
        self.lock = asyncio.Lock()

        # 持久化配置
        self.persistence_enabled = True
        self.persistence_key_prefix = "queue_manager"

    async def start(self):
        """启动队列管理器"""
        self.running = True

        # 启动队列监控协程
        asyncio.create_task(self._queue_monitor())
        asyncio.create_task(self._delayed_queue_processor())
        asyncio.create_task(self._dead_letter_processor())

        # 恢复持久化任务
        if self.persistence_enabled:
            await self._restore_tasks()

        logger.info("优先级队列管理器已启动")

    async def stop(self):
        """停止队列管理器"""
        self.running = False

        # 停止所有工作池
        for pool in self.worker_pools.values():
            await pool.stop()

        # 持久化剩余任务
        if self.persistence_enabled:
            await self._persist_tasks()

        logger.info("优先级队列管理器已停止")

    def register_worker_pool(self, queue_type: QueueType, name: str, size: int, handler: Callable):
        """注册工作池"""
        pool = WorkerPool(name, size, handler)
        self.worker_pools[queue_type] = pool
        logger.info(f"注册工作池: {name} for {queue_type.value}")

    async def submit_task(
        self,
        priority: Priority,
        payload: Dict[str, Any],
        queue_type: QueueType = QueueType.PRIORITY,
        delay_until: Optional[datetime] = None,
        max_retries: int = 3,
        timeout: int = 30,
        callback: Optional[Callable] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """提交任务"""
        async with self.lock:
            self.task_counter += 1
            task_id = f"task_{int(time.time())}_{self.task_counter}"

        task = Task(
            id=task_id,
            priority=priority,
            payload=payload,
            queue_type=queue_type,
            delay_until=delay_until,
            max_retries=max_retries,
            timeout=timeout,
            callback=callback,
            metadata=metadata or {}
        )

        # 持久化任务
        if self.persistence_enabled:
            await self._persist_task(task)

        # 提交到相应队列
        if queue_type == QueueType.DELAYED or delay_until:
            await self.queues[QueueType.DELAYED].put(task)
        else:
            await self.queues[queue_type].put(task)

        logger.debug(f"任务已提交: {task_id} - 优先级: {priority.name}")
        return task_id

    async def _queue_monitor(self):
        """队列监控协程"""
        while self.running:
            try:
                # 处理优先级队列
                priority_queue = self.queues[QueueType.PRIORITY]
                if not priority_queue.empty() and QueueType.PRIORITY in self.worker_pools:
                    task = await priority_queue.get()
                    await self.worker_pools[QueueType.PRIORITY].submit_task(task)

                await asyncio.sleep(0.01)  # 避免CPU占用过高

            except Exception as e:
                logger.error(f"队列监控错误: {e}")
                await asyncio.sleep(1)

    async def _delayed_queue_processor(self):
        """延时队列处理器"""
        while self.running:
            try:
                delayed_queue = self.queues[QueueType.DELAYED]
                if not delayed_queue.empty():
                    task = await delayed_queue.get()

                    # 检查是否到期
                    if task.delay_until and task.delay_until <= datetime.now():
                        # 转移到优先级队列
                        task.delay_until = None
                        await self.queues[QueueType.PRIORITY].put(task)
                        logger.debug(f"延时任务 {task.id} 已到期，转移到优先级队列")
                    else:
                        # 重新放入延时队列
                        await delayed_queue.put(task)

                await asyncio.sleep(1)

            except Exception as e:
                logger.error(f"延时队列处理错误: {e}")
                await asyncio.sleep(5)

    async def _dead_letter_processor(self):
        """死信队列处理器"""
        while self.running:
            try:
                dead_letter_queue = self.queues[QueueType.DEAD_LETTER]
                if not dead_letter_queue.empty():
                    task = await dead_letter_queue.get()

                    # 记录死信任务
                    logger.error(f"死信任务: {task.id} - {task.payload}")

                    # 可以在这里发送告警通知
                    await self._send_dead_letter_alert(task)

                await asyncio.sleep(10)

            except Exception as e:
                logger.error(f"死信队列处理错误: {e}")
                await asyncio.sleep(30)

    async def _persist_task(self, task: Task):
        """持久化任务"""
        try:
            task_data = {
                "id": task.id,
                "priority": task.priority.value,
                "payload": task.payload,
                "queue_type": task.queue_type.value,
                "retry_count": task.retry_count,
                "max_retries": task.max_retries,
                "delay_until": task.delay_until.isoformat() if task.delay_until else None,
                "created_at": task.created_at.isoformat(),
                "scheduled_at": task.scheduled_at.isoformat() if task.scheduled_at else None,
                "timeout": task.timeout,
                "metadata": task.metadata
            }

            key = f"{self.persistence_key_prefix}:task:{task.id}"
            self.redis.setex(key, 86400, json.dumps(task_data))  # 保存24小时

        except Exception as e:
            logger.error(f"持久化任务失败: {e}")

    async def _persist_tasks(self):
        """持久化所有队列中的任务"""
        try:
            # 持久化优先级队列任务
            priority_tasks = []
            priority_queue = self.queues[QueueType.PRIORITY]
            while not priority_queue.empty():
                task = priority_queue.get_nowait()
                priority_tasks.append(task)
                await self._persist_task(task)

            # 恢复任务到队列
            for task in priority_tasks:
                await priority_queue.put(task)

            # 类似处理其他队列...
            logger.info("任务持久化完成")

        except Exception as e:
            logger.error(f"持久化任务失败: {e}")

    async def _restore_tasks(self):
        """恢复持久化的任务"""
        try:
            pattern = f"{self.persistence_key_prefix}:task:*"
            keys = self.redis.keys(pattern)

            restored_count = 0
            for key in keys:
                task_data = self.redis.get(key)
                if task_data:
                    data = json.loads(task_data)

                    task = Task(
                        id=data["id"],
                        priority=Priority(data["priority"]),
                        payload=data["payload"],
                        queue_type=QueueType(data["queue_type"]),
                        retry_count=data["retry_count"],
                        max_retries=data["max_retries"],
                        delay_until=datetime.fromisoformat(data["delay_until"]) if data["delay_until"] else None,
                        created_at=datetime.fromisoformat(data["created_at"]),
                        scheduled_at=datetime.fromisoformat(data["scheduled_at"]) if data["scheduled_at"] else None,
                        timeout=data["timeout"],
                        metadata=data["metadata"]
                    )

                    # 重新加入队列
                    if task.delay_until:
                        await self.queues[QueueType.DELAYED].put(task)
                    else:
                        await self.queues[task.queue_type].put(task)

                    restored_count += 1

            logger.info(f"恢复了 {restored_count} 个任务")

        except Exception as e:
            logger.error(f"恢复任务失败: {e}")

    async def _send_dead_letter_alert(self, task: Task):
        """发送死信任务告警"""
        try:
            alert_data = {
                "timestamp": datetime.now().isoformat(),
                "task_id": task.id,
                "priority": task.priority.name,
                "retry_count": task.retry_count,
                "payload": task.payload,
                "error": "任务执行失败，超过最大重试次数"
            }

            alert_key = f"{self.persistence_key_prefix}:alerts:{task.id}"
            self.redis.setex(alert_key, 86400, json.dumps(alert_data))

            # 这里可以集成邮件、钉钉、企业微信等告警渠道
            logger.warning(f"死信任务告警: {task.id}")

        except Exception as e:
            logger.error(f"发送死信告警失败: {e}")

    async def get_queue_stats(self) -> Dict[str, Any]:
        """获取队列统计信息"""
        stats = {
            "timestamp": datetime.now().isoformat(),
            "queues": {},
            "worker_pools": {},
            "total_tasks": 0
        }

        # 队列统计
        for queue_type, queue in self.queues.items():
            queue_size = queue.qsize()
            stats["queues"][queue_type.value] = queue_size
            stats["total_tasks"] += queue_size

        # 工作池统计
        for queue_type, pool in self.worker_pools.items():
            stats["worker_pools"][queue_type.value] = pool.get_stats()

        return stats

    async def cancel_task(self, task_id: str) -> bool:
        """取消任务"""
        # 这里需要实现任务取消逻辑
        # 由于队列的特性，实现起来比较复杂
        # 可以通过维护一个取消任务列表来实现
        logger.info(f"任务取消请求: {task_id}")
        return True

    async def pause_queue(self, queue_type: QueueType):
        """暂停队列处理"""
        # 实现队列暂停逻辑
        logger.info(f"队列暂停: {queue_type.value}")

    async def resume_queue(self, queue_type: QueueType):
        """恢复队列处理"""
        # 实现队列恢复逻辑
        logger.info(f"队列恢复: {queue_type.value}")


class ExchangeTaskHandler:
    """交易所任务处理器"""

    def __init__(self, rate_limiter, api_client):
        self.rate_limiter = rate_limiter
        self.api_client = api_client

    async def handle_task(self, task: Task) -> Any:
        """处理任务"""
        payload = task.payload
        exchange = payload.get("exchange")
        endpoint = payload.get("endpoint")
        method = payload.get("method", "GET")
        params = payload.get("params", {})
        data = payload.get("data", {})

        # 检查限流
        # allowed, reason = await self.rate_limiter.check_rate_limit(exchange, task)
        # if not allowed:
        #     raise Exception(f"请求被限流: {reason}")

        # 执行API调用
        try:
            result = await self.api_client.request(
                method=method,
                endpoint=endpoint,
                params=params,
                data=data
            )
            return result

        except Exception as e:
            logger.error(f"API调用失败: {e}")
            raise


# 使用示例
async def example_priority_queue():
    """优先级队列使用示例"""

    # 初始化Redis客户端
    redis_client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)

    # 创建队列管理器
    queue_manager = PriorityQueueManager(redis_client)

    # 注册任务处理器
    async def sample_task_handler(task: Task) -> Any:
        """示例任务处理器"""
        print(f"处理任务: {task.id} - 优先级: {task.priority.name}")
        await asyncio.sleep(1)  # 模拟任务执行
        return {"status": "success", "task_id": task.id}

    # 启动队列管理器
    await queue_manager.start()

    # 注册工作池
    queue_manager.register_worker_pool(
        QueueType.PRIORITY,
        "main_pool",
        3,
        sample_task_handler
    )

    # 启动工作池
    await queue_manager.worker_pools[QueueType.PRIORITY].start()

    try:
        # 提交不同优先级的任务
        tasks = []

        # 高优先级任务
        for i in range(3):
            task_id = await queue_manager.submit_task(
                priority=Priority.HIGH,
                payload={"action": "high_priority_task", "index": i},
                callback=lambda result, error: print(f"高优先级任务完成: {result or error}")
            )
            tasks.append(task_id)

        # 中等优先级任务
        for i in range(5):
            task_id = await queue_manager.submit_task(
                priority=Priority.MEDIUM,
                payload={"action": "medium_priority_task", "index": i}
            )
            tasks.append(task_id)

        # 低优先级任务
        for i in range(2):
            task_id = await queue_manager.submit_task(
                priority=Priority.LOW,
                payload={"action": "low_priority_task", "index": i}
            )
            tasks.append(task_id)

        # 延时任务
        delay_task_id = await queue_manager.submit_task(
            priority=Priority.MEDIUM,
            payload={"action": "delayed_task"},
            delay_until=datetime.now() + timedelta(seconds=10)
        )

        print(f"提交了 {len(tasks) + 1} 个任务")

        # 运行一段时间
        await asyncio.sleep(15)

        # 查看队列统计
        stats = await queue_manager.get_queue_stats()
        print("\n=== 队列统计信息 ===")
        print(json.dumps(stats, indent=2, ensure_ascii=False))

    finally:
        # 停止队列管理器
        await queue_manager.stop()


if __name__ == "__main__":
    # 运行示例
    asyncio.run(example_priority_queue())