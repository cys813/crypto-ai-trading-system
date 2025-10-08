# 多Agent加密货币量化交易分析系统

基于Python的多Agent虚拟货币量化交易分析系统，集成5个专业agent（新闻收集、做多分析、做空分析、策略生成、交易执行）和LLM大模型，实现完全自动化的加密货币交易策略分析、决策和执行。

## 系统特性

### 🚀 核心功能
- **多交易所支持**: Binance、Coinbase、Kraken、Huobi、OKEx等主流交易所
- **智能限流**: 令牌桶、滑动窗口、漏桶等多种限流算法
- **优先级队列**: 支持高、中、低优先级任务调度
- **熔断器**: 自动故障检测和恢复
- **降级策略**: 多级降级保证系统可用性
- **重试机制**: 指数退避、线性退避等多种重试策略
- **实时监控**: 系统状态监控和告警

### 📊 技术特点
- **分布式架构**: 基于Redis的分布式状态管理
- **高性能**: 异步并发处理，支持高并发请求
- **可扩展**: 模块化设计，易于扩展新的交易所
- **高可用**: 容错机制，故障自动恢复
- **可观测**: 完整的日志、指标和链路追踪

## 项目结构

```
crypto_ai_trading/
├── main.py                     # 主程序入口
├── config.yaml                 # 配置文件
├── exchange_rate_limiter.py    # 基础限流器和API客户端
├── advanced_rate_limiting.py   # 高级限流算法实现
├── priority_queue_manager.py   # 优先级队列管理
├── fallback_retry_system.py    # 降级和重试系统
├── redis_config.lua           # Redis限流脚本
└── README.md                  # 项目文档
```

## 快速开始

### 1. 环境要求

- Python 3.8+
- Redis 6.0+
- 网络连接（访问交易所API）

### 2. 安装依赖

```bash
pip install redis httpx asyncio pyyaml
```

### 3. 配置环境变量

```bash
# 配置交易所API密钥（可选）
export BINANCE_API_KEY="your_binance_api_key"
export BINANCE_API_SECRET="your_binance_api_secret"
export COINBASE_API_KEY="your_coinbase_api_key"
# ... 其他交易所密钥
```

### 4. 启动Redis

```bash
# Docker方式
docker run -d -p 6379:6379 redis:6-alpine

# 或本地安装
redis-server
```

### 5. 运行系统

```bash
python main.py
```

## 详细配置

### Redis配置

```yaml
redis:
  host: "localhost"
  port: 6379
  db: 0
  connection_pool:
    max_connections: 50
    retry_on_timeout: true
```

### 交易所配置

```yaml
exchanges:
  binance:
    name: "Binance"
    base_url: "https://api.binance.com"
    rate_limits:
      - type: "token_bucket"
        capacity: 6000
        rate: 100
        window: 60
    circuit_breaker:
      failure_threshold: 5
      recovery_timeout: 60
```

### 限流策略配置

```yaml
retry_strategies:
  default:
    max_attempts: 3
    strategy: "exponential_backoff"
    base_delay: 1.0
    max_delay: 60.0
    jitter: true
```

## 使用示例

### 基础API调用

```python
from main import MultiExchangeTradingSystem

# 创建系统实例
system = MultiExchangeTradingSystem("config.yaml")
await system.start()

# 提交交易请求
task_id = await system.submit_trading_request(
    exchange="binance",
    endpoint="/api/v3/ticker/price",
    params={"symbol": "BTCUSDT"},
    priority="high"
)
```

### 高级限流使用

```python
from advanced_rate_limiting import ExchangeRateLimitManager

# 创建限流管理器
rate_manager = ExchangeRateLimitManager(redis_client)

# 检查请求限流
result = await rate_manager.check_request("binance", "public")
if result.allowed:
    # 执行API调用
    pass
else:
    print(f"被限流，重试时间: {result.retry_after}秒")
```

### 优先级队列使用

```python
from priority_queue_manager import PriorityQueueManager, Priority

# 创建队列管理器
queue_manager = PriorityQueueManager(redis_client)
await queue_manager.start()

# 提交高优先级任务
task_id = await queue_manager.submit_task(
    priority=Priority.HIGH,
    payload={"action": "urgent_trading"},
    callback=lambda result, error: print(f"结果: {result}")
)
```

### 降级策略使用

```python
from fallback_retry_system import FallbackConfig, ResilientAPIClient

# 配置降级策略
fallback_config = FallbackConfig(
    level="partial",
    timeout=10.0,
    cache_enabled=True,
    alternative_endpoints=["https://backup-api.example.com"]
)

# 创建弹性客户端
client = ResilientAPIClient(redis_client)

# 执行弹性请求
result = await client.resilient_request(
    exchange="binance",
    endpoint="/api/v3/ticker/price",
    fallback_config=fallback_config
)
```

## 限流算法详解

### 令牌桶算法 (Token Bucket)

- **适用场景**: 允许短时间突发，但限制平均速率
- **特点**: 平滑限流，适合处理突发流量
- **配置参数**: capacity（容量）、rate（生成速率）

```python
# 令牌桶配置示例
rate_limit = {
    "type": "token_bucket",
    "capacity": 6000,  # 桶容量
    "rate": 100,       # 每秒生成100个令牌
    "window": 60       # 时间窗口
}
```

### 滑动窗口算法 (Sliding Window)

- **适用场景**: 精确控制固定时间窗口内的请求数
- **特点**: 计算简单，限流精确
- **配置参数**: limit（限制数）、window（窗口大小）

```python
# 滑动窗口配置示例
rate_limit = {
    "type": "sliding_window",
    "limit": 10000,    # 窗口内最大请求数
    "window": 10       # 10秒窗口
}
```

### 漏桶算法 (Leaky Bucket)

- **适用场景**: 平滑输出流量，严格控制请求速率
- **特点**: 输出速率恒定，适合需要稳定流量的场景
- **配置参数**: capacity（容量）、leak_rate（漏出速率）

```python
# 漏桶配置示例
rate_limit = {
    "type": "leaky_bucket",
    "capacity": 100,   # 桶容量
    "leak_rate": 15,   # 每秒漏出15个请求
    "window": 60
}
```

## 重试策略

### 指数退避 (Exponential Backoff)

```python
retry_config = RetryConfig(
    max_attempts=5,
    strategy=RetryStrategy.EXPONENTIAL_BACKOFF,
    base_delay=1.0,
    multiplier=2.0,
    max_delay=60.0,
    jitter=True
)
```

### 线性退避 (Linear Backoff)

```python
retry_config = RetryConfig(
    max_attempts=3,
    strategy=RetryStrategy.LINEAR_BACKOFF,
    base_delay=2.0
)
```

### 斐波那契退避 (Fibonacci Backoff)

```python
retry_config = RetryConfig(
    max_attempts=5,
    strategy=RetryStrategy.FIBONACCI_BACKOFF,
    base_delay=1.0
)
```

## 降级策略

### 降级级别

- **NONE**: 不降级
- **PARTIAL**: 部分降级，返回缓存数据或简化结果
- **FULL**: 完全降级，返回默认值或静态数据
- **EMERGENCY**: 紧急降级，最小可用功能

### 降级配置

```python
fallback_config = FallbackConfig(
    level=FallbackLevel.PARTIAL,
    timeout=10.0,
    cache_enabled=True,
    cache_ttl=300,
    alternative_endpoints=[
        "https://backup-api.example.com"
    ],
    fallback_function=lambda error, *args, **kwargs: {
        "status": "degraded",
        "data": None
    }
)
```

## 监控和告警

### 系统状态监控

```python
# 获取系统状态
status = await system.get_system_status()

# 查看队列统计
queue_stats = status['queue_stats']
print(f"高优先级队列: {queue_stats['priority']}")

# 查看熔断器状态
circuit_breakers = status['circuit_breakers']
for name, breaker in circuit_breakers['breakers'].items():
    print(f"{name}: {breaker['state']}")
```

### 指标收集

系统自动收集以下指标：
- 请求成功/失败率
- 响应时间分布
- 队列积压情况
- 熔断器触发次数
- 降级使用频率
- Redis连接状态

### 告警配置

```yaml
monitoring:
  alerts:
    enabled: true
    thresholds:
      error_rate: 0.1        # 错误率超过10%
      response_time: 5000     # 响应时间超过5秒
      queue_size: 1000        # 队列积压超过1000
```

## 性能优化

### 连接池配置

```yaml
performance:
  connection_pooling:
    enabled: true
    max_connections_per_exchange: 20
    connection_timeout: 10
    read_timeout: 30
```

### 缓存策略

```yaml
performance:
  caching:
    enabled: true
    default_ttl: 300
    max_cache_size: 10000
    eviction_policy: "lru"
```

### 批处理

```yaml
performance:
  batch_processing:
    enabled: true
    batch_size: 50
    batch_timeout: 1.0
```

## 故障排除

### 常见问题

1. **Redis连接失败**
   ```bash
   # 检查Redis服务状态
   redis-cli ping

   # 检查网络连接
   telnet localhost 6379
   ```

2. **限流触发频繁**
   - 检查限流配置是否合理
   - 查看系统负载情况
   - 调整并发数量

3. **熔断器频繁开启**
   - 检查网络连接
   - 验证API端点可用性
   - 调整熔断器阈值

### 日志查看

```python
# 启用详细日志
import logging
logging.basicConfig(level=logging.DEBUG)

# 查看Redis中的日志
redis-cli LRANGE logs:system 0 -1
```

## 部署建议

### 生产环境配置

1. **Redis集群**: 使用Redis Cluster确保高可用
2. **负载均衡**: 部署多个实例实现负载均衡
3. **监控告警**: 集成Prometheus + Grafana监控
4. **日志聚合**: 使用ELK Stack收集日志

### Docker部署

```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
CMD ["python", "main.py"]
```

### Kubernetes部署

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: exchange-api-system
spec:
  replicas: 3
  selector:
    matchLabels:
      app: exchange-api-system
  template:
    metadata:
      labels:
        app: exchange-api-system
    spec:
      containers:
      - name: api-system
        image: exchange-api-system:latest
        ports:
        - containerPort: 8000
        env:
        - name: REDIS_HOST
          value: "redis-service"
```

## 贡献指南

1. Fork项目
2. 创建功能分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 打开Pull Request

## 许可证

本项目采用MIT许可证 - 查看 [LICENSE](LICENSE) 文件了解详情。

## 联系方式

- 项目链接: [https://github.com/yourusername/crypto_ai_trading](https://github.com/yourusername/crypto_ai_trading)
- 问题反馈: [Issues](https://github.com/yourusername/crypto_ai_trading/issues)

## 更新日志

### v1.0.0 (2024-01-01)
- 初始版本发布
- 支持5个主流交易所
- 完整的限流和降级功能
- 监控和告警系统