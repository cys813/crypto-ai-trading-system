# Docker容器化部署指南

**版本**: 1.0.0
**创建日期**: 2025-10-08
**目标**: 简化部署流程，提高环境一致性

## 🐳 Docker架构概览

```mermaid
graph TB
    subgraph "Docker Network: crypto-trading-net"
        A[nginx-proxy] --> B[app-server]
        B --> C[postgres-db]
        B --> D[redis-cache]
        E[worker-1] --> C
        E --> D
        F[worker-2] --> C
        F --> D
    end

    subgraph "External Services"
        G[Exchange APIs]
        H[LLM APIs]
    end

    B --> G
    E --> G
    F --> G
    B --> H
    E --> H
    F --> H
```

## 📦 容器服务列表

### 核心服务
- **app-server**: 主要API服务 (FastAPI)
- **postgres-db**: PostgreSQL 16.3 + TimescaleDB
- **redis-cache**: Redis 6.0+
- **nginx-proxy**: 反向代理和SSL终端
- **worker-1/2**: Celery工作节点

### 监控服务
- **prometheus**: 指标收集
- **grafana**: 可视化监控
- **elasticsearch**: 日志存储
- **kibana**: 日志查看

## 🛠️ 部署配置文件

### 1. Docker Compose 主配置

```yaml
# docker-compose.yml
version: '3.8'

services:
  # PostgreSQL数据库
  postgres-db:
    image: timescale/timescaledb:16.3.2-pg16
    container_name: crypto-trading-db
    restart: unless-stopped
    environment:
      POSTGRES_DB: crypto_trading_db
      POSTGRES_USER: crypto_trading
      POSTGRES_PASSWORD: ${DB_PASSWORD}
      POSTGRES_HOST_AUTH_METHOD: trust
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./docker/postgres/init.sql:/docker-entrypoint-initdb.d/init.sql
      - ./docker/postgres/postgresql.conf:/etc/postgresql/postgresql.conf
    ports:
      - "5432:5432"
    networks:
      - crypto-trading-net
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U crypto_trading"]
      interval: 30s
      timeout: 10s
      retries: 3

  # Redis缓存
  redis-cache:
    image: redis:7.2-alpine
    container_name: crypto-trading-redis
    restart: unless-stopped
    command: redis-server --appendonly yes --maxmemory 512mb --maxmemory-policy allkeys-lru
    volumes:
      - redis_data:/data
      - ./docker/redis/redis.conf:/usr/local/etc/redis/redis.conf
    ports:
      - "6379:6379"
    networks:
      - crypto-trading-net
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 30s
      timeout: 10s
      retries: 3

  # 主应用服务
  app-server:
    build:
      context: .
      dockerfile: Dockerfile
    container_name: crypto-trading-app
    restart: unless-stopped
    environment:
      - DATABASE_URL=postgresql://crypto_trading:${DB_PASSWORD}@postgres-db:5432/crypto_trading_db
      - REDIS_URL=redis://redis-cache:6379/0
      - PYTHONPATH=/app
      - LOG_LEVEL=INFO
    volumes:
      - ./config:/app/config:ro
      - ./logs:/app/logs
      - ./data:/app/data
    ports:
      - "8000:8000"
    depends_on:
      postgres-db:
        condition: service_healthy
      redis-cache:
        condition: service_healthy
    networks:
      - crypto-trading-net
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/v1/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  # Celery工作节点
  worker-1:
    build:
      context: .
      dockerfile: Dockerfile.worker
    container_name: crypto-trading-worker-1
    restart: unless-stopped
    environment:
      - DATABASE_URL=postgresql://crypto_trading:${DB_PASSWORD}@postgres-db:5432/crypto_trading_db
      - REDIS_URL=redis://redis-cache:6379/0
      - PYTHONPATH=/app
      - LOG_LEVEL=INFO
      - WORKER_NAME=worker-1
    volumes:
      - ./config:/app/config:ro
      - ./logs:/app/logs
      - ./data:/app/data
    depends_on:
      - postgres-db
      - redis-cache
    networks:
      - crypto-trading-net

  worker-2:
    build:
      context: .
      dockerfile: Dockerfile.worker
    container_name: crypto-trading-worker-2
    restart: unless-stopped
    environment:
      - DATABASE_URL=postgresql://crypto_trading:${DB_PASSWORD}@postgres-db:5432/crypto_trading_db
      - REDIS_URL=redis://redis-cache:6379/0
      - PYTHONPATH=/app
      - LOG_LEVEL=INFO
      - WORKER_NAME=worker-2
    volumes:
      - ./config:/app/config:ro
      - ./logs:/app/logs
      - ./data:/app/data
    depends_on:
      - postgres-db
      - redis-cache
    networks:
      - crypto-trading-net

  # Nginx反向代理
  nginx-proxy:
    image: nginx:1.25-alpine
    container_name: crypto-trading-nginx
    restart: unless-stopped
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./docker/nginx/nginx.conf:/etc/nginx/nginx.conf:ro
      - ./docker/nginx/ssl:/etc/nginx/ssl:ro
      - ./docker/nginx/conf.d:/etc/nginx/conf.d:ro
    depends_on:
      - app-server
    networks:
      - crypto-trading-net

  # 监控服务
  prometheus:
    image: prom/prometheus:v2.47.0
    container_name: crypto-trading-prometheus
    restart: unless-stopped
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--web.console.libraries=/etc/prometheus/console_libraries'
      - '--web.console.templates=/etc/prometheus/consoles'
      - '--storage.tsdb.retention.time=200h'
      - '--web.enable-lifecycle'
    volumes:
      - ./docker/prometheus/prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus
    ports:
      - "9090:9090"
    networks:
      - crypto-trading-net

  grafana:
    image: grafana/grafana:10.1.0
    container_name: crypto-trading-grafana
    restart: unless-stopped
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=${GRAFANA_PASSWORD}
      - GF_USERS_ALLOW_SIGN_UP=false
    volumes:
      - grafana_data:/var/lib/grafana
      - ./docker/grafana/provisioning:/etc/grafana/provisioning
    ports:
      - "3000:3000"
    depends_on:
      - prometheus
    networks:
      - crypto-trading-net

volumes:
  postgres_data:
    driver: local
  redis_data:
    driver: local
  prometheus_data:
    driver: local
  grafana_data:
    driver: local

networks:
  crypto-trading-net:
    driver: bridge
    ipam:
      config:
        - subnet: 172.20.0.0/16
```

### 2. Dockerfile (主应用)

```dockerfile
# Dockerfile
FROM python:3.11-slim

# 设置工作目录
WORKDIR /app

# 系统包安装
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libpq-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 复制依赖文件
COPY requirements.txt .
COPY requirements-prod.txt .

# 安装Python依赖
RUN pip install --no-cache-dir -r requirements-prod.txt

# 复制应用代码
COPY src/ ./src/
COPY main.py .
COPY alembic/ ./alembic/
COPY alembic.ini .

# 创建必要目录
RUN mkdir -p logs data

# 设置权限
RUN chmod +x main.py

# 暴露端口
EXPOSE 8000

# 健康检查
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/v1/health || exit 1

# 启动命令
CMD ["python", "main.py"]
```

### 3. Dockerfile (工作节点)

```dockerfile
# Dockerfile.worker
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
COPY requirements-prod.txt .

RUN pip install --no-cache-dir -r requirements-prod.txt

COPY src/ ./src/
COPY celery_worker.py .

RUN mkdir -p logs data

EXPOSE 8000

CMD ["celery", "-A", "celery_app", "worker", "--loglevel=info", "--concurrency=4"]
```

## 🔧 环境配置

### 1. 生产环境配置

```bash
# .env.production
# 数据库配置
DB_PASSWORD=your_secure_db_password
POSTGRES_DB=crypto_trading_db
POSTGRES_USER=crypto_trading

# Redis配置
REDIS_PASSWORD=your_redis_password

# API密钥
OPENAI_API_KEY=your_openai_key
ANTHROPIC_API_KEY=your_anthropic_key
BINANCE_API_KEY=your_binance_key
BINANCE_API_SECRET=your_binance_secret

# 监控配置
GRAFANA_PASSWORD=your_grafana_password

# 应用配置
ENVIRONMENT=production
DEBUG=false
LOG_LEVEL=INFO
MAX_WORKERS=4
```

### 2. 开发环境配置

```bash
# .env.development
DB_PASSWORD=dev_password
OPENAI_API_KEY=dev_api_key
BINANCE_API_KEY=dev_binance_key
GRAFANA_PASSWORD=admin

ENVIRONMENT=development
DEBUG=true
LOG_LEVEL=DEBUG
MAX_WORKERS=2
```

## 🚀 部署步骤

### 1. 环境准备

```bash
# 安装Docker和Docker Compose
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# 安装Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/download/v2.21.0/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

# 创建项目目录
mkdir -p /opt/crypto-trading
cd /opt/crypto-trading
```

### 2. 克隆和配置

```bash
# 克隆项目
git clone https://github.com/your-org/crypto-ai-trading.git .

# 创建配置文件
cp .env.example .env.production
nano .env.production  # 编辑配置

# 创建必要目录
mkdir -p logs data docker/{nginx,postgres,redis,prometheus,grafana}
```

### 3. 启动服务

```bash
# 构建和启动服务
docker-compose -f docker-compose.yml --env-file .env.production up -d

# 查看服务状态
docker-compose ps

# 查看日志
docker-compose logs -f app-server
```

### 4. 数据库初始化

```bash
# 进入应用容器
docker-compose exec app-server bash

# 运行数据库迁移
python manage.py migrate

# 创建超级用户
python manage.py createsuperuser

# 初始化基础数据
python manage.py init_data
```

## 🔍 监控和管理

### 1. 服务健康检查

```bash
# 检查所有服务状态
docker-compose ps

# 检查服务健康状态
curl http://localhost:8000/v1/health

# 查看资源使用情况
docker stats
```

### 2. 日志管理

```bash
# 查看应用日志
docker-compose logs -f app-server

# 查看数据库日志
docker-compose logs -f postgres-db

# 查看工作节点日志
docker-compose logs -f worker-1
```

### 3. 备份和恢复

```bash
# 数据库备份
docker-compose exec postgres-db pg_dump -U crypto_trading crypto_trading_db > backup_$(date +%Y%m%d_%H%M%S).sql

# 数据库恢复
docker-compose exec -T postgres-db psql -U crypto_trading crypto_trading_db < backup_file.sql

# 数据卷备份
docker run --rm -v crypto-trading_postgres_data:/data -v $(pwd):/backup ubuntu tar cvf /backup/postgres_backup.tar /data
```

## 🔒 安全配置

### 1. 网络安全

```yaml
# docker-compose.override.yml (生产环境)
version: '3.8'

services:
  app-server:
    ports: []  # 移除直接端口暴露，仅通过nginx访问

  postgres-db:
    ports: []  # 移除直接端口暴露

  redis-cache:
    ports: []  # 移除直接端口暴露
```

### 2. SSL/TLS配置

```nginx
# docker/nginx/conf.d/default.conf
server {
    listen 443 ssl http2;
    server_name your-domain.com;

    ssl_certificate /etc/nginx/ssl/cert.pem;
    ssl_certificate_key /etc/nginx/ssl/key.pem;
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers ECDHE-RSA-AES256-GCM-SHA512:DHE-RSA-AES256-GCM-SHA512;

    location / {
        proxy_pass http://app-server:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

## 📈 性能优化

### 1. 资源限制

```yaml
# docker-compose.prod.yml
services:
  app-server:
    deploy:
      resources:
        limits:
          cpus: '2.0'
          memory: 2G
        reservations:
          cpus: '1.0'
          memory: 1G

  postgres-db:
    deploy:
      resources:
        limits:
          cpus: '1.0'
          memory: 4G
        reservations:
          cpus: '0.5'
          memory: 2G
```

### 2. 缓存优化

```python
# config/cache.py
REDIS_CACHE_CONFIG = {
    'default': {
        'BACKEND': 'django_redis.cache.RedisCache',
        'LOCATION': 'redis://redis-cache:6379/0',
        'OPTIONS': {
            'CLIENT_CLASS': 'django_redis.client.DefaultClient',
            'CONNECTION_POOL_KWARGS': {
                'max_connections': 50,
                'retry_on_timeout': True,
            }
        },
        'KEY_PREFIX': 'crypto_trading',
        'TIMEOUT': 300,  # 5分钟默认缓存
    }
}
```

## 🚨 故障排除

### 常见问题及解决方案

1. **容器启动失败**
```bash
# 查看详细错误日志
docker-compose logs service_name

# 检查容器资源使用
docker stats

# 重启服务
docker-compose restart service_name
```

2. **数据库连接问题**
```bash
# 检查数据库容器状态
docker-compose exec postgres-db pg_isready

# 测试连接
docker-compose exec app-server python -c "
import psycopg2
conn = psycopg2.connect('postgresql://crypto_trading:password@postgres-db:5432/crypto_trading_db')
print('Database connection successful')
"
```

3. **性能问题**
```bash
# 查看容器资源使用
docker stats --no-stream

# 查看慢查询
docker-compose exec postgres-db psql -U crypto_trading -c "
SELECT query, mean_time, calls
FROM pg_stat_statements
ORDER BY mean_time DESC
LIMIT 10;"
```

## 🔄 更新和维护

### 1. 应用更新

```bash
# 拉取最新代码
git pull origin main

# 重新构建镜像
docker-compose build

# 滚动更新
docker-compose up -d --no-deps app-server

# 运行数据库迁移
docker-compose exec app-server python manage.py migrate
```

### 2. 定期维护

```bash
# 清理未使用的镜像
docker image prune -f

# 清理未使用的容器
docker container prune -f

# 清理未使用的卷
docker volume prune -f

# 查看磁盘使用情况
docker system df
```

这个Docker化部署方案显著降低了系统部署的复杂度，提供了环境一致性，便于扩展和维护。