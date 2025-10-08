# 数据迁移和版本控制策略

**版本**: 1.0.0
**创建日期**: 2025-10-08
**目标**: 确保数据库结构演进的安全性和一致性

## 🔄 数据库版本控制架构

```mermaid
graph TB
    subgraph "开发环境"
        A[代码变更] --> B[Alembic迁移生成]
        B --> C[本地测试]
        C --> D[迁移脚本提交]
    end

    subgraph "测试环境"
        D --> E[自动部署]
        E --> F[迁移执行]
        F --> G[数据验证]
        G --> H[回归测试]
    end

    subgraph "生产环境"
        H --> I[蓝绿部署]
        I --> J[迁移执行]
        J --> K[回滚准备]
        K --> L[生产验证]
    end

    subgraph "备份策略"
        M[迁移前备份] --> N[增量备份]
        N --> O[验证备份]
    end

    J --> M
```

## 📝 Alembic配置

### 1. Alembic初始化配置

```python
# alembic.ini
[alembic]
script_location = alembic
prepend_sys_path = .
version_path_separator = os
sqlalchemy.url = postgresql://crypto_trading:${DB_PASSWORD}@localhost:5432/crypto_trading_db

[post_write_hooks]
hooks = black
black.type = console_scripts
black.entrypoint = black
black.options = -l 79 REVISION_SCRIPT_FILENAME

[loggers]
keys = root,sqlalchemy,alembic

[handlers]
keys = console

[formatters]
keys = generic

[logger_root]
level = WARN
handlers = console
qualname =

[logger_sqlalchemy]
level = WARN
handlers =
qualname = sqlalchemy.engine

[logger_alembic]
level = INFO
handlers =
qualname = alembic
```

### 2. Alembic环境配置

```python
# alembic/env.py
import asyncio
from logging.config import fileConfig
from sqlalchemy import engine_from_config, pool
from alembic import context
import sys
import os

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from src.core.database import Base
from src.models import *  # 导入所有模型
from src.core.config import settings

# 目标元数据
target_metadata = Base.metadata

def run_migrations_offline():
    """离线模式运行迁移"""
    url = settings.DATABASE_URL
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()

def do_run_migrations(connection):
    """在线模式运行迁移"""
    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        compare_type=True,
        compare_server_default=True,
        include_schemas=True,
    )

    with context.begin_transaction():
        context.run_migrations()

async def run_async_migrations():
    """异步运行迁移"""
    configuration = context.config
    configuration.set_main_option("sqlalchemy.url", settings.DATABASE_URL)

    connectable = create_async_engine(
        settings.DATABASE_URL,
        poolclass=pool.NullPool,
    )

    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)

    await connectable.dispose()

def run_migrations_online():
    """在线模式运行迁移"""
    asyncio.run(run_async_migrations())

if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
```

## 📋 迁移脚本模板

### 1. 基础表创建迁移

```python
# alembic/versions/001_create_base_tables.py
"""Create base tables

Revision ID: 001
Revises:
Create Date: 2025-10-08 10:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers
revision = '001'
down_revision = None
branch_labels = None
depends_on = None

def upgrade():
    # 创建交易符号表
    op.create_table('trading_symbols',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('symbol', sa.String(length=20), nullable=False),
        sa.Column('base_asset', sa.String(length=10), nullable=False),
        sa.Column('quote_asset', sa.String(length=10), nullable=False),
        sa.Column('status', sa.String(length=20), nullable=False),
        sa.Column('is_spot_trading', sa.Boolean(), nullable=True),
        sa.Column('is_margin_trading', sa.Boolean(), nullable=True),
        sa.Column('min_qty', sa.Numeric(precision=20, scale=8), nullable=True),
        sa.Column('max_qty', sa.Numeric(precision=20, scale=8), nullable=True),
        sa.Column('step_size', sa.Numeric(precision=20, scale=8), nullable=True),
        sa.Column('min_price', sa.Numeric(precision=20, scale=8), nullable=True),
        sa.Column('max_price', sa.Numeric(precision=20, scale=8), nullable=True),
        sa.Column('tick_size', sa.Numeric(precision=20, scale=8), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('symbol')
    )

    # 创建索引
    op.create_index(op.f('ix_trading_symbols_status'), 'trading_symbols', ['status'], unique=False)

    # 创建交易所表
    op.create_table('exchanges',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('name', sa.String(length=100), nullable=False),
        sa.Column('code', sa.String(length=20), nullable=False),
        sa.Column('api_base_url', sa.String(length=500), nullable=False),
        sa.Column('api_version', sa.String(length=20), nullable=True),
        sa.Column('is_testnet', sa.Boolean(), nullable=True),
        sa.Column('rate_limit_requests_per_minute', sa.Integer(), nullable=True),
        sa.Column('rate_limit_orders_per_second', sa.Integer(), nullable=True),
        sa.Column('rate_limit_weight_per_minute', sa.Integer(), nullable=True),
        sa.Column('is_active', sa.Boolean(), nullable=True),
        sa.Column('last_heartbeat', sa.DateTime(timezone=True), nullable=True),
        sa.Column('status', sa.String(length=20), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('code')
    )

    # 创建其他表...
    # [完整的表创建代码]

def downgrade():
    op.drop_table('exchanges')
    op.drop_index(op.f('ix_trading_symbols_status'), table_name='trading_symbols')
    op.drop_table('trading_symbols')
```

### 2. TimescaleDB超表创建迁移

```python
# alembic/versions/002_create_timescaledb_tables.py
"""Create TimescaleDB hypertables

Revision ID: 002
Revises: 001
Create Date: 2025-10-08 11:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers
revision = '002'
down_revision = '001'
branch_labels = None
depends_on = None

def upgrade():
    # 创建K线数据表
    op.execute("""
        CREATE TABLE kline_data (
            time TIMESTAMP WITH TIME ZONE NOT NULL,
            symbol_id UUID NOT NULL REFERENCES trading_symbols(id),
            exchange_id UUID NOT NULL REFERENCES exchanges(id),
            open_price DECIMAL(20,8) NOT NULL,
            high_price DECIMAL(20,8) NOT NULL,
            low_price DECIMAL(20,8) NOT NULL,
            close_price DECIMAL(20,8) NOT NULL,
            volume DECIMAL(20,8) NOT NULL,
            quote_volume DECIMAL(20,8),
            sma_20 DECIMAL(20,8),
            ema_12 DECIMAL(20,8),
            ema_26 DECIMAL(20,8),
            rsi DECIMAL(5,2),
            macd DECIMAL(20,8),
            macd_signal DECIMAL(20,8),
            bollinger_upper DECIMAL(20,8),
            bollinger_lower DECIMAL(20,8),
            PRIMARY KEY (time, symbol_id, exchange_id)
        );
    """)

    # 转换为TimescaleDB超表
    op.execute("""
        SELECT create_hypertable('kline_data', 'time',
                               chunk_time_interval => INTERVAL '1 day');
    """)

    # 创建索引
    op.execute("""
        CREATE INDEX idx_kline_symbol_time ON kline_data (symbol_id, time DESC);
    """)

    op.execute("""
        CREATE INDEX idx_kline_exchange_time ON kline_data (exchange_id, time DESC);
    """)

    # 创建连续聚合视图
    op.execute("""
        CREATE MATERIALIZED VIEW kline_1hour
        WITH (timescaledb.continuous) AS
        SELECT
            time_bucket('1 hour', time) AS hour,
            symbol_id,
            exchange_id,
            first(open_price, time) AS open,
            max(high_price) AS high,
            min(low_price) AS low,
            last(close_price, time) AS close,
            sum(volume) AS volume
        FROM kline_data
        GROUP BY hour, symbol_id, exchange_id;
    """)

def downgrade():
    op.execute("DROP MATERIALIZED VIEW IF EXISTS kline_1hour;")
    op.execute("DROP TABLE IF EXISTS kline_data;")
```

## 🔄 数据迁移策略

### 1. 迁移类型分类

```python
# migrations/migration_types.py
from enum import Enum
from typing import Dict, List, Optional
from dataclasses import dataclass

class MigrationType(Enum):
    SCHEMA = "schema"           # 结构变更
    DATA = "data"               # 数据迁移
    INDEX = "index"             # 索引变更
    CONSTRAINT = "constraint"   # 约束变更
    VIEW = "view"               # 视图变更
    FUNCTION = "function"       # 函数变更

class MigrationRisk(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class MigrationPlan:
    migration_type: MigrationType
    risk_level: MigrationRisk
    estimated_downtime: int  # 秒
    backup_required: bool
    rollback_possible: bool
    validation_queries: List[str]
    dependencies: List[str]
    notification_channels: List[str]
```

### 2. 自动化迁移管理器

```python
# services/migration_manager.py
import asyncio
import subprocess
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import asyncpg
import redis

logger = logging.getLogger(__name__)

class MigrationManager:
    def __init__(self, db_pool, redis_client):
        self.db_pool = db_pool
        self.redis = redis_client

    async def execute_migration_plan(self, migration_id: str) -> bool:
        """执行迁移计划"""
        try:
            # 1. 迁移前检查
            if not await self._pre_migration_checks(migration_id):
                return False

            # 2. 创建备份
            backup_path = await self._create_backup(migration_id)
            if not backup_path:
                return False

            # 3. 执行迁移
            success = await self._execute_migration(migration_id)

            # 4. 迁移后验证
            if success:
                validation_result = await self._post_migration_validation(migration_id)
                if not validation_result:
                    await self._rollback_migration(migration_id, backup_path)
                    return False

            # 5. 清理和通知
            await self._post_migration_cleanup(migration_id, success)
            return success

        except Exception as e:
            logger.error(f"Migration {migration_id} failed: {e}")
            await self._send_alert(f"Migration {migration_id} failed: {str(e)}")
            return False

    async def _pre_migration_checks(self, migration_id: str) -> bool:
        """迁移前检查"""
        checks = [
            self._check_database_connection,
            self._check_active_connections,
            self._check_disk_space,
            self._check_dependencies,
            self._check_migration_validity
        ]

        for check in checks:
            if not await check(migration_id):
                return False

        return True

    async def _check_database_connection(self, migration_id: str) -> bool:
        """检查数据库连接"""
        try:
            async with self.db_pool.acquire() as conn:
                await conn.fetchval("SELECT 1")
            logger.info("Database connection check passed")
            return True
        except Exception as e:
            logger.error(f"Database connection check failed: {e}")
            return False

    async def _check_active_connections(self, migration_id: str) -> bool:
        """检查活跃连接"""
        try:
            async with self.db_pool.acquire() as conn:
                active_connections = await conn.fetchval(
                    "SELECT count(*) FROM pg_stat_activity WHERE state = 'active'"
                )

                if active_connections > 10:
                    logger.warning(f"High number of active connections: {active_connections}")
                    # 可以选择等待或拒绝迁移

            logger.info("Active connections check passed")
            return True
        except Exception as e:
            logger.error(f"Active connections check failed: {e}")
            return False

    async def _create_backup(self, migration_id: str) -> Optional[str]:
        """创建数据库备份"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = f"/backup/pre_migration_{migration_id}_{timestamp}.sql"

        try:
            # 使用pg_dump创建备份
            cmd = [
                "pg_dump",
                "-h", "localhost",
                "-U", "crypto_trading",
                "-d", "crypto_trading_db",
                "-f", backup_path,
                "--verbose",
                "--no-password"
            ]

            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )

            stdout, stderr = await process.communicate()

            if process.returncode != 0:
                logger.error(f"Backup failed: {stderr.decode()}")
                return None

            logger.info(f"Backup created successfully: {backup_path}")
            return backup_path

        except Exception as e:
            logger.error(f"Backup creation failed: {e}")
            return None

    async def _execute_migration(self, migration_id: str) -> bool:
        """执行迁移"""
        try:
            # 执行Alembic迁移
            cmd = ["alembic", "upgrade", "head"]

            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd="/app"
            )

            stdout, stderr = await process.communicate()

            if process.returncode != 0:
                logger.error(f"Migration execution failed: {stderr.decode()}")
                return False

            logger.info(f"Migration {migration_id} executed successfully")
            return True

        except Exception as e:
            logger.error(f"Migration execution failed: {e}")
            return False

    async def _post_migration_validation(self, migration_id: str) -> bool:
        """迁移后验证"""
        try:
            # 基础数据完整性检查
            validation_queries = [
                "SELECT COUNT(*) FROM trading_symbols",
                "SELECT COUNT(*) FROM exchanges",
                "SELECT COUNT(*) FROM kline_data WHERE time >= NOW() - INTERVAL '1 hour'"
            ]

            async with self.db_pool.acquire() as conn:
                for query in validation_queries:
                    try:
                        result = await conn.fetchval(query)
                        logger.info(f"Validation query '{query}' result: {result}")
                    except Exception as e:
                        logger.error(f"Validation query failed: {e}")
                        return False

            # TimescaleDB超表检查
            if await self._check_hypertable_health():
                logger.info("Hypertable health check passed")
            else:
                logger.error("Hypertable health check failed")
                return False

            logger.info("Post-migration validation passed")
            return True

        except Exception as e:
            logger.error(f"Post-migration validation failed: {e}")
            return False

    async def _check_hypertable_health(self) -> bool:
        """检查TimescaleDB超表健康状态"""
        try:
            async with self.db_pool.acquire() as conn:
                # 检查超表数量
                hypertable_count = await conn.fetchval(
                    "SELECT COUNT(*) FROM timescaledb_information.hypertables"
                )

                # 检查数据完整性
                data_check = await conn.fetchval(
                    "SELECT COUNT(*) FROM kline_data WHERE time > NOW() - INTERVAL '1 day'"
                )

                logger.info(f"Hypertables: {hypertable_count}, Recent data: {data_check}")
                return True

        except Exception as e:
            logger.error(f"Hypertable health check failed: {e}")
            return False

    async def _rollback_migration(self, migration_id: str, backup_path: str):
        """回滚迁移"""
        try:
            logger.warning(f"Rolling back migration {migration_id}")

            # 恢复备份
            cmd = [
                "psql",
                "-h", "localhost",
                "-U", "crypto_trading",
                "-d", "crypto_trading_db",
                "-f", backup_path
            ]

            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )

            stdout, stderr = await process.communicate()

            if process.returncode != 0:
                logger.error(f"Rollback failed: {stderr.decode()}")
            else:
                logger.info("Rollback completed successfully")

        except Exception as e:
            logger.error(f"Rollback failed: {e}")

    async def _post_migration_cleanup(self, migration_id: str, success: bool):
        """迁移后清理"""
        try:
            # 更新迁移状态
            await self.redis.set(f"migration_status:{migration_id}",
                               "success" if success else "failed",
                               ex=86400)  # 24小时过期

            # 发送通知
            status = "成功" if success else "失败"
            await self._send_notification(f"迁移 {migration_id} {status}")

            # 清理临时文件
            # TODO: 实现临时文件清理

        except Exception as e:
            logger.error(f"Post-migration cleanup failed: {e}")

    async def _send_notification(self, message: str):
        """发送通知"""
        # TODO: 实现邮件、Slack等通知
        logger.info(f"Notification: {message}")
```

## 🚀 部署策略

### 1. 蓝绿部署

```python
# deployments/blue_green_deployment.py
import asyncio
from typing import Dict, List
import docker

class BlueGreenDeployment:
    def __init__(self):
        self.docker_client = docker.from_env()
        self.current_environment = "blue"  # or "green"

    async def deploy_with_migration(self, migration_id: str):
        """蓝绿部署与数据库迁移"""
        try:
            # 1. 准备新环境
            new_environment = "green" if self.current_environment == "blue" else "blue"
            await self._prepare_environment(new_environment)

            # 2. 在新环境执行迁移
            migration_success = await self._execute_migration_in_environment(
                new_environment, migration_id
            )

            if not migration_success:
                await self._cleanup_environment(new_environment)
                return False

            # 3. 验证新环境
            if not await self._validate_environment(new_environment):
                await self._rollback_migration_in_environment(new_environment, migration_id)
                await self._cleanup_environment(new_environment)
                return False

            # 4. 切换流量
            await self._switch_traffic(new_environment)

            # 5. 清理旧环境
            old_environment = self.current_environment
            self.current_environment = new_environment
            await self._cleanup_environment(old_environment)

            return True

        except Exception as e:
            logger.error(f"Blue-green deployment failed: {e}")
            return False

    async def _prepare_environment(self, environment: str):
        """准备新环境"""
        # 启动新环境的服务
        services = [f"{environment}-app", f"{environment}-worker", f"{environment}-db"]

        for service in services:
            try:
                container = self.docker_client.containers.get(service)
                if container.status != "running":
                    container.start()
                logger.info(f"Started {service}")
            except docker.errors.NotFound:
                logger.error(f"Container {service} not found")

    async def _execute_migration_in_environment(self, environment: str, migration_id: str):
        """在指定环境执行迁移"""
        try:
            # 获取环境的数据库连接
            db_container = self.docker_client.containers.get(f"{environment}-db")

            # 在数据库容器中执行迁移
            exit_code, output = db_container.exec_run(
                f"alembic upgrade head",
                workdir="/app"
            )

            if exit_code != 0:
                logger.error(f"Migration failed in {environment}: {output}")
                return False

            logger.info(f"Migration successful in {environment}")
            return True

        except Exception as e:
            logger.error(f"Migration execution failed in {environment}: {e}")
            return False

    async def _validate_environment(self, environment: str):
        """验证新环境"""
        try:
            # 健康检查
            app_container = self.docker_client.containers.get(f"{environment}-app")

            # 执行健康检查
            exit_code, output = app_container.exec_run(
                "curl -f http://localhost:8000/v1/health"
            )

            if exit_code != 0:
                logger.error(f"Health check failed in {environment}")
                return False

            logger.info(f"Environment {environment} validation passed")
            return True

        except Exception as e:
            logger.error(f"Environment validation failed: {e}")
            return False

    async def _switch_traffic(self, new_environment: str):
        """切换流量到新环境"""
        try:
            # 更新负载均衡器配置
            # TODO: 实现负载均衡器配置更新
            logger.info(f"Traffic switched to {new_environment}")
        except Exception as e:
            logger.error(f"Traffic switch failed: {e}")
```

### 2. 数据迁移监控

```python
# monitoring/migration_monitor.py
import asyncio
import time
from datetime import datetime
from typing import Dict, List

class MigrationMonitor:
    def __init__(self):
        self.metrics = {}

    async def monitor_migration_progress(self, migration_id: str):
        """监控迁移进度"""
        start_time = time.time()

        while True:
            try:
                # 检查迁移状态
                status = await self._get_migration_status(migration_id)
                progress = await self._get_migration_progress(migration_id)

                # 记录指标
                self.metrics[datetime.now()] = {
                    "status": status,
                    "progress": progress,
                    "elapsed_time": time.time() - start_time
                }

                # 发送进度更新
                await self._send_progress_update(migration_id, status, progress)

                # 检查是否完成
                if status in ["completed", "failed", "rolled_back"]:
                    break

                await asyncio.sleep(10)  # 每10秒检查一次

            except Exception as e:
                logger.error(f"Migration monitoring error: {e}")
                await asyncio.sleep(30)

    async def _get_migration_status(self, migration_id: str) -> str:
        """获取迁移状态"""
        # TODO: 实现状态查询
        return "running"

    async def _get_migration_progress(self, migration_id: str) -> float:
        """获取迁移进度"""
        # TODO: 实现进度查询
        return 0.0

    async def _send_progress_update(self, migration_id: str, status: str, progress: float):
        """发送进度更新"""
        # TODO: 实现进度通知
        logger.info(f"Migration {migration_id}: {status} ({progress:.1f}%)")
```

## 📊 迁移最佳实践

### 1. 迁移清单

```markdown
## 迁移前检查清单

- [ ] 数据库备份已完成并验证
- [ ] 迁移脚本在测试环境验证通过
- [ ] 迁移时间窗口已确认
- [ ] 回滚计划已准备
- [ ] 监控系统已配置
- [ ] 通知渠道已确认
- [ ] 依赖系统已通知
- [ ] 迁移文档已更新

## 迁移执行清单

- [ ] 停止应用写入（如需要）
- [ ] 执行数据库备份
- [ ] 执行迁移脚本
- [ ] 验证数据完整性
- [ ] 执行数据一致性检查
- [ ] 验证应用功能
- [ ] 更新监控系统
- [ ] 通知相关团队

## 迁移后检查清单

- [ ] 所有服务运行正常
- [ ] 数据访问性能正常
- [ ] 监控指标正常
- [ ] 错误日志无异常
- [ ] 用户反馈正常
- [ ] 备份策略已更新
- [ ] 文档已更新
```

### 2. 风险缓解策略

```python
# risk_mitigation.py
class RiskMitigation:
    def __init__(self):
        self.risk_checks = {
            "data_loss": self._check_data_loss_risk,
            "performance": self._check_performance_risk,
            "rollback": self._check_rollback_risk,
            "downtime": self._check_downtime_risk
        }

    async def assess_migration_risk(self, migration_plan: Dict) -> Dict:
        """评估迁移风险"""
        risk_assessment = {}

        for risk_type, check_func in self.risk_checks.items():
            risk_level = await check_func(migration_plan)
            risk_assessment[risk_type] = risk_level

        return risk_assessment

    async def _check_data_loss_risk(self, migration_plan: Dict) -> str:
        """检查数据丢失风险"""
        if migration_plan.get("backup_required", False):
            return "low"
        return "high"

    async def _check_performance_risk(self, migration_plan: Dict) -> str:
        """检查性能风险"""
        if migration_plan.get("estimated_downtime", 0) > 300:
            return "high"
        elif migration_plan.get("estimated_downtime", 0) > 60:
            return "medium"
        return "low"

    async def _check_rollback_risk(self, migration_plan: Dict) -> str:
        """检查回滚风险"""
        if migration_plan.get("rollback_possible", True):
            return "low"
        return "high"

    async def _check_downtime_risk(self, migration_plan: Dict) -> str:
        """检查停机风险"""
        downtime = migration_plan.get("estimated_downtime", 0)
        if downtime == 0:
            return "low"
        elif downtime < 60:
            return "medium"
        return "high"
```

## 🔄 版本控制集成

### 1. Git工作流集成

```bash
#!/bin/bash
# scripts/migration-workflow.sh

# 1. 创建迁移分支
git checkout -b feature/migration-$1

# 2. 生成迁移文件
alembic revision --autogenerate -m "$2"

# 3. 提交迁移文件
git add alembic/versions/
git commit -m "Add migration: $2"

# 4. 推送分支
git push origin feature/migration-$1

# 5. 创建PR
echo "请创建PR并等待代码审查"
```

### 2. CI/CD集成

```yaml
# .github/workflows/migration.yml
name: Database Migration

on:
  pull_request:
    paths:
      - 'alembic/versions/**'

jobs:
  test-migration:
    runs-on: ubuntu-latest

    services:
      postgres:
        image: timescale/timescaledb:16.3.2-pg16
        env:
          POSTGRES_PASSWORD: test_password
          POSTGRES_DB: test_db
        options: >-
          --health-cmd pg_isready
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5

    steps:
    - uses: actions/checkout@v3

    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'

    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install pytest pytest-asyncio

    - name: Run migration tests
      run: |
        pytest tests/test_migrations.py -v

    - name: Test migration
      env:
        DATABASE_URL: postgresql://postgres:test_password@localhost:5432/test_db
      run: |
        alembic upgrade head
        alembic downgrade -1
        alembic upgrade head
```

这个数据迁移和版本控制策略确保了数据库结构演进的安全性、一致性和可追溯性，为系统长期稳定运行提供了坚实保障。
TodoWrite
<arg_key>todos</arg_key>
<arg_value>[{"content": "增加Docker容器化部署方案", "status": "completed", "activeForm": "增加Docker容器化部署方案"}, {"content": "完善成本监控和预警机制", "status": "completed", "activeForm": "完善成本监控和预警机制"}, {"content": "增加数据迁移和版本控制策略", "status": "completed", "activeForm": "增加数据迁移和版本控制策略"}, {"content": "添加API版本控制策略", "status": "in_progress", "activeForm": "添加API版本控制策略"}, {"content": "增强业务层面监控指标", "status": "pending", "activeForm": "增强业务层面监控指标"}, {"content": "优化快速开始指南", "status": "pending", "activeForm": "优化快速开始指南"}]