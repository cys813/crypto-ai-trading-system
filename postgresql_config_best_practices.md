# 加密货币量化交易系统PostgreSQL配置最佳实践

## 1. PostgreSQL版本选择建议

### 推荐版本：PostgreSQL 16+
- **PostgreSQL 16.0+**：相比15版本在只读工作负载中性能提升约15-20%，写入性能相当
- **PostgreSQL 17**：进一步增强的并行查询支持、改进的逻辑复制、更优的WAL压缩
- **稳定性考虑**：建议使用PostgreSQL 16.3+的稳定版本，避免最新小版本的潜在问题

### 金融交易系统特性要求
- **数据一致性**：绝对不能关闭fsync和synchronous_commit
- **高并发性**：需要处理大量并发交易请求
- **时间序列优化**：K线数据需要专门的时间序列优化
- **低延迟**：交易决策系统对查询延迟要求极高

## 2. 时间序列数据（K线）优化配置

### TimescaleDB扩展配置

```sql
-- 安装TimescaleDB扩展
CREATE EXTENSION IF NOT EXISTS timescaledb;

-- 创建K线数据超表
CREATE TABLE klines (
    time TIMESTAMPTZ NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    interval VARCHAR(10) NOT NULL,
    open_price DECIMAL(20,8),
    high_price DECIMAL(20,8),
    low_price DECIMAL(20,8),
    close_price DECIMAL(20,8),
    volume DECIMAL(30,8),
    quote_volume DECIMAL(30,8),
    trade_count INTEGER,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- 创建超表
SELECT create_hypertable('klines', 'time', 'symbol', 4);

-- 创建时间+符号的复合索引
CREATE INDEX idx_klines_symbol_time ON klines (symbol, time DESC);
CREATE INDEX idx_klines_time_symbol ON klines (time, symbol);

-- 创建连续聚合视图用于快速查询
CREATE MATERIALIZED VIEW klines_1h_7d
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 hour', time) AS hour,
    symbol,
    FIRST(open_price, time) AS open,
    MAX(high_price) AS high,
    MIN(low_price) AS low,
    LAST(close_price, time) AS close,
    SUM(volume) AS volume,
    SUM(quote_volume) AS quote_volume
FROM klines
WHERE time >= NOW() - INTERVAL '7 days'
GROUP BY hour, symbol;

-- 添加连续聚合策略
ADD CONTINUOUS AGGREGATE POLICY klines_1h_7d_policy
ON klines_1h_7d
WITH (
    start_offset => INTERVAL '1 hour',
    end_offset => INTERVAL '10 minutes',
    schedule_interval => INTERVAL '5 minutes'
);

-- 设置数据保留策略（例如保留1年数据）
SELECT add_retention_policy('klines', INTERVAL '1 year');
```

### 分区表配置（原生PostgreSQL）

```sql
-- 创建分区表
CREATE TABLE klines_partitioned (
    time TIMESTAMPTZ NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    interval VARCHAR(10) NOT NULL,
    open_price DECIMAL(20,8),
    high_price DECIMAL(20,8),
    low_price DECIMAL(20,8),
    close_price DECIMAL(20,8),
    volume DECIMAL(30,8),
    quote_volume DECIMAL(30,8),
    trade_count INTEGER,
    created_at TIMESTAMPTZ DEFAULT NOW()
) PARTITION BY RANGE (time);

-- 创建按月分区
CREATE TABLE klines_2024_01 PARTITION OF klines_partitioned
FOR VALUES FROM ('2024-01-01') TO ('2024-02-01');

CREATE TABLE klines_2024_02 PARTITION OF klines_partitioned
FOR VALUES FROM ('2024-02-01') TO ('2024-03-01');

-- 自动创建分区的存储过程
CREATE OR REPLACE FUNCTION create_monthly_partitions()
RETURNS void AS $$
DECLARE
    start_date date;
    end_date date;
    partition_name text;
BEGIN
    -- 为未来3个月创建分区
    FOR i IN 0..2 LOOP
        start_date := date_trunc('month', CURRENT_DATE + interval '1 month' * i);
        end_date := start_date + interval '1 month';
        partition_name := 'klines_' || to_char(start_date, 'YYYY_MM');

        EXECUTE format('CREATE TABLE IF NOT EXISTS %I PARTITION OF klines_partitioned
                       FOR VALUES FROM (%L) TO (%L)',
                       partition_name, start_date, end_date);
    END LOOP;
END;
$$ LANGUAGE plpgsql;

-- 创建索引
CREATE INDEX CONCURRENTLY idx_klines_symbol_time_partitioned
ON klines_partitioned (symbol, time DESC);
```

## 3. 高并发读写性能调优

### postgresql.conf核心配置

```ini
# 连接配置
listen_addresses = '*'
port = 5432
max_connections = 200
superuser_reserved_connections = 3

# 内存配置（假设32GB RAM）
shared_buffers = 8GB                    # 25% of RAM
effective_cache_size = 24GB             # 75% of RAM
work_mem = 64MB                         # 每个查询操作内存
maintenance_work_mem = 1GB              # 维护操作内存
max_parallel_maintenance_workers = 4    # 并行维护工作进程

# WAL配置
wal_buffers = 64MB
wal_level = replica
max_wal_size = 4GB
min_wal_size = 1GB
checkpoint_timeout = 15min
checkpoint_completion_target = 0.9
wal_writer_delay = 200ms
max_wal_senders = 10

# 查询规划器配置
random_page_cost = 1.1                  # SSD优化
effective_io_concurrency = 200          # SSD并发I/O
seq_page_cost = 1.0
default_statistics_target = 100         # 统计信息精度

# 并行查询配置
max_parallel_workers_per_gather = 4
max_parallel_workers = 8
parallel_tuple_cost = 0.1
parallel_setup_cost = 1000.0

# 自动清理配置
autovacuum = on
autovacuum_max_workers = 4
autovacuum_naptime = 1min
autovacuum_vacuum_threshold = 50
autovacuum_analyze_threshold = 50
autovacuum_vacuum_scale_factor = 0.05
autovacuum_analyze_scale_factor = 0.02
autovacuum_vacuum_cost_delay = 2ms
autovacuum_vacuum_cost_limit = -1

# 日志配置
logging_collector = on
log_directory = 'log'
log_filename = 'postgresql-%Y-%m-%d_%H%M%S.log'
log_rotation_age = 1d
log_rotation_size = 100MB
log_min_duration_statement = 1000       # 记录超过1秒的查询
log_checkpoints = on
log_connections = on
log_disconnections = on
log_lock_waits = on

# 预加载库
shared_preload_libraries = 'timescaledb,pg_stat_statements'
pg_stat_statements.max = 10000
pg_stat_statements.track = all
```

### 交易数据表优化

```sql
-- 订单表
CREATE TABLE orders (
    id BIGSERIAL PRIMARY KEY,
    order_id VARCHAR(64) UNIQUE NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    side VARCHAR(10) NOT NULL,
    order_type VARCHAR(20) NOT NULL,
    quantity DECIMAL(30,8) NOT NULL,
    price DECIMAL(20,8),
    status VARCHAR(20) NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    exchange_id VARCHAR(50),
    strategy_id VARCHAR(100)
) PARTITION BY RANGE (created_at);

-- 按日分区
CREATE TABLE orders_2024_01_01 PARTITION OF orders
FOR VALUES FROM ('2024-01-01') TO ('2024-01-02');

-- 索引优化
CREATE INDEX CONCURRENTLY idx_orders_symbol_status_created
ON orders (symbol, status, created_at DESC);
CREATE INDEX CONCURRENTLY idx_orders_strategy_created
ON orders (strategy_id, created_at DESC);
CREATE INDEX CONCURRENTLY idx_orders_exchange_created
ON orders (exchange_id, created_at DESC);

-- 交易记录表
CREATE TABLE trades (
    id BIGSERIAL PRIMARY KEY,
    trade_id VARCHAR(64) UNIQUE NOT NULL,
    order_id VARCHAR(64) NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    side VARCHAR(10) NOT NULL,
    quantity DECIMAL(30,8) NOT NULL,
    price DECIMAL(20,8) NOT NULL,
    fee DECIMAL(20,8) DEFAULT 0,
    fee_currency VARCHAR(10),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    exchange_id VARCHAR(50)
) PARTITION BY RANGE (created_at);

-- 按日分区
CREATE TABLE trades_2024_01_01 PARTITION OF trades
FOR VALUES FROM ('2024-01-01') TO ('2024-01-02');

-- 索引优化
CREATE INDEX CONCURRENTLY idx_trades_symbol_created
ON trades (symbol, created_at DESC);
CREATE INDEX CONCURRENTLY idx_trades_order_id
ON trades (order_id);
```

## 4. 数据备份和恢复策略

### WAL连续归档配置

```ini
# postgresql.conf
wal_level = replica
archive_mode = on
archive_command = 'cp %p /backup/wal/%f'
wal_compression = on
max_wal_senders = 10
wal_keep_segments = 64
```

### pgBackRest备份配置

```ini
# /etc/pgbackrest/pgbackrest.conf
[global]
repo1-path=/backup/pgbackrest
repo1-retention-full=30
repo1-retention-diff=7
process-max=4
log-level-console=info
log-level-file=debug

[my-db]
db-host=localhost
db-path=/var/lib/postgresql/16/main
db-port=5432
db-user=postgres
```

### 备份脚本

```bash
#!/bin/bash
# backup_script.sh

# 全量备份（每周日）
pgbackrest --config=/etc/pgbackrest/pgbackrest.conf \
    --stanza=my-db --type=full backup

# 增量备份（其他日期）
pgbackrest --config=/etc/pgbackrest/pgbackrest.conf \
    --stanza=my-db --type=incr backup

# 验证备份
pgbackrest --config=/etc/pgbackrest/pgbackrest.conf \
    --stanza=my-db --type=incr --no-archive-timeout backup
```

### 恢复策略

```bash
# 时间点恢复
pgbackrest --config=/etc/pgbackrest/pgbackrest.conf \
    --stanza=my-db --type=time \
    --target="2024-01-15 14:30:00" \
    delta restore

# 基于最新备份恢复
pgbackrest --config=/etc/pgbackrest/pgbackrest.conf \
    --stanza=my-db delta restore
```

### 主从复制配置

```ini
# 主服务器配置
max_wal_senders = 10
wal_level = replica
archive_mode = on
wal_keep_segments = 64

# 从服务器配置
hot_standby = on
max_standby_streaming_delay = 30s
max_standby_archive_delay = 30s
```

## 5. 连接池和索引优化

### PgBouncer配置

```ini
# /etc/pgbouncer/pgbouncer.ini
[databases]
trading_db = host=localhost port=5432 dbname=trading_db
klines_db = host=localhost port=5432 dbname=klines_db

[pgbouncer]
listen_port = 6432
listen_addr = 127.0.0.1
auth_type = md5
auth_file = /etc/pgbouncer/userlist.txt
logfile = /var/log/pgbouncer/pgbouncer.log
pidfile = /var/run/pgbouncer/pgbouncer.pid
admin_users = postgres
stats_users = stats, postgres

# 连接池模式
pool_mode = transaction

# 连接数配置
max_client_conn = 1000
default_pool_size = 20
min_pool_size = 5
reserve_pool_size = 5
reserve_pool_timeout = 5
max_db_connections = 50
max_user_connections = 50

# 超时配置
server_reset_query = DISCARD ALL
server_check_delay = 30
server_check_query = select 1
server_lifetime = 3600
server_idle_timeout = 600
```

### 应用端连接池配置

#### HikariCP (Java)
```properties
# 连接池配置
spring.datasource.hikari.minimum-idle=1
spring.datasource.hikari.maximum-pool-size=15
spring.datasource.hikari.idle-timeout=600000
spring.datasource.hikari.max-lifetime=1800000
spring.datasource.hikari.connection-timeout=30000

# 性能优化
spring.datasource.hikari.pool-name=TradingHikariPool
spring.datasource.hikari.connection-test-query=SELECT 1
spring.datasource.hikari.leak-detection-threshold=60000
```

#### GORM (Go)
```go
sqlDB, err := db.DB()
if err != nil {
    log.Fatal(err)
}

// 连接池配置
sqlDB.SetMaxIdleConns(1)                    // 最小空闲连接
sqlDB.SetMaxOpenConns(15)                   // 最大连接数
sqlDB.SetConnMaxLifetime(time.Hour)         // 连接最大生命周期
sqlDB.SetConnMaxIdleTime(10 * time.Minute)  // 空闲连接超时
```

### 索引优化策略

```sql
-- 复合索引设计（按查询频率排序）
CREATE INDEX CONCURRENTLY idx_klines_symbol_time_interval
ON klines (symbol, time DESC, interval);

-- 条件索引（针对活跃交易对）
CREATE INDEX CONCURRENTLY idx_active_klines
ON klines (time DESC)
WHERE symbol IN ('BTCUSDT', 'ETHUSDT', 'BNBUSDT');

-- 部分索引（只索引未完成订单）
CREATE INDEX CONCURRENTLY idx_active_orders
ON orders (symbol, created_at DESC)
WHERE status IN ('PENDING', 'PARTIALLY_FILLED');

-- 包含列的索引（覆盖索引）
CREATE INDEX CONCURRENTLY idx_trades_covering
ON trades (symbol, created_at DESC)
INCLUDE (price, quantity, side);

-- JSON字段索引（如果使用JSON存储策略配置）
CREATE INDEX CONCURRENTLY idx_strategy_config_gin
ON strategies USING GIN (config);

-- 表达式索引（按小时分组查询优化）
CREATE INDEX CONCURRENTLY idx_klines_hour
ON klines (date_trunc('hour', time), symbol);

-- 定期索引维护
REINDEX INDEX CONCURRENTLY idx_klines_symbol_time_interval;
ANALYZE klines;
```

### 查询优化示例

```sql
-- 使用时间范围查询优化
EXPLAIN (ANALYZE, BUFFERS)
SELECT * FROM klines
WHERE symbol = 'BTCUSDT'
  AND time >= '2024-01-01'
  AND time < '2024-01-02'
  AND interval = '1m'
ORDER BY time DESC
LIMIT 1000;

-- 使用并行查询优化
SET max_parallel_workers_per_gather = 4;
SET parallel_setup_cost = 1000;
SET parallel_tuple_cost = 0.1;

-- 批量插入优化
INSERT INTO klines (time, symbol, interval, open_price, high_price, low_price, close_price, volume)
VALUES
    ('2024-01-01 00:00:00', 'BTCUSDT', '1m', 42000.00, 42050.00, 41980.00, 42020.00, 15.5),
    ('2024-01-01 00:01:00', 'BTCUSDT', '1m', 42020.00, 42080.00, 42000.00, 42065.00, 18.2)
ON CONFLICT (time, symbol, interval) DO UPDATE SET
    open_price = EXCLUDED.open_price,
    high_price = EXCLUDED.high_price,
    low_price = EXCLUDED.low_price,
    close_price = EXCLUDED.close_price,
    volume = EXCLUDED.volume;
```

## 6. 监控和维护

### 性能监控查询

```sql
-- 查看慢查询
SELECT query, calls, total_exec_time, mean_exec_time, rows
FROM pg_stat_statements
ORDER BY mean_exec_time DESC
LIMIT 10;

-- 查看表大小
SELECT
    schemaname,
    tablename,
    pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) as size,
    pg_total_relation_size(schemaname||'.'||tablename) as size_bytes
FROM pg_tables
WHERE schemaname = 'public'
ORDER BY size_bytes DESC;

-- 查看索引使用情况
SELECT
    schemaname,
    tablename,
    indexname,
    idx_scan,
    idx_tup_read,
    idx_tup_fetch
FROM pg_stat_user_indexes
ORDER BY idx_scan DESC;

-- 查看锁等待
SELECT
    blocked_locks.pid AS blocked_pid,
    blocked_activity.usename AS blocked_user,
    blocking_locks.pid AS blocking_pid,
    blocking_activity.usename AS blocking_user,
    blocked_activity.query AS blocked_statement,
    blocking_activity.query AS current_statement_in_blocking_process
FROM pg_catalog.pg_locks blocked_locks
JOIN pg_catalog.pg_stat_activity blocked_activity ON blocked_activity.pid = blocked_locks.pid
JOIN pg_catalog.pg_locks blocking_locks ON blocking_locks.locktype = blocked_locks.locktype
JOIN pg_catalog.pg_stat_activity blocking_activity ON blocking_activity.pid = blocking_locks.pid
WHERE NOT blocked_locks.granted;
```

### 自动维护脚本

```bash
#!/bin/bash
# maintenance_script.sh

# 更新统计信息
psql -d trading_db -c "ANALYZE;"

# 重建索引（低峰期执行）
psql -d trading_db -c "REINDEX DATABASE trading_db;"

# 清理WAL文件
pgbackrest --config=/etc/pgbackrest/pgbackrest.conf --stanza=my-db expire

# 检查数据库健康状况
psql -d trading_db -c "
SELECT
    'Table Size (GB)' as metric,
    round(sum(pg_total_relation_size(schemaname||'.'||tablename))/1024.0/1024.0/1024.0, 2) as value
FROM pg_tables
WHERE schemaname = 'public'
UNION ALL
SELECT
    'Database Size (GB)',
    round(pg_database_size(current_database())/1024.0/1024.0/1024.0, 2)
UNION ALL
SELECT
    'Active Connections',
    count(*)
FROM pg_stat_activity
WHERE state = 'active';
"
```

## 总结

本配置方案针对加密货币量化交易系统的特殊需求，提供了完整的PostgreSQL优化配置。关键要点包括：

1. **版本选择**：推荐PostgreSQL 16+，平衡性能和稳定性
2. **时间序列优化**：使用TimescaleDB或原生分区表处理K线数据
3. **性能调优**：针对高并发和低延迟场景优化内存和连接参数
4. **数据安全**：实施WAL归档、主从复制和定期备份策略
5. **连接管理**：使用PgBouncer和应用端连接池优化连接使用
6. **索引策略**：设计适合交易查询模式的复合索引和部分索引

这套配置能够支持每秒数千次交易请求，处理TB级的时间序列数据，同时确保数据的一致性和可靠性。