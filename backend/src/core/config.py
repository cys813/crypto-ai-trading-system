"""
配置管理模块

提供环境变量和配置文件管理功能。
"""

import os
from pathlib import Path
from typing import List, Optional
from pydantic import BaseSettings, Field, validator
from pydantic_settings import SettingsConfigDict


class Settings(BaseSettings):
    """应用配置类"""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )

    # 应用基础配置
    APP_NAME: str = "Crypto AI Trading System"
    VERSION: str = "0.1.0"
    DEBUG: bool = Field(default=False, env="DEBUG")
    ENVIRONMENT: str = Field(default="development", env="ENVIRONMENT")
    LOG_LEVEL: str = Field(default="INFO", env="LOG_LEVEL")
    SECRET_KEY: str = Field(..., env="SECRET_KEY")

    # 服务器配置
    HOST: str = Field(default="0.0.0.0", env="HOST")
    PORT: int = Field(default=8000, env="PORT")
    ALLOWED_ORIGINS: List[str] = Field(
        default=["http://localhost:3000", "http://localhost:8080"],
        env="ALLOWED_ORIGINS"
    )
    ALLOWED_HOSTS: List[str] = Field(
        default=["localhost", "127.0.0.1"],
        env="ALLOWED_HOSTS"
    )

    # 数据库配置
    DATABASE_URL: str = Field(
        default="postgresql+asyncpg://crypto_trading:password@localhost:5432/crypto_trading_db",
        env="DATABASE_URL"
    )
    DATABASE_POOL_SIZE: int = Field(default=10, env="DATABASE_POOL_SIZE")
    DATABASE_MAX_OVERFLOW: int = Field(default=20, env="DATABASE_MAX_OVERFLOW")
    DATABASE_POOL_TIMEOUT: int = Field(default=30, env="DATABASE_POOL_TIMEOUT")

    # Redis配置
    REDIS_URL: str = Field(
        default="redis://localhost:6379/0",
        env="REDIS_URL"
    )
    REDIS_PASSWORD: Optional[str] = Field(default=None, env="REDIS_PASSWORD")
    REDIS_DB: int = Field(default=0, env="REDIS_DB")
    REDIS_MAX_CONNECTIONS: int = Field(default=20, env="REDIS_MAX_CONNECTIONS")

    # LLM API配置
    OPENAI_API_KEY: Optional[str] = Field(default=None, env="OPENAI_API_KEY")
    OPENAI_MODEL: str = Field(default="gpt-4", env="OPENAI_MODEL")
    OPENAI_MAX_TOKENS: int = Field(default=4096, env="OPENAI_MAX_TOKENS")
    OPENAI_TEMPERATURE: float = Field(default=0.1, env="OPENAI_TEMPERATURE")

    ANTHROPIC_API_KEY: Optional[str] = Field(default=None, env="ANTHROPIC_API_KEY")
    ANTHROPIC_MODEL: str = Field(default="claude-3-sonnet-20240229", env="ANTHROPIC_MODEL")
    ANTHROPIC_MAX_TOKENS: int = Field(default=4096, env="ANTHROPIC_MAX_TOKENS")

    # 交易所API配置
    BINANCE_API_KEY: Optional[str] = Field(default=None, env="BINANCE_API_KEY")
    BINANCE_API_SECRET: Optional[str] = Field(default=None, env="BINANCE_API_SECRET")
    BINANCE_TESTNET: bool = Field(default=False, env="BINANCE_TESTNET")

    COINBASE_API_KEY: Optional[str] = Field(default=None, env="COINBASE_API_KEY")
    COINBASE_API_SECRET: Optional[str] = Field(default=None, env="COINBASE_API_SECRET")
    COINBASE_API_PASSPHRASE: Optional[str] = Field(default=None, env="COINBASE_API_PASSPHRASE")

    KRAKEN_API_KEY: Optional[str] = Field(default=None, env="KRAKEN_API_KEY")
    KRAKEN_API_SECRET: Optional[str] = Field(default=None, env="KRAKEN_API_SECRET")

    # 交易配置
    TRADING_ENABLED: bool = Field(default=False, env="TRADING_ENABLED")
    MAX_CONCURRENT_ANALYSIS: int = Field(default=20, env="MAX_CONCURRENT_ANALYSIS")
    ANALYSIS_INTERVAL_MINUTES: int = Field(default=1, env="ANALYSIS_INTERVAL_MINUTES")
    DEFAULT_RISK_LEVEL: str = Field(default="medium", env="DEFAULT_RISK_LEVEL")
    MAX_POSITION_SIZE_PERCENT: float = Field(default=0.25, env="MAX_POSITION_SIZE_PERCENT")

    # 安全配置
    ACCESS_TOKEN_EXPIRE_MINUTES: int = Field(default=30, env="ACCESS_TOKEN_EXPIRE_MINUTES")
    REFRESH_TOKEN_EXPIRE_DAYS: int = Field(default=7, env="REFRESH_TOKEN_EXPIRE_DAYS")
    ALGORITHM: str = Field(default="HS256", env="ALGORITHM")

    # 监控配置
    PROMETHEUS_PORT: int = Field(default=9090, env="PROMETHEUS_PORT")
    METRICS_ENABLED: bool = Field(default=True, env="METRICS_ENABLED")
    HEALTH_CHECK_INTERVAL: int = Field(default=60, env="HEALTH_CHECK_INTERVAL")

    # 告警配置
    ALERT_EMAIL_ENABLED: bool = Field(default=False, env="ALERT_EMAIL_ENABLED")
    ALERT_EMAIL_SMTP_HOST: Optional[str] = Field(default=None, env="ALERT_EMAIL_SMTP_HOST")
    ALERT_EMAIL_SMTP_PORT: int = Field(default=587, env="ALERT_EMAIL_SMTP_PORT")
    ALERT_EMAIL_USERNAME: Optional[str] = Field(default=None, env="ALERT_EMAIL_USERNAME")
    ALERT_EMAIL_PASSWORD: Optional[str] = Field(default=None, env="ALERT_EMAIL_PASSWORD")
    ALERT_EMAIL_TO: Optional[str] = Field(default=None, env="ALERT_EMAIL_TO")

    SLACK_WEBHOOK_URL: Optional[str] = Field(default=None, env="SLACK_WEBHOOK_URL")
    SLACK_CHANNEL: str = Field(default="#trading-alerts", env="SLACK_CHANNEL")

    # Celery配置
    CELERY_BROKER_URL: str = Field(default="redis://localhost:6379/1", env="CELERY_BROKER_URL")
    CELERY_RESULT_BACKEND: str = Field(default="redis://localhost:6379/2", env="CELERY_RESULT_BACKEND")
    CELERY_WORKER_CONCURRENCY: int = Field(default=4, env="CELERY_WORKER_CONCURRENCY")

    @validator("ALLOWED_ORIGINS", pre=True)
    def parse_origins(cls, v):
        """解析允许的源"""
        if isinstance(v, str) and not v.startswith("["):
            return [i.strip() for i in v.split(",")]
        elif isinstance(v, (list, str)):
            return v
        raise ValueError(v)

    @validator("ALLOWED_HOSTS", pre=True)
    def parse_hosts(cls, v):
        """解析允许的主机"""
        if isinstance(v, str) and not v.startswith("["):
            return [i.strip() for i in v.split(",")]
        elif isinstance(v, (list, str)):
            return v
        raise ValueError(v)

    @validator("ENVIRONMENT")
    def validate_environment(cls, v):
        """验证环境变量"""
        allowed = ["development", "testing", "staging", "production"]
        if v not in allowed:
            raise ValueError(f"Environment must be one of: {allowed}")
        return v

    @validator("LOG_LEVEL")
    def validate_log_level(cls, v):
        """验证日志级别"""
        allowed = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in allowed:
            raise ValueError(f"Log level must be one of: {allowed}")
        return v.upper()

    @validator("DEFAULT_RISK_LEVEL")
    def validate_risk_level(cls, v):
        """验证风险级别"""
        allowed = ["low", "medium", "high"]
        if v not in allowed:
            raise ValueError(f"Risk level must be one of: {allowed}")
        return v

    def is_production(self) -> bool:
        """检查是否为生产环境"""
        return self.ENVIRONMENT == "production"

    def is_development(self) -> bool:
        """检查是否为开发环境"""
        return self.ENVIRONMENT == "development"

    def is_testing(self) -> bool:
        """检查是否为测试环境"""
        return self.ENVIRONMENT == "testing"

    def get_database_url_sync(self) -> str:
        """获取同步数据库URL（用于Alembic）"""
        return self.DATABASE_URL.replace("+asyncpg", "")

    def get_exchange_config(self, exchange_name: str) -> dict:
        """获取交易所配置"""
        exchange_configs = {
            "binance": {
                "api_key": self.BINANCE_API_KEY,
                "secret": self.BINANCE_API_SECRET,
                "sandbox": self.BINANCE_TESTNET,
                "enableRateLimit": True,
                "options": {"defaultType": "future"}
            },
            "coinbase": {
                "apiKey": self.COINBASE_API_KEY,
                "secret": self.COINBASE_API_SECRET,
                "passphrase": self.COINBASE_API_PASSPHRASE,
                "enableRateLimit": True
            },
            "kraken": {
                "apiKey": self.KRAKEN_API_KEY,
                "secret": self.KRAKEN_API_SECRET,
                "enableRateLimit": True
            }
        }
        return exchange_configs.get(exchange_name.lower(), {})

    class Config:
        env_file = ".env"
        case_sensitive = False


# 创建全局配置实例
settings = Settings()


# 配置文件路径
class ConfigPaths:
    """配置文件路径常量"""

    BASE_DIR = Path(__file__).parent.parent.parent
    ENV_FILE = BASE_DIR / ".env"
    CONFIG_DIR = BASE_DIR / "config"
    LOGS_DIR = BASE_DIR / "logs"
    DATA_DIR = BASE_DIR / "data"

    # 确保目录存在
    @classmethod
    def ensure_directories(cls):
        """确保必要的目录存在"""
        for dir_path in [cls.CONFIG_DIR, cls.LOGS_DIR, cls.DATA_DIR]:
            dir_path.mkdir(exist_ok=True)


# 初始化目录
ConfigPaths.ensure_directories()