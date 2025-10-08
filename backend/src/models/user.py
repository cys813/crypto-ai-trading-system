"""
用户相关模型

包含用户认证和权限管理相关的数据模型。
"""

from sqlalchemy import Column, String, Boolean, DateTime
from sqlalchemy.orm import relationship
from .base import BaseModel


class User(BaseModel):
    """用户模型"""

    __tablename__ = "users"

    username = Column(String(100), unique=True, nullable=False, index=True)
    email = Column(String(255), unique=True, nullable=True, index=True)

    # 认证信息
    password_hash = Column(String(255), nullable=False)
    api_key = Column(String(255), unique=True, nullable=True)
    api_secret = Column(String(255), nullable=True)

    # 用户配置
    timezone = Column(String(50), default="UTC")
    language = Column(String(10), default="en")
    risk_level = Column(String(20), default="medium")  # low, medium, high

    # 权限信息
    is_active = Column(Boolean, default=True)
    is_admin = Column(Boolean, default=False)

    # 登录信息
    last_login = Column(DateTime(timezone=True), nullable=True)

    # 关系
    trading_strategies = relationship("TradingStrategy", back_populates="creator")
    trading_orders = relationship("TradingOrder", back_populates="user")
    positions = relationship("Position", back_populates="user")

    def __repr__(self):
        return f"<User(username='{self.username}', active={self.is_active})>"