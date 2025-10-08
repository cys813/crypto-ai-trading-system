"""
API模块

包含FastAPI应用的所有路由和中间件配置。
"""

from fastapi import APIRouter

# 创建主路由器
api_router = APIRouter()

# 这里会自动导入并注册所有端点路由