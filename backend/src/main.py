"""
FastAPI应用主入口

多Agent加密货币量化交易分析系统的API服务。
"""

import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
import time
import uuid

from .core.cache import init_redis, close_redis
from .core.config import settings
from .core.logging import setup_logging
from .core.exceptions import setup_exception_handlers

# 设置日志
setup_logging()
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    # 启动时执行
    logger.info("正在启动Crypto AI Trading系统...")

    try:
        # 初始化Redis连接
        await init_redis(settings.REDIS_URL)

        # 初始化数据库连接（可选）
        # await init_database()

        logger.info("系统启动完成")

    except Exception as e:
        logger.error(f"系统启动失败: {e}")
        raise

    yield

    # 关闭时执行
    logger.info("正在关闭系统...")

    try:
        # 关闭Redis连接
        await close_redis()

        # 关闭其他资源
        # await close_database()

        logger.info("系统已安全关闭")

    except Exception as e:
        logger.error(f"系统关闭时出错: {e}")


# 创建FastAPI应用实例
app = FastAPI(
    title="多Agent加密货币量化交易分析系统",
    description="基于Python的多Agent虚拟货币量化交易分析系统API",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan
)

# 添加CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 添加受信任主机中间件
if settings.ENVIRONMENT == "production":
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=settings.ALLOWED_HOSTS
    )


# 请求ID中间件
@app.middleware("http")
async def add_request_id(request: Request, call_next):
    """添加请求ID和响应时间中间件"""
    request_id = str(uuid.uuid4())
    request.state.request_id = request_id

    start_time = time.time()

    response = await call_next(request)

    process_time = time.time() - start_time
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Process-Time"] = str(process_time)

    return response


# 请求日志中间件
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """记录请求日志"""
    start_time = time.time()

    logger.info(
        f"请求开始 - Method: {request.method}, Path: {request.url.path}, "
        f"Client: {request.client.host if request.client else 'unknown'}"
    )

    response = await call_next(request)

    process_time = time.time() - start_time

    logger.info(
        f"请求完成 - Method: {request.method}, Path: {request.url.path}, "
        f"Status: {response.status_code}, Process-Time: {process_time:.4f}s"
    )

    return response


# 设置异常处理器
setup_exception_handlers(app)

# 导入并注册API路由
from .api.endpoints import auth, news, trading, monitoring, strategies


# 健康检查端点
@app.get("/health", tags=["系统"])
async def health_check():
    """系统健康检查"""
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "version": "0.1.0",
        "environment": settings.ENVIRONMENT
    }


# 根路径
@app.get("/", tags=["系统"])
async def root():
    """根路径"""
    return {
        "message": "多Agent加密货币量化交易分析系统",
        "version": "0.1.0",
        "docs": "/docs",
        "health": "/health"
    }


# 注册API路由
app.include_router(
    auth.router,
    prefix="/api/v1/auth",
    tags=["认证"]
)

app.include_router(
    news.router,
    prefix="/api/v1/news",
    tags=["新闻"]
)

app.include_router(
    trading.router,
    prefix="/api/v1/trading",
    tags=["交易"]
)

app.include_router(
    monitoring.router,
    prefix="/api/v1/monitoring",
    tags=["监控"]
)

app.include_router(
    strategies.router,
    prefix="/api/v1",
    tags=["策略分析"]
)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.DEBUG,
        log_level=settings.LOG_LEVEL.lower()
    )