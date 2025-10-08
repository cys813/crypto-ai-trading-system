# Quickstart Guide: OpenAI Compatible API Configuration

**Date**: 2025-10-08
**Feature**: OpenAI Compatible API with User-Specified Configuration

## 快速开始

### 1. 基础配置

#### 使用OpenAI官方API（默认）
```bash
# .env 文件配置
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL=gpt-4
# OPENAI_BASE_URL 会自动使用 https://api.openai.com/v1
```

#### 使用自定义OpenAI兼容API
```bash
# SiliconFlow示例
OPENAI_API_KEY=your_siliconflow_api_key
OPENAI_BASE_URL=https://api.siliconflow.cn/v1
OPENAI_MODEL=gpt-4

# DeepSeek示例
OPENAI_API_KEY=your_deepseek_api_key
OPENAI_BASE_URL=https://api.deepseek.com/v1
OPENAI_MODEL=deepseek-chat

# Azure OpenAI示例
OPENAI_API_KEY=your_azure_openai_key
OPENAI_BASE_URL=https://your-resource.openai.azure.com/openai/deployments/your-deployment
OPENAI_MODEL=gpt-4
```

### 2. 完整配置示例

```bash
# .env 完整配置示例
# 应用基础配置
APP_NAME=Crypto AI Trading System
DEBUG=false
ENVIRONMENT=production
SECRET_KEY=your-super-secret-key-here

# OpenAI API配置
OPENAI_API_KEY=sk-your-api-key-here
OPENAI_BASE_URL=https://api.siliconflow.cn/v1  # 可选，自定义端点
OPENAI_ORGANIZATION=org-your-org-id             # 可选，组织ID
OPENAI_MODEL=gpt-4                              # 默认模型
OPENAI_MAX_TOKENS=4096                          # 最大token数
OPENAI_TEMPERATURE=0.1                          # 温度参数
OPENAI_TIMEOUT=60                               # 请求超时(秒)
OPENAI_MAX_RETRIES=3                            # 最大重试次数
```

### 3. 代码使用示例

#### 基础LLM调用（无需修改现有代码）
```python
from src.core.llm_integration import get_llm_service

# 获取LLM服务实例
llm_service = get_llm_service()

# 文本生成（自动使用配置的端点）
response = await llm_service.generate_completion(
    prompt="分析当前加密货币市场趋势",
    model="gpt-4",
    temperature=0.1
)

print(response)
```

#### 带完整响应的调用
```python
from src.core.llm_integration import get_llm_service, LLMRequest

llm_service = get_llm_service()

# 创建请求
request = LLMRequest(
    prompt="生成交易策略建议",
    model="gpt-4",
    temperature=0.1,
    max_tokens=2000,
    system_prompt="你是一个专业的加密货币交易分析师"
)

# 获取完整响应
response = await llm_service.generate_completion_with_response(request)

print(f"内容: {response.content}")
print(f"模型: {response.model}")
print(f"使用的端点: {getattr(response, 'base_url', 'default')}")
print(f"Token使用: {response.tokens_used}")
print(f"响应时间: {response.response_time_ms}ms")
print(f"成本: ${response.cost_usd}")
```

### 4. 连接测试

```python
from src.core.llm_integration import LLMProvider, get_llm_service

llm_service = get_llm_service()

# 测试OpenAI连接
is_connected = await llm_service.test_connection(LLMProvider.OPENAI)
if is_connected:
    print("✅ OpenAI连接成功")
else:
    print("❌ OpenAI连接失败")

# 获取提供商状态
status = llm_service.get_provider_status()
print(f"提供商状态: {status}")
```

### 5. 错误处理

```python
from src.core.llm_integration import LLMServiceError, get_llm_service

llm_service = get_llm_service()

try:
    response = await llm_service.generate_completion(
        prompt="测试提示",
        model="gpt-4"
    )
except LLMServiceError as e:
    print(f"LLM服务错误: {e.message}")
    print(f"错误代码: {e.error_code}")

    # 根据错误代码进行处理
    if e.error_code == "OPENAI_API_ERROR":
        print("API调用失败，请检查网络连接和配置")
    elif e.error_code == "OPENAI_NOT_INSTALLED":
        print("请安装OpenAI包: pip install openai")
```

## 配置验证

### 验证配置文件
```python
from src.core.config import settings

# 检查OpenAI配置
if not settings.OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY 未配置")

if settings.OPENAI_BASE_URL:
    print(f"使用自定义端点: {settings.OPENAI_BASE_URL}")
else:
    print("使用OpenAI官方端点")

print(f"默认模型: {settings.OPENAI_MODEL}")
print(f"超时设置: {settings.OPENAI_TIMEOUT}秒")
print(f"最大重试: {settings.OPENAI_MAX_RETRIES}次")
```

### 环境变量检查脚本
```python
# check_config.py
import os
from dotenv import load_dotenv

load_dotenv()

required_vars = ["OPENAI_API_KEY"]
optional_vars = [
    "OPENAI_BASE_URL",
    "OPENAI_ORGANIZATION",
    "OPENAI_MODEL",
    "OPENAI_MAX_TOKENS",
    "OPENAI_TEMPERATURE",
    "OPENAI_TIMEOUT",
    "OPENAI_MAX_RETRIES"
]

print("🔍 检查OpenAI配置...")

# 检查必需变量
missing_required = []
for var in required_vars:
    value = os.getenv(var)
    if value:
        print(f"✅ {var}: {'*' * len(value)}")  # 隐藏API密钥
    else:
        print(f"❌ {var}: 未配置")
        missing_required.append(var)

# 检查可选变量
for var in optional_vars:
    value = os.getenv(var)
    if value:
        print(f"✅ {var}: {value}")
    else:
        print(f"⚪ {var}: 使用默认值")

if missing_required:
    print(f"\n❌ 配置失败: 缺少必需变量 {missing_required}")
else:
    print("\n✅ 配置检查通过")
```

## 常见问题

### Q: 如何切换不同的API提供商？
A: 只需修改`.env`文件中的`OPENAI_BASE_URL`和`OPENAI_API_KEY`，然后重启服务即可。

### Q: 配置了错误的base_url怎么办？
A: 系统会在首次API调用时提供清晰的错误信息，包括连接失败的原因。检查URL格式和网络连接，然后重启服务。

### Q: 如何验证自定义端点的兼容性？
A: 使用连接测试功能：`await llm_service.test_connection(LLMProvider.OPENAI)`

### Q: 支持哪些OpenAI兼容的API？
A: 支持任何遵循OpenAI API规范的端点，包括SiliconFlow、DeepSeek、Azure OpenAI等。

### Q: 如何调试API调用问题？
A: 检查日志文件，系统会记录详细的API调用信息，包括使用的端点、请求参数和响应时间。

## 生产环境部署

### 安全配置
```bash
# 生产环境配置
OPENAI_API_KEY=${OPENAI_API_KEY}  # 从环境变量读取
OPENAI_BASE_URL=${OPENAI_BASE_URL}  # 从环境变量读取
OPENAI_TIMEOUT=30  # 生产环境建议较短超时
OPENAI_MAX_RETRIES=2  # 生产环境建议较少重试
```

### 监控配置
```python
# 添加监控和日志
import logging

logger = logging.getLogger(__name__)

async def monitored_llm_call(prompt: str) -> str:
    llm_service = get_llm_service()

    start_time = time.time()
    try:
        response = await llm_service.generate_completion(prompt)
        duration = time.time() - start_time

        logger.info(
            "LLM调用成功",
            prompt_length=len(prompt),
            response_length=len(response),
            duration=duration
        )
        return response
    except Exception as e:
        duration = time.time() - start_time
        logger.error(
            "LLM调用失败",
            error=str(e),
            duration=duration
        )
        raise
```

这个快速开始指南涵盖了从基础配置到生产部署的所有关键步骤。