# Phase 1: Data Model & Design

**Date**: 2025-10-08
**Feature**: OpenAI Compatible API with User-Specified Configuration

## Configuration Data Model

### Enhanced Settings Class

```python
# backend/src/core/config.py

class Settings(BaseSettings):
    # ... existing fields ...

    # Enhanced LLM API Configuration
    OPENAI_API_KEY: Optional[str] = Field(default=None, env="OPENAI_API_KEY")
    OPENAI_BASE_URL: Optional[str] = Field(default=None, env="OPENAI_BASE_URL")
    OPENAI_ORGANIZATION: Optional[str] = Field(default=None, env="OPENAI_ORGANIZATION")
    OPENAI_MODEL: str = Field(default="gpt-4", env="OPENAI_MODEL")
    OPENAI_MAX_TOKENS: int = Field(default=4096, env="OPENAI_MAX_TOKENS")
    OPENAI_TEMPERATURE: float = Field(default=0.1, env="OPENAI_TEMPERATURE")
    OPENAI_TIMEOUT: int = Field(default=60, env="OPENAI_TIMEOUT")
    OPENAI_MAX_RETRIES: int = Field(default=3, env="OPENAI_MAX_RETRIES")

    @validator("OPENAI_BASE_URL", pre=True)
    def validate_base_url(cls, v):
        """验证OpenAI Base URL格式"""
        if v is None:
            return v
        if not v.startswith(("http://", "https://")):
            raise ValueError("OPENAI_BASE_URL must start with http:// or https://")
        return v.rstrip("/")  # 移除尾部斜杠

    @validator("OPENAI_TIMEOUT", pre=True)
    def validate_timeout(cls, v):
        """验证超时设置"""
        if v is not None and v <= 0:
            raise ValueError("OPENAI_TIMEOUT must be positive")
        return v or 60

    @validator("OPENAI_MAX_RETRIES", pre=True)
    def validate_max_retries(cls, v):
        """验证重试次数"""
        if v is not None and v < 0:
            raise ValueError("OPENAI_MAX_RETRIES must be non-negative")
        return v or 3
```

### OpenAI Client Configuration

```python
# backend/src/core/llm_integration.py - Enhanced OpenAIClient

class OpenAIClient(BaseLLMClient):
    """增强的OpenAI客户端，支持自定义base_url"""

    def __init__(self):
        super().__init__(LLMProvider.OPENAI)
        self.client = None
        self.base_url = None  # 记录当前使用的base_url
        self._initialize_client()

    def _initialize_client(self):
        """初始化OpenAI客户端，支持自定义配置"""
        try:
            import openai

            # 构建客户端参数
            client_kwargs = {
                "api_key": getattr(settings, 'OPENAI_API_KEY', None),
                "timeout": getattr(settings, 'OPENAI_TIMEOUT', 60),
                "max_retries": getattr(settings, 'OPENAI_MAX_RETRIES', 3)
            }

            # 添加可选的base_url参数
            if base_url := getattr(settings, 'OPENAI_BASE_URL', None):
                client_kwargs["base_url"] = base_url
                self.base_url = base_url

            # 添加可选的组织ID
            if organization := getattr(settings, 'OPENAI_ORGANIZATION', None):
                client_kwargs["organization"] = organization

            self.client = openai.AsyncOpenAI(**client_kwargs)

            # 记录初始化信息（不包含敏感数据）
            self.logger.info(
                "OpenAI客户端初始化成功",
                base_url=self.base_url or "default",
                timeout=client_kwargs["timeout"],
                max_retries=client_kwargs["max_retries"]
            )

        except ImportError as e:
            raise LLMServiceError(
                message="OpenAI package not installed. Install with: pip install openai",
                error_code="OPENAI_NOT_INSTALLED",
                cause=e
            )
        except Exception as e:
            raise LLMServiceError(
                message=f"OpenAI客户端初始化失败: {str(e)}",
                error_code="OPENAI_CLIENT_INIT_FAILED",
                cause=e
            )
```

## Environment Configuration Template

```bash
# .env.example - Enhanced with custom base URL support

# Core Application Settings (existing)
APP_NAME=Crypto AI Trading System
DEBUG=false
ENVIRONMENT=development
SECRET_KEY=your-secret-key-here

# Enhanced OpenAI API Configuration
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_ORGANIZATION=your_organization_id
OPENAI_MODEL=gpt-4
OPENAI_MAX_TOKENS=4096
OPENAI_TEMPERATURE=0.1
OPENAI_TIMEOUT=60
OPENAI_MAX_RETRIES=3

# Alternative Provider Examples:
# SiliconFlow:
# OPENAI_BASE_URL=https://api.siliconflow.cn/v1
# OPENAI_API_KEY=your_siliconflow_key

# DeepSeek:
# OPENAI_BASE_URL=https://api.deepseek.com/v1
# OPENAI_API_KEY=your_deepseek_key

# Azure OpenAI:
# OPENAI_BASE_URL=https://your-resource.openai.azure.com/openai/deployments/your-deployment
# OPENAI_API_KEY=your_azure_key
# OPENAI_API_VERSION=2024-02-15-preview
```

## Data Validation Rules

### URL Validation
- 必须以 `http://` 或 `https://` 开头
- 自动移除尾部斜杠
- 支持标准OpenAI API格式和兼容端点

### Configuration Validation
- API Key: 可选，但使用时必需
- Base URL: 可选，未设置时使用OpenAI官方端点
- Timeout: 必须为正整数，默认60秒
- Max Retries: 必须为非负整数，默认3次

### Error Handling Strategy
1. **初始化错误**: 提供清晰的错误信息和解决建议
2. **连接错误**: 记录详细的错误上下文，包括使用的base_url
3. **配置错误**: 在启动时验证并报告配置问题

## Integration Points

### 1. Settings Integration
- 扩展现有配置类，添加新字段
- 保持向后兼容性，所有新字段都是可选的
- 添加适当的验证器确保配置有效性

### 2. Client Integration
- 修改现有OpenAIClient初始化逻辑
- 保持API接口不变，确保现有代码无需修改
- 增强日志记录，提供更好的可观测性

### 3. Testing Integration
- 扩展现有连接测试功能
- 支持不同base_url的连接验证
- 增强错误场景的测试覆盖

## Migration Strategy

### Phase 1: Configuration Enhancement
1. 添加新的配置字段到Settings类
2. 更新环境变量示例文件
3. 添加配置验证逻辑

### Phase 2: Client Enhancement
1. 修改OpenAIClient初始化方法
2. 增强错误处理和日志记录
3. 更新连接测试功能

### Phase 3: Testing & Documentation
1. 更新单元测试和集成测试
2. 更新API文档和使用示例
3. 验证向后兼容性

这种设计确保了最小化修改，同时提供了灵活的自定义端点支持。