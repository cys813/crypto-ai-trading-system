# API Enhancement Contract

**Date**: 2025-10-08
**Feature**: OpenAI Compatible API with User-Specified Configuration

## Configuration API Enhancement

### Environment Variables Schema

```yaml
# Enhanced OpenAI Configuration Schema
openai_config:
  type: object
  properties:
    OPENAI_API_KEY:
      type: string
      description: "OpenAI API密钥"
      format: password
      example: "sk-..."
    OPENAI_BASE_URL:
      type: string
      description: "自定义OpenAI API基础URL"
      format: uri
      examples:
        - "https://api.openai.com/v1"
        - "https://api.siliconflow.cn/v1"
        - "https://api.deepseek.com/v1"
      default: null
    OPENAI_ORGANIZATION:
      type: string
      description: "OpenAI组织ID"
      example: "org-..."
      default: null
    OPENAI_MODEL:
      type: string
      description: "默认使用的模型"
      enum: ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo", "gpt-4o"]
      default: "gpt-4"
    OPENAI_MAX_TOKENS:
      type: integer
      description: "最大生成token数"
      minimum: 1
      maximum: 128000
      default: 4096
    OPENAI_TEMPERATURE:
      type: number
      description: "生成温度参数"
      minimum: 0
      maximum: 2
      default: 0.1
    OPENAI_TIMEOUT:
      type: integer
      description: "请求超时时间(秒)"
      minimum: 1
      maximum: 600
      default: 60
    OPENAI_MAX_RETRIES:
      type: integer
      description: "最大重试次数"
      minimum: 0
      maximum: 10
      default: 3
  required:
    - OPENAI_API_KEY
```

### Client Configuration Response

```yaml
# OpenAI Client Configuration Status
client_status:
  type: object
  properties:
    provider:
      type: string
      example: "openai"
    status:
      type: string
      enum: ["initialized", "failed", "not_configured"]
    base_url:
      type: string
      description: "当前使用的API基础URL"
      example: "https://api.siliconflow.cn/v1"
    model:
      type: string
      example: "gpt-4"
    timeout:
      type: integer
      example: 60
    max_retries:
      type: integer
      example: 3
    error:
      type: object
      description: "初始化错误信息(如果有)"
      properties:
        code:
          type: string
        message:
          type: string
        details:
          type: object
```

## LLM Service API Enhancement

### Connection Test Enhancement

```yaml
# Enhanced Connection Test Request
connection_test_request:
  type: object
  properties:
    provider:
      type: string
      enum: ["openai"]
      description: "要测试的LLM提供商"
  required:
    - provider

# Enhanced Connection Test Response
connection_test_response:
  type: object
  properties:
    success:
      type: boolean
      description: "连接测试是否成功"
    provider:
      type: string
      example: "openai"
    base_url:
      type: string
      description: "测试使用的API基础URL"
      example: "https://api.siliconflow.cn/v1"
    response_time_ms:
      type: integer
      description: "测试响应时间(毫秒)"
      example: 1250
    model:
      type: string
      description: "测试使用的模型"
      example: "gpt-4"
    error:
      type: object
      description: "错误信息(测试失败时)"
      properties:
        code:
          type: string
        message:
          type: string
        type:
          type: string
          enum: ["connection_error", "authentication_error", "rate_limit", "invalid_response"]
    timestamp:
      type: string
      format: date-time
```

### LLM Generation Enhancement

```yaml
# Enhanced LLM Generation Request (existing API, enhanced response)
generation_request:
  type: object
  properties:
    prompt:
      type: string
      description: "输入提示文本"
    model:
      type: string
      description: "使用的模型"
      default: "gpt-4"
    provider:
      type: string
      description: "LLM提供商"
      default: "openai"
    temperature:
      type: number
      minimum: 0
      maximum: 2
      default: 0.7
    max_tokens:
      type: integer
      minimum: 1
      maximum: 128000
    system_prompt:
      type: string
      description: "系统提示词"
  required:
    - prompt

# Enhanced LLM Generation Response
generation_response:
  type: object
  properties:
    content:
      type: string
      description: "生成的文本内容"
    model:
      type: string
      description: "实际使用的模型"
    provider:
      type: string
      example: "openai"
    base_url:
      type: string
      description: "使用的API基础URL"
      example: "https://api.siliconflow.cn/v1"
    tokens_used:
      type: integer
      description: "使用的token数量"
    cost_usd:
      type: number
      description: "API调用成本(美元)"
    response_time_ms:
      type: integer
      description: "响应时间(毫秒)"
    metadata:
      type: object
      properties:
        finish_reason:
          type: string
          enum: ["stop", "length", "content_filter", "function_call", "tool_calls"]
        usage:
          type: object
          properties:
            prompt_tokens:
              type: integer
            completion_tokens:
              type: integer
            total_tokens:
              type: integer
    timestamp:
      type: string
      format: date-time
```

## Error Response Schema

```yaml
# Enhanced Error Response
error_response:
  type: object
  properties:
    error:
      type: object
      properties:
        code:
          type: string
          description: "错误代码"
          enum:
            - "OPENAI_NOT_INSTALLED"
            - "OPENAI_CLIENT_INIT_FAILED"
            - "OPENAI_API_ERROR"
            - "INVALID_CONFIGURATION"
            - "CONNECTION_FAILED"
            - "AUTHENTICATION_FAILED"
        message:
          type: string
          description: "错误消息"
        type:
          type: string
          enum: ["client_error", "server_error", "configuration_error"]
        details:
          type: object
          description: "错误详细信息"
          properties:
            provider:
              type: string
            base_url:
              type: string
            model:
              type: string
            validation_errors:
              type: array
              items:
                type: object
                properties:
                  field:
                    type: string
                  message:
                    type: string
            original_error:
              type: string
    timestamp:
      type: string
      format: date-time
    request_id:
      type: string
      description: "请求追踪ID"
```

## Configuration Validation API

### Validation Request/Response

```yaml
# Configuration Validation Request
config_validation_request:
  type: object
  properties:
    openai_config:
      type: object
      properties:
        api_key:
          type: string
        base_url:
          type: string
        organization:
          type: string
        model:
          type: string
        timeout:
          type: integer
        max_retries:
          type: integer

# Configuration Validation Response
config_validation_response:
  type: object
  properties:
    valid:
      type: boolean
      description: "配置是否有效"
    errors:
      type: array
      items:
        type: object
        properties:
          field:
            type: string
          message:
            type: string
          code:
            type: string
    warnings:
      type: array
      items:
        type: object
        properties:
          field:
            type: string
          message:
            type: string
          severity:
            type: string
            enum: ["info", "warning"]
    connection_test:
      type: object
      properties:
        status:
          type: string
          enum: ["not_tested", "success", "failed"]
        response_time_ms:
          type: integer
        error:
          type: string
```

## Migration Notes

### Backward Compatibility
- 所有现有的API端点和响应格式保持不变
- 新的配置字段都是可选的
- 现有代码无需修改即可继续工作

### Deprecation Notices
- 无计划弃用现有功能
- 建议逐步迁移到新的配置选项以获得更好的灵活性

### Version Impact
- 当前API版本保持兼容
- 新增字段不影响现有客户端集成