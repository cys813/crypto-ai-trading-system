# Implementation Plan: OpenAI Compatible API with User-Specified Configuration

**Branch**: `003-openai-api-base` | **Date**: 2025-10-08 | **Spec**: [spec.md](./spec.md)
**Input**: Feature specification from `/specs/003-openai-api-base/spec.md`

## Summary

基于现有LLM集成框架的最小化扩展，通过添加OPENAI_BASE_URL配置支持，使系统能够使用任何OpenAI兼容的API端点。主要修改现有的OpenAI客户端和系统配置类，保持100%向后兼容性。

## Technical Context

**Language/Version**: Python 3.11+
**Primary Dependencies**: FastAPI, OpenAI SDK, asyncio, SQLAlchemy, Redis, pydantic-settings
**Storage**: PostgreSQL (配置存储), Redis (缓存和状态)
**Testing**: pytest, pytest-asyncio
**Target Platform**: Linux server
**Project Type**: Web application (backend API service)
**Performance Goals**: 100并发API请求，5秒平均响应时间
**Constraints**: 必须保持向后兼容，通过HTTPS保护传输，支持服务重启配置更新
**Scale/Scope**: 单一配置字段支持，影响现有OpenAI客户端集成

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

### Required Gates (based on Crypto AI Trading Constitution v1.1.0):

- [x] **简化优先检查**: ✅ 采用最小化修改策略，仅添加配置字段和修改客户端初始化，避免过度抽象
- [x] **测试先行检查**: ✅ 制定完整的测试策略，包括单元测试、集成测试和连接测试
- [x] **集成优先检查**: ✅ 完全基于现有系统架构，保持API接口标准化，无需修改现有代码
- [x] **模块复用检查**: ✅ 复用现有的Settings、OpenAIClient和错误处理机制，接口设计稳定
- [x] **高内聚低耦合检查**: ✅ 配置管理与客户端职责分离，依赖关系最小化
- [x] **代码可读性检查**: ✅ 采用清晰的命名规范，提供完整文档和示例
- [x] **系统架构检查**: ✅ 保持现有分层架构，支持扩展但不引入额外复杂性

## Project Structure

### Documentation (this feature)

```
specs/[###-feature]/
├── plan.md              # This file (/speckit.plan command output)
├── research.md          # Phase 0 output (/speckit.plan command)
├── data-model.md        # Phase 1 output (/speckit.plan command)
├── quickstart.md        # Phase 1 output (/speckit.plan command)
├── contracts/           # Phase 1 output (/speckit.plan command)
└── tasks.md             # Phase 2 output (/speckit.tasks command - NOT created by /speckit.plan)
```

### Source Code (repository root)

```
backend/
├── src/
│   ├── core/
│   │   ├── config.py              # 需要修改：添加OPENAI_BASE_URL配置
│   │   └── llm_integration.py     # 需要修改：增强OpenAIClient初始化
│   └── ...                        # 现有代码保持不变
└── tests/
    ├── unit/
    │   ├── test_config.py         # 需要扩展：测试新配置字段
    │   └── test_llm_integration.py # 需要扩展：测试自定义base_url
    └── integration/
        └── test_llm_endpoints.py  # 需要扩展：集成测试
```

**Structure Decision**: 基于现有Web application架构，仅需修改backend/src/core/中的两个核心文件，其他代码保持不变。

## Complexity Tracking

本设计无需任何Constitution违规的复杂度权衡，所有修改都遵循简化优先原则：

| 复杂度方面 | 设计选择 | 简化策略 |
|-----------|----------|----------|
| 架构复杂度 | 最小化修改现有组件 | 无需新建模块或抽象层 |
| 配置管理 | 扩展现有Settings类 | 避免独立配置系统 |
| 客户端集成 | 增强现有OpenAIClient | 不创建新的客户端类 |
| 测试策略 | 扩展现有测试框架 | 复用已有测试基础设施 |
| 向后兼容 | 所有新功能可选 | 现有代码无需修改 |
