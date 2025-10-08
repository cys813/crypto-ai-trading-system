<!--
Sync Impact Report:
Version change: 1.0.0 → 1.1.0 (MINOR: Added 7 new core principles based on user requirements)
Modified principles: All 5 placeholder principles replaced with concrete principles
Added sections: Code Quality Standards, Architecture Guidelines
Removed sections: None
Templates requiring updates: ✅ plan-template.md (Constitution Check section updated)
                      ⚠ spec-template.md (needs alignment with new principles)
                      ⚠ tasks-template.md (needs task categorization updates)
Follow-up TODOs: None - all placeholders filled with concrete values
-->

# Crypto AI Trading Constitution

## Core Principles

### I. 简化优先原则 (Simplicity First)
系统设计必须追求最简实现，拒绝过度抽象。每个组件都应该有明确的单一职责，避免不必要的复杂性和过度设计。选择最直接的解决方案，只有在明确需求时才增加复杂性。

### II. 测试先行原则 (Test-First Development)
所有功能必须采用测试驱动开发(TDD)方式实现。先编写测试用例，确保测试失败，然后实现最小化代码使测试通过。每个功能点都必须有对应的自动化测试，包括单元测试和集成测试。

### III. 集成优先原则 (Integration-First)
优先考虑系统集成和模块间协作。在实现独立功能时，必须考虑与现有系统的集成方式。API设计必须简洁明确，数据格式必须标准化，确保模块间的无缝集成。

### IV. 模块复用原则 (Module Reusability)
所有模块设计必须考虑复用性。通用功能应该抽象为独立模块，避免重复实现。模块接口必须清晰稳定，内部实现细节应该封装良好，便于在不同场景中复用。

### V. 高内聚低耦合原则 (High Cohesion, Low Coupling)
每个模块必须保持高内聚性 - 相关功能集中在同一模块内。模块间必须保持低耦合性 - 依赖关系最小化，接口明确。避免模块间的紧耦合，确保系统架构的灵活性和可维护性。

### VI. 代码可读性原则 (Code Readability)
代码必须像文档一样可读。使用清晰的命名规范，有意义的变量和函数名。复杂的逻辑必须包含必要的注释和文档。代码结构必须逻辑清晰，便于其他开发者理解和维护。

### VII. 系统架构原则 (System Architecture)
系统架构必须保持良好的分层和模块化。各层职责明确，依赖方向清晰。架构演进必须向后兼容，重大变更需要经过充分的评审和测试。系统必须支持水平扩展和高可用性。

## Code Quality Standards

### 编码规范
- 使用统一的代码格式化工具和linting规则
- 所有公共API必须有完整的文档说明
- 代码审查必须检查是否符合核心原则
- 禁止在生产代码中包含调试代码或临时解决方案

### 性能要求
- 所有API响应时间必须在可接受范围内
- 数据库查询必须优化，避免N+1问题
- 内存使用必须合理，避免内存泄漏
- 关键路径必须经过性能测试

## Architecture Guidelines

### 模块设计
- 每个模块必须有单一明确的职责
- 模块间通过定义良好的接口通信
- 核心业务逻辑与技术实现分离
- 配置管理必须集中化和标准化

### 数据管理
- 数据模型设计必须简单直观
- 数据访问层必须统一管理
- 敏感数据必须加密存储
- 数据迁移必须向前兼容

## Governance

### 修订流程
- 本宪法是项目的最高指导原则
- 任何修订必须经过充分讨论和团队同意
- 修订必须记录变更原因和影响范围
- 重大修订需要逐步迁移，确保系统稳定性

### 合规检查
- 所有代码提交必须符合宪法原则
- 代码审查必须验证原则遵循情况
- 定期进行架构审查和重构
- 违反原则的情况必须有充分的理由说明

### 质量保证
- 每个功能发布前必须通过完整测试
- 集成测试必须覆盖关键业务流程
- 性能测试必须在每次重大变更后执行
- 安全审查必须定期进行

**Version**: 1.1.0 | **Ratified**: 2025-10-08 | **Last Amended**: 2025-10-08