# Feature Specification: OpenAI Compatible API with User-Specified Configuration

**Feature Branch**: `003-openai-api-base`
**Created**: 2025-10-08
**Status**: Draft
**Input**: User description: "要求兼用openai的通用api，base url和key用户指定"

## Clarifications

### Session 2025-10-08

- **Q**: What security measures should be implemented for storing and transmitting user-specified API keys and configuration data? → **A**: Store API keys in plain text but use HTTPS for all transmissions; no additional encryption required
- **Q**: What performance and scalability targets should the system support for custom API configurations? → **B**: Support up to 100 concurrent API requests per minute with 5-second average response time
- **Q**: 运行时动态base_url切换的具体实现机制是什么？ → **A**: 简化实现：仅支持服务重启时读取新配置，移除真正的热更新功能

## User Scenarios & Testing *(mandatory)*

### User Story 1 - 扩展现有OpenAI客户端支持自定义Base URL (Priority: P1)

基于现有的OpenAI客户端进行最小化修改，支持用户指定自定义的API基础URL，使系统能够使用任何符合OpenAI API规范的LLM提供商服务。

**Why this priority**: 这是基于现有框架的最小化扩展，复用已有的OpenAI客户端架构和LLM集成服务，只需添加base_url配置支持，实现成本最低且风险最小。

**Independent Test**: 可以通过修改现有OpenAI客户端的base_url参数，测试不同OpenAI兼容端点（如SiliconFlow、DeepSeek、企业内部服务等），验证现有LLM服务能够无缝使用自定义端点。

**Acceptance Scenarios**:

1. **Given** 现有的OpenAI客户端架构，**When** 在配置中添加自定义base_url，**Then** OpenAIClient能够使用自定义端点进行API调用
2. **Given** 已配置自定义base_url，**When** 通过现有的LLMIntegrationService调用，**Then** 系统路由到自定义端点且功能保持不变
3. **Given** 多个自定义base_url配置，**When** 在现有Settings中切换配置，**Then** 系统能够使用不同的端点服务

---

### User Story 2 - 增强现有配置支持自定义Base URL (Priority: P2)

扩展现有的系统配置类，添加对自定义OpenAI API base URL的支持，同时保持向后兼容性。

**Why this priority**: 基于现有配置系统的最小化扩展，只需添加新的配置字段，不影响现有功能，确保平滑升级。

**Independent Test**: 可以通过在现有Settings中添加OPENAI_BASE_URL配置项，验证现有OpenAI客户端能够正确读取和使用自定义base_url。

**Acceptance Scenarios**:

1. **Given** 现有的Settings配置结构，**When** 添加OPENAI_BASE_URL字段，**Then** OpenAIClient能够读取并使用自定义base_url
2. **Given** 未配置自定义base_url，**When** 系统启动，**Then** 使用默认的OpenAI官方端点保持向后兼容
3. **Given** 配置了无效的自定义base_url，**When** OpenAIClient初始化，**Then** 系统在首次API调用时提供清晰的错误提示

---

### User Story 3 - 服务重启配置切换 (Priority: P3)

简化实现：系统支持通过服务重启来应用新的OPENAI_BASE_URL配置，避免复杂的热更新机制，保持架构简单可靠。

**Why this priority**: 最简单的实现方式，通过标准的服务重启流程来更新配置，降低系统复杂度和潜在风险。

**Independent Test**: 可以通过修改环境变量并重启服务，验证新的base_url配置被正确读取和使用。

**Acceptance Scenarios**:

1. **Given** 修改了OPENAI_BASE_URL环境变量，**When** 重启服务，**Then** 系统使用新的自定义端点
2. **Given** 未配置OPENAI_BASE_URL，**When** 重启服务，**Then** 系统使用OpenAI官方默认端点
3. **Given** 配置了无效的base_url，**When** 重启后首次API调用，**Then** 系统提供清晰的错误信息

---

### Edge Cases

- 当配置的OPENAI_BASE_URL无法访问时，OpenAIClient如何处理连接错误？
- 当自定义端点的API响应格式与OpenAI官方API不完全一致时，现有客户端如何兼容？
- 当配置了无效的base_url时，系统如何在启动时提供清晰的错误提示？

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: 现有的OpenAIClient必须支持读取OPENAI_BASE_URL配置并使用自定义端点
- **FR-002**: 系统必须保持向后兼容，未配置OPENAI_BASE_URL时使用OpenAI官方端点
- **FR-003**: Settings类必须添加OPENAI_BASE_URL字段支持环境变量配置
- **FR-004**: 系统必须支持通过服务重启来应用新的base_url配置
- **FR-005**: 当自定义端点不可用时，系统必须提供清晰的错误信息

### Non-Functional Requirements

- **NFR-001**: 添加base_url配置不得影响现有性能，保持100并发请求和5秒响应时间
- **NFR-002**: 自定义端点必须支持HTTPS协议保护API密钥传输
- **NFR-003**: 配置变更通过标准服务重启流程生效，确保配置一致性
- **NFR-004**: 系统必须保持现有LLMIntegrationService的所有功能不变

### Key Entities *(include if feature involves data)*

- **自定义API配置**: 存储用户指定的API基础URL、认证密钥等信息，通过环境变量管理
- **API端点配置**: 系统支持的自定义OpenAI兼容API端点信息
- **请求/响应格式**: 保持现有格式不变，确保向后兼容性

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: 通过添加OPENAI_BASE_URL环境变量，用户能够在1分钟内完成自定义端点配置
- **SC-002**: 100%向后兼容，现有功能无需修改即可继续使用
- **SC-003**: 支持通过服务重启切换base_url，配置变更在服务重启后生效
- **SC-004**: 当自定义端点不可用时，系统提供明确错误信息并保持原有错误处理机制
- **SC-005**: 性能无影响，保持100并发请求和5秒响应时间指标