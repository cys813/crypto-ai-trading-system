#!/usr/bin/env python3
"""
OpenAI配置验证脚本

用于验证OpenAI API配置的正确性，检测常见配置问题并提供修复建议。
"""

import sys
import os
import asyncio
import json
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass

# 添加backend/src到Python路径
backend_src = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(backend_src))

from core.config import settings
from core.llm_integration import get_llm_service, LLMProvider


@dataclass
class ValidationResult:
    """验证结果"""
    is_valid: bool
    field: str
    value: Optional[str]
    error: Optional[str]
    suggestion: str


class OpenAIConfigValidator:
    """OpenAI配置验证器"""

    def __init__(self):
        self.results: List[ValidationResult] = []

    def validate_basic_config(self) -> None:
        """验证基础配置"""
        # API密钥验证
        api_key = getattr(settings, 'OPENAI_API_KEY', None)
        if not api_key:
            self.results.append(ValidationResult(
                is_valid=False,
                field="OPENAI_API_KEY",
                value=None,
                error="API密钥未配置",
                suggestion="请设置OPENAI_API_KEY环境变量，例如：export OPENAI_API_KEY=sk-..."
            ))
        elif api_key and not api_key.startswith("sk-"):
            self.results.append(ValidationResult(
                is_valid=False,
                field="OPENAI_API_KEY",
                value=api_key[:8] + "...",
                error="API密钥格式可能不正确",
                suggestion="OpenAI API密钥通常以'sk-'开头"
            ))
        else:
            self.results.append(ValidationResult(
                is_valid=True,
                field="OPENAI_API_KEY",
                value=api_key[:8] + "...",
                error=None,
                suggestion="API密钥配置正确"
            ))

        # Base URL验证
        base_url = getattr(settings, 'OPENAI_BASE_URL', None)
        if base_url:
            if not base_url.startswith(("http://", "https://")):
                self.results.append(ValidationResult(
                    is_valid=False,
                    field="OPENAI_BASE_URL",
                    value=base_url,
                    error="Base URL格式不正确",
                    suggestion="Base URL必须以http://或https://开头"
                ))
            else:
                self.results.append(ValidationResult(
                    is_valid=True,
                    field="OPENAI_BASE_URL",
                    value=base_url,
                    error=None,
                    suggestion="自定义Base URL配置正确"
                ))
        else:
            self.results.append(ValidationResult(
                is_valid=True,
                field="OPENAI_BASE_URL",
                value="default (OpenAI)",
                error=None,
                suggestion="使用默认OpenAI端点"
            ))

        # 其他配置验证
        self.validate_optional_config()

    def validate_optional_config(self) -> None:
        """验证可选配置"""
        # 模型验证
        model = getattr(settings, 'OPENAI_MODEL', 'gpt-4')
        valid_models = ['gpt-3.5-turbo', 'gpt-4', 'gpt-4-turbo', 'gpt-4o']
        if model not in valid_models:
            self.results.append(ValidationResult(
                is_valid=False,
                field="OPENAI_MODEL",
                value=model,
                error="未知的模型",
                suggestion=f"建议使用: {', '.join(valid_models)}"
            ))
        else:
            self.results.append(ValidationResult(
                is_valid=True,
                field="OPENAI_MODEL",
                value=model,
                error=None,
                suggestion="模型配置正确"
            ))

        # 超时设置验证
        timeout = getattr(settings, 'OPENAI_TIMEOUT', 60)
        if timeout <= 0:
            self.results.append(ValidationResult(
                is_valid=False,
                field="OPENAI_TIMEOUT",
                value=str(timeout),
                error="超时时间必须为正数",
                suggestion="建议设置30-300秒之间的值"
            ))
        elif timeout > 600:
            self.results.append(ValidationResult(
                is_valid=False,
                field="OPENAI_TIMEOUT",
                value=str(timeout),
                error="超时时间过长",
                suggestion="建议设置在600秒以内"
            ))
        else:
            self.results.append(ValidationResult(
                is_valid=True,
                field="OPENAI_TIMEOUT",
                value=str(timeout),
                error=None,
                suggestion="超时设置合理"
            ))

        # 重试次数验证
        max_retries = getattr(settings, 'OPENAI_MAX_RETRIES', 3)
        if max_retries < 0:
            self.results.append(ValidationResult(
                is_valid=False,
                field="OPENAI_MAX_RETRIES",
                value=str(max_retries),
                error="重试次数不能为负数",
                suggestion="建议设置0-10之间的值"
            ))
        elif max_retries > 10:
            self.results.append(ValidationResult(
                is_valid=False,
                field="OPENAI_MAX_RETRIES",
                value=str(max_retries),
                error="重试次数过多",
                suggestion="建议设置在10次以内"
            ))
        else:
            self.results.append(ValidationResult(
                is_valid=True,
                field="OPENAI_MAX_RETRIES",
                value=str(max_retries),
                error=None,
                suggestion="重试设置合理"
            ))

    async def test_connection(self) -> ValidationResult:
        """测试连接"""
        try:
            llm_service = get_llm_service()
            is_connected = await llm_service.test_connection(LLMProvider.OPENAI)

            if is_connected:
                return ValidationResult(
                    is_valid=True,
                    field="API_CONNECTION",
                    value="success",
                    error=None,
                    suggestion="API连接测试成功"
                )
            else:
                return ValidationResult(
                    is_valid=False,
                    field="API_CONNECTION",
                    value="failed",
                    error="无法连接到API端点",
                    suggestion="请检查API密钥、网络连接或端点配置"
                )
        except Exception as e:
            return ValidationResult(
                is_valid=False,
                field="API_CONNECTION",
                value="error",
                error=str(e),
                suggestion="请检查配置详情或联系技术支持"
            )

    def print_results(self) -> None:
        """打印验证结果"""
        print("🔍 OpenAI配置验证结果")
        print("=" * 50)

        valid_count = sum(1 for r in self.results if r.is_valid)
        total_count = len(self.results)

        for result in self.results:
            status = "✅" if result.is_valid else "❌"
            print(f"{status} {result.field}: {result.value or 'N/A'}")

            if result.error:
                print(f"   ❌ 错误: {result.error}")
            if result.suggestion:
                print(f"   💡 建议: {result.suggestion}")
            print()

        print(f"总结: {valid_count}/{total_count} 项配置正确")

    def print_provider_info(self) -> None:
        """打印提供商信息"""
        try:
            llm_service = get_llm_service()
            status = llm_service.get_provider_status()

            print("📊 LLM提供商状态")
            print("=" * 50)

            for provider, info in status.items():
                print(f"提供商: {provider}")
                print(f"  可用性: {'✅' if info.get('available') else '❌'}")
                print(f"  客户端类型: {info.get('client_type', 'unknown')}")

                if provider == 'openai':
                    print(f"  Base URL: {info.get('base_url', 'default')}")
                    print(f"  超时: {info.get('timeout', 'unknown')}秒")
                    print(f"  重试次数: {info.get('max_retries', 'unknown')}")
                    print(f"  模型: {info.get('model', 'unknown')}")
                    print(f"  API密钥: {'✅ 已配置' if info.get('configured_api_key') else '❌ 未配置'}")

                print()
        except Exception as e:
            print(f"❌ 无法获取提供商状态: {e}")

    def generate_env_file_suggestions(self) -> None:
        """生成环境变量配置建议"""
        print("📝 环境变量配置建议")
        print("=" * 50)

        # 检查当前配置
        current_config = {}
        for result in self.results:
            if result.field.startswith("OPENAI_"):
                config_key = result.field
                current_config[config_key] = result.value

        print("# 推荐的.env配置")
        print("# OpenAI API配置")

        api_key_result = next((r for r in self.results if r.field == "OPENAI_API_KEY"), None)
        if api_key_result and not api_key_result.is_valid:
            print("OPENAI_API_KEY=your_openai_api_key_here")

        base_url_result = next((r for r in self.results if r.field == "OPENAI_BASE_URL"), None)
        if base_url_result and base_url_result.value:
            print(f"OPENAI_BASE_URL={base_url_result.value}")
        else:
            print("# OPENAI_BASE_URL=https://api.openai.com/v1  # 可选，自定义端点")

        print("OPENAI_MODEL=gpt-4")
        print("OPENAI_MAX_TOKENS=4096")
        print("OPENAI_TEMPERATURE=0.1")
        print("OPENAI_TIMEOUT=60")
        print("OPENAI_MAX_RETRIES=3")
        print("# OPENAI_ORGANIZATION=your_organization_id  # 可选")

    def run_validation(self, test_connection: bool = False) -> bool:
        """运行完整验证"""
        print("🚀 开始OpenAI配置验证")
        print()

        # 基础配置验证
        self.validate_basic_config()

        # 打印结果
        self.print_results()

        # 提供商信息
        self.print_provider_info()

        # 生成配置建议
        self.generate_env_file_suggestions()

        # 连接测试
        if test_connection:
            print("🔗 测试API连接...")
            connection_result = asyncio.run(self.test_connection())
            status = "✅" if connection_result.is_valid else "❌"
            print(f"{status} {connection_result.field}: {connection_result.value}")
            if connection_result.error:
                print(f"   错误: {connection_result.error}")
            if connection_result.suggestion:
                print(f"   建议: {connection_result.suggestion}")
            print()

        # 总体结果
        all_valid = all(result.is_valid for result in self.results)
        if all_valid:
            print("🎉 所有配置验证通过！系统已准备就绪。")
            return True
        else:
            print("⚠️  配置验证发现问题，请按照上述建议进行修复。")
            return False


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description="OpenAI配置验证工具")
    parser.add_argument("--test-connection", action="store_true",
                       help="测试API连接")
    parser.add_argument("--output-json", action="store_true",
                       help="输出JSON格式结果")

    args = parser.parse_args()

    validator = OpenAIConfigValidator()

    if args.output_json:
        # JSON输出模式
        validator.validate_basic_config()
        results_data = [
            {
                "field": r.field,
                "is_valid": r.is_valid,
                "value": r.value,
                "error": r.error,
                "suggestion": r.suggestion
            }
            for r in validator.results
        ]
        print(json.dumps(results_data, indent=2, ensure_ascii=False))
    else:
        # 交互模式
        success = validator.run_validation(test_connection=args.test_connection)
        sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()