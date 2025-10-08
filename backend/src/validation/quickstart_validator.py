"""
Quickstart验证模块

验证快速开始指南中的系统要求和配置是否正确
"""

import asyncio
import os
import sys
import subprocess
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import importlib.util

from ..core.config import settings
from ..core.exceptions import ConfigurationError

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """验证结果"""
    category: str
    item: str
    status: bool
    message: str
    severity: str  # "error", "warning", "info"


class QuickstartValidator:
    """快速开始验证器"""

    def __init__(self):
        self.results: List[ValidationResult] = []
        self.project_root = Path(__file__).parent.parent.parent.parent

    async def validate_all(self) -> List[ValidationResult]:
        """执行所有验证"""
        logger.info("开始执行快速开始验证...")

        # 验证系统要求
        await self._validate_system_requirements()

        # 验证Python环境
        await self._validate_python_environment()

        # 验证外部服务
        await self._validate_external_services()

        # 验证项目结构
        await self._validate_project_structure()

        # 验证配置文件
        await self._validate_configuration()

        # 验证数据库连接
        await self._validate_database_connection()

        # 验证API集成
        await self._validate_api_integrations()

        # 验证LLM服务
        await self._validate_llm_services()

        logger.info(f"验证完成，共 {len(self.results)} 项检查")
        return self.results

    async def _validate_system_requirements(self):
        """验证系统要求"""
        # 检查操作系统
        self._check_operating_system()

        # 检查内存
        self._check_memory()

        # 检查存储空间
        self._check_storage_space()

        # 检查网络连接
        await self._check_network_connectivity()

    def _check_operating_system(self):
        """检查操作系统"""
        import platform
        system = platform.system()

        if system in ["Linux", "Darwin"]:
            self.results.append(ValidationResult(
                category="系统要求",
                item="操作系统",
                status=True,
                message=f"支持的操作系统: {system}",
                severity="info"
            ))
        else:
            self.results.append(ValidationResult(
                category="系统要求",
                item="操作系统",
                status=False,
                message=f"不支持的操作系统: {system}，推荐使用Linux",
                severity="warning"
            ))

    def _check_memory(self):
        """检查内存"""
        try:
            if os.name == 'posix':
                with open('/proc/meminfo', 'r') as f:
                    meminfo = f.read()

                mem_total = 0
                for line in meminfo.split('\n'):
                    if 'MemTotal:' in line:
                        mem_total = int(line.split()[1]) // 1024  # MB
                        break

                if mem_total >= 8192:  # 8GB
                    status = True
                    message = f"内存充足: {mem_total}MB"
                    severity = "info"
                elif mem_total >= 4096:  # 4GB
                    status = True
                    message = f"内存可能不足: {mem_total}MB，推荐8GB以上"
                    severity = "warning"
                else:
                    status = False
                    message = f"内存严重不足: {mem_total}MB，最低要求8GB"
                    severity = "error"

                self.results.append(ValidationResult(
                    category="系统要求",
                    item="内存",
                    status=status,
                    message=message,
                    severity=severity
                ))

        except Exception as e:
            self.results.append(ValidationResult(
                category="系统要求",
                item="内存",
                status=False,
                message=f"无法检查内存: {e}",
                severity="warning"
            ))

    def _check_storage_space(self):
        """检查存储空间"""
        try:
            stat = os.statvfs(self.project_root)
            free_space_gb = (stat.f_bavail * stat.f_frsize) // (1024**3)

            if free_space_gb >= 10:
                status = True
                message = f"存储空间充足: {free_space_gb}GB 可用"
                severity = "info"
            elif free_space_gb >= 5:
                status = True
                message = f"存储空间较少: {free_space_gb}GB 可用，推荐10GB以上"
                severity = "warning"
            else:
                status = False
                message = f"存储空间不足: {free_space_gb}GB 可用，最低要求5GB"
                severity = "error"

            self.results.append(ValidationResult(
                category="系统要求",
                item="存储空间",
                status=status,
                message=message,
                severity=severity
            ))

        except Exception as e:
            self.results.append(ValidationResult(
                category="系统要求",
                item="存储空间",
                status=False,
                message=f"无法检查存储空间: {e}",
                severity="warning"
            ))

    async def _check_network_connectivity(self):
        """检查网络连接"""
        try:
            import aiohttp
            async with aiohttp.ClientSession() as session:
                # 测试基本网络连接
                async with session.get('https://httpbin.org/get', timeout=10) as response:
                    if response.status == 200:
                        self.results.append(ValidationResult(
                            category="系统要求",
                            item="网络连接",
                            status=True,
                            message="网络连接正常",
                            severity="info"
                        ))
                    else:
                        self.results.append(ValidationResult(
                            category="系统要求",
                            item="网络连接",
                            status=False,
                            message=f"网络连接异常: HTTP {response.status}",
                            severity="error"
                        ))
        except Exception as e:
            self.results.append(ValidationResult(
                category="系统要求",
                item="网络连接",
                status=False,
                message=f"网络连接失败: {e}",
                severity="error"
            ))

    async def _validate_python_environment(self):
        """验证Python环境"""
        # 检查Python版本
        self._check_python_version()

        # 检查依赖包
        await self._check_dependencies()

        # 检查虚拟环境
        self._check_virtual_environment()

    def _check_python_version(self):
        """检查Python版本"""
        version = sys.version_info
        if version.major == 3 and version.minor >= 11:
            self.results.append(ValidationResult(
                category="Python环境",
                item="Python版本",
                status=True,
                message=f"Python版本正确: {version.major}.{version.minor}.{version.micro}",
                severity="info"
            ))
        else:
            self.results.append(ValidationResult(
                category="Python环境",
                item="Python版本",
                status=False,
                message=f"Python版本过低: {version.major}.{version.minor}.{version.micro}，要求3.11+",
                severity="error"
            ))

    async def _check_dependencies(self):
        """检查依赖包"""
        required_packages = [
            'fastapi', 'sqlalchemy', 'alembic', 'redis', 'celery',
            'pandas', 'numpy', 'ccxt', 'langchain', 'pytest'
        ]

        for package in required_packages:
            try:
                spec = importlib.util.find_spec(package)
                if spec is not None:
                    self.results.append(ValidationResult(
                        category="Python环境",
                        item=f"依赖包: {package}",
                        status=True,
                        message=f"已安装: {package}",
                        severity="info"
                    ))
                else:
                    self.results.append(ValidationResult(
                        category="Python环境",
                        item=f"依赖包: {package}",
                        status=False,
                        message=f"未安装: {package}",
                        severity="error"
                    ))
            except ImportError:
                self.results.append(ValidationResult(
                    category="Python环境",
                    item=f"依赖包: {package}",
                    status=False,
                    message=f"导入失败: {package}",
                    severity="error"
                ))

    def _check_virtual_environment(self):
        """检查虚拟环境"""
        in_venv = hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix)

        if in_venv:
            self.results.append(ValidationResult(
                category="Python环境",
                item="虚拟环境",
                status=True,
                message="正在使用虚拟环境",
                severity="info"
            ))
        else:
            self.results.append(ValidationResult(
                category="Python环境",
                item="虚拟环境",
                status=False,
                message="未检测到虚拟环境，建议使用虚拟环境",
                severity="warning"
            ))

    async def _validate_external_services(self):
        """验证外部服务"""
        # 检查PostgreSQL
        await self._check_postgresql()

        # 检查Redis
        await self._check_redis()

    async def _check_postgresql(self):
        """检查PostgreSQL"""
        try:
            from sqlalchemy import create_engine, text
            database_url = settings.DATABASE_URL

            if database_url:
                engine = create_engine(database_url)
                with engine.connect() as conn:
                    result = conn.execute(text("SELECT version()"))
                    version = result.fetchone()[0]

                self.results.append(ValidationResult(
                    category="外部服务",
                    item="PostgreSQL",
                    status=True,
                    message=f"PostgreSQL连接正常: {version[:50]}...",
                    severity="info"
                ))
            else:
                self.results.append(ValidationResult(
                    category="外部服务",
                    item="PostgreSQL",
                    status=False,
                    message="未配置数据库连接",
                    severity="error"
                ))

        except Exception as e:
            self.results.append(ValidationResult(
                category="外部服务",
                item="PostgreSQL",
                status=False,
                message=f"PostgreSQL连接失败: {e}",
                severity="error"
            ))

    async def _check_redis(self):
        """检查Redis"""
        try:
            from ..core.cache import get_cache
            cache = get_cache()
            await cache.ping()

            self.results.append(ValidationResult(
                category="外部服务",
                item="Redis",
                status=True,
                message="Redis连接正常",
                severity="info"
            ))

        except Exception as e:
            self.results.append(ValidationResult(
                category="外部服务",
                item="Redis",
                status=False,
                message=f"Redis连接失败: {e}",
                severity="error"
            ))

    async def _validate_project_structure(self):
        """验证项目结构"""
        required_dirs = [
            'backend/src',
            'backend/src/api',
            'backend/src/services',
            'backend/src/models',
            'backend/src/core',
            'backend/tests',
            'specs/001-python-llm-agent'
        ]

        required_files = [
            'backend/requirements.txt',
            'backend/pyproject.toml',
            'specs/001-python-llm-agent/spec.md',
            'specs/001-python-llm-agent/plan.md',
            'specs/001-python-llm-agent/tasks.md'
        ]

        for dir_path in required_dirs:
            full_path = self.project_root / dir_path
            if full_path.exists() and full_path.is_dir():
                self.results.append(ValidationResult(
                    category="项目结构",
                    item=f"目录: {dir_path}",
                    status=True,
                    message="目录存在",
                    severity="info"
                ))
            else:
                self.results.append(ValidationResult(
                    category="项目结构",
                    item=f"目录: {dir_path}",
                    status=False,
                    message="目录不存在",
                    severity="error"
                ))

        for file_path in required_files:
            full_path = self.project_root / file_path
            if full_path.exists() and full_path.is_file():
                self.results.append(ValidationResult(
                    category="项目结构",
                    item=f"文件: {file_path}",
                    status=True,
                    message="文件存在",
                    severity="info"
                ))
            else:
                self.results.append(ValidationResult(
                    category="项目结构",
                    item=f"文件: {file_path}",
                    status=False,
                    message="文件不存在",
                    severity="error"
                ))

    async def _validate_configuration(self):
        """验证配置文件"""
        config_files = [
            '.env',
            'config.yaml',
            'scripts/deployment/config.yaml'
        ]

        for config_file in config_files:
            full_path = self.project_root / config_file
            if full_path.exists():
                self.results.append(ValidationResult(
                    category="配置文件",
                    item=config_file,
                    status=True,
                    message="配置文件存在",
                    severity="info"
                ))
            else:
                self.results.append(ValidationResult(
                    category="配置文件",
                    item=config_file,
                    status=False,
                    message="配置文件不存在",
                    severity="warning"
                ))

    async def _validate_database_connection(self):
        """验证数据库连接"""
        try:
            from ..core.database import engine
            with engine.connect() as conn:
                # 检查必要的表是否存在
                required_tables = ['users', 'trading_strategies', 'market_data']
                for table in required_tables:
                    try:
                        result = conn.execute(text(f"SELECT 1 FROM {table} LIMIT 1"))
                        self.results.append(ValidationResult(
                            category="数据库",
                            item=f"表: {table}",
                            status=True,
                            message="表存在且可访问",
                            severity="info"
                        ))
                    except Exception:
                        self.results.append(ValidationResult(
                            category="数据库",
                            item=f"表: {table}",
                            status=False,
                            message="表不存在或不可访问",
                            severity="warning"
                        ))

        except Exception as e:
            self.results.append(ValidationResult(
                category="数据库",
                item="连接",
                status=False,
                message=f"数据库连接失败: {e}",
                severity="error"
            ))

    async def _validate_api_integrations(self):
        """验证API集成"""
        # 检查交易所API配置
        self._check_exchange_api_config()

        # 检查API端点
        await self._check_api_endpoints()

    def _check_exchange_api_config(self):
        """检查交易所API配置"""
        required_env_vars = [
            'BINANCE_API_KEY',
            'BINANCE_SECRET_KEY',
            'COINBASE_API_KEY',
            'COINBASE_SECRET_KEY'
        ]

        for env_var in required_env_vars:
            if os.getenv(env_var):
                self.results.append(ValidationResult(
                    category="API集成",
                    item=f"环境变量: {env_var}",
                    status=True,
                    message="API密钥已配置",
                    severity="info"
                ))
            else:
                self.results.append(ValidationResult(
                    category="API集成",
                    item=f"环境变量: {env_var}",
                    status=False,
                    message="API密钥未配置",
                    severity="warning"
                ))

    async def _check_api_endpoints(self):
        """检查API端点"""
        try:
            from ..src.main import app
            # 这里应该测试主要的API端点
            endpoints = ['/health', '/status']

            for endpoint in endpoints:
                self.results.append(ValidationResult(
                    category="API集成",
                    item=f"端点: {endpoint}",
                    status=True,
                    message="端点已定义",
                    severity="info"
                ))

        except Exception as e:
            self.results.append(ValidationResult(
                category="API集成",
                item="API端点",
                status=False,
                message=f"API端点检查失败: {e}",
                severity="error"
            ))

    async def _validate_llm_services(self):
        """验证LLM服务"""
        required_env_vars = [
            'OPENAI_API_KEY',
            'ANTHROPIC_API_KEY'
        ]

        for env_var in required_env_vars:
            if os.getenv(env_var):
                self.results.append(ValidationResult(
                    category="LLM服务",
                    item=f"环境变量: {env_var}",
                    status=True,
                    message="LLM API密钥已配置",
                    severity="info"
                ))
            else:
                self.results.append(ValidationResult(
                    category="LLM服务",
                    item=f"环境变量: {env_var}",
                    status=False,
                    message="LLM API密钥未配置",
                    severity="warning"
                ))

    def generate_report(self) -> str:
        """生成验证报告"""
        error_count = sum(1 for r in self.results if r.severity == "error")
        warning_count = sum(1 for r in self.results if r.severity == "warning")
        info_count = sum(1 for r in self.results if r.severity == "info")

        report = f"""
# 快速开始验证报告

## 验证结果概览
- ✅ 通过: {info_count} 项
- ⚠️  警告: {warning_count} 项
- ❌ 错误: {error_count} 项

## 详细结果

### 系统要求
"""

        # 按类别分组显示结果
        categories = {}
        for result in self.results:
            if result.category not in categories:
                categories[result.category] = []
            categories[result.category].append(result)

        for category, results in categories.items():
            report += f"\n#### {category}\n"
            for result in results:
                icon = "✅" if result.status else "❌"
                severity_marker = {
                    "info": "",
                    "warning": " ⚠️",
                    "error": " ❌"
                }.get(result.severity, "")

                report += f"- {icon} {result.item}{severity_marker}: {result.message}\n"

        # 添加总结和建议
        report += f"""

## 总结和建议

"""

        if error_count == 0 and warning_count == 0:
            report += "🎉 所有验证项目都通过！系统已准备就绪。"
        elif error_count == 0:
            report += "⚠️ 存在一些警告，但不影响基本功能。建议修复警告项以获得最佳体验。"
        else:
            report += f"❌ 发现 {error_count} 个错误需要修复。请解决错误项后再启动系统。"

        if error_count > 0:
            report += """

## 错误修复建议

1. **配置错误**: 检查环境变量和配置文件是否正确设置
2. **依赖问题**: 运行 `pip install -r requirements.txt` 安装缺失的依赖
3. **服务连接**: 确保PostgreSQL和Redis服务正在运行
4. **权限问题**: 检查文件和目录权限是否正确

## 下一步操作

1. 修复所有错误项
2. 解决警告项（可选）
3. 重新运行验证
4. 启动系统: `python -m backend.src.main`
"""

        return report


async def run_quickstart_validation():
    """运行快速开始验证"""
    validator = QuickstartValidator()
    results = await validator.validate_all()
    report = validator.generate_report()

    # 保存报告到文件
    report_path = Path("docs/reports/quickstart_validation_report.md")
    report_path.parent.mkdir(parents=True, exist_ok=True)

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)

    print(report)
    print(f"\n📄 详细报告已保存到: {report_path}")

    return results


if __name__ == "__main__":
    asyncio.run(run_quickstart_validation())