#!/usr/bin/env python3
"""
项目管理工具
提供便捷的命令来运行各种分析和管理工具
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path

class ProjectManager:
    """项目管理器"""

    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.tools_dir = project_root / "tools"
        self.scripts_dir = project_root / "scripts"
        self.docs_dir = project_root / "docs"

    def run_performance_analysis(self, analysis_type="all"):
        """运行性能分析"""
        print(f"🚀 运行性能分析: {analysis_type}")

        if analysis_type == "all" or analysis_type == "simple":
            print("📊 运行简单性能分析...")
            subprocess.run([
                sys.executable,
                self.tools_dir / "performance" / "simple_performance_analysis.py",
                "--project-root", str(self.project_root)
            ])

        if analysis_type == "all" or analysis_type == "database":
            print("🗄️ 运行数据库性能分析...")
            subprocess.run([
                sys.executable,
                self.tools_dir / "performance" / "database_optimization_analysis.py",
                "--project-root", str(self.project_root)
            ])

        if analysis_type == "all" or analysis_type == "api":
            print("🌐 运行API性能分析...")
            subprocess.run([
                sys.executable,
                self.tools_dir / "performance" / "api_performance_analysis.py",
                "--project-root", str(self.project_root)
            ])

    def run_tests(self, test_type="all"):
        """运行测试"""
        print(f"🧪 运行测试: {test_type}")

        if test_type == "all" or test_type == "quick":
            print("📋 运行快速测试分析...")
            subprocess.run([
                sys.executable,
                self.tools_dir / "analysis" / "run_tests.py"
            ])

        if test_type == "all" or test_type == "generate_report":
            print("📄 生成测试报告...")
            subprocess.run([
                sys.executable,
                self.tools_dir / "analysis" / "test_report_generator.py"
            ])

    def setup_development(self):
        """设置开发环境"""
        print("🔧 设置开发环境...")

        # 复制环境配置
        env_example = self.scripts_dir / "deployment" / ".env.example"
        if env_example.exists() and not Path(".env").exists():
            print("📝 复制环境配置文件...")
            import shutil
            shutil.copy(env_example, Path(".env"))

        # 安装依赖
        requirements = self.scripts_dir / "deployment" / "requirements.txt"
        if requirements.exists():
            print("📦 安装Python依赖...")
            subprocess.run([
                sys.executable, "-m", "pip", "install", "-r", str(requirements)
            ])

        print("✅ 开发环境设置完成!")

    def start_services(self):
        """启动服务"""
        print("🚀 启动服务...")

        compose_file = self.scripts_dir / "deployment" / "docker-compose.yml"
        if compose_file.exists():
            subprocess.run([
                "docker-compose", "-f", str(compose_file), "up", "-d"
            ])
        else:
            print("❌ docker-compose.yml 文件不存在")

    def stop_services(self):
        """停止服务"""
        print("🛑 停止服务...")

        compose_file = self.scripts_dir / "deployment" / "docker-compose.yml"
        if compose_file.exists():
            subprocess.run([
                "docker-compose", "-f", str(compose_file), "down"
            ])

    def show_project_structure(self):
        """显示项目结构"""
        print("📁 项目结构:")
        print("""
crypto-ai-trading-system/
├── 📋 README.md                     # 项目说明
├── 🤖 project_manager.py           # 项目管理工具 (本文件)
├── 🏗️ backend/                     # 后端API服务
│   ├── src/api/                    # REST API接口
│   ├── src/services/               # AI Agent服务
│   ├── src/models/                 # 数据模型
│   ├── src/core/                   # 核心功能
│   └── tests/                      # 测试套件
├── 🔧 tools/                       # 开发工具
│   ├── analysis/                   # 分析工具
│   ├── performance/                # 性能分析
│   └── monitoring/                 # 监控工具
├── 📜 scripts/                     # 脚本文件
│   ├── deployment/                 # 部署脚本
│   └── utils/                      # 工具脚本
├── 📚 docs/                        # 文档
│   ├── reports/                    # 分析报告
│   ├── api/                        # API文档
│   ├── setup/                      # 安装指南
│   └── architecture/               # 架构文档
├── ⚙️ config/                      # 配置文件
│   ├── redis/                      # Redis配置
│   ├── postgres/                   # PostgreSQL配置
│   └── nginx/                      # Nginx配置
├── 📱 mobile/                      # 移动端
├── 📋 specs/                       # 技术规格
└── 🔄 .github/                     # GitHub配置
        """)

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="多Agent加密货币量化交易系统 - 项目管理工具")
    parser.add_argument("command", choices=[
        "analyze", "test", "setup", "start", "stop", "structure"
    ], help="要执行的命令")
    parser.add_argument("--type", help="分析或测试类型")
    parser.add_argument("--project-root", default=".", help="项目根目录")

    args = parser.parse_args()

    project_root = Path(args.project_root)
    manager = ProjectManager(project_root)

    if args.command == "analyze":
        analysis_type = args.type or "all"
        manager.run_performance_analysis(analysis_type)

    elif args.command == "test":
        test_type = args.type or "all"
        manager.run_tests(test_type)

    elif args.command == "setup":
        manager.setup_development()

    elif args.command == "start":
        manager.start_services()

    elif args.command == "stop":
        manager.stop_services()

    elif args.command == "structure":
        manager.show_project_structure()

if __name__ == "__main__":
    main()