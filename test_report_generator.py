#!/usr/bin/env python3
"""
测试报告生成器

用于生成详细的测试报告，包括测试覆盖率、性能指标和问题分析。
"""

import os
import sys
import json
import time
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any
import argparse

class TestReportGenerator:
    """测试报告生成器"""

    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.backend_root = self.project_root / "backend"
        self.test_results = {}
        self.report_data = {
            "metadata": {
                "generated_at": datetime.now().isoformat(),
                "project_root": str(project_root),
                "python_version": sys.version,
                "platform": sys.platform
            },
            "test_results": {},
            "coverage": {},
            "performance": {},
            "issues": [],
            "summary": {}
        }

    def run_tests(self) -> Dict[str, Any]:
        """运行所有测试"""
        print("🧪 开始运行测试套件...")

        test_suites = [
            {
                "name": "Contract Tests",
                "path": "tests/contract",
                "pattern": "test_*.py",
                "description": "API端点合约测试"
            },
            {
                "name": "Integration Tests",
                "path": "tests/integration",
                "pattern": "test_*.py",
                "description": "系统集成测试"
            },
            {
                "name": "Unit Tests",
                "path": "tests/unit",
                "pattern": "test_*.py",
                "description": "单元测试"
            }
        ]

        for suite in test_suites:
            print(f"\n📋 运行 {suite['name']}...")
            result = self._run_test_suite(suite)
            self.report_data["test_results"][suite["name"]] = result

            # 输出测试结果
            self._print_test_result(suite["name"], result)

        return self.report_data["test_results"]

    def _run_test_suite(self, suite: Dict[str, Any]) -> Dict[str, Any]:
        """运行单个测试套件"""
        test_path = self.backend_root / suite["path"]

        if not test_path.exists():
            return {
                "status": "skipped",
                "reason": f"测试目录不存在: {test_path}",
                "total": 0,
                "passed": 0,
                "failed": 0,
                "skipped": 0,
                "errors": [],
                "duration": 0,
                "test_files": []
            }

        # 查找测试文件
        test_files = list(test_path.glob(suite["pattern"]))

        if not test_files:
            return {
                "status": "skipped",
                "reason": f"没有找到测试文件: {suite['pattern']}",
                "total": 0,
                "passed": 0,
                "failed": 0,
                "skipped": 0,
                "errors": [],
                "duration": 0,
                "test_files": []
            }

        # 运行测试
        return self._run_pytest(test_path, suite["name"], test_files)

    def _run_pytest(self, test_path: Path, suite_name: str, test_files: List[Path]) -> Dict[str, Any]:
        """使用pytest运行测试"""
        start_time = time.time()

        try:
            # 构建pytest命令
            cmd = [
                sys.executable, "-m", "pytest",
                str(test_path),
                "-v",
                "--tb=short",
                "--json-report",
                "--json-report-file=/tmp/pytest_report.json",
                "--html=/tmp/test_report.html",
                "--self-contained-html",
                "--cov=" + str(self.backend_root / "src"),
                "--cov-report=term-missing",
                "--cov-report=html:/tmp/coverage_html",
                "--cov-report=json:/tmp/coverage.json"
            ]

            # 切换到backend目录
            original_cwd = os.getcwd()
            os.chdir(self.backend_root)

            # 运行pytest
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300  # 5分钟超时
            )

            duration = time.time() - start_time

            # 解析结果
            if result.returncode == 0:
                return self._parse_pytest_success(result, test_files, duration)
            else:
                return self._parse_pytest_failure(result, test_files, duration)

        except subprocess.TimeoutExpired:
            return {
                "status": "timeout",
                "reason": "测试执行超时",
                "total": 0,
                "passed": 0,
                "failed": 0,
                "skipped": 0,
                "errors": ["测试执行超时"],
                "duration": 300,
                "test_files": [f.name for f in test_files]
            }
        except Exception as e:
            return {
                "status": "error",
                "reason": f"执行错误: {str(e)}",
                "total": 0,
                "passed": 0,
                "failed": 0,
                "skipped": 0,
                "errors": [str(e)],
                "duration": time.time() - start_time,
                "test_files": [f.name for f in test_files]
            }
        finally:
            os.chdir(original_cwd)

    def _parse_pytest_success(self, result: subprocess.CompletedProcess, test_files: List[Path], duration: float) -> Dict[str, Any]:
        """解析pytest成功结果"""
        try:
            # 尝试读取JSON报告
            with open("/tmp/pytest_report.json", "r") as f:
                json_report = json.load(f)

            summary = json_report.get("summary", {})

            return {
                "status": "passed",
                "total": summary.get("total", 0),
                "passed": summary.get("passed", 0),
                "failed": summary.get("failed", 0),
                "skipped": summary.get("skipped", 0),
                "errors": [],
                "duration": duration,
                "test_files": [f.name for f in test_files],
                "details": {
                    "summary": summary,
                    "collected": summary.get("collected", 0)
                }
            }
        except Exception as e:
            # 如果JSON解析失败，尝试从stdout解析
            return self._parse_pytest_output(result.stdout, test_files, duration, "passed")

    def _parse_pytest_failure(self, result: subprocess.CompletedProcess, test_files: List[Path], duration: float) -> Dict[str, Any]:
        """解析pytest失败结果"""
        try:
            with open("/tmp/pytest_report.json", "r") as f:
                json_report = json.load(f)

            summary = json_report.get("summary", {})

            # 提取错误信息
            errors = []
            if "errors" in json_report:
                for error in json_report["errors"]:
                    errors.append(f"{error.get('test_id', 'Unknown')}: {error.get('message', 'Unknown error')}")

            # 从输出中提取额外错误信息
            output_lines = result.stdout.split('\n')
            for line in output_lines:
                if "FAILED" in line and "test_" in line:
                    if line not in errors:
                        errors.append(line.strip())

            return {
                "status": "failed",
                "total": summary.get("total", 0),
                "passed": summary.get("passed", 0),
                "failed": summary.get("failed", 0),
                "skipped": summary.get("skipped", 0),
                "errors": errors[:10],  # 限制错误数量
                "duration": duration,
                "test_files": [f.name for f in test_files],
                "details": {
                    "summary": summary,
                    "collected": summary.get("collected", 0),
                    "returncode": result.returncode,
                    "stdout": result.stdout[-1000:],  # 保留最后1000字符
                    "stderr": result.stderr[-1000:]
                }
            }
        except Exception as e:
            return self._parse_pytest_output(result.stdout, test_files, duration, "failed")

    def _parse_pytest_output(self, output: str, test_files: List[Path], duration: float, status: str) -> Dict[str, Any]:
        """从pytest输出解析结果"""
        lines = output.split('\n')

        total = 0
        passed = 0
        failed = 0
        skipped = 0
        errors = []

        for line in lines:
            if "=" in line and "passed" in line.lower():
                # 解析类似 "2 passed, 1 failed in 5.23s" 的行
                parts = line.split()
                for part in parts:
                    if part.isdigit():
                        num = int(part)
                        if "passed" in line:
                            passed = num
                        elif "failed" in line:
                            failed = num
                        elif "skipped" in line:
                            skipped = num
                        total = max(total, num)
            elif "FAILED" in line:
                errors.append(line.strip())
            elif "ERROR" in line:
                errors.append(line.strip())

        return {
            "status": status,
            "total": total,
            "passed": passed,
            "failed": failed,
            "skipped": skipped,
            "errors": errors[:10],
            "duration": duration,
            "test_files": [f.name for f in test_files],
            "details": {
                "raw_output": output
            }
        }

    def _print_test_result(self, suite_name: str, result: Dict[str, Any]):
        """打印测试结果"""
        status_icon = "✅" if result["status"] == "passed" else "❌" if result["status"] == "failed" else "⚠️"

        print(f"{status_icon} {suite_name}")
        print(f"   总计: {result['total']}, 通过: {result['passed']}, 失败: {result['failed']}, 跳过: {result['skipped']}")
        print(f"   耗时: {result['duration']:.2f}s")

        if result["status"] != "passed":
            print(f"   状态: {result['status']}")
            if result.get("reason"):
                print(f"   原因: {result['reason']}")
            if result["errors"]:
                print(f"   错误: {len(result['errors'])}个")
                for error in result["errors"][:3]:  # 显示前3个错误
                    print(f"     - {error}")
        print()

    def check_code_quality(self) -> Dict[str, Any]:
        """检查代码质量"""
        print("🔍 检查代码质量...")

        quality_checks = {
            "flake8": self._run_flake8(),
            "isort": self._run_isort_check(),
            "black": self._run_black_check(),
            "mypy": self._run_mypy_check()
        }

        self.report_data["code_quality"] = quality_checks
        return quality_checks

    def _run_flake8(self) -> Dict[str, Any]:
        """运行flake8代码检查"""
        try:
            original_cwd = os.getcwd()
            os.chdir(self.backend_root)

            result = subprocess.run(
                [sys.executable, "-m", "flake8", "src/", "--count"],
                capture_output=True,
                text=True
            )

            os.chdir(original_cwd)

            return {
                "status": "passed" if result.returncode == 0 else "failed",
                "issues": int(result.stdout.strip()) if result.stdout.strip() else 0,
                "output": result.stdout
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}

    def _run_isort_check(self) -> Dict[str, Any]:
        """运行isort检查"""
        try:
            original_cwd = os.getcwd()
            os.chdir(self.backend_root)

            result = subprocess.run(
                [sys.executable, "-m", "isort", "src/", "--check-only"],
                capture_output=True,
                text=True
            )

            os.chdir(original_cwd)

            return {
                "status": "passed" if result.returncode == 0 else "failed",
                "output": result.stdout
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}

    def _run_black_check(self) -> Dict[str, Any]:
        """运行black格式检查"""
        try:
            original_cwd = os.getcwd()
            os.chdir(self.backend_root)

            result = subprocess.run(
                [sys.executable, "-m", "black", "src/", "--check"],
                capture_output=True,
                text=True
            )

            os.chdir(original_cwd)

            return {
                "status": "passed" if result.returncode == 0 else "failed",
                "output": result.stdout
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}

    def _run_mypy_check(self) -> Dict[str, Any]:
        """运行mypy类型检查"""
        try:
            original_cwd = os.getcwd()
            os.chdir(self.backend_root)

            result = subprocess.run(
                [sys.executable, "-m", "mypy", "src/", "--ignore-missing-imports"],
                capture_output=True,
                text=True,
                timeout=60
            )

            os.chdir(original_cwd)

            return {
                "status": "passed" if result.returncode == 0 else "failed",
                "output": result.stdout
            }
        except subprocess.TimeoutExpired:
            return {"status": "timeout", "error": "类型检查超时"}
        except Exception as e:
            return {"status": "error", "error": str(e)}

    def analyze_code_complexity(self) -> Dict[str, Any]:
        """分析代码复杂度"""
        print("📊 分析代码复杂度...")

        complexity_metrics = {
            "total_files": 0,
            "total_lines": 0,
            "python_files": 0,
            "test_files": 0,
            "service_files": 0,
            "model_files": 0,
            "largest_files": [],
            "complexity_analysis": {}
        }

        # 扫描Python文件
        for root, dirs, files in os.walk(self.backend_root):
            # 跳过特定目录
            dirs[:] = [d for d in dirs if d not in ['.git', '__pycache__', 'node_modules', '.pytest_cache']]

            for file in files:
                if file.endswith('.py'):
                    file_path = Path(root) / file
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            lines = f.readlines()
                            line_count = len(lines)

                            complexity_metrics["python_files"] += 1
                            complexity_metrics["total_lines"] += line_count
                            complexity_metrics["total_files"] += 1

                            # 分类文件
                            if "test" in file_path.parts:
                                complexity_metrics["test_files"] += 1
                            elif "services" in file_path.parts:
                                complexity_metrics["service_files"] += 1
                            elif "models" in file_path.parts:
                                complexity_metrics["model_files"] += 1

                            # 记录最大文件
                            complexity_metrics["largest_files"].append({
                                "path": str(file_path.relative_to(self.backend_root)),
                                "lines": line_count
                            })

                    except Exception as e:
                        print(f"警告: 无法读取文件 {file_path}: {e}")

        # 排序最大的文件
        complexity_metrics["largest_files"] = sorted(
            complexity_metrics["largest_files"],
            key=lambda x: x["lines"],
            reverse=True
        )[:10]

        self.report_data["complexity"] = complexity_metrics
        return complexity_metrics

    def generate_report(self, output_file: str = "test_report.json"):
        """生成测试报告"""
        print("📋 生成测试报告...")

        # 计算汇总统计
        total_tests = sum(r.get("total", 0) for r in self.report_data["test_results"].values())
        total_passed = sum(r.get("passed", 0) for r in self.report_data["test_results"].values())
        total_failed = sum(r.get("failed", 0) for r in self.report_data["test_results"].values())
        total_skipped = sum(r.get("skipped", 0) for r in self.report_data["test_results"].values())

        self.report_data["summary"] = {
            "total_tests": total_tests,
            "total_passed": total_passed,
            "total_failed": total_failed,
            "total_skipped": total_skipped,
            "success_rate": (total_passed / total_tests * 100) if total_tests > 0 else 0,
            "test_suites": len(self.report_data["test_results"]),
            "all_passed": total_failed == 0
        }

        # 保存JSON报告
        report_path = self.project_root / output_file
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(self.report_data, f, indent=2, ensure_ascii=False, default=str)

        print(f"✅ 测试报告已保存到: {report_path}")

        # 生成Markdown报告
        self._generate_markdown_report()

        return report_path

    def _generate_markdown_report(self):
        """生成Markdown格式的测试报告"""
        md_content = self._create_markdown_content()

        md_path = self.project_root / "test_report.md"
        with open(md_path, 'w', encoding='utf-8') as f:
            f.write(md_content)

        print(f"✅ Markdown报告已保存到: {md_path}")

    def _create_markdown_content(self) -> str:
        """创建Markdown报告内容"""
        summary = self.report_data["summary"]
        complexity = self.report_data.get("complexity", {})

        md = f"""# 多Agent加密货币量化交易系统 - 测试报告

## 📊 测试汇总

- **测试时间**: {self.report_data["metadata"]["generated_at"]}
- **总测试数**: {summary["total_tests"]}
- **通过测试**: {summary["total_passed"]}
- **失败测试**: {summary["total_failed"]}
- **跳过测试**: {summary["total_skipped"]}
- **成功率**: {summary["success_rate"]:.1f}%
- **测试套件**: {summary["test_suites"]}
- **整体状态**: {"✅ 全部通过" if summary["all_passed"] else "❌ 存在失败"}

## 🧪 测试结果详情

"""

        # 测试套件详情
        for suite_name, result in self.report_data["test_results"].items():
            status_icon = "✅" if result["status"] == "passed" else "❌" if result["status"] == "failed" else "⚠️"
            md += f"""
### {status_icon} {suite_name}

- **状态**: {result["status"]}
- **总计**: {result["total"]}
- **通过**: {result["passed"]}
- **失败**: {result["failed"]}
- **跳过**: {result["skipped"]}
- **耗时**: {result["duration"]:.2f}s
- **测试文件**: {len(result.get("test_files", []))}

"""

            if result["errors"]:
                md += "#### 错误详情:\n"
                for error in result["errors"]:
                    md += f"- `{error}`\n"
                md += "\n"

        # 代码质量
        if "code_quality" in self.report_data:
            md += "## 🔍 代码质量检查\n\n"
            quality = self.report_data["code_quality"]

            for check_name, result in quality.items():
                status_icon = "✅" if result["status"] == "passed" else "❌" if result["status"] == "failed" else "⚠️"
                md += f"### {status_icon} {check_name.upper()}\n"
                md += f"- **状态**: {result['status']}\n"

                if check_name == "flake8":
                    md += f"- **问题数**: {result.get('issues', 0)}\n"

                if "output" in result and result["output"]:
                    output_lines = result["output"].split('\n')[:5]
                    for line in output_lines:
                        if line.strip():
                            md += f"- {line.strip()}\n"
                md += "\n"

        # 代码复杂度
        if complexity:
            md += "## 📊 代码复杂度分析\n\n"
            md += f"- **Python文件总数**: {complexity.get('python_files', 0)}\n"
            md += f"- **代码总行数**: {complexity.get('total_lines', 0)}\n"
            md += f"- **测试文件数**: {complexity.get('test_files', 0)}\n"
            md += f"- **服务文件数**: {complexity.get('service_files', 0)}\n"
            md += f"- **模型文件数**: {complexity.get('model_files', 0)}\n"

            if complexity.get("largest_files"):
                md += "\n#### 最大文件 (按行数):\n"
                for file_info in complexity["largest_files"][:5]:
                    md += f"- `{file_info['path']}`: {file_info['lines']} 行\n"
            md += "\n"

        # 系统信息
        md += "## ℹ️ 系统信息\n\n"
        metadata = self.report_data["metadata"]
        md += f"- **Python版本**: {metadata['python_version']}\n"
        md += f"- **平台**: {metadata['platform']}\n"
        md += f"- **项目路径**: {metadata['project_root']}\n"

        md += """
## 📝 结论

"""

        if summary["all_passed"]:
            md += """✅ **测试结果**: 所有测试均通过，系统功能正常

🎉 **系统状态**: 核心功能已完整实现，具备生产部署条件

"""
        else:
            md += f"""❌ **测试结果**: 存在 {summary['total_failed']} 个失败测试

⚠️ **建议**: 需要修复失败的测试后才能部署

"""

        md += "---\n"
        md += f"*报告生成时间: {self.report_data['metadata']['generated_at']}*\n"

        return md

    def print_summary(self):
        """打印测试总结"""
        summary = self.report_data["summary"]

        print("\n" + "="*60)
        print("🧪 测试总结")
        print("="*60)
        print(f"总测试数: {summary['total_tests']}")
        print(f"通过: {summary['total_passed']}")
        print(f"失败: {summary['total_failed']}")
        print(f"跳过: {summary['total_skipped']}")
        print(f"成功率: {summary['success_rate']:.1f}%")
        print("="*60)

        if summary["all_passed"]:
            print("🎉 所有测试通过！系统已准备好部署。")
        else:
            print("⚠️ 存在失败的测试，请检查并修复问题。")

        print("="*60)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="运行测试并生成报告")
    parser.add_argument("--project-root", default=".", help="项目根目录路径")
    parser.add_argument("--output", default="test_report.json", help="测试报告输出文件")
    parser.add_argument("--skip-quality", action="store_true", help="跳过代码质量检查")
    parser.add_argument("--skip-complexity", action="store_true", help="跳过代码复杂度分析")

    args = parser.parse_args()

    print("🚀 多Agent加密货币量化交易系统 - 测试报告生成器")
    print("="*60)

    generator = TestReportGenerator(args.project_root)

    try:
        # 运行测试
        generator.run_tests()

        # 代码质量检查
        if not args.skip_quality:
            generator.check_code_quality()

        # 代码复杂度分析
        if not args.skip_complexity:
            generator.analyze_code_complexity()

        # 生成报告
        generator.generate_report(args.output)

        # 打印总结
        generator.print_summary()

    except KeyboardInterrupt:
        print("\n⚠️ 测试被用户中断")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ 测试过程中发生错误: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()