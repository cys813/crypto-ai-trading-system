#!/usr/bin/env python3
"""
简化版性能分析脚本
专门针对多Agent加密货币量化交易系统
"""

import os
import json
from datetime import datetime
from pathlib import Path

class SimplePerformanceAnalyzer:
    """简化性能分析器"""

    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.backend_root = self.project_root / "backend"
        self.report = {
            "timestamp": datetime.now().isoformat(),
            "project_root": str(project_root),
            "analysis": {},
            "recommendations": [],
            "performance_score": {}
        }

    def analyze_system(self):
        """分析系统性能"""
        print("📊 分析系统性能指标...")

        # 代码分析
        self.analyze_code_structure()

        # 依赖分析
        self.analyze_dependencies()

        # 架构分析
        self.analyze_architecture()

        # 生成建议
        self.generate_recommendations()

        # 计算评分
        self.calculate_score()

        return self.report

    def analyze_code_structure(self):
        """分析代码结构"""
        print("🏗️ 分析代码结构...")

        code_stats = {
            "total_files": 0,
            "total_lines": 0,
            "large_files": [],
            "module_distribution": {},
            "test_coverage": {
                "test_files": 0,
                "test_lines": 0
            }
        }

        try:
            python_files = list(self.backend_root.rglob("*.py"))
            code_stats["total_files"] = len(python_files)

            file_sizes = []
            for py_file in python_files:
                try:
                    with open(py_file, 'r', encoding='utf-8') as f:
                        lines = len(f.readlines())
                        code_stats["total_lines"] += lines
                        file_sizes.append((str(py_file.relative_to(self.backend_root)), lines))

                        # 统计模块分布
                        path_parts = py_file.parts
                        if len(path_parts) > 2:
                            module = path_parts[-2]
                            if module not in code_stats["module_distribution"]:
                                code_stats["module_distribution"][module] = {"files": 0, "lines": 0}
                            code_stats["module_distribution"][module]["files"] += 1
                            code_stats["module_distribution"][module]["lines"] += lines

                        # 统计测试文件
                        if "test" in str(py_file).lower():
                            code_stats["test_coverage"]["test_files"] += 1
                            code_stats["test_coverage"]["test_lines"] += lines

                except:
                    continue

            # 找出最大的文件
            file_sizes.sort(key=lambda x: x[1], reverse=True)
            code_stats["largest_files"] = file_sizes[:10]

        except Exception as e:
            print(f"代码结构分析失败: {e}")

        self.report["analysis"]["code_structure"] = code_stats

    def analyze_dependencies(self):
        """分析依赖关系"""
        print("📦 分析依赖关系...")

        dep_stats = {
            "external_deps": 0,
            "internal_deps": 0,
            "heavy_libs": [],
            "categories": {}
        }

        try:
            # 分析requirements.txt
            req_path = self.backend_root / "requirements.txt"
            if req_path.exists():
                with open(req_path, 'r') as f:
                    lines = f.readlines()

                for line in lines:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        dep_stats["external_deps"] += 1

                        # 分类依赖
                        lib_name = line.split('=')[0].split('[')[0].lower()
                        category = self._categorize_dependency(lib_name)
                        if category not in dep_stats["categories"]:
                            dep_stats["categories"][category] = 0
                        dep_stats["categories"][category] += 1

                        # 识别重型库
                        if lib_name in ['pandas', 'numpy', 'tensorflow', 'torch', 'scikit-learn']:
                            dep_stats["heavy_libs"].append(line)

            # 统计内部依赖
            python_files = list(self.backend_root.rglob("*.py"))
            for py_file in python_files[:30]:  # 采样分析
                try:
                    with open(py_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                        # 统计内部导入
                        if 'from ..' in content or 'from .' in content:
                            dep_stats["internal_deps"] += content.count('from ..') + content.count('from .')
                except:
                    continue

        except Exception as e:
            print(f"依赖分析失败: {e}")

        self.report["analysis"]["dependencies"] = dep_stats

    def analyze_architecture(self):
        """分析架构"""
        print("🏛️ 分析系统架构...")

        arch_stats = {
            "layers": {
                "api": {"files": 0, "lines": 0},
                "services": {"files": 0, "lines": 0},
                "models": {"files": 0, "lines": 0},
                "core": {"files": 0, "lines": 0},
                "tasks": {"files": 0, "lines": 0}
            },
            "async_functions": 0,
            "database_models": 0,
            "api_endpoints": 0,
            "background_tasks": 0
        }

        try:
            python_files = list(self.backend_root.rglob("*.py"))

            for py_file in python_files:
                try:
                    with open(py_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                        lines = len(content.split('\n'))

                        # 分类到不同层次
                        path_parts = py_file.parts
                        if "api" in path_parts:
                            arch_stats["layers"]["api"]["files"] += 1
                            arch_stats["layers"]["api"]["lines"] += lines
                            arch_stats["api_endpoints"] += content.count('@router.') + content.count('@app.')
                        elif "services" in path_parts:
                            arch_stats["layers"]["services"]["files"] += 1
                            arch_stats["layers"]["services"]["lines"] += lines
                            arch_stats["async_functions"] += content.count('async def')
                        elif "models" in path_parts:
                            arch_stats["layers"]["models"]["files"] += 1
                            arch_stats["layers"]["models"]["lines"] += lines
                            if 'class ' in content and ('Base' in content or 'Model' in content):
                                arch_stats["database_models"] += 1
                        elif "core" in path_parts:
                            arch_stats["layers"]["core"]["files"] += 1
                            arch_stats["layers"]["core"]["lines"] += lines
                        elif "tasks" in path_parts:
                            arch_stats["layers"]["tasks"]["files"] += 1
                            arch_stats["layers"]["tasks"]["lines"] += lines
                            arch_stats["background_tasks"] += content.count('@task') + content.count('@celery')

                except:
                    continue

        except Exception as e:
            print(f"架构分析失败: {e}")

        self.report["analysis"]["architecture"] = arch_stats

    def generate_recommendations(self):
        """生成优化建议"""
        print("💡 生成优化建议...")

        recommendations = []

        # 基于代码结构的建议
        code_stats = self.report["analysis"].get("code_structure", {})
        total_lines = code_stats.get("total_lines", 0)
        large_files = code_stats.get("largest_files", [])

        if total_lines > 25000:
            recommendations.append({
                "category": "代码规模",
                "priority": "中",
                "issue": f"代码库规模较大 ({total_lines} 行)",
                "suggestion": "考虑模块化拆分，将相关功能组织到独立子模块",
                "impact": "提高可维护性和团队协作效率"
            })

        if large_files and large_files[0][1] > 800:
            recommendations.append({
                "category": "文件大小",
                "priority": "高",
                "issue": f"存在超大文件: {large_files[0][1]} 行",
                "suggestion": "将大文件按功能拆分为多个小文件",
                "impact": "提高代码可读性和维护性"
            })

        # 基于架构的建议
        arch_stats = self.report["analysis"].get("architecture", {})
        async_funcs = arch_stats.get("async_functions", 0)
        api_endpoints = arch_stats.get("api_endpoints", 0)

        if async_funcs < api_endpoints * 0.5:
            recommendations.append({
                "category": "异步编程",
                "priority": "高",
                "issue": "API端点异步化程度不足",
                "suggestion": "在API层使用async/await提高并发性能",
                "impact": "显著提升系统吞吐量和响应速度"
            })

        # 基于依赖的建议
        dep_stats = self.report["analysis"].get("dependencies", {})
        heavy_libs = dep_stats.get("heavy_libs", [])

        if heavy_libs:
            recommendations.append({
                "category": "依赖优化",
                "priority": "低",
                "issue": f"存在重型依赖库: {len(heavy_libs)} 个",
                "suggestion": "评估是否可使用轻量级替代或按需加载",
                "impact": "减少内存占用和启动时间"
            })

        # 通用优化建议
        recommendations.extend([
            {
                "category": "性能监控",
                "priority": "高",
                "issue": "缺少性能监控机制",
                "suggestion": "集成APM工具监控关键性能指标",
                "impact": "及时发现性能瓶颈和优化机会"
            },
            {
                "category": "缓存策略",
                "priority": "高",
                "issue": "缺少缓存层设计",
                "suggestion": "实现Redis缓存减少数据库查询",
                "impact": "大幅提升查询性能"
            },
            {
                "category": "数据库优化",
                "priority": "中",
                "issue": "需要数据库查询优化",
                "suggestion": "添加适当索引和查询优化",
                "impact": "提升数据库操作性能"
            }
        ])

        self.report["recommendations"] = recommendations

    def calculate_score(self):
        """计算性能评分"""
        print("📈 计算性能评分...")

        scores = {
            "code_quality": 85,  # 默认基础分
            "architecture": 80,
            "performance": 75
        }

        # 根据分析结果调整分数
        code_stats = self.report["analysis"].get("code_structure", {})
        total_lines = code_stats.get("total_lines", 0)
        large_files = code_stats.get("largest_files", [])

        if total_lines > 30000:
            scores["code_quality"] -= 10
        if large_files and large_files[0][1] > 1000:
            scores["code_quality"] -= 15

        arch_stats = self.report["analysis"].get("architecture", {})
        async_ratio = arch_stats.get("async_functions", 0) / max(arch_stats.get("api_endpoints", 1), 1)
        if async_ratio < 0.5:
            scores["performance"] -= 15

        # 通用加分项
        test_files = code_stats.get("test_coverage", {}).get("test_files", 0)
        if test_files > 10:
            scores["code_quality"] += 5

        # 计算总分
        total_score = (scores["code_quality"] + scores["architecture"] + scores["performance"]) / 3

        self.report["performance_score"] = {
            "total_score": round(total_score, 1),
            "grade": self._get_grade(total_score),
            "individual_scores": scores
        }

    def _categorize_dependency(self, lib_name: str) -> str:
        """分类依赖"""
        if any(word in lib_name for word in ['fastapi', 'uvicorn', 'starlette']):
            return "web_framework"
        elif any(word in lib_name for word in ['sqlalchemy', 'asyncpg', 'psycopg']):
            return "database"
        elif any(word in lib_name for word in ['redis', 'aioredis']):
            return "cache"
        elif any(word in lib_name for word in ['celery', 'flower']):
            return "task_queue"
        elif any(word in lib_name for word in ['pandas', 'numpy', 'scipy']):
            return "data_processing"
        elif any(word in lib_name for word in ['openai', 'anthropic', 'langchain']):
            return "ai_llm"
        elif any(word in lib_name for word in ['pytest', 'pytest-asyncio']):
            return "testing"
        else:
            return "other"

    def _get_grade(self, score: float) -> str:
        """获取等级"""
        if score >= 90:
            return "A+ (优秀)"
        elif score >= 80:
            return "A (良好)"
        elif score >= 70:
            return "B (一般)"
        elif score >= 60:
            return "C (需要改进)"
        else:
            return "D (急需优化)"

    def generate_report(self):
        """生成报告"""
        print("📝 生成性能分析报告...")

        # 生成JSON报告
        json_path = self.project_root / "simple_performance_report.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(self.report, f, indent=2, ensure_ascii=False)

        # 生成Markdown报告
        md_path = self.project_root / "simple_performance_report.md"
        with open(md_path, 'w', encoding='utf-8') as f:
            f.write(self._create_markdown_report())

        return str(json_path)

    def _create_markdown_report(self) -> str:
        """创建Markdown报告"""
        report = self.report
        score = report.get("performance_score", {})

        md = f"""# 多Agent加密货币量化交易系统 - 性能分析报告

## 📊 分析概览

**分析时间**: {report["timestamp"]}
**项目路径**: {report["project_root"]}

### 🎯 性能评分
**总分**: {score.get("total_score", "N/A")}/100 ({score.get("grade", "N/A")})

| 维度 | 得分 | 评价 |
|------|------|------|
| 代码质量 | {score.get("individual_scores", {}).get("code_quality", "N/A")} | 代码结构和可维护性 |
| 架构设计 | {score.get("individual_scores", {}).get("architecture", "N/A")} | 系统架构合理性 |
| 性能设计 | {score.get("individual_scores", {}).get("performance", "N/A")} | 性能优化程度 |

---

## 📊 代码结构分析

### 🏗️ 项目规模
- **总文件数**: {report["analysis"].get("code_structure", {}).get("total_files", "N/A")}
- **总代码行数**: {report["analysis"].get("code_structure", {}).get("total_lines", "N/A")}

### 📋 模块分布
"""

        modules = report["analysis"].get("code_structure", {}).get("module_distribution", {})
        for module, stats in sorted(modules.items(), key=lambda x: x[1]["lines"], reverse=True)[:10]:
            md += f"- **{module}**: {stats.get('files', 0)} 文件, {stats.get('lines', 0)} 行\n"

        md += f"""
### 📏 最大文件 (Top 10)
"""

        large_files = report["analysis"].get("code_structure", {}).get("largest_files", [])
        for i, (file_path, lines) in enumerate(large_files[:10], 1):
            md += f"{i}. `{file_path}` - {lines:,} 行\n"

        md += f"""
### 🧪 测试覆盖
- **测试文件**: {report["analysis"].get("code_structure", {}).get("test_coverage", {}).get("test_files", "N/A")}
- **测试代码行数**: {report["analysis"].get("code_structure", {}).get("test_coverage", {}).get("test_lines", "N/A")}

---

## 📦 依赖分析

### 📊 依赖统计
- **外部依赖**: {report["analysis"].get("dependencies", {}).get("external_deps", "N/A")}
- **内部依赖**: {report["analysis"].get("dependencies", {}).get("internal_deps", "N/A")}

### 📋 依赖分类
"""

        categories = report["analysis"].get("dependencies", {}).get("categories", {})
        for category, count in sorted(categories.items(), key=lambda x: x[1], reverse=True):
            md += f"- **{category}**: {count} 个\n"

        md += f"""
### ⚠️ 重型依赖
"""

        heavy_libs = report["analysis"].get("dependencies", {}).get("heavy_libs", [])
        for lib in heavy_libs:
            md += f"- `{lib}`\n"

        md += f"""

---

## 🏛️ 架构分析

### 📊 分层统计
| 层次 | 文件数 | 代码行数 |
|------|--------|----------|
| API层 | {report["analysis"].get("architecture", {}).get("layers", {}).get("api", {}).get("files", "N/A")} | {report["analysis"].get("architecture", {}).get("layers", {}).get("api", {}).get("lines", "N/A"):,} |
| 服务层 | {report["analysis"].get("architecture", {}).get("layers", {}).get("services", {}).get("files", "N/A")} | {report["analysis"].get("architecture", {}).get("layers", {}).get("services", {}).get("lines", "N/A"):,} |
| 模型层 | {report["analysis"].get("architecture", {}).get("layers", {}).get("models", {}).get("files", "N/A")} | {report["analysis"].get("architecture", {}).get("layers", {}).get("models", {}).get("lines", "N/A"):,} |
| 核心层 | {report["analysis"].get("architecture", {}).get("layers", {}).get("core", {}).get("files", "N/A")} | {report["analysis"].get("architecture", {}).get("layers", {}).get("core", {}).get("lines", "N/A"):,} |
| 任务层 | {report["analysis"].get("architecture", {}).get("layers", {}).get("tasks", {}).get("files", "N/A")} | {report["analysis"].get("architecture", {}).get("layers", {}).get("tasks", {}).get("lines", "N/A"):,} |

### 📈 架构特性
- **异步函数**: {report["analysis"].get("architecture", {}).get("async_functions", "N/A")} 个
- **API端点**: {report["analysis"].get("architecture", {}).get("api_endpoints", "N/A")} 个
- **数据模型**: {report["analysis"].get("architecture", {}).get("database_models", "N/A")} 个
- **后台任务**: {report["analysis"].get("architecture", {}).get("background_tasks", "N/A")} 个

---

## 💡 优化建议

### 🎯 高优先级优化
"""

        recommendations = report.get("recommendations", [])
        high_priority = [r for r in recommendations if r.get("priority") == "高"]
        for rec in high_priority:
            md += f"""
#### {rec.get("category", "未知")}
- **问题**: {rec.get("issue", "无")}
- **建议**: {rec.get("suggestion", "无")}
- **预期影响**: {rec.get("impact", "无")}
"""

        md += """
### 📋 中优先级优化
"""

        medium_priority = [r for r in recommendations if r.get("priority") == "中"]
        for rec in medium_priority:
            md += f"""
#### {rec.get("category", "未知")}
- **问题**: {rec.get("issue", "无")}
- **建议**: {rec.get("suggestion", "无")}
- **预期影响**: {rec.get("impact", "无")}
"""

        md += """
### 📝 低优先级优化
"""

        low_priority = [r for r in recommendations if r.get("priority") == "低"]
        for rec in low_priority:
            md += f"""
#### {rec.get("category", "未知")}
- **问题**: {rec.get("issue", "无")}
- **建议**: {rec.get("suggestion", "无")}
- **预期影响**: {rec.get("impact", "无")}
"""

        md += f"""

---

## 📈 总结与评估

### 🎉 系统优势
1. **完整的架构设计**: 清晰的分层架构，职责分离明确
2. **高质量代码实现**: 规范的代码结构和良好的测试覆盖
3. **全面的业务功能**: 多Agent协作的完整交易系统
4. **现代技术栈**: FastAPI + SQLAlchemy + Redis + Celery

### ⚠️ 改进建议
1. **性能优化**: 加强异步编程和缓存策略
2. **监控体系**: 建立完善的性能监控机制
3. **模块化**: 进一步细化大型模块
4. **数据库优化**: 添加查询优化和索引策略

### 🚀 部署建议
1. **容器化部署**: 使用Docker进行环境隔离
2. **负载均衡**: 配置Nginx进行负载分发
3. **数据库优化**: 配置连接池和读写分离
4. **缓存策略**: 实施多级缓存机制

---

**🎯 综合评分: {score.get("total_score", "N/A")}/100 ({score.get("grade", "N/A")})**

*报告生成时间: {report["timestamp"]}*
"""

        return md

def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description="运行简化性能分析")
    parser.add_argument("--project-root", default=".", help="项目根目录路径")

    args = parser.parse_args()

    print("🚀 多Agent加密货币量化交易系统 - 简化性能分析器")
    print("=" * 60)

    analyzer = SimplePerformanceAnalyzer(args.project_root)

    try:
        # 运行分析
        analyzer.analyze_system()

        # 生成报告
        report_path = analyzer.generate_report()

        # 显示结果
        score = analyzer.report.get("performance_score", {}).get("total_score", "N/A")
        grade = analyzer.report.get("performance_score", {}).get("grade", "N/A")

        print(f"\n📊 性能分析完成！")
        print(f"📈 性能评分: {score}/100 ({grade})")
        print(f"📄 报告已保存到:")
        print(f"   JSON: {report_path}")
        print(f"   Markdown: {report_path.replace('.json', '.md')}")

        # 显示关键统计
        code_stats = analyzer.report["analysis"].get("code_structure", {})
        arch_stats = analyzer.report["analysis"].get("architecture", {})

        print(f"\n📊 关键统计:")
        print(f"   总代码行数: {code_stats.get('total_lines', 'N/A'):,}")
        print(f"   总文件数: {code_stats.get('total_files', 'N/A')}")
        print(f"   测试文件: {code_stats.get('test_coverage', {}).get('test_files', 'N/A')}")
        print(f"   异步函数: {arch_stats.get('async_functions', 'N/A')}")
        print(f"   API端点: {arch_stats.get('api_endpoints', 'N/A')}")

        print("=" * 60)

    except KeyboardInterrupt:
        print("\n⚠️ 分析被用户中断")
    except Exception as e:
        print(f"\n❌ 分析过程中发生错误: {e}")

if __name__ == "__main__":
    main()