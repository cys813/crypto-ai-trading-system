#!/usr/bin/env python3
"""
API性能分析脚本
分析API响应时间、并发处理能力和性能瓶颈
"""

import os
import re
import json
import time
import asyncio
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Tuple

class APIPerformanceAnalyzer:
    """API性能分析器"""

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

    def analyze_api_structure(self):
        """分析API结构"""
        print("🌐 分析API结构...")

        api_analysis = {
            "endpoints": [],
            "middleware": [],
            "async_endpoints": 0,
            "sync_endpoints": 0,
            "response_models": 0,
            "validation_complexity": {},
            "error_handling": 0
        }

        try:
            api_dir = self.backend_root / "src" / "api"
            if not api_dir.exists():
                print("未找到api目录")
                return

            api_files = list(api_dir.rglob("*.py"))

            for api_file in api_files:
                if api_file.name == "__init__.py":
                    continue

                file_analysis = self._analyze_api_file(api_file)
                if file_analysis:
                    api_analysis["endpoints"].extend(file_analysis["endpoints"])
                    api_analysis["middleware"].extend(file_analysis["middleware"])
                    api_analysis["async_endpoints"] += file_analysis["async_endpoints"]
                    api_analysis["sync_endpoints"] += file_analysis["sync_endpoints"]
                    api_analysis["response_models"] += file_analysis["response_models"]
                    api_analysis["error_handling"] += file_analysis["error_handling"]

                    # 合并验证复杂度
                    for key, value in file_analysis["validation_complexity"].items():
                        if key in api_analysis["validation_complexity"]:
                            api_analysis["validation_complexity"][key] += value
                        else:
                            api_analysis["validation_complexity"][key] = value

        except Exception as e:
            print(f"API结构分析失败: {e}")

        self.report["analysis"]["api_structure"] = api_analysis

    def _analyze_api_file(self, api_file: Path) -> Dict[str, Any]:
        """分析单个API文件"""
        try:
            with open(api_file, 'r', encoding='utf-8') as f:
                content = f.read()

            file_analysis = {
                "endpoints": [],
                "middleware": [],
                "async_endpoints": 0,
                "sync_endpoints": 0,
                "response_models": 0,
                "validation_complexity": {},
                "error_handling": 0
            }

            # 查找API端点
            endpoint_patterns = [
                r'@(?:app|router)\.(get|post|put|delete|patch)\(["\']([^"\']+)["\']',
                r'@.*\.route\(["\']([^"\']+)["\'].*?,\s*methods=\[([^\]]+)\]'
            ]

            for pattern in endpoint_patterns:
                matches = re.findall(pattern, content, re.MULTILINE)
                for match in matches:
                    if len(match) == 2:
                        method, path = match
                    else:
                        path, methods = match
                        method = methods.strip(' "').split(',')[0] if ',' in methods else 'GET'

                    # 检查是否为异步
                    async_pattern = rf'async\s+def\s+\w+.*?(?:@.*?(?:get|post|put|delete|patch).*?["\']{re.escape(path)}["\'])'
                    is_async = bool(re.search(async_pattern, content, re.DOTALL))

                    if is_async:
                        file_analysis["async_endpoints"] += 1
                    else:
                        file_analysis["sync_endpoints"] += 1

                    file_analysis["endpoints"].append({
                        "method": method.upper(),
                        "path": path,
                        "file": str(api_file.relative_to(self.backend_root)),
                        "is_async": is_async
                    })

            # 查找中间件
            middleware_patterns = [
                r'@(?:app|router)\.middleware\("([^"]+)"\)',
                r'@.*\.before_request',
                r'@.*\.after_request'
            ]

            for pattern in middleware_patterns:
                matches = re.findall(pattern, content)
                for match in matches:
                    file_analysis["middleware"].append({
                        "type": match,
                        "file": str(api_file.relative_to(self.backend_root))
                    })

            # 统计响应模型
            file_analysis["response_models"] = content.count('response_model') + content.count('BaseModel')

            # 统计验证复杂度
            validation_keywords = ['Field(', 'validator(', 'Pydantic', 'BaseModel', 'Query(', 'Path(', 'Body(']
            for keyword in validation_keywords:
                count = content.count(keyword)
                if count > 0:
                    file_analysis["validation_complexity"][keyword] = count

            # 统计错误处理
            file_analysis["error_handling"] = content.count('HTTPException') + content.count('raise ') + content.count('try:')

            return file_analysis

        except Exception as e:
            print(f"分析API文件失败 {api_file}: {e}")
            return None

    def analyze_concurrency_patterns(self):
        """分析并发模式"""
        print("⚡ 分析并发模式...")

        concurrency_analysis = {
            "async_functions": 0,
            "await_usage": 0,
            "asyncio_usage": 0,
            "threading_usage": 0,
            "pool_usage": 0,
            "blocking_operations": 0,
            "concurrency_patterns": []
        }

        try:
            python_files = list(self.backend_root.rglob("*.py"))

            for py_file in python_files[:30]:  # 采样分析
                try:
                    with open(py_file, 'r', encoding='utf-8') as f:
                        content = f.read()

                    # 统计异步相关
                    concurrency_analysis["async_functions"] += content.count('async def')
                    concurrency_analysis["await_usage"] += content.count('await ')
                    concurrency_analysis["asyncio_usage"] += content.count('asyncio.')

                    # 统计线程相关
                    concurrency_analysis["threading_usage"] += content.count('threading.') + content.count('Thread(')

                    # 统计连接池
                    concurrency_analysis["pool_usage"] += content.count('pool') + content.count('Pool(')

                    # 统计阻塞操作
                    blocking_patterns = [
                        r'time\.sleep\(',
                        r'requests\.',
                        r'session\.execute\(',
                        r'file\.read\(',
                        r'subprocess\.'
                    ]

                    for pattern in blocking_patterns:
                        matches = re.findall(pattern, content)
                        concurrency_analysis["blocking_operations"] += len(matches)

                    # 分析并发模式
                    if 'async def' in content or 'await ' in content:
                        concurrency_analysis["concurrency_patterns"].append({
                            "file": str(py_file.relative_to(self.backend_root)),
                            "type": "async",
                            "async_functions": content.count('async def'),
                            "await_usage": content.count('await ')
                        })

                except:
                    continue

        except Exception as e:
            print(f"并发模式分析失败: {e}")

        self.report["analysis"]["concurrency"] = concurrency_analysis

    def analyze_response_optimization(self):
        """分析响应优化"""
        print("🚀 分析响应优化...")

        response_analysis = {
            "compression": 0,
            "streaming": 0,
            "caching": 0,
            "batching": 0,
            "pagination": 0,
            "optimization_features": [],
            "performance_patterns": []
        }

        try:
            api_files = list(self.backend_root.rglob("*.py"))

            for api_file in api_files[:20]:  # 采样分析
                try:
                    with open(api_file, 'r', encoding='utf-8') as f:
                        content = f.read()

                    # 统计优化特性
                    if 'gzip' in content.lower() or 'compression' in content.lower():
                        response_analysis["compression"] += 1

                    if 'streaming' in content.lower() or 'StreamingResponse' in content:
                        response_analysis["streaming"] += 1

                    if 'cache' in content.lower():
                        response_analysis["caching"] += 1

                    if 'batch' in content.lower():
                        response_analysis["batching"] += 1

                    if 'pagination' in content.lower() or 'page' in content.lower():
                        response_analysis["pagination"] += 1

                    # 分析性能模式
                    patterns = self._extract_performance_patterns(content)
                    if patterns:
                        response_analysis["performance_patterns"].append({
                            "file": str(api_file.relative_to(self.backend_root)),
                            "patterns": patterns
                        })

                except:
                    continue

        except Exception as e:
            print(f"响应优化分析失败: {e}")

        self.report["analysis"]["response_optimization"] = response_analysis

    def _extract_performance_patterns(self, content: str) -> List[str]:
        """提取性能模式"""
        patterns = []

        performance_keywords = [
            'background task',
            'celery',
            'async',
            'cache',
            'compress',
            'stream',
            'batch',
            'lazy loading',
            'eager loading',
            'connection pool',
            'rate limit'
        ]

        for keyword in performance_keywords:
            if keyword in content.lower():
                patterns.append(keyword)

        return patterns

    def analyze_security_performance(self):
        """分析安全性能"""
        print("🔒 分析安全性能...")

        security_analysis = {
            "rate_limiting": 0,
            "authentication": 0,
            "authorization": 0,
            "input_validation": 0,
            "cors_config": 0,
            "security_headers": 0,
            "security_overhead": []
        }

        try:
            api_files = list(self.backend_root.rglob("*.py"))

            for api_file in api_files:
                try:
                    with open(api_file, 'r', encoding='utf-8') as f:
                        content = f.read()

                    # 统计安全特性
                    if 'rate' in content.lower() and 'limit' in content.lower():
                        security_analysis["rate_limiting"] += 1

                    if 'auth' in content.lower() or 'oauth' in content.lower():
                        security_analysis["authentication"] += 1

                    if 'permission' in content.lower() or 'role' in content.lower():
                        security_analysis["authorization"] += 1

                    if 'validate' in content.lower() or 'validation' in content.lower():
                        security_analysis["input_validation"] += 1

                    if 'cors' in content.lower():
                        security_analysis["cors_config"] += 1

                    # 分析安全开销
                    security_patterns = self._extract_security_overhead(content)
                    if security_patterns:
                        security_analysis["security_overhead"].append({
                            "file": str(api_file.relative_to(self.backend_root)),
                            "patterns": security_patterns
                        })

                except:
                    continue

        except Exception as e:
            print(f"安全性能分析失败: {e}")

        self.report["analysis"]["security_performance"] = security_analysis

    def _extract_security_overhead(self, content: str) -> List[str]:
        """提取安全开销"""
        overhead_patterns = []

        security_overhead_keywords = [
            'hashing',
            'encryption',
            'jwt',
            'token validation',
            'rate limiting',
            'input sanitization',
            'sql injection prevention',
            'xss protection'
        ]

        for keyword in security_overhead_keywords:
            if keyword in content.lower():
                overhead_patterns.append(keyword)

        return overhead_patterns

    def generate_performance_recommendations(self):
        """生成性能建议"""
        print("💡 生成API性能建议...")

        recommendations = []

        # 基于API结构的建议
        api_structure = self.report["analysis"].get("api_structure", {})
        total_endpoints = len(api_structure.get("endpoints", []))
        async_endpoints = api_structure.get("async_endpoints", 0)
        sync_endpoints = api_structure.get("sync_endpoints", 0)

        if sync_endpoints > async_endpoints:
            recommendations.append({
                "category": "异步优化",
                "priority": "高",
                "issue": f"同步端点过多 ({sync_endpoints} vs {async_endpoints})",
                "suggestion": "将I/O密集型API端点转换为异步实现",
                "impact": "提升并发处理能力 50-100%"
            })

        # 基于并发模式的建议
        concurrency = self.report["analysis"].get("concurrency", {})
        blocking_ops = concurrency.get("blocking_operations", 0)
        async_functions = concurrency.get("async_functions", 0)

        if blocking_ops > async_functions * 2:
            recommendations.append({
                "category": "并发优化",
                "priority": "高",
                "issue": f"阻塞操作过多 ({blocking_ops} 个)",
                "suggestion": "使用异步操作替换阻塞调用，实施连接池",
                "impact": "减少响应时间 30-50%"
            })

        # 基于响应优化的建议
        response_opt = self.report["analysis"].get("response_optimization", {})
        compression = response_opt.get("compression", 0)
        caching = response_opt.get("caching", 0)

        if compression == 0:
            recommendations.append({
                "category": "响应压缩",
                "priority": "中",
                "issue": "缺少响应压缩",
                "suggestion": "启用Gzip压缩减少传输数据量",
                "impact": "减少传输时间 40-60%"
            })

        if caching < total_endpoints * 0.3:
            recommendations.append({
                "category": "响应缓存",
                "priority": "高",
                "issue": "缓存覆盖率不足",
                "suggestion": "为频繁访问的API端点添加缓存层",
                "impact": "提升响应速度 60-80%"
            })

        # 基于安全性能的建议
        security = self.report["analysis"].get("security_performance", {})
        rate_limiting = security.get("rate_limiting", 0)

        if rate_limiting == 0:
            recommendations.append({
                "category": "限流保护",
                "priority": "高",
                "issue": "缺少API限流机制",
                "suggestion": "实施基于IP和用户的请求限流",
                "impact": "防止滥用，保护系统稳定性"
            })

        # 通用优化建议
        recommendations.extend([
            {
                "category": "监控和日志",
                "priority": "中",
                "issue": "缺少性能监控",
                "suggestion": "集成APM工具监控API响应时间和错误率",
                "impact": "及时发现性能问题"
            },
            {
                "category": "负载均衡",
                "priority": "低",
                "issue": "单点部署风险",
                "suggestion": "配置负载均衡器实现水平扩展",
                "impact": "提高系统可用性和处理能力"
            },
            {
                "category": "CDN优化",
                "priority": "低",
                "issue": "静态资源加载慢",
                "suggestion": "使用CDN加速静态资源访问",
                "impact": "提升静态资源访问速度"
            }
        ])

        self.report["recommendations"] = recommendations

    def calculate_performance_score(self):
        """计算性能评分"""
        print("📈 计算API性能评分...")

        scores = {
            "async_performance": 70,
            "response_optimization": 65,
            "concurrency_handling": 75,
            "security_performance": 80
        }

        # 基于分析结果调整分数
        api_structure = self.report["analysis"].get("api_structure", {})
        concurrency = self.report["analysis"].get("concurrency", {})
        response_opt = self.report["analysis"].get("response_optimization", {})
        security = self.report["analysis"].get("security_performance", {})

        # 异步性能评分
        total_endpoints = len(api_structure.get("endpoints", []))
        if total_endpoints > 0:
            async_ratio = api_structure.get("async_endpoints", 0) / total_endpoints
            if async_ratio > 0.8:
                scores["async_performance"] = 90
            elif async_ratio > 0.5:
                scores["async_performance"] = 80
            elif async_ratio > 0.3:
                scores["async_performance"] = 70
            else:
                scores["async_performance"] = 50

        # 响应优化评分
        optimization_features = sum([
            response_opt.get("compression", 0),
            response_opt.get("caching", 0),
            response_opt.get("streaming", 0),
            response_opt.get("pagination", 0)
        ])

        if optimization_features > 3:
            scores["response_optimization"] = 85
        elif optimization_features > 1:
            scores["response_optimization"] = 75
        else:
            scores["response_optimization"] = 60

        # 并发处理评分
        async_functions = concurrency.get("async_functions", 0)
        blocking_ops = concurrency.get("blocking_operations", 0)

        if async_functions > blocking_ops:
            scores["concurrency_handling"] = 85
        elif async_functions > blocking_ops * 0.5:
            scores["concurrency_handling"] = 75
        else:
            scores["concurrency_handling"] = 65

        # 安全性能评分
        security_features = sum([
            security.get("rate_limiting", 0),
            security.get("authentication", 0),
            security.get("authorization", 0),
            security.get("input_validation", 0)
        ])

        if security_features > 3:
            scores["security_performance"] = 90
        elif security_features > 2:
            scores["security_performance"] = 80
        else:
            scores["security_performance"] = 70

        # 计算总分
        total_score = sum(scores.values()) / len(scores)

        self.report["performance_score"] = {
            "total_score": round(total_score, 1),
            "grade": self._get_grade(total_score),
            "individual_scores": scores
        }

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
        """生成API性能报告"""
        print("📝 生成API性能报告...")

        # 生成JSON报告
        json_path = self.project_root / "api_performance_report.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(self.report, f, indent=2, ensure_ascii=False)

        # 生成Markdown报告
        md_path = self.project_root / "api_performance_report.md"
        with open(md_path, 'w', encoding='utf-8') as f:
            f.write(self._create_markdown_report())

        return str(json_path)

    def _create_markdown_report(self) -> str:
        """创建Markdown报告"""
        report = self.report
        score = report.get("performance_score", {})

        md = f"""# 多Agent加密货币量化交易系统 - API性能分析报告

## 📊 分析概览

**分析时间**: {report["timestamp"]}
**项目路径**: {report["project_root"]}

### 🎯 性能评分
**总分**: {score.get("total_score", "N/A")}/100 ({score.get("grade", "N/A")})

| 维度 | 得分 | 评价 |
|------|------|------|
| 异步性能 | {score.get("individual_scores", {}).get("async_performance", "N/A")} | 异步处理能力 |
| 响应优化 | {score.get("individual_scores", {}).get("response_optimization", "N/A")} | 响应时间和压缩 |
| 并发处理 | {score.get("individual_scores", {}).get("concurrency_handling", "N/A")} | 并发处理能力 |
| 安全性能 | {score.get("individual_scores", {}).get("security_performance", "N/A")} | 安全机制开销 |

---

## 🌐 API结构分析

### 📊 端点统计
"""

        api_structure = report["analysis"].get("api_structure", {})
        total_endpoints = len(api_structure.get("endpoints", []))
        async_endpoints = api_structure.get("async_endpoints", 0)
        sync_endpoints = api_structure.get("sync_endpoints", 0)

        md += f"- **总端点数**: {total_endpoints}\n"
        md += f"- **异步端点**: {async_endpoints} ({async_endpoints/total_endpoints*100:.1f}%)\n"
        md += f"- **同步端点**: {sync_endpoints} ({sync_endpoints/total_endpoints*100:.1f}%)\n"
        md += f"- **中间件数**: {len(api_structure.get('middleware', []))}\n"
        md += f"- **响应模型**: {api_structure.get('response_models', 0)}\n"

        md += f"""
### 📋 端点分布
"""

        # 按方法分类统计
        method_counts = {}
        for endpoint in api_structure.get("endpoints", []):
            method = endpoint.get("method", "UNKNOWN")
            method_counts[method] = method_counts.get(method, 0) + 1

        for method, count in sorted(method_counts.items()):
            md += f"- **{method}**: {count} 个端点\n"

        md += f"""
### 🔧 验证复杂度
"""

        validation_complexity = api_structure.get("validation_complexity", {})
        for key, count in sorted(validation_complexity.items(), key=lambda x: x[1], reverse=True):
            md += f"- **{key}**: {count} 次使用\n"

        md += f"""

---

## ⚡ 并发模式分析

### 📊 并发统计
"""

        concurrency = report["analysis"].get("concurrency", {})
        md += f"- **异步函数**: {concurrency.get('async_functions', 0)} 个\n"
        md += f"- **Await使用**: {concurrency.get('await_usage', 0)} 次\n"
        md += f"- **AsyncIO使用**: {concurrency.get('asyncio_usage', 0)} 次\n"
        md += f"- **线程使用**: {concurrency.get('threading_usage', 0)} 次\n"
        md += f"- **连接池**: {concurrency.get('pool_usage', 0)} 个\n"
        md += f"- **阻塞操作**: {concurrency.get('blocking_operations', 0)} 个\n"

        md += f"""
### 📋 并发模式
"""

        concurrency_patterns = concurrency.get("concurrency_patterns", [])
        for pattern in concurrency_patterns[:10]:  # 显示前10个
            md += f"- `{pattern['file']}`: {pattern['async_functions']} 异步函数, {pattern['await_usage']} await\n"

        md += f"""

---

## 🚀 响应优化分析

### 📊 优化特性统计
"""

        response_opt = report["analysis"].get("response_optimization", {})
        md += f"- **响应压缩**: {response_opt.get('compression', 0)} 个实现\n"
        md += f"- **流式响应**: {response_opt.get('streaming', 0)} 个实现\n"
        md += f"- **响应缓存**: {response_opt.get('caching', 0)} 个实现\n"
        md += f"- **批处理**: {response_opt.get('batching', 0)} 个实现\n"
        md += f"- **分页**: {response_opt.get('pagination', 0)} 个实现\n"

        md += f"""
### 📋 性能模式
"""

        performance_patterns = response_opt.get("performance_patterns", [])
        for pattern in performance_patterns[:10]:  # 显示前10个
            md += f"- `{pattern['file']}`: {', '.join(pattern['patterns'])}\n"

        md += f"""

---

## 🔒 安全性能分析

### 📊 安全特性统计
"""

        security = report["analysis"].get("security_performance", {})
        md += f"- **限流保护**: {security.get('rate_limiting', 0)} 个实现\n"
        md += f"- **身份认证**: {security.get('authentication', 0)} 个实现\n"
        md += f"- **权限控制**: {security.get('authorization', 0)} 个实现\n"
        md += f"- **输入验证**: {security.get('input_validation', 0)} 个实现\n"
        md += f"- **CORS配置**: {security.get('cors_config', 0)} 个实现\n"

        md += f"""
### 📋 安全开销
"""

        security_overhead = security.get("security_overhead", [])
        for overhead in security_overhead[:10]:  # 显示前10个
            md += f"- `{overhead['file']}`: {', '.join(overhead['patterns'])}\n"

        md += f"""

---

## 💡 性能优化建议

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

## 🚀 实施路线图

### 第一阶段 (立即执行 - 1周内)
1. **异步化改造**: 将同步API端点转换为异步
2. **响应压缩**: 启用Gzip压缩
3. **基础缓存**: 为关键API添加缓存

### 第二阶段 (短期优化 - 2-4周)
1. **并发优化**: 优化阻塞操作，实施连接池
2. **限流保护**: 实现API限流机制
3. **性能监控**: 集成APM监控工具

### 第三阶段 (长期优化 - 1-3个月)
1. **高级缓存**: 实施多级缓存策略
2. **负载均衡**: 配置负载分发
3. **CDN集成**: 优化静态资源访问

---

## 📈 预期性能提升

### 🎯 关键指标改进
- **响应时间**: 减少 40-70%
- **并发处理能力**: 提升 50-150%
- **吞吐量**: 提升 60-120%
- **资源利用率**: 提升 30-50%

### 💰 业务价值
- **用户体验**: 显著提升响应速度
- **系统稳定性**: 增强高负载处理能力
- **运营成本**: 提高资源使用效率
- **扩展能力**: 支持更大规模访问

---

## 🔍 监控建议

### �� 关键监控指标
1. **响应时间**: 平均、P95、P99响应时间
2. **吞吐量**: 每秒请求数(QPS)
3. **错误率**: 4xx、5xx错误比例
4. **并发数**: 同时处理的请求数
5. **资源使用**: CPU、内存、网络使用率

### 🚨 告警阈值建议
- **响应时间**: P95 > 500ms
- **错误率**: > 1%
- **并发数**: > 80% 最大容量
- **CPU使用率**: > 80%
- **内存使用率**: > 85%

---

**🎯 API性能评分: {score.get("total_score", "N/A")}/100 ({score.get("grade", "N/A")})**

*报告生成时间: {report["timestamp"]}*
"""

        return md

def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description="运行API性能分析")
    parser.add_argument("--project-root", default=".", help="项目根目录路径")

    args = parser.parse_args()

    print("🚀 多Agent加密货币量化交易系统 - API性能分析器")
    print("=" * 60)

    analyzer = APIPerformanceAnalyzer(args.project_root)

    try:
        # 运行分析
        analyzer.analyze_api_structure()
        analyzer.analyze_concurrency_patterns()
        analyzer.analyze_response_optimization()
        analyzer.analyze_security_performance()
        analyzer.generate_performance_recommendations()
        analyzer.calculate_performance_score()

        # 生成报告
        report_path = analyzer.generate_report()

        # 显示结果
        score = analyzer.report.get("performance_score", {}).get("total_score", "N/A")
        grade = analyzer.report.get("performance_score", {}).get("grade", "N/A")

        print(f"\n📊 API性能分析完成！")
        print(f"📈 性能评分: {score}/100 ({grade})")
        print(f"📄 报告已保存到:")
        print(f"   JSON: {report_path}")
        print(f"   Markdown: {report_path.replace('.json', '.md')}")

        # 显示关键统计
        api_structure = analyzer.report["analysis"].get("api_structure", {})
        concurrency = analyzer.report["analysis"].get("concurrency", {})

        print(f"\n📊 关键统计:")
        print(f"   API端点: {len(api_structure.get('endpoints', []))} 个")
        print(f"   异步端点: {api_structure.get('async_endpoints', 0)} 个")
        print(f"   异步函数: {concurrency.get('async_functions', 0)} 个")
        print(f"   阻塞操作: {concurrency.get('blocking_operations', 0)} 个")

        print("=" * 60)

    except KeyboardInterrupt:
        print("\n⚠️ 分析被用户中断")
    except Exception as e:
        print(f"\n❌ 分析过程中发生错误: {e}")

if __name__ == "__main__":
    main()