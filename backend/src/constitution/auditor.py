"""
宪法合规性审计模块

确保所有实施的代码都符合Crypto AI Trading Constitution的要求
"""

import ast
import os
import re
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class ComplianceLevel(str, Enum):
    """合规性级别"""
    COMPLIANT = "compliant"      # 完全合规
    WARNING = "warning"          # 轻微违规
    VIOLATION = "violation"      # 严重违规
    CRITICAL = "critical"        # 致命违规


@dataclass
class ComplianceIssue:
    """合规性问题"""
    principle: str
    file_path: str
    line_number: int
    severity: ComplianceLevel
    message: str
    suggestion: str
    code_snippet: Optional[str] = None


class ConstitutionAuditor:
    """宪法审计器"""

    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.backend_root = project_root / "backend"
        self.issues: List[ComplianceIssue] = []
        self.constitution_path = project_root / ".specify/memory/constitution.md"
        self.constitution_principles = self._load_constitution_principles()

    def _load_constitution_principles(self) -> Dict[str, Dict[str, Any]]:
        """加载宪法原则"""
        principles = {}

        try:
            if self.constitution_path.exists():
                with open(self.constitution_path, 'r', encoding='utf-8') as f:
                    content = f.read()

                # 解析宪法原则
                current_principle = None
                current_section = None

                for line in content.split('\n'):
                    line = line.strip()

                    # 识别原则
                    if line.startswith('### ') and '原则' in line:
                        principle_name = line.replace('### ', '').replace('原则', '').strip()
                        current_principle = principle_name
                        principles[current_principle] = {
                            "name": principle_name,
                            "description": "",
                            "requirements": [],
                            "must_items": [],
                            "should_items": []
                        }

                    # 识别要求
                    elif line.startswith('-') and current_principle:
                        requirement = line[1:].strip()
                        if '必须' in requirement or 'MUST' in requirement:
                            principles[current_principle]["must_items"].append(requirement)
                        elif '应该' in requirement or 'SHOULD' in requirement:
                            principles[current_principle]["should_items"].append(requirement)
                        else:
                            principles[current_principle]["requirements"].append(requirement)

        except Exception as e:
            logger.error(f"加载宪法原则失败: {e}")

        return principles

    async def audit_all_code(self) -> List[ComplianceIssue]:
        """审计所有代码"""
        logger.info("开始宪法合规性审计...")

        # 获取所有Python文件
        python_files = list(self.backend_root.rglob("*.py"))

        for file_path in python_files:
            if '__pycache__' in str(file_path):
                continue

            try:
                await self._audit_file(file_path)
            except Exception as e:
                logger.error(f"审计文件失败 {file_path}: {e}")

        logger.info(f"审计完成，发现 {len(self.issues)} 个合规性问题")
        return self.issues

    async def _audit_file(self, file_path: Path):
        """审计单个文件"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # 解析AST
            tree = ast.parse(content)
            lines = content.split('\n')

            # 审计各项原则
            await self._audit_simplicity_first(file_path, tree, lines)
            await self._audit_test_first(file_path, tree, lines)
            await self._audit_integration_first(file_path, tree, lines)
            await self._audit_module_reusability(file_path, tree, lines)
            await self._audit_high_cohesion_low_coupling(file_path, tree, lines)
            await self._audit_code_readability(file_path, tree, lines)
            await self._audit_system_architecture(file_path, tree, lines)

        except SyntaxError as e:
            self.issues.append(ComplianceIssue(
                principle="代码质量",
                file_path=str(file_path),
                line_number=e.lineno or 0,
                severity=ComplianceLevel.CRITICAL,
                message=f"语法错误: {e}",
                suggestion="修复语法错误"
            ))
        except Exception as e:
            logger.error(f"解析文件失败 {file_path}: {e}")

    async def _audit_simplicity_first(self, file_path: Path, tree: ast.AST, lines: List[str]):
        """审计简化优先原则"""
        # 检查函数复杂度
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                complexity = self._calculate_function_complexity(node)
                if complexity > 10:
                    self.issues.append(ComplianceIssue(
                        principle="简化优先原则",
                        file_path=str(file_path),
                        line_number=node.lineno,
                        severity=ComplianceLevel.WARNING,
                        message=f"函数 {node.name} 复杂度过高 ({complexity})",
                        suggestion="将函数拆分为更小的函数",
                        code_snippet=lines[node.lineno-1] if node.lineno <= len(lines) else None
                    ))

            # 检查类的大小
            elif isinstance(node, ast.ClassDef):
                class_size = len(node.body)
                if class_size > 20:
                    self.issues.append(ComplianceIssue(
                        principle="简化优先原则",
                        file_path=str(file_path),
                        line_number=node.lineno,
                        severity=ComplianceLevel.WARNING,
                        message=f"类 {node.name} 过大 ({class_size} 个方法)",
                        suggestion="考虑将类拆分为多个小类",
                        code_snippet=lines[node.lineno-1] if node.lineno <= len(lines) else None
                    ))

    async def _audit_test_first(self, file_path: Path, tree: ast.AST, lines: List[str]):
        """审计测试先行原则"""
        file_str = str(file_path)

        # 检查测试文件
        if 'test' in file_str.lower():
            # 检查是否有测试类
            has_test_class = any(
                isinstance(node, ast.ClassDef) and 'test' in node.name.lower()
                for node in ast.walk(tree)
            )

            if not has_test_class:
                self.issues.append(ComplianceIssue(
                    principle="测试先行原则",
                    file_path=str(file_path),
                    line_number=1,
                    severity=ComplianceLevel.WARNING,
                    message="测试文件缺少测试类",
                    suggestion="添加测试类和测试方法"
                ))

        # 检查生产代码是否有对应的测试文件
        elif 'src' in file_str:
            test_file = file_str.replace('/src/', '/tests/').replace('.py', '_test.py')
            if not Path(test_file).exists():
                relative_path = str(file_path).replace(str(self.project_root), '')
                self.issues.append(ComplianceIssue(
                    principle="测试先行原则",
                    file_path=str(file_path),
                    line_number=1,
                    severity=ComplianceLevel.WARNING,
                    message=f"缺少测试文件: {relative_path}",
                    suggestion=f"创建测试文件: {test_file}"
                ))

    async def _audit_integration_first(self, file_path: Path, tree: ast.AST, lines: List[str]):
        """审计集成优先原则"""
        # 检查API设计
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                # 检查是否有RESTful API设计
                if any(keyword in node.name.lower() for keyword in ['get_', 'post_', 'put_', 'delete_']):
                    # 检查是否有标准化响应
                    has_response_model = False
                    for child in ast.walk(node):
                        if isinstance(child, ast.Return) and isinstance(child.value, ast.Dict):
                            has_response_model = True
                            break

                    if not has_response_model:
                        self.issues.append(ComplianceIssue(
                            principle="集成优先原则",
                            file_path=str(file_path),
                            line_number=node.lineno,
                            severity=ComplianceLevel.WARNING,
                            message=f"API端点 {node.name} 缺少标准化响应",
                            suggestion="返回标准化的JSON响应格式"
                        ))

            # 检查是否有配置管理
            elif isinstance(node, ast.ImportFrom):
                if 'config' in str(node.module) or 'settings' in str(node.module):
                    # 检查配置是否集中管理
                    config_files = ['config.yaml', '.env', 'settings.py']
                    has_central_config = any(
                        (self.project_root / config_file).exists()
                        for config_file in config_files
                    )

                    if not has_central_config:
                        self.issues.append(ComplianceIssue(
                            principle="集成优先原则",
                            file_path=str(file_path),
                            line_number=node.lineno,
                            severity=ComplianceLevel.WARNING,
                            message="缺少集中化配置管理",
                            suggestion="创建统一的配置文件"
                        ))

    async def _audit_module_reusability(self, file_path: Path, tree: ast.AST, lines: List[str]):
        """审计模块复用原则"""
        # 检查是否有重复代码
        function_signatures = []
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                signature = (node.name, len(node.args.args))
                function_signatures.append(signature)

        # 检查是否有重复的函数签名
        from collections import Counter
        signature_counts = Counter(function_signatures)
        duplicates = [sig for sig, count in signature_counts.items() if count > 1]

        if duplicates:
            for sig, count in duplicates:
                self.issues.append(ComplianceIssue(
                    principle="模块复用原则",
                    file_path=str(file_path),
                    line_number=1,
                    severity=ComplianceLevel.WARNING,
                    message=f"发现重复函数签名: {sig[0]} (出现{count}次)",
                    suggestion="将重复函数提取为公共模块"
                ))

        # 检查是否有抽象接口
        if 'services' in str(file_path):
            has_abstract_base = any(
                isinstance(node, ast.ClassDef) and any(
                    isinstance(base, ast.Name) and base.id in ['ABC', 'Protocol']
                    for base in node.bases
                )
                for node in ast.walk(tree)
            )

            if not has_abstract_base:
                self.issues.append(ComplianceIssue(
                    principle="模块复用原则",
                    file_path=str(file_path),
                    line_number=1,
                    severity=ComplianceLevel.WARNING,
                    message="服务模块缺少抽象接口",
                    suggestion="定义抽象基类或协议以增强复用性"
                ))

    async def _audit_high_cohesion_low_coupling(self, file_path: Path, tree: ast.AST, lines: List[str]):
        """审计高内聚低耦合原则"""
        # 检查导入依赖
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imports.append(node.module)

        # 检查依赖数量
        if len(imports) > 10:
            self.issues.append(ComplianceIssue(
                principle="高内聚低耦合原则",
                file_path=str(file_path),
                line_number=1,
                severity=ComplianceLevel.WARNING,
                message=f"文件依赖过多 ({len(imports)} 个导入)",
                suggestion="减少不必要的依赖，使用依赖注入"
            ))

        # 检查是否有循环依赖
        for import_name in imports:
            if 'src/services' in str(file_path) and 'src/services' in import_name:
                # 可能的循环依赖
                self.issues.append(ComplianceIssue(
                    principle="高内聚低耦合原则",
                    file_path=str(file_path),
                    line_number=1,
                    severity=ComplianceLevel.WARNING,
                    message=f"可能的服务间循环依赖: {import_name}",
                    suggestion="使用事件驱动架构减少直接依赖"
                ))

    async def _audit_code_readability(self, file_path: Path, tree: ast.AST, lines: List[str]):
        """审计代码可读性原则"""
        # 检查文档字符串
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                if not ast.get_docstring(node):
                    self.issues.append(ComplianceIssue(
                        principle="代码可读性原则",
                        file_path=str(file_path),
                        line_number=node.lineno,
                        severity=ComplianceLevel.WARNING,
                        message=f"{node.__class__.__name__} {node.name} 缺少文档字符串",
                        suggestion="添加清晰的文档字符串"
                    ))

        # 检查命名规范
        for i, line in enumerate(lines, 1):
            line = line.strip()

            # 检查函数命名
            if re.match(r'^def [a-z]', line):
                continue  # 好的命名
            elif re.match(r'^def [A-Z]', line):
                self.issues.append(ComplianceIssue(
                    principle="代码可读性原则",
                    file_path=str(file_path),
                    line_number=i,
                    severity=ComplianceLevel.WARNING,
                    message="函数名应使用小写字母和下划线",
                    suggestion=f"修改函数命名: {line}",
                    code_snippet=line
                ))

            # 检查变量命名
            if '=' in line and not line.startswith('#'):
                var_name = line.split('=')[0].strip()
                if var_name and not re.match(r'^[a-z_]', var_name):
                    # 检查是否是常量（全大写）
                    if not var_name.isupper():
                        self.issues.append(ComplianceIssue(
                            principle="代码可读性原则",
                            file_path=str(file_path),
                            line_number=i,
                            severity=ComplianceLevel.WARNING,
                            message="变量名应使用小写字母和下划线",
                            suggestion=f"修改变量命名: {var_name}",
                            code_snippet=line
                        ))

    async def _audit_system_architecture(self, file_path: Path, tree: ast.AST, lines: List[str]):
        """审计系统架构原则"""
        file_str = str(file_path)

        # 检查分层架构
        layer_indicators = {
            'api': ['FastAPI', 'router', 'endpoint', 'HTTPException'],
            'services': ['Service', 'Manager', 'Handler', 'Processor'],
            'models': ['BaseModel', 'Column', 'Table', 'relationship'],
            'core': ['config', 'cache', 'database', 'exception']
        }

        for layer, indicators in layer_indicators.items():
            if layer in file_str:
                # 检查是否遵循分层原则
                content = '\n'.join(lines)
                violations = []

                for other_layer, other_indicators in layer_indicators.items():
                    if other_layer != layer:
                        for indicator in other_indicators:
                            if indicator in content:
                                violations.append(other_layer)

                if len(violations) > 2:  # 超过2个其他层的内容
                    self.issues.append(ComplianceIssue(
                        principle="系统架构原则",
                        file_path=str(file_path),
                        line_number=1,
                        severity=ComplianceLevel.WARNING,
                        message=f"{layer.title()}层包含其他层的内容: {', '.join(violations)}",
                        suggestion="将代码移动到相应的分层中"
                    ))

    def _calculate_function_complexity(self, node: ast.FunctionDef) -> int:
        """计算函数复杂度"""
        complexity = 1  # 基础复杂度

        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.For, ast.While, ast.Try)):
                complexity += 1
            elif isinstance(child, ast.BoolOp):
                complexity += len(child.values) - 1

        return complexity

    def generate_report(self) -> str:
        """生成审计报告"""
        severity_counts = {
            ComplianceLevel.COMPLIANT: 0,
            ComplianceLevel.WARNING: 0,
            ComplianceLevel.VIOLATION: 0,
            ComplianceLevel.CRITICAL: 0
        }

        for issue in self.issues:
            severity_counts[issue.severity] += 1

        report = f"""
# 宪法合规性审计报告

## 审计概览
- **审计文件数**: {len([i for i in self.issues if i.severity != ComplianceLevel.COMPLIANT])}
- **发现问题总数**: {len(self.issues)}

### 问题严重程度分布
- ✅ 合规: {severity_counts[ComplianceLevel.COMPLIANT]} 个
- ⚠️ 警告: {severity_counts[ComplianceLevel.WARNING]} 个
- ❌ 违规: {severity_counts[ComplianceLevel.VIOLATION]} 个
- 🚨 致命: {severity_counts[ComplianceLevel.CRITICAL]} 个

## 宪法原则合规情况

"""

        # 按原则分组显示问题
        principle_issues = {}
        for issue in self.issues:
            if issue.principle not in principle_issues:
                principle_issues[issue.principle] = []
            principle_issues[issue.principle].append(issue)

        for principle, issues in principle_issues.items():
            report += f"### {principle}\n\n"

            for issue in issues:
                severity_icon = {
                    ComplianceLevel.WARNING: "⚠️",
                    ComplianceLevel.VIOLATION: "❌",
                    ComplianceLevel.CRITICAL: "🚨"
                }.get(issue.severity, "✅")

                report += f"{severity_icon} **{Path(issue.file_path).name}:{issue.line_number}**\n"
                report += f"- **问题**: {issue.message}\n"
                report += f"- **建议**: {issue.suggestion}\n"

                if issue.code_snippet:
                    report += f"- **代码**: `{issue.code_snippet}`\n"

                report += "\n"

        # 添加合规性评分
        total_issues = len(self.issues)
        if total_issues == 0:
            compliance_score = 100
            grade = "A+"
        else:
            weighted_score = (
                severity_counts[ComplianceLevel.WARNING] * 1 +
                severity_counts[ComplianceLevel.VIOLATION] * 5 +
                severity_counts[ComplianceLevel.CRITICAL] * 10
            )
            max_score = total_issues * 10
            compliance_score = max(0, 100 - (weighted_score / max_score * 100))

            if compliance_score >= 90:
                grade = "A"
            elif compliance_score >= 80:
                grade = "B"
            elif compliance_score >= 70:
                grade = "C"
            else:
                grade = "D"

        report += f"""
## 合规性评分

**总分**: {compliance_score:.1f}/100 ({grade})

### 评分标准
- A+ (90-100): 完全合规
- A (80-89): 良好合规
- B (70-79): 基本合规
- C (60-69): 需要改进
- D (0-59): 严重违规

## 改进建议

### 高优先级
1. 修复所有🚨级别的致命问题
2. 解决❌级别的违规问题
3. 处理⚠️级别的警告

### 中优先级
1. 完善文档字符串和注释
2. 优化函数和类的设计
3. 减少模块间的耦合

### 长期改进
1. 建立自动化合规检查
2. 定期进行代码审查
3. 培养团队合规意识

"""

        return report


async def run_constitution_audit(project_root: str = None):
    """运行宪法合规性审计"""
    if project_root is None:
        project_root = Path.cwd()
    else:
        project_root = Path(project_root)

    auditor = ConstitutionAuditor(project_root)
    issues = await auditor.audit_all_code()
    report = auditor.generate_report()

    # 保存报告
    report_path = project_root / "docs" / "reports" / "constitution_audit_report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)

    print(report)
    print(f"\n📄 详细报告已保存到: {report_path}")

    return issues


if __name__ == "__main__":
    asyncio.run(run_constitution_audit())