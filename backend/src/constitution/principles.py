"""
宪法原则定义

定义所有宪法原则的基础类和具体实现。
"""

import ast
import logging
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


class ConstitutionPrinciple(ABC):
    """宪法原则基础类"""

    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description

    @abstractmethod
    def validate(self, target: Any) -> Dict[str, Any]:
        """
        验证目标是否符合本原则

        Args:
            target: 要验证的目标（文件路径、代码字符串、模块等）

        Returns:
            验证结果字典，包含是否合规和相关详情
        """
        pass

    def __str__(self):
        return f"{self.name}: {self.description}"


class SimplicityFirstPrinciple(ConstitutionPrinciple):
    """简化优先原则"""

    def __init__(self):
        super().__init__(
            "简化优先原则",
            "系统设计必须追求最简实现，拒绝过度抽象"
        )

    def validate(self, target: Any) -> Dict[str, Any]:
        """验证简化原则"""
        result = {
            "principle": self.name,
            "compliant": True,
            "issues": [],
            "suggestions": []
        }

        if isinstance(target, Path) and target.suffix == ".py":
            # 验证Python文件的简化性
            try:
                with open(target, 'r', encoding='utf-8') as f:
                    content = f.read()
                    tree = ast.parse(content)

                # 检查类复杂度
                for node in ast.walk(tree):
                    if isinstance(node, ast.ClassDef):
                        methods = [n for n in node.body if isinstance(n, ast.FunctionDef)]
                        if len(methods) > 10:
                            result["issues"].append(
                                f"类 {node.name} 有 {len(methods)} 个方法，可能过于复杂"
                            )
                            result["compliant"] = False

                    elif isinstance(node, ast.FunctionDef):
                        # 检查函数长度
                        if hasattr(node, 'end_lineno') and node.end_lineno:
                            lines = node.end_lineno - node.lineno
                            if lines > 50:
                                result["issues"].append(
                                    f"函数 {node.name} 有 {lines} 行，建议拆分"
                                )
                                result["suggestions"].append(
                                    f"考虑将 {node.name} 拆分为更小的函数"
                                )

            except Exception as e:
                logger.error(f"解析文件失败 {target}: {e}")
                result["issues"].append(f"无法解析文件: {e}")

        return result


class TestFirstPrinciple(ConstitutionPrinciple):
    """测试先行原则"""

    def __init__(self):
        super().__init__(
            "测试先行原则",
            "所有功能必须采用测试驱动开发(TDD)方式实现"
        )

    def validate(self, target: Any) -> Dict[str, Any]:
        """验证测试先行原则"""
        result = {
            "principle": self.name,
            "compliant": True,
            "issues": [],
            "suggestions": []
        }

        if isinstance(target, Path):
            # 检查是否有对应的测试文件
            if target.suffix == ".py" and not target.name.startswith("test_"):
                # 查找对应的测试文件
                test_dir = target.parent.parent / "tests"
                if test_dir.exists():
                    test_files = list(test_dir.rglob(f"test_{target.stem}.py"))
                    if not test_files:
                        result["issues"].append(
                            f"缺少 {target.name} 的测试文件"
                        )
                        result["compliant"] = False
                        result["suggestions"].append(
                            f"创建 test_{target.stem}.py 测试文件"
                        )

        return result


class IntegrationFirstPrinciple(ConstitutionPrinciple):
    """集成优先原则"""

    def __init__(self):
        super().__init__(
            "集成优先原则",
            "优先考虑系统集成和模块间协作"
        )

    def validate(self, target: Any) -> Dict[str, Any]:
        """验证集成优先原则"""
        result = {
            "principle": self.name,
            "compliant": True,
            "issues": [],
            "suggestions": []
        }

        if isinstance(target, Path) and target.suffix == ".py":
            try:
                with open(target, 'r', encoding='utf-8') as f:
                    content = f.read()

                # 检查是否有API端点定义
                if "router" in content and "FastAPI" in content:
                    if "openapi" not in content.lower():
                        result["suggestions"].append(
                            "添加OpenAPI文档注释以提高API可集成性"
                        )

                # 检查是否有明确的接口定义
                if "class" in content and "def " in content:
                    if "interface" not in content.lower() and "abstract" not in content.lower():
                        result["suggestions"].append(
                            "考虑定义明确的接口以提高模块间集成性"
                        )

            except Exception as e:
                logger.error(f"读取文件失败 {target}: {e}")

        return result


class ModuleReusabilityPrinciple(ConstitutionPrinciple):
    """模块复用原则"""

    def __init__(self):
        super().__init__(
            "模块复用原则",
            "所有模块设计必须考虑复用性"
        )

    def validate(self, target: Any) -> Dict[str, Any]:
        """验证模块复用原则"""
        result = {
            "principle": self.name,
            "compliant": True,
            "issues": [],
            "suggestions": []
        }

        if isinstance(target, Path) and target.suffix == ".py":
            try:
                with open(target, 'r', encoding='utf-8') as f:
                    content = f.read()
                    tree = ast.parse(content)

                # 检查是否有硬编码的值
                hardcoded_values = []
                for node in ast.walk(tree):
                    if isinstance(node, ast.Constant) and isinstance(node.value, str):
                        if len(node.value) > 20:  # 长字符串可能是硬编码的配置
                            hardcoded_values.append(node.value[:50] + "...")

                if hardcoded_values:
                    result["issues"].append(
                        f"发现可能的硬编码值: {len(hardcoded_values)} 个"
                    )
                    result["suggestions"].append(
                        "将硬编码值提取为配置参数"
                    )

                # 检查函数是否过于具体
                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef):
                        if "bitcoin" in node.name.lower() or "ethereum" in node.name.lower():
                            result["suggestions"].append(
                                f"函数 {node.name} 可能过于具体，考虑泛化处理"
                            )

            except Exception as e:
                logger.error(f"解析文件失败 {target}: {e}")

        return result


class HighCohesionLowCouplingPrinciple(ConstitutionPrinciple):
    """高内聚低耦合原则"""

    def __init__(self):
        super().__init__(
            "高内聚低耦合原则",
            "模块必须保持高内聚性，模块间保持低耦合性"
        )

    def validate(self, target: Any) -> Dict[str, Any]:
        """验证高内聚低耦合原则"""
        result = {
            "principle": self.name,
            "compliant": True,
            "issues": [],
            "suggestions": []
        }

        if isinstance(target, Path) and target.suffix == ".py":
            try:
                with open(target, 'r', encoding='utf-8') as f:
                    content = f.read()

                # 检查导入依赖
                import_lines = [line for line in content.split('\n') if line.strip().startswith('import')]
                from_lines = [line for line in content.split('\n') if line.strip().startswith('from')]

                total_imports = len(import_lines) + len(from_lines)
                if total_imports > 15:
                    result["issues"].append(
                        f"模块有 {total_imports} 个导入，可能耦合度过高"
                    )
                    result["compliant"] = False

                # 检查全局变量
                if content.count(' = ') > 20:  # 简单估算赋值语句数量
                    result["suggestions"].append(
                        "考虑减少全局变量，提高模块内聚性"
                    )

            except Exception as e:
                logger.error(f"读取文件失败 {target}: {e}")

        return result


class CodeReadabilityPrinciple(ConstitutionPrinciple):
    """代码可读性原则"""

    def __init__(self):
        super().__init__(
            "代码可读性原则",
            "代码必须像文档一样可读"
        )

    def validate(self, target: Any) -> Dict[str, Any]:
        """验证代码可读性原则"""
        result = {
            "principle": self.name,
            "compliant": True,
            "issues": [],
            "suggestions": []
        }

        if isinstance(target, Path) and target.suffix == ".py":
            try:
                with open(target, 'r', encoding='utf-8') as f:
                    lines = f.readlines()

                # 检查行长度
                long_lines = []
                for i, line in enumerate(lines, 1):
                    if len(line.strip()) > 100:
                        long_lines.append(i)

                if long_lines:
                    result["issues"].append(
                        f"发现 {len(long_lines)} 行超过100字符"
                    )
                    result["suggestions"].append(
                        "将长行拆分为多行以提高可读性"
                    )

                # 检查是否有足够的注释
                code_lines = len([line for line in lines if line.strip() and not line.strip().startswith('#')])
                comment_lines = len([line for line in lines if line.strip().startswith('#')])

                if code_lines > 20 and comment_lines / code_lines < 0.1:  # 注释比例低于10%
                    result["suggestions"].append(
                        "增加注释以提高代码可读性"
                    )

                # 检查函数和类的文档字符串
                tree = ast.parse(''.join(lines))
                documented_items = 0
                total_items = 0

                for node in ast.walk(tree):
                    if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                        total_items += 1
                        if ast.get_docstring(node):
                            documented_items += 1

                if total_items > 0 and documented_items / total_items < 0.8:
                    result["issues"].append(
                        f"只有 {documented_items}/{total_items} 个函数/类有文档字符串"
                    )
                    result["compliant"] = False

            except Exception as e:
                logger.error(f"解析文件失败 {target}: {e}")

        return result


class SystemArchitecturePrinciple(ConstitutionPrinciple):
    """系统架构原则"""

    def __init__(self):
        super().__init__(
            "系统架构原则",
            "系统架构必须保持良好的分层和模块化"
        )

    def validate(self, target: Any) -> Dict[str, Any]:
        """验证系统架构原则"""
        result = {
            "principle": self.name,
            "compliant": True,
            "issues": [],
            "suggestions": []
        }

        if isinstance(target, Path):
            # 检查目录结构是否符合分层架构
            if target.is_dir():
                expected_layers = ["models", "services", "api", "core"]
                actual_dirs = [d.name for d in target.iterdir() if d.is_dir()]

                missing_layers = set(expected_layers) - set(actual_dirs)
                if missing_layers:
                    result["suggestions"].append(
                        f"考虑添加缺失的架构层: {missing_layers}"
                    )

                # 检查是否有循环依赖
                src_path = target / "src"
                if src_path.exists():
                    python_files = list(src_path.rglob("*.py"))
                    if len(python_files) > 10:
                        result["suggestions"].append(
                            "使用工具检查循环依赖，确保架构清晰"
                        )

        return result


# 获取所有原则实例
def get_all_principles() -> List[ConstitutionPrinciple]:
    """获取所有宪法原则实例"""
    return [
        SimplicityFirstPrinciple(),
        TestFirstPrinciple(),
        IntegrationFirstPrinciple(),
        ModuleReusabilityPrinciple(),
        HighCohesionLowCouplingPrinciple(),
        CodeReadabilityPrinciple(),
        SystemArchitecturePrinciple(),
    ]