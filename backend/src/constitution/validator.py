"""
宪法合规性验证器

提供代码和实现的宪法合规性检查功能。
"""

import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from .principles import get_all_principles, ConstitutionPrinciple

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """验证结果"""
    principle: str
    compliant: bool
    issues: List[str]
    suggestions: List[str]
    score: float  # 0.0 - 1.0


@dataclass
class ComplianceReport:
    """合规性报告"""
    overall_score: float
    compliant: bool
    principle_results: List[ValidationResult]
    summary: str
    recommendations: List[str]


class ConstitutionValidator:
    """宪法合规性验证器"""

    def __init__(self):
        self.principles = get_all_principles()
        self.logger = logging.getLogger(f"{__name__}.ConstitutionValidator")

    def validate_file(self, file_path: Path) -> ValidationResult:
        """验证单个文件"""
        results = []
        for principle in self.principles:
            try:
                result = principle.validate(file_path)
                score = 1.0 if result["compliant"] else 0.5
                if result["issues"]:
                    score -= len(result["issues"]) * 0.1
                score = max(0.0, min(1.0, score))

                results.append(ValidationResult(
                    principle=principle.name,
                    compliant=result["compliant"],
                    issues=result["issues"],
                    suggestions=result["suggestions"],
                    score=score
                ))

            except Exception as e:
                self.logger.error(f"验证文件 {file_path} 时出错 {principle.name}: {e}")
                results.append(ValidationResult(
                    principle=principle.name,
                    compliant=False,
                    issues=[f"验证失败: {str(e)}"],
                    suggestions=[],
                    score=0.0
                ))

        # 计算总体得分
        overall_score = sum(r.score for r in results) / len(results) if results else 0.0
        overall_compliant = overall_score >= 0.8

        all_issues = []
        all_suggestions = []
        for result in results:
            all_issues.extend(result.issues)
            all_suggestions.extend(result.suggestions)

        return ValidationResult(
            principle="Overall",
            compliant=overall_compliant,
            issues=all_issues,
            suggestions=all_suggestions,
            score=overall_score
        )

    def validate_directory(self, dir_path: Path, pattern: str = "*.py") -> ComplianceReport:
        """验证目录"""
        self.logger.info(f"开始验证目录: {dir_path}")

        if not dir_path.exists():
            raise FileNotFoundError(f"目录不存在: {dir_path}")

        # 获取所有Python文件
        python_files = list(dir_path.rglob(pattern))
        if not python_files:
            self.logger.warning(f"目录中没有找到Python文件: {dir_path}")
            return ComplianceReport(
                overall_score=1.0,
                compliant=True,
                principle_results=[],
                summary="没有文件需要验证",
                recommendations=[]
            )

        # 验证每个文件
        file_results = []
        principle_scores = {principle.name: [] for principle in self.principles}

        for file_path in python_files:
            try:
                file_result = self.validate_file(file_path)
                file_results.append(file_result)

                # 收集每个原则的得分
                for principle in self.principles:
                    principle_scores[principle.name].append(
                        next((r.score for r in file_results if r.principle == principle), 1.0)
                    )

            except Exception as e:
                self.logger.error(f"验证文件失败 {file_path}: {e}")

        # 计算每个原则的平均得分
        principle_results = []
        for principle in self.principles:
            scores = principle_scores[principle.name]
            avg_score = sum(scores) / len(scores) if scores else 1.0

            # 收集所有问题和建议
            principle_issues = []
            principle_suggestions = []
            for result in file_results:
                if result.issues:
                    principle_issues.extend(result.issues)
                if result.suggestions:
                    principle_suggestions.extend(result.suggestions)

            principle_results.append(ValidationResult(
                principle=principle.name,
                compliant=avg_score >= 0.8,
                issues=principle_issues[:5],  # 限制显示数量
                suggestions=principle_suggestions[:5],
                score=avg_score
            ))

        # 计算总体得分
        overall_score = sum(r.score for r in principle_results) / len(principle_results)
        overall_compliant = overall_score >= 0.8

        # 生成总结和建议
        summary = self._generate_summary(principle_results, overall_score)
        recommendations = self._generate_recommendations(principle_results)

        return ComplianceReport(
            overall_score=overall_score,
            compliant=overall_compliant,
            principle_results=principle_results,
            summary=summary,
            recommendations=recommendations
        )

    def _generate_summary(self, results: List[ValidationResult], score: float) -> str:
        """生成总结"""
        compliant_count = sum(1 for r in results if r.compliant)
        total_count = len(results)

        if score >= 0.9:
            return f"优秀！代码完全符合宪法原则，得分 {score:.2f}"
        elif score >= 0.8:
            return f"良好！代码基本符合宪法原则，得分 {score:.2f}"
        elif score >= 0.6:
            return f"及格，但需要改进。得分 {score:.2f}，{total_count - compliant_count} 个原则需要调整"
        else:
            return f"需要重大改进。得分 {score:.2f}，多个宪法原则未被遵循"

    def _generate_recommendations(self, results: List[ValidationResult]) -> List[str]:
        """生成改进建议"""
        recommendations = []

        # 根据最严重的问题生成建议
        worst_principles = sorted(results, key=lambda x: x.score)[:3]

        for result in worst_principles:
            if result.score < 0.8:
                if result.issues:
                    recommendations.append(
                        f"优先修复 {result.principle}: {result.issues[0]}"
                    )
                if result.suggestions:
                    recommendations.append(
                        f"改进建议 {result.principle}: {result.suggestions[0]}"
                    )

        # 通用建议
        low_score_count = sum(1 for r in results if r.score < 0.6)
        if low_score_count > 2:
            recommendations.append(
                "建议进行全面的代码重构以提高宪法合规性"
            )

        return recommendations[:10]  # 限制建议数量

    def generate_report(self, report: ComplianceReport) -> str:
        """生成格式化的合规性报告"""
        lines = [
            "# 宪法合规性报告",
            "=" * 50,
            f"总体得分: {report.overall_score:.2f}",
            f"合规状态: {'✅ 合规' if report.compliant else '❌ 不合规'}",
            "",
            "## 原则检查结果",
        ]

        for result in report.principle_results:
            status = "✅" if result.compliant else "❌"
            lines.append(f"### {status} {result.principle} (得分: {result.score:.2f})")

            if result.issues:
                lines.append("**问题:**")
                for issue in result.issues[:3]:  # 限制显示数量
                    lines.append(f"- {issue}")

            if result.suggestions:
                lines.append("**建议:**")
                for suggestion in result.suggestions[:3]:
                    lines.append(f"- {suggestion}")

            lines.append("")

        lines.extend([
            "## 总结",
            report.summary,
            "",
            "## 改进建议",
        ])

        for i, rec in enumerate(report.recommendations, 1):
            lines.append(f"{i}. {rec}")

        lines.extend([
            "",
            "---",
            "*报告由宪法合规性验证器生成*"
        ])

        return "\n".join(lines)


# 便捷函数
def validate_implementation(target: Path, pattern: str = "*.py") -> ComplianceReport:
    """验证实现的宪法合规性"""
    validator = ConstitutionValidator()

    if target.is_file():
        # 验证单个文件
        file_result = validator.validate_file(target)
        return ComplianceReport(
            overall_score=file_result.score,
            compliant=file_result.compliant,
            principle_results=[file_result],
            summary=f"文件 {target.name} 的合规性检查完成",
            recommendations=file_result.suggestions
        )
    elif target.is_dir():
        # 验证目录
        return validator.validate_directory(target, pattern)
    else:
        raise ValueError(f"不支持的目标类型: {target}")


def quick_check(file_path: Path) -> bool:
    """快速检查文件是否合规"""
    try:
        report = validate_implementation(file_path)
        return report.compliant
    except Exception:
        return False