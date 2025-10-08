"""
宪法合规性验证模块

确保所有代码和实现都符合项目宪法原则。
"""

from .validator import ConstitutionValidator, validate_implementation
from .principles import (
    SimplicityFirstPrinciple,
    TestFirstPrinciple,
    IntegrationFirstPrinciple,
    ModuleReusabilityPrinciple,
    HighCohesionLowCouplingPrinciple,
    CodeReadabilityPrinciple,
    SystemArchitecturePrinciple
)

__all__ = [
    "ConstitutionValidator",
    "validate_implementation",
    "SimplicityFirstPrinciple",
    "TestFirstPrinciple",
    "IntegrationFirstPrinciple",
    "ModuleReusabilityPrinciple",
    "HighCohesionLowCouplingPrinciple",
    "CodeReadabilityPrinciple",
    "SystemArchitecturePrinciple",
]