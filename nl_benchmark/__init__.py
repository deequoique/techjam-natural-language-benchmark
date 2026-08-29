"""Standalone target-exact natural-language benchmark."""

from .catalog import Catalog, Product, match_fact
from .agent_loader import AgentProcessError, SubprocessAgent
from .evaluator import evaluate_dataset, evaluate_sample, normalize_recommendations
from .generator import GeneratorConfig, GenerationError, TargetGenerator, generate_samples
from .question_interpreter import QuestionInterpretation, interpret_question
from .schema import Fact, Sample, project_for_agent
from .simulator import IntelligentSimulator, SimulatorReply
from .validator import ValidationConfig, ValidationResult, validate_dataset, validate_sample

__all__ = [
    "Catalog", "Product", "match_fact", "Fact", "Sample", "project_for_agent",
    "SubprocessAgent", "AgentProcessError",
    "GeneratorConfig", "GenerationError", "TargetGenerator", "generate_samples",
    "QuestionInterpretation", "interpret_question",
    "IntelligentSimulator", "SimulatorReply", "ValidationConfig", "ValidationResult",
    "validate_dataset", "validate_sample", "evaluate_dataset", "evaluate_sample",
    "normalize_recommendations",
]
