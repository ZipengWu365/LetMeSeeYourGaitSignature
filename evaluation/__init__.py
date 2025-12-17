# Evaluation module
from .evaluate_sequence_labeling import (
    SequenceLabelingEvaluator,
    create_evaluator,
)

# Import metrics_report from eval.py if it exists
try:
    # This is imported by classifier.py
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from eval import metrics_report
except (ImportError, ModuleNotFoundError):
    pass

__all__ = ['SequenceLabelingEvaluator', 'create_evaluator']
