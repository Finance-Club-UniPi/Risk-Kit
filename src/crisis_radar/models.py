"""Model definitions module."""

import logging
from typing import Dict

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import CrisisRadarConfig

logger = logging.getLogger(__name__)


def get_models(config: CrisisRadarConfig) -> Dict[str, Pipeline]:
    """
    Get model pipelines based on configuration.
    
    Args:
        config: CrisisRadarConfig instance
    
    Returns:
        Dictionary mapping model names to sklearn Pipeline objects
    """
    models = {}
    
    # Logistic Regression baseline
    logreg = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(
            class_weight="balanced",
            random_state=config.random_seed,
            max_iter=1000,
        )),
    ])
    models["logreg"] = logreg
    
    # Random Forest
    rf = RandomForestClassifier(
        n_estimators=config.n_estimators,
        max_depth=config.max_depth,
        class_weight="balanced",
        random_state=config.random_seed,
        n_jobs=-1,
    )
    models["rf"] = rf
    
    # Gradient Boosting
    gb = GradientBoostingClassifier(
        n_estimators=config.n_estimators,
        max_depth=config.max_depth if config.max_depth else 3,
        random_state=config.random_seed,
    )
    models["gb"] = gb
    
    logger.info(f"Created {len(models)} models: {list(models.keys())}")
    return models

