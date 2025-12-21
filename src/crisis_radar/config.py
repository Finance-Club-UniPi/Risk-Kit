"""Configuration dataclass for Crisis Radar pipeline."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


@dataclass
class CrisisRadarConfig:
    """Configuration for the Crisis Radar ML pipeline."""
    
    # Data parameters
    start_date: str = "2000-01-01"
    end_date: Optional[str] = None  # None means use today
    
    # Target definition
    horizon_days: int = 20  # N: prediction horizon in trading days
    drawdown_threshold: float = 0.08  # D: -8% drawdown threshold
    
    # Train/test split
    train_start: str = "2005-01-01"
    train_end: str = "2015-12-31"
    test_start: str = "2016-01-01"
    test_end: Optional[str] = None  # None means use end_date
    
    # Model parameters
    model_type: str = "rf"  # "logreg" or "rf" or "gb"
    calibration_method: str = "isotonic"  # "isotonic" or "sigmoid"
    n_estimators: int = 100  # For RF/GB
    max_depth: Optional[int] = None  # For RF/GB
    random_seed: int = 42
    
    # Data paths
    data_dir: str = "data"
    artifacts_dir: str = "artifacts"
    reports_dir: str = "reports"
    
    def __post_init__(self):
        """Validate and set defaults after initialization."""
        if self.end_date is None:
            from datetime import date
            self.end_date = date.today().strftime("%Y-%m-%d")
        
        if self.test_end is None:
            self.test_end = self.end_date
        
        # Validate dates
        train_start_dt = datetime.strptime(self.train_start, "%Y-%m-%d")
        train_end_dt = datetime.strptime(self.train_end, "%Y-%m-%d")
        test_start_dt = datetime.strptime(self.test_start, "%Y-%m-%d")
        test_end_dt = datetime.strptime(self.test_end, "%Y-%m-%d")
        
        if train_end_dt >= test_start_dt:
            raise ValueError("train_end must be before test_start")
        
        if test_start_dt > test_end_dt:
            raise ValueError("test_start must be before test_end")
        
        if self.drawdown_threshold <= 0 or self.drawdown_threshold >= 1:
            raise ValueError("drawdown_threshold must be between 0 and 1")
        
        if self.horizon_days <= 0:
            raise ValueError("horizon_days must be positive")

