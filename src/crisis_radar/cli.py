"""Command-line interface for Crisis Radar."""

import argparse
import logging
import sys
from datetime import date
from pathlib import Path

from .config import CrisisRadarConfig
from .live import get_live_risk_score
from .pipeline import run_pipeline
from .utils import setup_logging

logger = logging.getLogger(__name__)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Crisis Radar: ML pipeline for predicting S&P 500 drawdown events"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Run command
    run_parser = subparsers.add_parser("run", help="Run the complete pipeline")
    run_parser.add_argument("--horizon", type=int, default=20, help="Prediction horizon in trading days (default: 20)")
    run_parser.add_argument("--dd", type=float, default=0.08, help="Drawdown threshold as fraction (default: 0.08 for -8%%)")
    run_parser.add_argument("--train-start", type=str, default="2005-01-01", help="Training start date (YYYY-MM-DD)")
    run_parser.add_argument("--train-end", type=str, default="2015-12-31", help="Training end date (YYYY-MM-DD)")
    run_parser.add_argument("--test-start", type=str, default="2016-01-01", help="Test start date (YYYY-MM-DD)")
    run_parser.add_argument("--test-end", type=str, default=None, help="Test end date (YYYY-MM-DD, default: today)")
    run_parser.add_argument("--model", type=str, choices=["logreg", "rf", "gb"], default="rf", help="Model type (default: rf)")
    run_parser.add_argument("--calibration", type=str, choices=["isotonic", "sigmoid"], default="isotonic", help="Calibration method (default: isotonic)")
    run_parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    run_parser.add_argument("--start-date", type=str, default="2000-01-01", help="Data start date (YYYY-MM-DD)")
    run_parser.add_argument("--end-date", type=str, default=None, help="Data end date (YYYY-MM-DD, default: today)")
    
    # Live command
    live_parser = subparsers.add_parser("live", help="Get live risk score using latest market data")
    live_parser.add_argument("--model", type=str, default=None, help="Path to model file (default: artifacts/models/calibrated_model.pkl)")
    live_parser.add_argument("--config", type=str, default=None, help="Path to config file (default: artifacts/configs/pipeline_config.json)")
    live_parser.add_argument("--refresh", action="store_true", help="Force fresh data download (ignore cache)")
    
    # Report command (for future use)
    report_parser = subparsers.add_parser("report", help="Generate report from saved predictions")
    report_parser.add_argument("--predictions", type=str, required=True, help="Path to predictions CSV")
    
    return parser.parse_args()


def main():
    """Main CLI entry point."""
    args = parse_args()
    
    if args.command == "run":
        setup_logging()
        
        # Create config from args
        config = CrisisRadarConfig(
            start_date=args.start_date,
            end_date=args.end_date,
            horizon_days=args.horizon,
            drawdown_threshold=args.dd,
            train_start=args.train_start,
            train_end=args.train_end,
            test_start=args.test_start,
            test_end=args.test_end,
            model_type=args.model,
            calibration_method=args.calibration,
            random_seed=args.seed,
        )
        
        try:
            artifacts = run_pipeline(config)
            
            # Compute "Risk Score Today"
            logger.info("\n" + "=" * 60)
            logger.info("Computing Risk Score for Latest Date...")
            logger.info("=" * 60)
            
            # Load predictions to get latest probability
            import pandas as pd
            pred_df = pd.read_csv(artifacts["predictions"])
            latest = pred_df.iloc[-1]
            
            risk_score = latest["predicted_probability"]
            risk_date = latest["date"]
            
            print("\n" + "=" * 60)
            print("RISK SCORE TODAY")
            print("=" * 60)
            print(f"Date: {risk_date}")
            print(f"Crisis Probability: {risk_score:.2%}")
            print(f"Risk Level: ", end="")
            if risk_score < 0.1:
                print("LOW")
            elif risk_score < 0.3:
                print("MEDIUM")
            else:
                print("HIGH")
            print("=" * 60)
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}", exc_info=True)
            sys.exit(1)
    
    elif args.command == "live":
        setup_logging()
        try:
            result = get_live_risk_score(
                model_path=args.model,
                config_path=args.config,
                force_refresh=args.refresh,
            )
            
            print("\n" + "=" * 60)
            print("LIVE RISK SCORE")
            print("=" * 60)
            print(f"Date: {result['date']}")
            print(f"Prediction Horizon: {result['horizon_days']} trading days")
            print(f"Drawdown Threshold: -{result['drawdown_threshold']:.1%}")
            print(f"\nCrisis Probability: {result['probability']:.2%}")
            print(f"Risk Level: {result['risk_level']}")
            print("=" * 60)
            print("\nInterpretation:")
            print(f"  There is a {result['probability']:.1%} probability that the S&P 500")
            print(f"  will experience a drawdown of at least {result['drawdown_threshold']:.1%}")
            print(f"  within the next {result['horizon_days']} trading days.")
            print("=" * 60)
            
        except Exception as e:
            logger.error(f"Failed to get live risk score: {e}", exc_info=True)
            sys.exit(1)
    
    elif args.command == "report":
        logger.warning("Report command not yet implemented")
        sys.exit(1)
    
    else:
        parse_args().print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()

