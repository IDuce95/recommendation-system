
import os
import logging
from typing import Dict, Any, Optional, Union
from pathlib import Path

try:
    import mlflow
    import mlflow.pytorch
    import mlflow.sklearn
    from mlflow.tracking import MlflowClient
    _HAS_MLFLOW = True
except ImportError:
    _HAS_MLFLOW = False
    mlflow = None
    MlflowClient = None

from config.config_manager import get_config

logger = logging.getLogger(__name__)

class MLflowTracker:

    def __init__(self, experiment_name: Optional[str] = None):
        if not _HAS_MLFLOW:
            raise ImportError("MLflow is required. Install it with: pip install mlflow")

        self.config = get_config()
        self.mlflow_config = self.config.get_mlflow_config()

        tracking_uri = self.mlflow_config.get('tracking_uri', 'http://localhost:5555')
        mlflow.set_tracking_uri(tracking_uri)

        self.experiment_name = experiment_name or self.mlflow_config.get('experiment_name', 'recommendation_system')
        self._setup_experiment()

        self.client = MlflowClient()

        logger.info(f"MLflow tracking initialized with URI: {tracking_uri}")
        logger.info(f"Using experiment: {self.experiment_name}")

    def _setup_experiment(self):
        try:
            experiment = mlflow.get_experiment_by_name(self.experiment_name)
            if experiment is None:
                experiment_id = mlflow.create_experiment(
                    name=self.experiment_name,
                    artifact_location=self.mlflow_config.get('artifact_root', './mlflow_artifacts')
                )
                logger.info(f"Created new experiment: {self.experiment_name} (ID: {experiment_id})")
            else:
                logger.info(f"Using existing experiment: {self.experiment_name} (ID: {experiment.experiment_id})")

            mlflow.set_experiment(self.experiment_name)
        except Exception as e:
            logger.error(f"Failed to setup MLflow experiment: {e}")
            raise

    def start_run(self, run_name: Optional[str] = None, tags: Optional[Dict[str, str]] = None) -> mlflow.ActiveRun:
        return mlflow.start_run(run_name=run_name, tags=tags)

    def log_params(self, params: Dict[str, Union[str, int, float]]):
        for key, value in params.items():
            mlflow.log_param(key, value)

    def log_metrics(self, metrics: Dict[str, Union[int, float]], step: Optional[int] = None):
        for key, value in metrics.items():
            mlflow.log_metric(key, value, step=step)

    def log_artifact(self, local_path: Union[str, Path], artifact_path: Optional[str] = None):
        mlflow.log_artifact(str(local_path), artifact_path)

    def log_model(self, model: Any, artifact_path: str, **kwargs):
        if hasattr(model, 'save_pretrained'):
            model_dir = Path(f"temp_model_{artifact_path}")
            model_dir.mkdir(exist_ok=True)

            try:
                model.save_pretrained(model_dir)
                mlflow.log_artifacts(str(model_dir), artifact_path)
                logger.info(f"Logged HuggingFace model to {artifact_path}")
            finally:
                import shutil
                if model_dir.exists():
                    shutil.rmtree(model_dir)
        else:
            try:
                mlflow.sklearn.log_model(model, artifact_path, **kwargs)
                logger.info(f"Logged sklearn model to {artifact_path}")
            except Exception:
                try:
                    import pickle
                    model_path = f"temp_{artifact_path}.pkl"
                    with open(model_path, 'wb') as f:
                        pickle.dump(model, f)
                    mlflow.log_artifact(model_path, artifact_path)
                    os.remove(model_path)
                    logger.info(f"Logged pickled model to {artifact_path}")
                except Exception as e:
                    logger.error(f"Failed to log model: {e}")
                    raise

    def end_run(self, status: str = "FINISHED"):
        mlflow.end_run(status=status)

    def get_experiment_runs(self, experiment_name: Optional[str] = None) -> list:
        exp_name = experiment_name or self.experiment_name
        experiment = mlflow.get_experiment_by_name(exp_name)
        if experiment:
            return mlflow.search_runs(experiment_ids=[experiment.experiment_id])
        return []

    def get_best_run(self, metric_name: str, experiment_name: Optional[str] = None, ascending: bool = False):
        runs = self.get_experiment_runs(experiment_name)
        if not runs.empty and metric_name in runs.columns:
            best_run = runs.loc[runs[metric_name].idxmin() if ascending else runs[metric_name].idxmax()]
            return best_run
        return None

_mlflow_tracker: Optional[MLflowTracker] = None

def get_mlflow_tracker(experiment_name: Optional[str] = None) -> MLflowTracker:
    global _mlflow_tracker
    if _mlflow_tracker is None:
        _mlflow_tracker = MLflowTracker(experiment_name)
    return _mlflow_tracker

def init_mlflow_tracker(experiment_name: Optional[str] = None) -> MLflowTracker:
    global _mlflow_tracker
    _mlflow_tracker = MLflowTracker(experiment_name)
    return _mlflow_tracker

class MLflowRun:

    def __init__(self, run_name: Optional[str] = None, experiment_name: Optional[str] = None,
                 tags: Optional[Dict[str, str]] = None):
        self.run_name = run_name
        self.experiment_name = experiment_name
        self.tags = tags
        self.tracker = get_mlflow_tracker(experiment_name)
        self.run = None

    def __enter__(self):
        self.run = self.tracker.start_run(self.run_name, self.tags)
        return self.tracker

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            self.tracker.end_run("FAILED")
            logger.error(f"MLflow run failed: {exc_val}")
        else:
            self.tracker.end_run("FINISHED")
        return False

def setup_mlflow():
    try:
        config = get_config()
        mlflow_config = config.get_mlflow_config()

        tracking_uri = mlflow_config.get('tracking_uri', 'http://localhost:5555')
        mlflow.set_tracking_uri(tracking_uri)

        logger.info(f"MLflow setup completed with tracking URI: {tracking_uri}")
        return True

    except Exception as e:
        logger.error(f"Failed to setup MLflow: {e}")
        return False
