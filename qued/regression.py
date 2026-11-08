# requirements
import json
from pathlib import Path

import numpy as np
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import (
    mean_absolute_error,
    root_mean_squared_error,
    r2_score,
)
from sklearn.linear_model import Ridge
import optuna
import xgboost as xgb

class QUEDRegressor():
    def __init__(self, estimator, transformers=None, scale=False):
        self.estimator = estimator
        self.transformers = transformers or []
        self.scale = scale
        self.pipeline = self._build_pipeline()

    def _build_pipeline(self):
        # Transform data: Polynomial features 
        steps = [(f"transform_{i}", clone(transformer)) for i, transformer in enumerate(self.transformers)]
        # Standard scaler
        if self.scale:
            steps.append(("scaler", StandardScaler()))
        # Estimator: XGBoost, Linear Regression, etc.
        steps.append(("model", clone(self.estimator)))
        return Pipeline(steps)

    def cross_validate(self, X, y, cv=None, n_jobs=1):
        # Define a cross-validation strategy
        cv = KFold(n_splits=5, shuffle=True, random_state=20251210)
        # Perform cross validation
        ## Use negative root mean square error as the scoring metric 
        ## because cross_val_score maximizes the score
        scores = cross_val_score(self.pipeline, X, y,
                                 cv=cv, scoring='neg_root_mean_squared_error', 
                                 n_jobs=n_jobs)
        
        # Calculate the average RMSE 
        # take the negative sign to convert it to a positive RMSE
        return -np.mean(scores), scores

    def train(self, X, y):
        self.pipeline.fit(X, y)
        return self

    def predict(self, X):
        return self.pipeline.predict(X)

    def get_performance(self, X, y):
            # Return predictions and (MAE, RMSE, R²).
            y_pred = self.predict(X)

            return y_pred, {
            "MAE": mean_absolute_error(y, y_pred),
            "RMSE": root_mean_squared_error(y, y_pred),
            "R2": r2_score(y, y_pred),
        }

def hpo(estimator,
        X_train, 
        y_train, 
        search_space, 
        n_trials=100, 
        output_path=None, 
        suffix='', 
        cv=None, 
        n_jobs=1):
    """
    Optimize an estimator using Optuna and cross-validation RMSE.

    Parameters
    ----------
    estimator : sklearn-compatible estimator
        The model to optimize, e.g. XGBRegressor or Ridge.
    X_train, y_train : array-like
        Training features and targets.
    search_space : dict
        Hyperparameter definitions. Supported types:
        categorical, int, float, and float_log.
    n_trials : int
        Number of additional Optuna trials.
    output_path : str or Path
        Where to save the best hyperparameters and trials DB.
    cv : cross-validation splitter, optional
        CV strategy passed to QUEDRegressor.
    n_jobs: int, optional
        Number of jobs for HPO optimization

    Returns
    -------
    dict
        Best hyperparameters.
    """

    def suggest_params(trial):
        params = {}

        for name, spec in search_space.items():
            kind = spec['type']
            if kind=='categorical':
                params[name] = trial.suggest_categorical(name, spec['choices'])
            elif kind=='int':
                params[name] = trial.suggest_int(name, spec['low'], spec['high'], 
                                                 step=spec.get('step', 1), log=spec.get('log', False))
            elif kind=='float':
                params[name] = trial.suggest_float(name, spec['low'], spec['high'], 
                                                   step=spec.get('step', 1.), log=False)
            elif kind=='float_log':
                params[name] = trial.suggest_float(name, spec['low'], spec['high'], 
                                                   log=True)
            elif kind=='fixed':
                params[name] = spec['value']
            else:
                raise ValueError(f"Unsupported search-space type {kind!r} for parameter {name!r}")
        return params  

    # Prepare the output path before optimization, if requested.
    output_path = Path.cwd() if output_path is None else output_path
    path = Path(output_path) / f"hyperparams_{suffix}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    
    def objective(trial):
        # Define the hyperparameter search space
        params = suggest_params(trial)

        # Cross-validation of regressor
        regressor = QUEDRegressor(clone(estimator).set_params(**params))
        mean_rmse, _ = regressor.cross_validate(X_train, y_train, cv=cv)
        return float(mean_rmse)
    
    def save_best_params(study, trial):
        # Persist the best completed trial whenever it improves.
        if path is not None and study.best_trial.number == trial.number:
            with path.open("w") as f:
                json.dump(study.best_params, f, indent=4)

    # Define Study
    ## Use a distinct study for each descriptor/target/split.
    study_name = f"xgb_hpo_{suffix}" if suffix else "xgb_hpo"
    db_path =  Path(output_path) / Path(f"optuna_xgb_hpo_{suffix}.db")
    db_path.parent.mkdir(parents=True, exist_ok=True)
    storage = f"sqlite:///{db_path.resolve().as_posix()}"
    study = optuna.create_study(direction='minimize',
                                study_name=study_name,
                                storage=storage,
                                load_if_exists=True)
    study.optimize(
        objective,
        n_trials=n_trials,
        show_progress_bar=True,
        gc_after_trial=True,
        callbacks=[save_best_params] if path is not None else None,
        n_jobs=n_jobs,
    )
    hyperparams = study.best_params
    ## Save best model hyperparameters
    if path is not None:
        # Ensure the final result is saved, too.
        with path.open("w") as f:
            json.dump(hyperparams, f, indent=4)

        print(
            f"Best parameters saved to {path} "
            f"(CV RMSE={study.best_value:.6f})"
        )

    del study
    return hyperparams

al_choices = [float(f"{m}e{e}") for e in [-3, -2, -1, 0, 1] for m in [1, 2, 4, 6, 8] if float(f"{m}e{e}") <= 10.0]
xgb_space = {
    "reg_lambda": {"type": "categorical", "choices": al_choices},
    "reg_alpha": {"type": "categorical", "choices": al_choices},
    "colsample_bytree": {"type": "categorical", "choices": [0.1 * i for i in range(1, 11)]},
    "subsample": {"type": "categorical", "choices": [0.1 * i for i in range(1, 11)]},
    "learning_rate": {"type": "categorical", "choices": [0.008 + 0.002 * i for i in range(7)]},
    "n_estimators": {"type": "categorical", "choices": [1000, 1500, 2000, 3000, 4000]},
    "max_depth": {"type": "categorical", "choices": [5, 7, 9, 11, 13, 15, 17, 20]},
    "min_child_weight": {"type": "int", "low": 1, "high": 300},
    "random_state": {"type": "fixed", "value": 20251210},
    "n_jobs": {"type": "fixed", "value": 1}
}
