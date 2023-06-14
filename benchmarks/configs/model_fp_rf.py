from dataclasses import dataclass


@dataclass
class ModelFPRF:
    n_jobs: int = 8
    n_estimators: int = 500
    min_samples_leaf: int = 2
    min_samples_split: int = 10
    min_impurity_decrease: int = 0
    warm_start: bool = True
