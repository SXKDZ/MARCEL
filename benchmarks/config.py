from dataclasses import dataclass, field

from configs.model_1d import Model1D
from configs.model_2d import Model2D
from configs.model_3d import Model3D
from configs.model_4d import Model4D
from configs.model_fp_rf import ModelFPRF


@dataclass
class ReduceLROnPlateau:
    mode: str = 'min'
    factor: int = 0.8
    patience: int = 20


@dataclass
class CosineAnnealingLR:
    eta_min: float = 1e-6


@dataclass
class LinearWarmupCosineAnnealingLR:
    warmup_steps: int = 200
    max_epochs: int = 2000


@dataclass
class OneCycleLR:
    max_lr: float = 0.001
    steps_per_epoch: int = 100


@dataclass
class Config:
    dataset: str = 'Kraken'
    target: str = 'qpoletens_xx'
    max_num_molecules: int = None
    max_num_conformers: int = 20
    train_ratio: float = 0.7
    valid_ratio: float = 0.1

    batch_size: int = 256
    hidden_dim: int = 128
    num_epochs: int = 2000
    patience: int = 200
    activation: str = 'relu'
    seed: int = 123
    device: str = 'cuda:5'
    dropout: float = 0.5

    scheduler: str = None
    reduce_lr_on_plateau: ReduceLROnPlateau = ReduceLROnPlateau()
    cosine_annealing_lr: CosineAnnealingLR = CosineAnnealingLR()
    linear_warmup_cosine_annealing_lr: LinearWarmupCosineAnnealingLR = LinearWarmupCosineAnnealingLR()
    one_cycle_lr: OneCycleLR = OneCycleLR()

    learning_rate: float = 0.001
    weight_decay: float = 1e-4

    modelfprf: ModelFPRF = ModelFPRF()
    model1d: Model1D = Model1D()
    model2d: Model2D = Model2D()
    model3d: Model3D = Model3D()
    model4d: Model4D = Model4D()
