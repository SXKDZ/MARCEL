from dataclasses import dataclass


@dataclass
class Model1D:
    model: str = 'LSTM'
    input_type: str = 'SMILES'
    embedding_dim: int = 128
    num_layers: int = 4
    num_heads: int = 4
