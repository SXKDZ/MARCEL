from dataclasses import dataclass


@dataclass
class GIN:
    num_layers: int = 6
    virtual_node: bool = False


@dataclass
class GPS:
    num_layers: int = 6
    walk_length: int = 20
    num_heads: int = 4


@dataclass
class ChemProp:
    num_layers: int = 6


@dataclass
class Model2D:
    model: str = 'GIN'
    gin: GIN = GIN()
    gps: GPS = GPS()
    chemprop: ChemProp = ChemProp()
