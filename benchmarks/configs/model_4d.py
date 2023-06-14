from dataclasses import dataclass

from configs.model_3d import SchNet, DimeNet, DimeNetPlusPlus, GemNet, PaiNN, ClofNet


@dataclass
class TransformerPooling:
    num_heads: int = 8
    num_layers: int = 2


@dataclass
class Model4D:
    graph_encoder: str = 'SchNet'
    set_encoder: str = 'Attention'

    schnet: SchNet = SchNet()
    dimenet: DimeNet = DimeNet()
    dimenetplusplus: DimeNetPlusPlus = DimeNetPlusPlus()
    gemnet: GemNet = GemNet()
    painn: PaiNN = PaiNN()
    clofnet: ClofNet = ClofNet()

    transformer: TransformerPooling = TransformerPooling()
