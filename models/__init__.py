from .lstm import LSTM
from .mamba import Mamba
from .gru import GRU
from .blend import BlendModel

MODEL = {
    "LSTM": LSTM,
    "Mamba": Mamba,
    "GRU": GRU,
    "blend": BlendModel
}