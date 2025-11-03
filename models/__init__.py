from .lstm import LSTM
from .mamba import Mamba
from .gru import GRU


MODEL = {
    "LSTM": LSTM,
    "Mamba": Mamba,
    "GRU": GRU,
}