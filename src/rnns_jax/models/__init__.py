from .blend import BlendModel
from .gru import GRU
from .lstm import LSTM
from .mamba import Mamba

MODEL = {"LSTM": LSTM, "Mamba": Mamba, "GRU": GRU, "blend": BlendModel}
