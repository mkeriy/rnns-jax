from .lstm import LSTM
from .mamba import Mamba
from .gru import GRU
from .blend import BlendModel
from .mlstm import mLSTM

MODEL = {
    "LSTM": LSTM,
    "mLSTM": mLSTM,
    "Mamba": Mamba,
    "GRU": GRU,
    "blend": BlendModel
}