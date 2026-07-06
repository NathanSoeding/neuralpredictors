import torch
from torch import nn
from unittest.mock import patch

from neuralpredictors.layers.shifters import MLP

def test_xavier_initialization_is_used():
    with patch("neuralpredictors.layers.shifters.mlp.xavier_normal_") as mock_xavier:
        MLP(shift_layers=3)

    # One Linear layer per shift layer
    assert mock_xavier.call_count == 3