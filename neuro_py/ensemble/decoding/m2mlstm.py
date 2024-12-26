from typing import List, Tuple, Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F
import lightning as L

from torch import nn


class M2MLSTM(L.LightningModule):
    """
    Many-to-Many Long Short-Term Memory (LSTM) model.

    This class implements a Many-to-Many LSTM model using PyTorch Lightning.

    Parameters
    ----------
    in_dim : int, optional
        Dimensionality of input data, by default 100
    out_dim : int, optional
        Number of output columns, by default 2
    hidden_dims : Tuple[int, int, float], optional
        Architectural parameters of the model (hidden_size, num_layers, dropout),
        by default (400, 1, 0.0)
    use_bias : bool, optional
        Whether to use bias or not in the final linear layer, by default True
    args : Dict, optional
        Additional arguments for model configuration, by default {}

    Attributes
    ----------
    lstm : nn.LSTM
        LSTM layer
    fc : nn.Linear
        Fully connected layer
    hidden_state : Optional[torch.Tensor]
        Hidden state of the LSTM
    cell_state : Optional[torch.Tensor]
        Cell state of the LSTM
    """
    def __init__(self, in_dim: int = 100, out_dim: int = 2,
                 hidden_dims: Tuple[int, int, float] = (400, 1, 0.0),
                 use_bias: bool = True, args: Dict = {}):
        super().__init__()
        self.save_hyperparameters()
        self.in_dim = in_dim
        self.out_dim = out_dim
        if len(hidden_dims) != 3:
            raise ValueError('`hidden_dims` should be of size 3')
        self.hidden_size, self.nlayers, self.dropout = hidden_dims
        self.args = args

        self.lstm = nn.LSTM(input_size=in_dim, hidden_size=self.hidden_size,
                            num_layers=self.nlayers, batch_first=True, 
                            dropout=self.dropout, bidirectional=False)
        self.fc = nn.Linear(in_features=self.hidden_size, out_features=out_dim, bias=use_bias)
        self.hidden_state: Optional[torch.Tensor] = None
        self.cell_state: Optional[torch.Tensor] = None

        self._init_params()

    def _init_params(self) -> None:
        """Initialize model parameters."""
        def init_params(m: nn.Module) -> None:
            if isinstance(m, nn.Linear):
                torch.nn.init.kaiming_uniform_(m.weight, nonlinearity='leaky_relu')
                if m.bias is not None:
                    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
                    bound = 1 / np.sqrt(fan_in)
                    nn.init.uniform_(m.bias, -bound, bound)  # LeCunn init
        init_params(self.fc)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the LSTM model.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, sequence_length, input_dim)

        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch_size, sequence_length, output_dim)
        """
        B, L, N = x.shape
        self.hidden_state = self.hidden_state.to(x.device)
        self.cell_state = self.cell_state.to(x.device)
        self.hidden_state.data.fill_(0.0)
        self.cell_state.data.fill_(0.0)
        lstm_outs = []
        for i in range(L):
            lstm_out, (self.hidden_state, self.cell_state) = \
                self.lstm(x[:, i].unsqueeze(1), (self.hidden_state, self.cell_state))
            lstm_outs.append(lstm_out)

        lstm_outs = torch.stack(lstm_outs, dim=1)  # B, L, N
        out = self.fc(lstm_outs)
        out = out.view(B, L, self.out_dim)
        if self.args.get('clf', False):
            out = F.log_softmax(out, dim=-1)

        return out

    def init_hidden(self, batch_size: int) -> None:
        """
        Initialize hidden state and cell state.

        Parameters
        ----------
        batch_size : int
            Batch size for initialization
        """
        self.batch_size = batch_size
        self.hidden_state = torch.zeros((self.nlayers, batch_size, self.hidden_size), requires_grad=False)
        self.cell_state = torch.zeros((self.nlayers, batch_size, self.hidden_size), requires_grad=False)

    def _step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """
        Perform a single step (forward pass + loss calculation).

        Parameters
        ----------
        batch : Tuple[torch.Tensor, torch.Tensor]
            Batch of input data and labels
        batch_idx : int
            Index of the current batch

        Returns
        -------
        torch.Tensor
            Computed loss
        """
        xs, ys = batch
        outs = self(xs)
        loss = self.args['criterion'](outs, ys)
        return loss

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """
        Lightning method for training step.

        Parameters
        ----------
        batch : Tuple[torch.Tensor, torch.Tensor]
            Batch of input data and labels
        batch_idx : int
            Index of the current batch

        Returns
        -------
        torch.Tensor
            Computed loss
        """
        loss = self._step(batch, batch_idx)
        self.log('train_loss', loss)
        return loss

    def on_after_backward(self) -> None:
        """Lightning method called after backpropagation."""
        self.hidden_state = self.hidden_state.detach()
        self.cell_state = self.cell_state.detach()

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """
        Lightning method for validation step.

        Parameters
        ----------
        batch : Tuple[torch.Tensor, torch.Tensor]
            Batch of input data and labels
        batch_idx : int
            Index of the current batch

        Returns
        -------
        torch.Tensor
            Computed loss
        """
        loss = self._step(batch, batch_idx)
        self.log('val_loss', loss)
        return loss

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """
        Lightning method for test step.

        Parameters
        ----------
        batch : Tuple[torch.Tensor, torch.Tensor]
            Batch of input data and labels
        batch_idx : int
            Index of the current batch

        Returns
        -------
        torch.Tensor
            Computed loss
        """
        loss = self._step(batch, batch_idx)
        self.log('test_loss', loss)
        return loss

    def configure_optimizers(self) -> Tuple[List[torch.optim.Optimizer], List[Dict]]:
        """
        Configure optimizers and learning rate schedulers.

        Returns
        -------
        Tuple[List[torch.optim.Optimizer], List[Dict]]
            Tuple containing a list of optimizers and a list of scheduler configurations
        """
        optimizer = torch.optim.AdamW(
            self.parameters(), weight_decay=self.args['weight_decay'])
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=self.args['lr'],
            epochs=self.args['epochs'],
            total_steps=self.trainer.estimated_stepping_batches
        )
        lr_scheduler = {'scheduler': scheduler, 'interval': 'step'}
        return [optimizer], [lr_scheduler]


class NSVDataset(torch.utils.data.Dataset):
    """
    Custom Dataset for neural state vector (binned spike train) data.

    Parameters
    ----------
    nsv : List[np.ndarray]
        List of trial-segmented neural state vector arrays
    dv : List[np.ndarray]
        List of trial-segmented behavioral state vector arrays

    Attributes
    ----------
    nsv : List[np.ndarray]
        List of trial-segmented neural state vector arrays as float32
    dv : List[np.ndarray]
        List of trial-segmented behavioral state vector arrays as float32
    """
    def __init__(self, nsv: List[np.ndarray], dv: List[np.ndarray]):
        self.nsv = [i.astype(np.float32) for i in nsv]
        self.dv = [i.astype(np.float32) for i in dv]

    def __len__(self) -> int:
        """
        Get the length of the dataset.

        Returns
        -------
        int
            Number of samples in the dataset
        """
        return len(self.nsv)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get a sample from the dataset.

        Parameters
        ----------
        idx : int
            Index of the sample

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Tuple containing NSV and DV arrays
        """
        nsv, dv = self.nsv[idx], self.dv[idx]
        return nsv, dv
