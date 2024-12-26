from typing import List, Dict, Tuple, Optional

import torch
import torch.nn.functional as F
import lightning as L

from torch import nn


class LSTM(L.LightningModule):
    """
    Long Short-Term Memory (LSTM) model.

    This class implements an LSTM model using PyTorch Lightning.

    Parameters
    ----------
    in_dim : int, optional
        Dimensionality of input data, by default 100
    out_dim : int, optional
        Dimensionality of output data, by default 2
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
                            dropout=self.dropout, bidirectional=True)
        self.fc = nn.Linear(in_features=2*self.hidden_size, out_features=out_dim, bias=use_bias)
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
                    bound = 1 / torch.math.sqrt(fan_in)
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
            Output tensor of shape (batch_size, output_dim)
        """
        lstm_out, (self.hidden_state, self.cell_state) = \
            self.lstm(x, (self.hidden_state, self.cell_state))
        lstm_out = lstm_out[:, -1, :].contiguous()
        out = self.fc(lstm_out)
        if self.args.get('clf', False):
            out = F.log_softmax(out, dim=1)
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
        h0 = torch.zeros(
            (2*self.nlayers, batch_size, self.hidden_size),
            requires_grad=False
        )
        c0 = torch.zeros(
            (2*self.nlayers, batch_size, self.hidden_size),
            requires_grad=False
        )
        self.hidden_state = h0
        self.cell_state = c0

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Make predictions using the LSTM model.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor

        Returns
        -------
        torch.Tensor
            Predicted output
        """
        self.hidden_state = self.hidden_state.to(x.device)
        self.cell_state = self.cell_state.to(x.device)
        preds = []
        batch_size = self.batch_size
        for i in range(batch_size, x.shape[0]+batch_size, batch_size):
            iptensor = x[i-batch_size:i]
            if i > x.shape[0]:
                iptensor = F.pad(iptensor, (0,0,0,0,0,i-x.shape[0]))
            pred_loc = self.forward(iptensor)
            if i > x.shape[0]:
                pred_loc = pred_loc[:batch_size-(i-x.shape[0])]
            preds.extend(pred_loc)
        out = torch.stack(preds)
        if self.args.get('clf', False):
            out = F.log_softmax(out, dim=1)
        return out

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
        self.hidden_state.detach_()
        self.cell_state.detach_()

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
            steps_per_epoch=len(
                self.trainer._data_connector._train_dataloader_source.dataloader()
            )
        )
        lr_scheduler = {'scheduler': scheduler, 'interval': 'step'}
        return [optimizer], [lr_scheduler]
