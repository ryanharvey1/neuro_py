from typing import Tuple, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L

class PositionalEncoding(nn.Module):
    """
    Positional Encoding module for Transformer models.

    Parameters
    ----------
    in_dim : int
        Input dimension of the model
    max_context_len : int
        Maximum context length
    args : Dict
        Additional arguments (not used in this implementation)

    Attributes
    ----------
    pe : torch.Tensor
        Positional encoding tensor
    """
    def __init__(self, in_dim: int, max_context_len: int, args: Dict):
        super().__init__()
        pe = torch.zeros(max_context_len, in_dim)
        position = torch.arange(0, max_context_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, in_dim, 2).float() * (-np.log(1e4) / in_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)  # t x 1 x d
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to the input tensor.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (seq_len, batch_size, in_dim)

        Returns
        -------
        torch.Tensor
            Input tensor with added positional encoding
        """
        self.pe = self.pe.to(x.device)
        x = x + self.pe[:x.size(0), :]  # t x 1 x d, # t x b x d
        return x

class NDT(L.LightningModule):
    """
    Transformer encoder-based dynamical systems decoder.

    This class implements a Transformer-based decoder trained on MLM loss.
    It returns loss and predicted rates.

    Parameters
    ----------
    in_dim : int, optional
        Dimensionality of input data, by default 100
    out_dim : int, optional
        Number of output columns, by default 2
    hidden_dims : Tuple[int], optional
        Architectural parameters of the model
        (dim_feedforward, num_layers, nhead, dropout, rate_dropout), 
        by default [400, 1, 1, 0.0, 0.0]
    max_context_len : int, optional
        Maximum context length, by default 2
    args : Optional[Dict], optional
        Dictionary containing the hyperparameters of the model, by default None

    Attributes
    ----------
    pos_encoder : PositionalEncoding
        Positional encoding module
    transformer_encoder : nn.TransformerEncoder
        Transformer encoder module
    rate_dropout : nn.Dropout
        Dropout layer for rates
    decoder : nn.Sequential
        Decoder network
    src_mask : Dict[str, torch.Tensor]
        Dictionary to store source masks for different devices
    """
    def __init__(self, in_dim: int = 100, out_dim: int = 2,
                 hidden_dims: Tuple[int] = (400, 1, 1, 0.0, 0.0),
                 max_context_len: int = 2, args: Optional[Dict] = None):
        super().__init__()
        self.save_hyperparameters()
        self.max_context_len = max_context_len
        self.in_dim = in_dim
        self.args = args if args is not None else {}
        activations = nn.CELU if self.args.get('activations') is None else self.args['activations']

        self.src_mask: Dict[str, torch.Tensor] = {}

        self.pos_encoder = PositionalEncoding(in_dim, max_context_len, self.args)

        encoder_lyr = nn.TransformerEncoderLayer(
            in_dim,
            nhead=hidden_dims[2],
            dim_feedforward=hidden_dims[0],
            dropout=hidden_dims[3],
            activation=nn.functional.relu
        )

        self.transformer_encoder = nn.TransformerEncoder(
            encoder_lyr, hidden_dims[1], nn.LayerNorm(in_dim))
        
        self.rate_dropout = nn.Dropout(hidden_dims[4])

        self.decoder = nn.Sequential(
            nn.Linear(in_dim, 16), activations(), nn.Linear(16, out_dim)
        )

        self._init_params()

    def _init_params(self) -> None:
        """Initialize the parameters of the decoder."""
        def init_params(m: nn.Module) -> None:
            if isinstance(m, nn.Linear):
                torch.nn.init.kaiming_uniform_(m.weight, nonlinearity='leaky_relu')
                if m.bias is not None:
                    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
                    bound = 1 / np.sqrt(fan_in)
                    nn.init.uniform_(m.bias, -bound, bound)  # LeCunn init
        self.decoder.apply(init_params)

    def forward(self, x: torch.Tensor, mask_labels: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass of the model.

        Parameters
        ----------
        x : torch.Tensor
            Input data of shape (batch_size, seq_len, in_dim)
        mask_labels : Optional[torch.Tensor], optional
            Masking labels for the input data, by default None

        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch_size, seq_len, out_dim)
        """
        x = x.permute(1, 0, 2)  # LxBxN
        x = self.pos_encoder(x)
        x_mask = self._get_or_generate_context_mask(x)
        z = self.transformer_encoder(x, x_mask)
        z = self.rate_dropout(z)
        out = self.decoder(z).permute(1, 0, 2)  # B x L x out_dim
        if self.args.get('clf', False):
            out = F.log_softmax(out, dim=-1)
        return out

    def _get_or_generate_context_mask(self, src: torch.Tensor) -> torch.Tensor:
        """
        Get or generate the context mask for the input tensor.

        Parameters
        ----------
        src : torch.Tensor
            Input tensor

        Returns
        -------
        torch.Tensor
            Context mask for the input tensor
        """
        context_forward = 4
        size = src.size(0)  # T
        mask = (torch.triu(torch.ones(size, size, device=src.device), diagonal=-context_forward) == 1).transpose(0, 1)
        mask = mask.float()
        self.src_mask[str(src.device)] = mask
        return self.src_mask[str(src.device)]

    def _step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        """
        Perform a single step (forward pass + loss calculation).

        Parameters
        ----------
        batch : tuple
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

    def training_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        """
        Lightning method for training step.

        Parameters
        ----------
        batch : tuple
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

    def validation_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        """
        Lightning method for validation step.

        Parameters
        ----------
        batch : tuple
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

    def test_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        """
        Lightning method for test step.

        Parameters
        ----------
        batch : tuple
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

    def configure_optimizers(self) -> tuple:
        """
        Configure optimizers and learning rate schedulers.

        Returns
        -------
        tuple
            List of optimizers and a list of scheduler configurations
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
