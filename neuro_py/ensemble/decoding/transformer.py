import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L


class PositionalEncoding(nn.Module):
    def __init__(self, in_dim, max_context_len, args):
        super().__init__()
        pe = torch.zeros(max_context_len, in_dim) # * Can optim to empty
        position = torch.arange(0, max_context_len, dtype=torch.float).unsqueeze(1)

        div_term = torch.exp(torch.arange(0, in_dim, 2).float() * (-np.log(1e4) / in_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)  # t x 1 x d
        self.register_buffer('pe', pe)

    def forward(self, x):
        self.pe = self.pe.to(x.device)
        x = x + self.pe[:x.size(0), :]  # t x 1 x d, # t x b x d
        return x

class NDT(L.LightningModule):
    """Transformer encoder-based dynamical systems decoder.
    * Trained on MLM loss
    * Returns loss & predicted rates."""
    def __init__(self, in_dim=100, out_dim=2, hidden_dims=[400, 1, 1, .0, .0], max_context_len=2, args=None):
        """Constructs a Transformer-based decoder.

        Parameters
        ----------
        in_dim : int
            Dimensionality of input data
        out_dim : int
            Number of output columns
        hidden_dims : list
            Containing the architectural parameters of the model
            (dim_feedforward, num_layers, nhead, dropout, rate_dropout) 
        max_context_len : int
            Maximum context length
        args : dict
            Dictionary containing the hyperparameters of the model
        """
        super().__init__()
        self.save_hyperparameters()
        self.max_context_len = max_context_len
        self.in_dim = in_dim
        self.args = args
        activations = nn.CELU if self.args['activations'] is None else self.args['activations']

        self.src_mask = {}  # full context, by default

        # self.scale = np.sqrt(in_dim)

        self.pos_encoder = PositionalEncoding(in_dim, max_context_len, args)

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

        def init_params(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.kaiming_uniform_(m.weight, nonlinearity='leaky_relu')
                if m.bias is not None:
                    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
                    bound = 1 / np.sqrt(fan_in)
                    nn.init.uniform_(m.bias, -bound, bound)  # LeCunn init
        self.decoder.apply(init_params)


    def forward(self, x, mask_labels=None):
        """Forward pass of the model.

        Parameters
        ----------
        x : torch.tensor (BxLxN)
            Input data
        mask_labels : torch.tensor (LxL)
            Masking labels for the input data
        """
        x = x.permute(1, 0, 2)  # LxBxN
        x = self.pos_encoder(x)
        x_mask = self._get_or_generate_context_mask(x)
        z = self.transformer_encoder(x, x_mask)
        z = self.rate_dropout(z)
        out = self.decoder(z).permute(1, 0, 2)  # B x L x out_dim
        if self.args['clf']:
            out = F.log_softmax(out, dim=-1)
        return out

    def _get_or_generate_context_mask(self, src):
        context_forward = 4
        size = src.size(0)  # T
        mask = (torch.triu(torch.ones(size, size, device=src.device), diagonal=-context_forward) == 1).transpose(0, 1)
        mask = mask.float()
        self.src_mask[str(src.device)] = mask
        return self.src_mask[str(src.device)]

    def _step(self, batch, batch_idx) -> torch.Tensor:
        xs, ys = batch  # unpack the batch
        B, L, N = xs.shape
        outs = self(xs)
        loss = self.args['criterion'](outs, ys)
        return loss

    def training_step(self, batch, batch_idx) -> torch.Tensor:
        loss = self._step(batch, batch_idx)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx) -> torch.Tensor:
        loss = self._step(batch, batch_idx)
        self.log('val_loss', loss)
        return loss

    def test_step(self, batch, batch_idx) -> torch.Tensor:
        loss = self._step(batch, batch_idx)
        self.log('test_loss', loss)
        return loss

    def configure_optimizers(self):
        args = self.args
        optimizer = torch.optim.AdamW(
            self.parameters(), weight_decay=args['weight_decay'])
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=args['lr'],
            epochs=args['epochs'],
            total_steps=self.trainer.estimated_stepping_batches
        )
        lr_scheduler = {'scheduler': scheduler, 'interval': 'step'}
        return [optimizer], [lr_scheduler]
