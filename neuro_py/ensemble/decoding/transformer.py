import numpy as np
import torch
import torch.nn as nn
import lightning as L


class PositionalEncoding(nn.Module):
    def __init__(self, in_dim, trial_len, args):
        super().__init__()
        pe = torch.zeros(trial_len, in_dim).to(args['device']) # * Can optim to empty
        position = torch.arange(0, trial_len, dtype=torch.float).unsqueeze(1)

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
    def __init__(self, in_dim=100, out_dim=2, trial_len=2, args=None):
        super().__init__()
        self.save_hyperparameters()
        self.trial_len = trial_len
        self.in_dim = in_dim
        self.args = args
        activations = nn.CELU if self.args['activations'] is None else self.args['activations']

        self.src_mask = {}  # full context, by default

        # self.scale = np.sqrt(in_dim)

        self.pos_encoder = PositionalEncoding(in_dim, trial_len, args)

        encoder_lyr = nn.TransformerEncoderLayer(
            in_dim,
            nhead=args['nhead'],
            dim_feedforward=args['hidden_size'],
            dropout=args['dropout'],
            activation=args['activation']
        )

        self.transformer_encoder = nn.TransformerEncoder(
            encoder_lyr, args['nlayers'], nn.LayerNorm(in_dim))
        
        self.rate_dropout = nn.Dropout(args['dropout_rates'])

        self.decoder = nn.Sequential(
            nn.Linear(in_dim, 16), activations(), nn.Linear(16, out_dim)
        )
        self.regressor = nn.MSELoss()

        def init_params(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.kaiming_uniform_(m.weight, nonlinearity='leaky_relu')
                if m.bias is not None:
                    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
                    bound = 1 / np.sqrt(fan_in)
                    nn.init.uniform_(m.bias, -bound, bound)  # LeCunn init
        self.decoder.apply(init_params)


    def forward(self, x, mask_labels=None):
        """_summary_

        Parameters
        ----------
        x : torch.tensor (BxLxN)
            _description_
        mask_labels : _type_
            _description_
        """
        x = x.permute(1, 0, 2)  # LxBxN
        x = self.pos_encoder(x)
        x_mask = self._get_or_generate_context_mask(x)
        z = self.transformer_encoder(x, x_mask)
        z = self.rate_dropout(z)
        out = self.decoder(z).permute(1, 0, 2)  # BxLxout_dim
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
            epochs=args['epochs'], steps_per_epoch=len(self.trainer._data_connector._train_dataloader_source.dataloader())
        )
        lr_scheduler = {'scheduler': scheduler, 'interval': 'step'}
        return [optimizer], [lr_scheduler]
