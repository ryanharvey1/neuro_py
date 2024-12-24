import numpy as np
import torch
import torch.nn.functional as F
import lightning as L

from torch import nn


class M2MLSTM(L.LightningModule):
    """Many-to-Many Long Short-Term Memory (LSTM) model."""
    def __init__(self, in_dim=100, out_dim=2, hidden_dims=(400, 1, .0), use_bias=True, args={}):
        """
        Constructs a Many-to-Many LSTM

        Parameters
        ----------
        in_dim : int
            Dimensionality of input data
        out_dim : int
            Number of output columns
        hidden_dims : List
            Architectural parameters of the model
            (hidden_size, num_layers, dropout)
        use_bias : bool
            Whether to use bias or not in the final linear layer
        """
        super().__init__()
        self.save_hyperparameters()
        self.in_dim = in_dim
        self.out_dim = out_dim
        if len(hidden_dims) != 3:
            raise ValueError('`hidden_dims` should be of size 3')
        hidden_size, nlayers, dropout = hidden_dims
        self.nlayers = nlayers
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.args = args

        # Add final layer to the number of classes
        self.lstm = nn.LSTM(input_size=in_dim, hidden_size=hidden_size,
            num_layers=nlayers, batch_first=True, dropout=dropout, bidirectional=False)
        self.fc = nn.Linear(in_features=hidden_size, out_features=out_dim, bias=use_bias)
        self.hidden_state = None
        self.cell_state = None

        def init_params(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.kaiming_uniform_(m.weight, nonlinearity='leaky_relu')
                if m.bias is not None:
                    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
                    bound = 1 / np.sqrt(fan_in)
                    nn.init.uniform_(m.bias, -bound, bound)  # LeCunn init
        init_params(self.fc)

    def forward(self, x):
        B, L, N = x.shape
        self.hidden_state = self.hidden_state.to(x.device)
        self.cell_state = self.cell_state.to(x.device)
        self.hidden_state.data.fill_(.0)
        self.cell_state.data.fill_(.0)
        lstm_outs = []
        for i in range(L):
            lstm_out, (self.hidden_state, self.cell_state) = \
                self.lstm(x[:, i].unsqueeze(1), (self.hidden_state, self.cell_state))
            # Shape: [batch_size x max_length x hidden_dim]
            lstm_outs.append(lstm_out)

        lstm_outs = torch.stack(lstm_outs, dim=1)  # B, L, N
        # Select the activation of the last Hidden Layer
        # lstm_outs = lstm_outs.contiguous()
        # lstm_outs = lstm_outs.view(-1, lstm_outs.shape[2])  # B*L, N
        
        # Shape: [batch_size x hidden_dim]

        # Fully connected layer
        out = self.fc(lstm_outs)
        out = out.view(B, L, self.out_dim)
        if self.args['clf']:
            out = F.log_softmax(out, dim=-1)

        return out

    def init_hidden(self, batch_size):
        ''' Initializes hidden state '''
        # Create two new tensors with sizes n_layers x batch_size x hidden_dim,
        # initialized to zero, for hidden state and cell state of LSTM
        self.batch_size = batch_size
        h0 = torch.zeros((self.nlayers,batch_size,self.hidden_size), requires_grad=False)
        c0 = torch.zeros((self.nlayers,batch_size,self.hidden_size), requires_grad=False)
        self.hidden_state = h0
        self.cell_state = c0

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

    def on_after_backward(self):
        # LSTM specific
        self.hidden_state = self.hidden_state.detach()
        self.cell_state = self.cell_state.detach()

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


class NSVDataset(torch.utils.data.Dataset):
    def __init__(self, nsv, dv):
        self.nsv = [i.astype(np.float32) for i in nsv]
        self.dv = [i.astype(np.float32) for i in dv]

    def __len__(self):
        return len(self.nsv)

    def __getitem__(self, idx):
        nsv, dv = self.nsv[idx], self.dv[idx]
        return nsv, dv
