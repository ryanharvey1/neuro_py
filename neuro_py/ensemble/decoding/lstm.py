import torch
import torch.nn.functional as F
import lightning as L

from torch import nn


class LSTM(L.LightningModule):
    """MLPs in Pytorch of an arbitrary number of hidden
    layers of potentially different sizes.
    """
    def __init__(self, in_dim=100, out_dim=2, hidden_dims=(400, 1, .0), use_bias=True, args={}):
        """
        Constructs a MultiLayerPerceptron

        Args:
            in_dim: Integer
                dimensionality of input data (784)
            out_dim: Integer
                number of output columns
            hidden_dims: List
                containing the dimensions of the hidden layers,
                empty list corresponds to a linear model (in_dim, out_dim)
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
            num_layers=nlayers, batch_first=True, dropout=dropout, bidirectional=True)
        self.fc = nn.Linear(in_features=2*hidden_size, out_features=out_dim, bias=use_bias)
        self.hidden_state = None
        self.cell_state = None

        def init_params(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.kaiming_uniform_(m.weight, nonlinearity='leaky_relu')
                if m.bias is not None:
                    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
                    bound = 1 / torch.math.sqrt(fan_in)
                    nn.init.uniform_(m.bias, -bound, bound)  # LeCunn init
        init_params(self.fc)

    def forward(self, x):
        lstm_out, (self.hidden_state, self.cell_state) = self.lstm(x, (self.hidden_state, self.cell_state))
        B, L, H = lstm_out.shape
        # Shape: [batch_size x max_length x hidden_dim]

        # Select the activation of the last Hidden Layer
        # lstm_out = lstm_out.view(B, L, 2, -1).sum(dim=2)
        lstm_out = lstm_out[:,-1,:].contiguous()
        
        # Shape: [batch_size x hidden_dim]

        # Fully connected layer
        out = self.fc(lstm_out)
        if self.args['clf']:
            out = F.log_softmax(out, dim=1)

        return out

    def init_hidden(self, batch_size):
        ''' Initializes hidden state '''
        # Create two new tensors with sizes n_layers x batch_size x hidden_dim,
        # initialized to zero, for hidden state and cell state of LSTM
        self.batch_size = batch_size
        h0 = torch.zeros((2*self.nlayers,batch_size,self.hidden_size), requires_grad=False)
        c0 = torch.zeros((2*self.nlayers,batch_size,self.hidden_size), requires_grad=False)
        self.hidden_state = h0
        self.cell_state = c0

    def predict(self, x):
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
        if self.args['clf']:
            out = F.log_softmax(out, dim=1)
        return out

    def _step(self, batch, batch_idx) -> torch.Tensor:
        xs, ys = batch  # unpack the batch
        outs = self(xs)  # apply the model
        loss = self.args['criterion'](outs, ys)
        return loss

    def training_step(self, batch, batch_idx) -> torch.Tensor:
        loss = self._step(batch, batch_idx)
        self.log('train_loss', loss)
        return loss

    def on_after_backward(self):
        # LSTM specific
        self.hidden_state.detach_()
        self.cell_state.detach_()
        # self.hidden_state.data.fill_(.0)
        # self.cell_state.data.fill_(.0)

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
            steps_per_epoch=len(
                self.trainer._data_connector._train_dataloader_source.dataloader()
            )
        )
        lr_scheduler = {'scheduler': scheduler, 'interval': 'step'}
        return [optimizer], [lr_scheduler]
