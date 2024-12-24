import torch
import lightning as L

from torch import nn


class MLP(L.LightningModule):
    """MLPs in Pytorch of an arbitrary number of hidden
    layers of potentially different sizes.
    """
    def __init__(self, in_dim=100, out_dim=2, hidden_dims=(), use_bias=True, args=None):
        """
        Constructs a MultiLayerPerceptron

        Parameters
        ----------
        in_dim : int
            Dimensionality of input data
        out_dim : int
            Dimensionality of output data
        hidden_dims : List
            List containing architectural parameters of the model. If element is
            int, it is a hidden layer of that size. If element is float, it is a
            dropout layer with that probability.
        use_bias : bool
            Whether to use bias or not in the all linear layers
        args : dict
            Dictionary containing the hyperparameters of the model
        """
        super().__init__()
        self.save_hyperparameters()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.args = args
        activations = nn.CELU if self.args['activations'] is None else self.args['activations']

        # If we have no hidden layer, just initialize a linear model (e.g. in logistic regression)
        if len(hidden_dims) == 0:
            layers = [nn.Linear(in_dim, out_dim, bias=use_bias)]
        else:
            # 'Actual' MLP with dimensions in_dim - num_hidden_layers*[hidden_dim] - out_dim
            layers = []
            hidden_dims = [in_dim] + hidden_dims

            # Loop until before the last layer
            for i, hidden_dim in enumerate(hidden_dims[:-1]):
                if isinstance(hidden_dim, float):
                    continue
                if isinstance(hidden_dims[i+1], float):
                    layers += [nn.Linear(hidden_dim, hidden_dims[i + 2], bias=use_bias),
                                        nn.Dropout(p=hidden_dims[i+1]),
                                        activations() if i < len(hidden_dims)-1 else nn.Tanh()]
                else:
                    layers += [nn.Linear(hidden_dim, hidden_dims[i + 1], bias=use_bias),
                                        activations() if i < len(hidden_dims)-1 else nn.Tanh()]

            # Add final layer to the number of classes
            layers += [nn.Linear(hidden_dims[-1], out_dim, bias=use_bias)]
            if args['clf']:
                layers += [nn.LogSoftmax(dim=1)]

        self.main = nn.Sequential(*layers)

        def init_params(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.kaiming_uniform_(m.weight, nonlinearity='leaky_relu')
                if m.bias is not None:
                    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
                    bound = 1 / torch.math.sqrt(fan_in)
                    nn.init.uniform_(m.bias, -bound, bound)  # LeCunn init
        self.main.apply(init_params)

    def forward(self, x):
        """
        Defines the network structure and flow from input to output

        Parameters
        ----------
        x : torch.Tensor
            Input data
        
        Returns
        -------
        torch.Tensor
            Output data
        """
        return self.main(x)

    def _step(self, batch, batch_idx):
        xs, ys = batch  # unpack the batch
        outs = self(xs)  # apply the model
        loss = self.args['criterion'](outs, ys)  # compute the (squared error) loss
        return loss

    def training_step(self, batch, batch_idx):
        loss = self._step(batch, batch_idx)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._step(batch, batch_idx)
        self.log('val_loss', loss)
        return loss

    def test_step(self, batch, batch_idx):
        loss = self._step(batch, batch_idx)
        self.log('test_loss', loss)
        return loss

    def configure_optimizers(self):
        args = self.args
        optimizer = torch.optim.AdamW(
            self.parameters(), weight_decay=args['weight_decay'], betas=(0.9, 0.999),
            amsgrad=True)
        # scheduler = torch.optim.lr_scheduler.OneCycleLR(
        #   optimizer, max_lr=args['lr'],
        #   epochs=args['epochs'], steps_per_epoch=len(self.trainer._data_connector._train_dataloader_source.dataloader())
        # )
        # write a cycliclr scheduler
        scheduler = torch.optim.lr_scheduler.CyclicLR(
            optimizer, base_lr=args['base_lr'], max_lr=args['lr'],
            step_size_up=self.args['scheduler_step_size_multiplier']*self.args['num_training_batches'],  # assuming 1 batch_size, multiply if more: https://discuss.pytorch.org/t/step-size-for-cyclic-scheduler/69262/4
            cycle_momentum=False,
            mode='triangular2', gamma=0.99994,
            last_epoch=-1, verbose=False
        )
        lr_scheduler = {'scheduler': scheduler, 'interval': 'step'}
        return [optimizer], [lr_scheduler]
