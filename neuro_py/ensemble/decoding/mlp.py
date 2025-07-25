from typing import List, Union, Dict, Optional

import torch
import lightning as L

from torch import nn


class MLP(L.LightningModule):
    """
    Multi-Layer Perceptron (MLP) in PyTorch with an arbitrary number of hidden layers.

    This class implements an MLP model using PyTorch Lightning, allowing for flexible
    architecture with varying hidden layer sizes and dropout probabilities.

    Parameters
    ----------
    in_dim : int, optional
        Dimensionality of input data, by default 100
    out_dim : int, optional
        Dimensionality of output data, by default 2
    hidden_dims : List[Union[int, float]], optional
        List containing architectural parameters of the model. If an element is
        an int, it represents a hidden layer of that size. If an element is a float,
        it represents a dropout layer with that probability. By default ()
    use_bias : bool, optional
        Whether to use bias in all linear layers, by default True
    args : Optional[Dict], optional
        Dictionary containing the hyperparameters of the model, by default None

    Attributes
    ----------
    main : nn.Sequential
        The main sequential container of the MLP layers
    """

    def __init__(
        self,
        in_dim: int = 100,
        out_dim: int = 2,
        hidden_dims: List[Union[int, float]] = (),
        use_bias: bool = True,
        args: Optional[Dict] = None,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.args = args if args is not None else {}
        activations = (
            nn.CELU
            if self.args.get("activations") is None
            else self.args["activations"]
        )

        layers = self._build_layers(in_dim, out_dim, hidden_dims, use_bias, activations)
        self.main = nn.Sequential(*layers)
        self._init_params()

    def _build_layers(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dims: List[Union[int, float]],
        use_bias: bool,
        activations: nn.Module,
    ) -> List[nn.Module]:
        """
        Build the layers of the MLP.

        Parameters
        ----------
        in_dim : int
            Dimensionality of input data
        out_dim : int
            Dimensionality of output data
        hidden_dims : List[Union[int, float]]
            List of hidden layer sizes and dropout probabilities
        use_bias : bool
            Whether to use bias in linear layers
        activations : nn.Module
            Activation function to use

        Returns
        -------
        List[nn.Module]
            List of layers for the MLP
        """
        if len(hidden_dims) == 0:
            return [nn.Linear(in_dim, out_dim, bias=use_bias)]

        layers = []
        hidden_dims = [in_dim] + hidden_dims

        for i, hidden_dim in enumerate(hidden_dims[:-1]):
            if isinstance(hidden_dim, float):
                continue
            if isinstance(hidden_dims[i + 1], float):
                layers.extend(
                    [
                        nn.Linear(hidden_dim, hidden_dims[i + 2], bias=use_bias),
                        nn.Dropout(p=hidden_dims[i + 1]),
                        activations() if i < len(hidden_dims) - 1 else nn.Tanh(),
                    ]
                )
            else:
                layers.extend(
                    [
                        nn.Linear(hidden_dim, hidden_dims[i + 1], bias=use_bias),
                        activations() if i < len(hidden_dims) - 1 else nn.Tanh(),
                    ]
                )

        layers.append(nn.Linear(hidden_dims[-1], out_dim, bias=use_bias))
        if self.args.get("clf", False):
            layers.append(nn.LogSoftmax(dim=1))

        return layers

    def _init_params(self) -> None:
        """Initialize the parameters of the model."""

        def init_params(m: nn.Module) -> None:
            if isinstance(m, nn.Linear):
                torch.nn.init.kaiming_uniform_(m.weight, nonlinearity="leaky_relu")
                if m.bias is not None:
                    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
                    bound = 1 / torch.math.sqrt(fan_in)
                    nn.init.uniform_(m.bias, -bound, bound)  # LeCunn init

        self.main.apply(init_params)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Defines the network structure and flow from input to output.

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
        loss = self.args["criterion"](outs, ys)
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
        self.log("train_loss", loss)
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
        self.log("val_loss", loss)
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
        self.log("test_loss", loss)
        return loss

    def configure_optimizers(self):
        """
        Configure optimizers and learning rate schedulers.

        Returns
        -------
        tuple
            Tuple containing a list of optimizers and a list of scheduler configurations
        """
        optimizer = torch.optim.AdamW(
            self.parameters(),
            weight_decay=self.args["weight_decay"],
            betas=(0.9, 0.999),
            amsgrad=True,
        )
        scheduler = torch.optim.lr_scheduler.CyclicLR(
            optimizer,
            base_lr=self.args["base_lr"],
            max_lr=self.args["lr"],
            step_size_up=self.args["scheduler_step_size_multiplier"]
            * self.args["num_training_batches"],
            cycle_momentum=False,
            mode="triangular2",
            gamma=0.99994,
            last_epoch=-1,
        )
        lr_scheduler = {"scheduler": scheduler, "interval": "step"}
        return [optimizer], [lr_scheduler]
