"""
ConvLSTM implementation taken from https://github.com/ndrplz/ConvLSTM_pytorch/blob/master/convlstm.py
"""

from __future__ import annotations

import lightning as pl
import torch
import torch.nn as nn
import torchmetrics
import torchmetrics.regression
from torch.utils.data import DataLoader, random_split

from xai_crop_yield.config import DEVICE, RAW_DATA_DIR
from xai_crop_yield.dataset import SustainBenchCropYieldTimeseries


class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, bias):
        """
        Initialize ConvLSTM cell.

        Parameters
        ----------
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        kernel_size: (int, int)
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.
        """

        super(ConvLSTMCell, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias

        self.conv = nn.Conv2d(
            in_channels=self.input_dim + self.hidden_dim,
            out_channels=4 * self.hidden_dim,
            kernel_size=self.kernel_size,
            padding=self.padding,
            bias=self.bias,
            device=DEVICE,
        )

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state

        combined = torch.cat(
            [input_tensor, h_cur], dim=1
        )  # concatenate along channel axis

        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(
            combined_conv, self.hidden_dim, dim=1
        )
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        return (
            torch.zeros(
                batch_size,
                self.hidden_dim,
                height,
                width,
                device=self.conv.weight.device,
            ),
            torch.zeros(
                batch_size,
                self.hidden_dim,
                height,
                width,
                device=self.conv.weight.device,
            ),
        )


class ConvLSTM(nn.Module):
    """

    Parameters:
        input_dim: Number of channels in input
        hidden_dim: Number of hidden channels
        kernel_size: Size of kernel in convolutions
        num_layers: Number of LSTM layers stacked on each other
        batch_first: Whether or not dimension 0 is the batch or not
        bias: Bias or no bias in Convolution
        return_all_layers: Return the list of computations for all layers
        Note: Will do same padding.

    Input:
        A tensor of size B, T, C, H, W or T, B, C, H, W
    Output:
        A tuple of two lists of length num_layers (or length 1 if return_all_layers is False).
            0 - layer_output_list is the list of lists of length T of each output
            1 - last_state_list is the list of last states
                    each element of the list is a tuple (h, c) for hidden state and memory
    Example:
        >> x = torch.rand((32, 10, 64, 128, 128))
        >> convlstm = ConvLSTM(64, 16, 3, 1, True, True, False)
        >> _, last_states = convlstm(x)
        >> h = last_states[0][0]  # 0 for layer index, 0 for h index
    """

    def __init__(
        self,
        input_dim,
        hidden_dim,
        kernel_size,
        num_layers,
        batch_first=False,
        bias=True,
        return_all_layers=False,
    ):
        super(ConvLSTM, self).__init__()

        self._check_kernel_size_consistency(kernel_size)

        # Make sure that both `kernel_size` and `hidden_dim` are lists having len == num_layers
        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        hidden_dim = self._extend_for_multilayer(hidden_dim, num_layers)
        if not len(kernel_size) == len(hidden_dim) == num_layers:
            raise ValueError('Inconsistent list length.')

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        self.return_all_layers = return_all_layers

        cell_list = []
        for i in range(0, self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i - 1]

            cell_list.append(
                ConvLSTMCell(
                    input_dim=cur_input_dim,
                    hidden_dim=self.hidden_dim[i],
                    kernel_size=self.kernel_size[i],
                    bias=self.bias,
                )
            )

        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, input_tensor, hidden_state=None):
        """

        Parameters
        ----------
        input_tensor: todo
            5-D Tensor either of shape (t, b, c, h, w) or (b, t, c, h, w)
        hidden_state: todo
            None. todo implement stateful

        Returns
        -------
        last_state_list, layer_output
        """
        if not self.batch_first:
            # (t, b, c, h, w) -> (b, t, c, h, w)
            input_tensor = input_tensor.permute(1, 0, 2, 3, 4)

        b, _, _, h, w = input_tensor.size()

        # Implement stateful ConvLSTM
        if hidden_state is not None:
            raise NotImplementedError()
        else:
            # Since the init is done in forward. Can send image size here
            hidden_state = self._init_hidden(batch_size=b, image_size=(h, w))

        layer_output_list = []
        last_state_list = []

        seq_len = input_tensor.size(1)
        cur_layer_input = input_tensor

        for layer_idx in range(self.num_layers):
            h, c = hidden_state[layer_idx]
            output_inner = []
            for t in range(seq_len):
                h, c = self.cell_list[layer_idx](
                    input_tensor=cur_layer_input[:, t, :, :, :],
                    cur_state=[h, c],
                )
                output_inner.append(h)

            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output

            layer_output_list.append(layer_output)
            last_state_list.append([h, c])

        if not self.return_all_layers:
            layer_output_list = layer_output_list[-1:]
            last_state_list = last_state_list[-1:]

        return layer_output_list, last_state_list

    def _init_hidden(self, batch_size, image_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(
                self.cell_list[i].init_hidden(batch_size, image_size)
            )
        return init_states

    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        if not (
            isinstance(kernel_size, tuple)
            or (
                isinstance(kernel_size, list)
                and all([isinstance(elem, tuple) for elem in kernel_size])
            )
        ):
            raise ValueError('`kernel_size` must be tuple or list of tuples')

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param


class ConvLSTMRegressor(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        kernel_size,
        num_layers,
        batch_first=False,
        bias=True,
        return_all_layers=False,
    ):
        super().__init__()

        self.convlstm = ConvLSTM(
            input_dim,
            hidden_dim,
            kernel_size,
            num_layers,
            batch_first,
            bias,
            return_all_layers,
        )
        self.conv = nn.Conv2d(
            hidden_dim[-1],
            128,
            kernel_size=(3, 3),
            bias=bias,
            device=DEVICE,
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(self.conv.out_channels, 1, device=DEVICE)

    def forward(self, input_tensor: torch.Tensor):
        layer_output, last_states = self.convlstm(input_tensor)
        h, c = last_states[-1]
        h = self.conv(h)
        pooled = self.pool(h).view(h.size(0), -1)
        output = self.fc(pooled)
        return (output, layer_output, last_states)


class RegressionMetricCollection(torchmetrics.MetricCollection):
    def __init__(self, prefix: str | None = None):
        super().__init__(
            {
                'r2': torchmetrics.regression.R2Score(),
                'mse': torchmetrics.regression.MeanSquaredError(),
                'mae': torchmetrics.regression.MeanAbsoluteError(),
            }
        )


class ConvLSTMModel(pl.LightningModule):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        self.criterion = nn.MSELoss()
        self.train_metrics = RegressionMetricCollection(prefix='train_')
        self.val_metrics = self.train_metrics.clone(prefix='val_')
        self.test_metrics = self.train_metrics.clone(prefix='test_')

    def forward(self, x):
        return self.model(x)[0].view(-1)

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        logits = self.forward(x)
        self.train_metrics.update(logits, y)
        loss = self.criterion(logits, y.float())
        self.log('train_loss', loss, sync_dist=True)
        return loss

    def on_train_epoch_end(self):
        self.log_dict(
            self.train_metrics.compute(), sync_dist=True, prog_bar=True
        )
        self.train_metrics.reset()

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        logits = self.forward(x)
        self.val_metrics.update(logits, y)
        loss = self.criterion(logits, y.float())
        self.log('val_loss', loss, sync_dist=True, prog_bar=True)
        return loss

    def test_step(self, test_batch, batch_idx):
        x, y = test_batch
        logits = self.forward(x)
        self.test_metrics.update(logits, y)
        loss = self.criterion(logits, y.float())
        self.log_dict(self.test_metrics.compute(), sync_dist=True)
        self.log('test_loss', loss, sync_dist=True)

    def on_validation_epoch_end(self):
        self.log_dict(self.val_metrics.compute(), sync_dist=True, prog_bar=True)
        self.val_metrics.reset()

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=0.01)


if __name__ == '__main__':
    dataset = SustainBenchCropYieldTimeseries(
        RAW_DATA_DIR, country='usa', years=list(range(2005, 2016))
    )
    dataset._load()
    train, test, val = random_split(dataset, (0.8, 0.1, 0.1))
    train_dataloader = DataLoader(train, batch_size=32, shuffle=True)
    val_dataloader = DataLoader(val, batch_size=16, shuffle=False)
    test_dataloader = DataLoader(test, batch_size=16, shuffle=False)

    convlstm = ConvLSTMRegressor(
        input_dim=9,
        hidden_dim=[10],
        kernel_size=(3, 3),
        num_layers=1,
        batch_first=True,
        bias=True,
        return_all_layers=True,
    )
    model = ConvLSTMModel(convlstm)
    trainer = pl.Trainer(
        accelerator='gpu',
        max_epochs=200,
        enable_model_summary=True,
        enable_progress_bar=True,
        log_every_n_steps=1,
    )
    trainer.fit(
        model,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
    )
    trainer.test(model, dataloaders=test_dataloader)
