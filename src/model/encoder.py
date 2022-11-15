import torch.nn as nn
import torch


class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, seq_size, batch_size) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.seq_size = seq_size
        self.batch_size = batch_size
        self.blstm = nn.LSTM(input_size, hidden_size, bidirectional=True)
        self.linear = nn.Linear(input_size, 256)

    def forward(self, batch, hidden=None, linear=False):
        preprocess = (
            torch.transpose(torch.transpose(torch.flatten(batch, -2, -1), -2, -1), 0, 1)
            if batch.shape != (self.seq_size, self.batch_size, self.input_size)
            else batch
        )
        output, (hidden, _) = (
            self.blstm(preprocess, hidden) if hidden else self.blstm(preprocess)
        )
        if linear:
            output, hidden = self.linear(output), self.linear(
                hidden.transpose(0, 1).flatten(-2, -1)
            )
        return output, hidden
