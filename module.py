import torch

from torch import nn
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence


class LSTMEncoder(nn.Module):
    def __init__(self, input_size, rnn_size, num_layers, dropout,
                 bidirectional, batch_first=True):
        super(LSTMEncoder, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size,
                            hidden_size=rnn_size,
                            num_layers=num_layers,
                            dropout=dropout,
                            bidirectional=bidirectional,
                            batch_first=True)
        self.dropout = nn.Dropout(dropout)  # ???
        self.feature_size = rnn_size
        if bidirectional:
            self.feature_size *= 2
        self.batch_first = batch_first
        self.num_layers = num_layers

    @staticmethod
    def last_by_index(outputs, lengths):
        idx = (lengths - 1).view(-1, 1).expand(outputs.size(0),
                                               outputs.size(2)).unsqueeze(1)
        return outputs.gather(1, idx).squeeze()

    @staticmethod
    def split_directions(outputs):
        direction_size = int(outputs.size(-1) / 2)
        forward = outputs[:, :, :direction_size]
        backward = outputs[:, :, direction_size:]
        return forward, backward

    def last_timestep(self, outputs, lengths, bi=False):
        if bi:
            forward, backward = self.split_directions(outputs)
            last_forward = self.last_by_index(forward, lengths)
            last_backward = backward[:, 0, :]
            return torch.cat((last_forward, last_backward), dim=-1)

        return self.last_by_index(outputs, lengths)

    def forward(self, x, lengths):
        packed = pack_padded_sequence(x, list(lengths.data),
                                      batch_first=self.batch_first)
        out_packed, _ = self.lstm(packed)
        outputs, _ = pad_packed_sequence(out_packed,
                                         batch_first=self.batch_first)
        last_outputs = self.last_timestep(outputs, lengths,
                                          self.lstm.bidirectional)

        if self.num_layers < 2:
            last_outputs = self.dropout(last_outputs)
        return outputs, last_outputs

class GenderLoss(nn.Module):
    def __init__(self):
        super(GenderLoss, self).__init__()

    def forward(self, y_pred, y_true, y_counts):
        error = torch.abs(y_pred - y_true)
        error = torch.mul(error, y_counts)
        penalty = torch.where(y_pred > y_true, 2 * error, error)
        return penalty.mean()