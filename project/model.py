import torch.nn as nn


class LinearFF(nn.Module):
    def __init__(self, input_dim, seq_length, hidden_dim, output_dim=1):
        super(LinearFF, self).__init__()
        self.fc1 = nn.Linear(input_dim*seq_length, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # (batch size x seq length x input dim)
        x = x.view(x.shape[0], -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x[:, 0]


class FF(nn.Module):
    def __init__(self, input_dim, seq_length, hidden_dim, output_dim=1):
        super(FF, self).__init__()
        self.fc1 = nn.Linear(input_dim*seq_length, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        # (batch size x seq length x input dim)
        x = x.view(x.shape[0], -1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x[:, 0]


class RNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(RNN, self).__init__()
        self.rnn = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=0.0,
            batch_first=True,
        )
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.rnn(x)[0]
        x = x[:, -1, :]
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x[:, 0]


class TemporalConvolution(nn.Module):

    def __init__(self, input_dim, hidden_dim):
        super(TemporalConvolution, self).__init__()
        self.conv1d = nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=1)

    def forward(self, x):
        # Permute the input for 1D convolution
        x = x.permute(1, 2, 0)
        conv_out = self.conv1d(x)
        # Revert the permutation
        conv_out = conv_out.permute(2, 0, 1)
        return conv_out


class Transformer(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, num_heads, seq_len, output_dim=1):
        super(Transformer, self).__init__()
        self.embedding = nn.Linear(input_dim, hidden_dim)
        self.encoder_layers = nn.TransformerEncoderLayer(hidden_dim, num_heads, dropout=0.5)
        self.encoder = nn.TransformerEncoder(self.encoder_layers, num_layers)
        self.temporal_conv = TemporalConvolution(hidden_dim, hidden_dim)
        self.fc1 = nn.Linear(hidden_dim*seq_len, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.embedding(x)
        x = self.temporal_conv(x)
        x = x.permute(1, 0, 2)
        encoder_output = self.encoder(x)
        encoder_output = encoder_output.permute(1, 0, 2)
        encoder_output = encoder_output.reshape(encoder_output.shape[0], -1)
        x = self.relu(self.fc1(encoder_output))
        x = self.fc2(x)
        return encoder_output[:, 0]