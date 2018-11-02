import torch
import torch.nn as nn

class customRNN(nn.Module):
    """
        Custom RNN : Used to initialize one layer LSTMs with different sizes
                     initHidden() initializes the hidden_state, and the cell_state of the LSTM
    """
    def __init__(self, input_size, hidden_size, isbidirectional, device):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.device = device
        self.isbidirectional = isbidirectional
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True, bidirectional=isbidirectional)

    def forward(self, input, hidden_state, cell_state):
        output, (hidden_state, cell_state) = self.lstm(input, (hidden_state,cell_state))
        return output, hidden_state, cell_state

    def initHidden(self):
        if self.isbidirectional:
            hidden_state = torch.zeros([2, self.hidden_size], device=self.device)
            cell_state = torch.zeros([2, self.hidden_size], device=self.device)
        else:
            hidden_state = torch.zeros([1, self.hidden_size], device=self.device)
            cell_state = torch.zeros([1, self.hidden_size], device=self.device)
        return hidden_state, cell_state
