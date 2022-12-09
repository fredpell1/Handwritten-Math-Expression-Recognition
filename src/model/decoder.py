import torch


class Decoder(torch.nn.Module):
    def __init__(
        self, input_size, hidden_size, output_size, num_features, batch_size, device
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_features = num_features
        self.batch_size = batch_size
        self.lstm = torch.nn.LSTM(self.input_size, self.hidden_size)
        self.Wout = torch.nn.Linear(self.hidden_size, self.output_size, bias=False)
        self.Wf = torch.nn.Linear(self.hidden_size, 1, bias=False)
        self.Wh = torch.nn.Linear(self.hidden_size, 1, bias=False)
        self.Wc = torch.nn.Linear(2 * self.hidden_size, self.hidden_size, bias=False)
        self.softmax_out = torch.nn.Softmax(dim=-1)
        self.softmax_alpha = torch.nn.Softmax(dim=0)
        self.device = device

    def forward(self, input, hidden, encoder_output):
        """Batched input but only one word at a time"""
        hn, cn = hidden
        
        #coefficients for attention
        alphas = self.softmax_alpha(
            torch.tanh(
                self.Wh(hn.expand(self.num_features, self.batch_size, self.hidden_size))
                + self.Wf(encoder_output)
            )
        ).to(self.device)
        
        #context vector
        c = torch.sum(alphas * encoder_output, 0, keepdim=True).to(self.device)
        new_h = torch.cat((hn, c), -1).to(self.device)
        new_h = torch.tanh(self.Wc(new_h)).to(self.device)
        output, (hidden, cn) = self.lstm(input, (new_h, cn))
        probability = self.Wout(output)
        return output, (new_h, cn), probability
