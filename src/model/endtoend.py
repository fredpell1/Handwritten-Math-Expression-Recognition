import torch

class HME2LaTeX(torch.nn.Module):

    def __init__(self, cnn, encoder, decoder, seq_size, batch_size, language_size, EOS_TOKEN, SOS_token, max_sequence_size, device)-> None:
        super().__init__()
        self.cnn = cnn
        self.encoder = encoder
        self.decoder = decoder
        self.seq_size = seq_size
        self.batch_size = batch_size
        self.language_size = language_size
        self.EOS_token = EOS_TOKEN
        self.SOS_token = SOS_token
        self.max_sequence_size = max_sequence_size
        self.device = device


    def forward(self,images,labels):
        out = self.cnn(images)
        o,h,c = self.encoder(out, None, False)
        init_hidden = torch.cat((h[0], h[1]), -1).unsqueeze(0).to(self.device)
        init_c = torch.cat((c[0], c[1]), -1).unsqueeze(0).to(self.device)
        probs = torch.zeros((self.seq_size,self.batch_size,self.language_size)).to(self.device)
        #training time
        if labels is not None: 
            out,hidden,probs[0] = self.decoder(labels[0].unsqueeze(0), (init_hidden, init_c), o)
            for i in range(1, self.seq_size):
                out,hidden,prob = self.decoder(labels[i].unsqueeze(0), hidden, o)
                probs[i,:,:] += prob[0]
        #testing time
        else:
            SOS_tensor = torch.tensor(self.SOS_token, dtype=torch.float32).expand((1, self.batch_size, 1)).to(self.device)
            out,hidden,probs[0] = self.decoder(SOS_tensor,(init_hidden, init_c), o)
            for i in range(1, self.max_sequence_size):
                predicted_word = probs.topk(1)[1][0].unsqueeze(0).type(torch.float32)
                out,hidden,prob = self.decoder(predicted_word, hidden, o)
                probs[i,:,:] += prob[0]
        return probs
