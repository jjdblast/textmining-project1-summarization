import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F

## Move models to GPU
USE_CUDA = True

from preprocess import *

## Encoding Layer
class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers=1):
        super(EncoderRNN, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers)
        
    def forward(self, word_inputs, hidden):
        # Note: we run this all at once (over the whole input sequence)
        seq_len = len(word_inputs)
        embedded = self.embedding(word_inputs).view(seq_len, 1, -1)
        output, hidden = self.gru(embedded, hidden)
        return output, hidden

    def init_hidden(self):
        hidden = Variable(torch.zeros(self.n_layers, 1, self.hidden_size))
        if USE_CUDA: hidden = hidden.cuda()
        return hidden
    
## Attention Mech
class Attn(nn.Module):
    def __init__(self, method, hidden_size):
        super().__init__()
        
        self.method = method
        self.hidden_size = hidden_size
        
        if method == 'general':
            self.attn = nn.Linear(hidden_size, hidden_size)
        elif method == 'concat':
            self.attn = nn.Linear(hidden_size*2, hidden_size)
            self.other = nn.Parameter(torch.FloatTensor(hidden_size))
    
    def forward(self, hidden, encoder_outputs):
        seq_len = len(encoder_outputs)
        attn_energies = Variable(torch.zeros(seq_len)).cuda()
        for i in range(seq_len):
            attn_energies[i] = self.score(hidden, encoder_outputs[i])
        return F.softmax(attn_energies).unsqueeze(0).unsqueeze(0)
    
    def score(self, hidden, encoder_output):
        if self.method == 'dot':
            return hidden.squeeze().dot(encoder_output.squeeze())
        if self.method == 'general':
            energy = self.attn(encoder_output)
            energy = hidden.squeeze().dot(energy.squeeze())
            return  energy
        if self.method == 'concat':
            energy = self.attn(torch.cat(hidden, encoder_output), 1)
            energy = self.other.dot(energy.squeeze())
            return energy

## Decoding Layer with Attention Mech
class AttnDecoderRNN(nn.Module):
    def __init__(self, attn_mode, hidden_size, output_size, n_layers=1, dropout=0.1):
        super().__init__()
        # Parameters
        self.attn_mode = attn_mode
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout = dropout
        
        # Layers
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size * 2, hidden_size, n_layers, dropout=dropout)
        self.out = nn.Linear(hidden_size * 2, output_size)
        
        #Attn mode
        if attn_mode != None:
            self.attn = Attn(attn_mode, hidden_size)
            
    def forward(self, word_input, last_hidden, last_context, encoder_outputs):
        word_embeded = self.embedding(word_input).view(1, 1, -1)
        
        rnn_input = torch.cat((word_embeded, last_context.unsqueeze(0)), 2)
        rnn_output, hidden = self.gru(rnn_input, last_hidden)
        
        attn_weight = self.attn(rnn_output.squeeze(0), encoder_outputs)
        context = attn_weight.bmm(encoder_outputs.transpose(0, 1))
        
        rnn_output = rnn_output.squeeze(0)
        context = context.squeeze(1)
        output = F.log_softmax(self.out(torch.cat((rnn_output, context), 1)))
        return output, context, hidden, attn_weight

## Function to test model structure on GPU
def test_model():
    encoder_test = EncoderRNN(10, 10, 2).cuda()
    decoder_test = AttnDecoderRNN('general', 10, 10, 2).cuda()
    print(encoder_test)
    print(decoder_test)

    encoder_hidden = encoder_test.init_hidden()
    word_input =  Variable(torch.LongTensor([1, 2, 3])).cuda()
    encoder_outputs, encoder_hidden = encoder_test(word_input, encoder_hidden)

    decoder_attns = torch.zeros(1, 3, 3)
    decoder_hidden = encoder_hidden
    decoder_context = Variable(torch.zeros(1, decoder_test.hidden_size)).cuda()

    for i in range(3):
        decoder_output, decoder_context, decoder_hidden, decoder_attn = decoder_test(word_input[i], decoder_hidden,
                                                                                   decoder_context, encoder_outputs)
        print(decoder_output.size(), decoder_hidden.size(), decoder_attn.size())
        decoder_attns[0, i] = decoder_attn.squeeze(0).cpu().data

## Function to get encode-decoder model
def get_model(voc):
    ## Model parameters
    attn_model = 'general'
    hidden_size = 500
    n_layers = 2
    dropout_p = 0.05
    ## Initialize models
    encoder = EncoderRNN(voc.n_words, hidden_size, n_layers)
    decoder = AttnDecoderRNN(attn_model, hidden_size, voc.n_words, n_layers, dropout=dropout_p)
    return encoder,decoder