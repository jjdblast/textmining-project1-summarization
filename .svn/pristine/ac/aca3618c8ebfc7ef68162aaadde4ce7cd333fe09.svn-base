import random

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
## Show attention
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
#%matplotlib inline

## Move models to GPU
USE_CUDA = True

from preprocess import *

## Function to evaluate training result
def evaluate(encoder, decoder, voc, sentence, max_length=200):
    input_variable = variable_from_sentence(voc, sentence)
    input_length = input_variable.size()[0]
    
    # Run through encoder
    encoder_hidden = encoder.init_hidden()
    encoder_outputs, encoder_hidden = encoder(input_variable, encoder_hidden)

    # Create starting vectors for decoder
    decoder_input = Variable(torch.LongTensor([[SOS_token]])) # SOS
    decoder_context = Variable(torch.zeros(1, decoder.hidden_size))
    if USE_CUDA:
        decoder_input = decoder_input.cuda()
        decoder_context = decoder_context.cuda()

    decoder_hidden = encoder_hidden
    
    decoded_words = []
    decoder_attentions = torch.zeros(max_length, max_length)
    
    # Run through decoder
    for di in range(max_length):
        decoder_output, decoder_context, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, decoder_context, encoder_outputs)
        decoder_attentions[di,:decoder_attention.size(2)] += decoder_attention.squeeze(0).squeeze(0).cpu().data

        # Choose top word from output
        topv, topi = decoder_output.data.topk(1)
        ni = topi[0][0]
        if ni == EOS_token:
            decoded_words.append('<EOS>')
            break
        else:
            decoded_words.append(voc.index2word[ni])
            
        # Next input is chosen word
        decoder_input = Variable(torch.LongTensor([[ni]]))
        if USE_CUDA: decoder_input = decoder_input.cuda()
    
    return decoded_words, decoder_attentions[:di+1, :len(encoder_outputs)]

def evaluate_randomly(encoder, decoder, voc, pairs):
    pair = random.choice(pairs)
    output_words, decoder_attn = evaluate(encoder, decoder, voc, pair[0])
    output_sentence = ' '.join(output_words)
    
    print('>', pair[0])
    print('=', pair[1])
    print('<', output_sentence)
    print('')

    
## Function to show attention
def show_attention(input_sentence, output_words, attentions):
    # Set up figure with colorbar
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(attentions.numpy(), cmap='bone')
    fig.colorbar(cax)

    # Set up axes
    ax.set_xticklabels([''] + input_sentence.split(' ') + ['<EOS>'], rotation=90)
    ax.set_yticklabels([''] + output_words)

    # Show label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.show()
    plt.close()

def evaluate_and_show_attention(encoder, decoder, voc, input_sentence):
    output_words, attentions = evaluate(encoder, decoder, voc, input_sentence)
    print('input =', input_sentence)
    print('output =', ' '.join(output_words))
    show_attention(input_sentence, output_words, attentions)
    
    
def generate_result(encoder, decoder, voc, reference_file, result_file):
    ## reference_file is the extractive output file with human refrence summary
    ## reference_file = './data/extractive_ouput/test_output.txt'
    ## result_file is the location which store the model generated result
    ## result_file = './data/abstractive_output/test_result.txt'
    with open(reference_file, encoding='utf8') as ref_file:
        lines = ref_file.read().strip().split('\n') 
        ## save the abstractive model output summary
        with open(result_file, 'w+') as res_file:
            for line in lines:
                ## remove "</s><s> " with ""         
                res_input = re.sub(r"</s><s> ", "", line.split('\t')[0])
                res_output = ' '.join(evaluate(encoder,decoder,voc,res_input)[0])
                ref_output = '\t'.join([ref for ref in line.split('\t')[1:]])
                ## 
                result_line = '\t'.join([res_output, ref_output])
                res_file.write(result_line)
                res_file.write('\n')