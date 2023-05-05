"""NMT.py

Bradley Selee
ECE8560
Project 3
April 28, 2023

References: https://github.com/JasonFengGit/Neural-Model-Translation

Dataset location (vietnamese): https://nlp.stanford.edu/projects/nmt/
"""
from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import cv2
import numpy as np
from pathlib import Path
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
import matplotlib.pyplot as plt 
#from tqdm import tqdm
import random
import unicodedata
import string
import time
import math
import sys

from utils import read_corpus, batch_iter
from vocab import Vocab
from network import NMT, Hypothesis

#from vocab import Vocab

import re
import math
# BLEU score library
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

#from torchtext.legacy.data import Field, BucketIterator


SOS_token = 0
EOS_token = 1
PAD_ID = 2
UNKNOWN_ID = 3
#BUCKETS = [(19, 19), (28, 28), (33, 33), (40, 43), (50, 53), (60, 63)]
MAX_LENGTH = 50

class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS", 2: "PAD", 3: "UNK"}
        self.n_words = 4  # Count SOS and EOS

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1


def filterPair(p):
    return len(p[0].split(' ')) < MAX_LENGTH and \
        len(p[1].split(' ')) < MAX_LENGTH


def filterPairs(pairs):
    return [pair for pair in pairs if filterPair(pair)]


def read_data(input_file, source, vocab_size=21000):
    
    input_lines = open(input_file, encoding='utf-8').read().strip().split('\n')
    
    data = []
    for sentence in input_lines:
        words = sentence.split()
        print(words)
        exit()
        if source == 'target':
            words = ['<s>'] + words + ['</s>']
        pairs.append(pair)

    return pairs, input_lang, output_lang


def _pad_input(input_, size):
    return input_ + [PAD_ID] * (size - len(input_))


class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, device):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output = embedded
        output, hidden = self.gru(output, hidden)
        return output, hidden

    def initHidden(self, device):
        return torch.zeros(1, 1, self.hidden_size, device=device)


class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size,device):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        output = self.embedding(input).view(1, 1, -1)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden

    def initHidden(self, device):
        return torch.zeros(1, 1, self.hidden_size, device=device)


class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.1, max_length=MAX_LENGTH):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)

        attn_weights = F.softmax(
            self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                                 encoder_outputs.unsqueeze(0))

        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        output = F.relu(output)
        output, hidden = self.gru(output, hidden)

        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden, attn_weights

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)



def train(train_src_file, train_tgt_file, test_src_file, test_tgt_file, device):

    # Hyperparameters
    train_batch_size = 32
    epochs = 10
    learning_rate = 0.001
    valid_niter = 200#2000
    dropout = 0.3
    clip_grad = 5.0
    log_every = 10
    max_patience = 5
    lr_decay = 0.5
    max_num_trial = 5
    model_save_path = 'model/model.ckpt'
    Path('model/').mkdir(parents=True, exist_ok=True)
    
    train_data_src = read_corpus(train_src_file, source='source')
    train_data_tgt = read_corpus(train_tgt_file, source='target')
    
    test_data_src = read_corpus(test_src_file, source='source')
    test_data_tgt = read_corpus(test_tgt_file, source='target')

    train_data = list(zip(train_data_src, train_data_tgt))
    test_data = list(zip(test_data_src, test_data_tgt))

    vocab = Vocab.build(train_data_src, train_data_tgt)

    model = NMT(embed_size=512,
                hidden_size=512,
                dropout_rate=dropout,
                vocab=vocab,
                rnn_layer=nn.LSTM,
                bidirectional=True)

    model.train()

    uniform_init = 0.1
    if np.abs(uniform_init) > 0.:
        #print('uniformly initialize parameters [-%f, +%f]' % (uniform_init, uniform_init))
        for p in model.parameters():
            p.data.uniform_(-uniform_init, uniform_init)
    
    vocab_mask = torch.ones(len(vocab.tgt))
    vocab_mask[vocab.tgt['<pad>']] = 0

    model = model.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    num_trial = 0
    train_iter = patience = cum_loss = report_loss = cum_tgt_words = report_tgt_words = 0
    cum_examples = report_examples = epoch = valid_num = 0
    hist_valid_scores = []
    train_time = begin_time = time.time()
    #print('begin Maximum Likelihood training')
    train_ppl_log = open("ppl.log", "w")
    dev_ppl_log = open("dev_ppl.log", "w")
    
    while True:
        epoch +=1

        for src_sents, tgt_sents in batch_iter(train_data, batch_size=train_batch_size, shuffle=True):
            train_iter += 1
            
            optimizer.zero_grad()

            batch_size = len(src_sents)

            example_losses = -model(src_sents, tgt_sents) # (batch size,)
            batch_loss = example_losses.sum()
            loss = batch_loss / batch_size

            loss.backward()

            # clip gradient
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)

            optimizer.step()

            batch_losses_val = batch_loss.item()
            report_loss += batch_losses_val
            cum_loss += batch_losses_val

            tgt_words_num_to_predict = sum(len(s[1:]) for s in tgt_sents) # omitting leading `<s>`
            report_tgt_words += tgt_words_num_to_predict
            cum_tgt_words += tgt_words_num_to_predict
            report_examples += batch_size
            cum_examples += batch_size

            if train_iter % log_every == 0:
                # print('epoch %d, iter %d, avg. loss %.2f, avg. ppl %.2f ' \
                #       'cum. examples %d, speed %.2f words/sec, time elapsed %.2f sec' % (epoch, train_iter,
                #                                                                          report_loss / report_examples,
                #                                                                          math.exp(report_loss / report_tgt_words),
                #                                                                          cum_examples,
                #                                                                          report_tgt_words / (time.time() - train_time),
                #                                                                          time.time() - begin_time), file=sys.stderr)
                print(f'Epoch: {epoch} \t Loss {cum_examples}: {report_loss/report_examples}')
                train_ppl_log.write("{} {}\n".format(train_iter, math.exp(report_loss / report_tgt_words)))
                train_time = time.time()
                report_loss = report_tgt_words = report_examples = 0.
            
            # perform validation
            if train_iter % valid_niter == 0:
                print('epoch %d, iter %d, cum. loss %.2f, cum. ppl %.2f cum. examples %d' % (epoch, train_iter,
                                                                                         cum_loss / cum_examples,
                                                                                         np.exp(cum_loss / cum_tgt_words),
                                                                                         cum_examples), file=sys.stderr)
                cum_loss = cum_examples = cum_tgt_words = 0.
                valid_num += 1

                #print('\nValidation ...', file=sys.stderr)

                # compute dev. ppl and bleu
                dev_ppl = evaluate_ppl(model, test_data, batch_size=128)   # dev batch size can be a bit larger
                valid_metric = -dev_ppl

                #print('validation: iter %d, dev. ppl %f' % (train_iter, dev_ppl), file=sys.stderr)
                dev_ppl_log.write("{} {}\n".format(train_iter, dev_ppl))
                is_better = len(hist_valid_scores) == 0 or valid_metric > max(hist_valid_scores)
                hist_valid_scores.append(valid_metric)

                if is_better:
                    patience = 0
                    print('save currently the best model to [%s]' % model_save_path, file=sys.stderr)
                    model.save(model_save_path)

                    # also save the optimizers' state
                    torch.save(optimizer.state_dict(), model_save_path + '.optim')
                elif patience < max_patience:
                    patience += 1
                    print('hit patience %d' % patience, file=sys.stderr)

                    if patience == max_patience:
                        num_trial += 1
                        print('hit #%d trial' % num_trial, file=sys.stderr)
                        if num_trial == max_num_trial:
                            print('early stop!', file=sys.stderr)
                            exit(0)

                        # decay lr, and restore from previously best checkpoint
                        lr = optimizer.param_groups[0]['lr'] * lr_decay
                        print('load previously best model and decay learning rate to %f' % lr, file=sys.stderr)

                        # load model
                        params = torch.load(model_save_path, map_location=lambda storage, loc: storage)
                        model.load_state_dict(params['state_dict'])
                        model = model.to(device)

                        print('restore parameters of the optimizers', file=sys.stderr)
                        optimizer.load_state_dict(torch.load(model_save_path + '.optim'))

                        # set new lr
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = lr

                        # reset patience
                        patience = 0

                if epoch == epochs:
                    print('reached maximum number of epochs!', file=sys.stderr)
                    exit(0)


def beam_search(model, test_data_src, beam_size, max_decoding_time_step):
    """Run beam search to construct hypotheses for a list of src-language sentences"""
    was_training = model.training
    model.eval()

    hypotheses = []
    with torch.no_grad():
        for index, src_sent in enumerate(test_data_src):
            example_hyps = model.beam_search(src_sent, beam_size=beam_size, max_decoding_time_step=max_decoding_time_step)
            hypotheses.append(example_hyps)

    if was_training: 
        model.train(was_training)

    return hypotheses


def compute_corpus_level_bleu_score(references, hypotheses):
    """Given decoding results and reference sentences, compute corpus-level BLEU score"""
    chencherry = SmoothingFunction()
    # remove the start and end tokens
    if references[0][0] == '<s>':
        references = [ref[1:-1] for ref in references]

    running_bleu = 0
    for hypothesis, reference in zip(hypotheses, references):
        #print(f'gt: {reference}')
        #print(f'prediction: {hypothesis.value}\n')
        print(' '.join(hypothesis.value))
        running_bleu += sentence_bleu([reference], hypothesis.value, smoothing_function=chencherry.method1)
    print(f'\naverage bleu: {(running_bleu/len(references))*100:.2f}%')

    return (running_bleu/len(references))*100

def evaluate_ppl(model, test_data, batch_size=32):
    """ Evaluate perplexity on dev sentences"""
    was_training = model.training
    model.eval()

    cum_loss = 0.
    cum_tgt_words = 0.

    with torch.no_grad():
        for src_sents, tgt_sents in batch_iter(test_data, batch_size):
            loss = -model(src_sents, tgt_sents).sum()

            cum_loss += loss.item()
            tgt_word_num_to_predict = sum(len(s[1:]) for s in tgt_sents)  # omitting leading `<s>`
            cum_tgt_words += tgt_word_num_to_predict

        ppl = np.exp(cum_loss / cum_tgt_words)

    if was_training:
        model.train()

    return ppl 


def test(test_src_file, test_tgt_file, device):
    """ Performs decoding on a test set, and save the best-scoring decoding results.
    If the target gold-standard sentences are given, the function also computes
    corpus-level BLEU score.
    """

    batch_size = 1
    model_save_path = 'model/model.ckpt'

    test_data_src = read_corpus(test_src_file, source='source')
    test_data_tgt = read_corpus(test_tgt_file, source='target')

    test_data = list(zip(test_data_src, test_data_tgt))

    print("load model from {}".format(model_save_path), file=sys.stderr)
    print('Testing. This takes about 90 seconds on the tst20212.vi...')
    model = NMT.load(model_save_path)
    model = model.to(device)

    hypotheses = beam_search(model, test_data_src,
                             beam_size=10,
                             max_decoding_time_step=70)

    #if args['TEST_TARGET_FILE']:
    if True:
        top_hypotheses = [hyps[0] for hyps in hypotheses]
        bleu_score = compute_corpus_level_bleu_score(test_data_tgt, top_hypotheses)



def translate(user_input, device):
    """Translate a user input to another language"""

    batch_size = 1
    model_save_path = 'model/model.ckpt'

    test_data_src = read_corpus(user_input, source='translate')

    test_data = list(test_data_src)

    model = NMT.load(model_save_path)
    model = model.to(device)

    hypotheses = beam_search(model, test_data_src,
                             beam_size=10,
                             max_decoding_time_step=70)

    #if args['TEST_TARGET_FILE']:
    if True:
        top_hypotheses = [hyps[0] for hyps in hypotheses]
        print(' '.join(top_hypotheses[0].value))



def inference(model, device, image, classes, args):
    """A single forward pass of our trained model used for inferencing a single image"""
    model.eval()
    with torch.no_grad():
        image = image.to(device)
        output = model(image, args)
        prediction = torch.argmax(output)
        print(f'prediction result: {classes[prediction]}')


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('mode', type=str,
                        help='train or evaluate the model (train or eval)')
    parser.add_argument('image', type=str, nargs='?',
                        help='path to .png image for inference (32x32x3)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=True,
                        help='For Saving the current Model')
    args = parser.parse_args()
    use_cuda = 'cuda' if torch.cuda.is_available() else 'cpu'

    #torch.manual_seed(args.seed)

    # Assign cuda specific parameters if a GPU is available
    if use_cuda:
        device = torch.device("cuda")
        #print("Using GPU")
    else:
        device = torch.device("cpu")

    #print('Reading train files...')

    input_file_test = 'tst2012.en'
    label_file_test = 'tst2012.vi'
    # Loop through every epoch and train our model, evaluating the test accuracy on every epoch
    if args.mode == 'train':
        input_file = 'train.en'
        input_vocab ='vocab.en'
        label_file = 'train.vi'
        train(input_file, label_file, input_file_test, label_file_test, device)
    elif args.mode == 'test': # Test our trained model on a single image by doing a single forward pass
        test(input_file_test, label_file_test, device)
    elif args.mode == 'translate':
        while True:
            val = input('> ')
            translate(val, device)


if __name__ == '__main__':
    main()
