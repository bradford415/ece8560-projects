"""network.py
NMT model definition and embeddings
"""

import sys
from collections import namedtuple
from typing import Dict, List, Set, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils
# explantion on pack_padded_sequence
# https://stackoverflow.com/questions/51030782/why-do-we-pack-the-sequences-in-pytorch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

Hypothesis = namedtuple('Hypothesis', ['value', 'score'])

class NMT(nn.Module):
    """Neural Machine Translation Model
        - 
    """
    def __init__(self, embed_size, hidden_size, vocab, device='cuda', dropout_rate=0.2,
                 rnn_layer=nn.LSTM, num_layers=1, activation=torch.tanh, bidirectional=False):
        super(NMT, self).__init__()
        self.model_embeddings = ModelEmbeddings(embed_size, vocab) # (vocab_len, embed_size)
        self.hidden_size = hidden_size
        self.dropout_rate = dropout_rate
        self.vocab = vocab
        self.rnn_layer = rnn_layer
        self.num_layers = num_layers
        self.activation = activation
        self.bidirectional = bidirectional

        # default values - not sure what this is
        self.encoder = None
        self.decoder = None
        self.h_projection = None
        self.c_projection = None
        self.att_projection = None
        self.combined_output_projection = None
        self.target_vocab_projection = None
        self.dropout = None
        #print("******") # ????
        self.is_lstm = (rnn_layer == nn.LSTM)

        self.encoder = rnn_layer(input_size=embed_size, hidden_size=hidden_size, bidirectional=bidirectional,
                                 bias=True, num_layers=num_layers)
        # LSTMCell is an lstm with the 'for loop' - it won't loop through all the timesteps by itself
        # I believe these are also not optimized through cuDNN
        self.decoder = nn.LSTMCell(input_size=embed_size+hidden_size, hidden_size=hidden_size, bias=True)
        
        self.h_projection = nn.Linear(hidden_size*2, hidden_size, bias=False)
        if self.is_lstm:
            self.c_projection = nn.Linear(hidden_size*2, hidden_size, bias=False)
        self.att_projection = nn.Linear(hidden_size*2, hidden_size, bias=False)
        self.combined_output_projection = nn.Linear(hidden_size*3, hidden_size, bias=False)
        self.target_vocab_projection = nn.Linear(hidden_size, len(vocab.tgt), bias=False)
        self.dropout = nn.Dropout(p=self.dropout_rate)

        self.device = device

    def forward(self, source, target):
        # Compute sentence lengths
        source_lengths = [len(s) for s in source]

        # Convert list of lists into tensors
        source_padded = self.vocab.src.to_input_tensor(source, device=self.device)
        target_padded = self.vocab.tgt.to_input_tensor(target, device=self.device)
        enc_hiddens, dec_init_state = self.encode(source_padded, source_lengths)
        enc_masks = self.generate_sent_masks(enc_hiddens, source_lengths)
        combined_outputs = self.decode(enc_hiddens, enc_masks, dec_init_state, target_padded)
        P = F.log_softmax(self.target_vocab_projection(combined_outputs), dim=-1)

        # Zero out probabilities for which we have nothing in the target text
        target_masks = (target_padded != self.vocab.tgt['<pad>']).float()

        # Compute log probability of generating true target words
        target_gold_words_log_prob = torch.gather(P, index=target_padded[1:].unsqueeze(-1), dim=-1).squeeze(-1) * target_masks[1:]
        scores = target_gold_words_log_prob.sum(dim=0)
        return scores

    def encode(self, source_padded, source_lengths):
        """Apply the encoder to source sentences to obtain encoder hidden states.
        Then, take the final state of the encoder and project them to obtain initial state for decoder.
        """
        enc_hiddens, dec_init_state = None, None
        last_hidden, last_cell = None, None
        X = self.model_embeddings.source(source_padded) # (src_len, b, e)
        enc_hiddens, last_state = self.encoder(pack_padded_sequence(X, source_lengths))
        if self.is_lstm:
            last_hidden, last_cell = last_state
        else:
            last_hidden = last_state
        enc_hiddens, _ = pad_packed_sequence(enc_hiddens)
        enc_hiddens = enc_hiddens.permute(1, 0, 2)

        # if not using bidirectional then last hidden will only have a size of 1 and you do not have to concatenate
        init_decoder_hidden = self.h_projection(torch.cat((last_hidden[0], last_hidden[1]), 1))
        if self.is_lstm:
            init_decoder_cell = self.c_projection(torch.cat((last_cell[0], last_hidden[1]), 1))
            dec_init_state = (init_decoder_hidden, init_decoder_cell)
        else:
            dec_init_state = (init_decoder_hidden, torch.zeros_like(init_decoder_hidden))

        return enc_hiddens, dec_init_state

    def decode(self, enc_hiddens, enc_masks, dec_init_state, target_padded):
            # Chop of the <END> token for max length sentences.
        target_padded = target_padded[:-1]

        # Initialize the decoder state (hidden and cell)
        dec_state = dec_init_state

        # Initialize previous combined output vector o_{t-1} as zero
        batch_size = enc_hiddens.size(0)
        o_prev = torch.zeros(batch_size, self.hidden_size, device=self.device)

        # Initialize a list we will use to collect the combined output o_t on each step
        combined_outputs = []

        enc_hiddens_proj = self.att_projection(enc_hiddens) # (b, src_len, h)
        Y = self.model_embeddings.target(target_padded) # (tgt_len, b, e)
        for Y_t in torch.split(Y, 1, dim=0):
            Y_t = torch.squeeze(Y_t, 0) # (b, e)
            Ybar_t = torch.cat((Y_t, o_prev), 1)
            dec_state, o_t, e_t = self.step(Ybar_t, dec_state, enc_hiddens, enc_hiddens_proj, enc_masks)
            combined_outputs.append(o_t)
            o_prev = o_t
        combined_outputs = torch.stack(combined_outputs, dim=0) # (tgt_len, b, h)

        return combined_outputs

    def beam_search(self, src_sent: List[str], beam_size: int=5, max_decoding_time_step: int=70) -> List[Hypothesis]:
        """ Given a single source sentence, perform beam search, yielding translations in the target language."""
        src_sents_var = self.vocab.src.to_input_tensor([src_sent], self.device)

        src_encodings, dec_init_vec = self.encode(src_sents_var, [len(src_sent)])
        src_encodings_att_linear = self.att_projection(src_encodings)

        h_tm1 = dec_init_vec
        att_tm1 = torch.zeros(1, self.hidden_size, device=self.device)

        eos_id = self.vocab.tgt['</s>']

        hypotheses = [['<s>']]
        hyp_scores = torch.zeros(len(hypotheses), dtype=torch.float, device=self.device)
        completed_hypotheses = []

        t = 0
        while len(completed_hypotheses) < beam_size and t < max_decoding_time_step:
            t += 1
            hyp_num = len(hypotheses)

            exp_src_encodings = src_encodings.expand(hyp_num,
                                                     src_encodings.size(1),
                                                     src_encodings.size(2))

            exp_src_encodings_att_linear = src_encodings_att_linear.expand(hyp_num,
                                                                           src_encodings_att_linear.size(1),
                                                                           src_encodings_att_linear.size(2))

            y_tm1 = torch.tensor([self.vocab.tgt[hyp[-1]] for hyp in hypotheses], dtype=torch.long, device=self.device)
            y_t_embed = self.model_embeddings.target(y_tm1)

            x = torch.cat([y_t_embed, att_tm1], dim=-1)

            (h_t, cell_t), att_t, _  = self.step(x, h_tm1,
                                                      exp_src_encodings, exp_src_encodings_att_linear, enc_masks=None)

            # log probabilities over target words
            log_p_t = F.log_softmax(self.target_vocab_projection(att_t), dim=-1)

            live_hyp_num = beam_size - len(completed_hypotheses)
            contiuating_hyp_scores = (hyp_scores.unsqueeze(1).expand_as(log_p_t) + log_p_t).view(-1)
            top_cand_hyp_scores, top_cand_hyp_pos = torch.topk(contiuating_hyp_scores, k=live_hyp_num)

            prev_hyp_ids = top_cand_hyp_pos // len(self.vocab.tgt)
            hyp_word_ids = top_cand_hyp_pos % len(self.vocab.tgt)

            new_hypotheses = []
            live_hyp_ids = []
            new_hyp_scores = []

            for prev_hyp_id, hyp_word_id, cand_new_hyp_score in zip(prev_hyp_ids, hyp_word_ids, top_cand_hyp_scores):
                prev_hyp_id = prev_hyp_id.item()
                hyp_word_id = hyp_word_id.item()
                cand_new_hyp_score = cand_new_hyp_score.item()

                hyp_word = self.vocab.tgt.id2word[hyp_word_id]
                new_hyp_sent = hypotheses[prev_hyp_id] + [hyp_word]
                if hyp_word == '</s>':
                    completed_hypotheses.append(Hypothesis(value=new_hyp_sent[1:-1],
                                                           score=cand_new_hyp_score))
                else:
                    new_hypotheses.append(new_hyp_sent)
                    live_hyp_ids.append(prev_hyp_id)
                    new_hyp_scores.append(cand_new_hyp_score)

            if len(completed_hypotheses) == beam_size:
                break

            live_hyp_ids = torch.tensor(live_hyp_ids, dtype=torch.long, device=self.device)
            h_tm1 = (h_t[live_hyp_ids], cell_t[live_hyp_ids])
            att_tm1 = att_t[live_hyp_ids]

            hypotheses = new_hypotheses
            hyp_scores = torch.tensor(new_hyp_scores, dtype=torch.float, device=self.device)

        if len(completed_hypotheses) == 0:
            completed_hypotheses.append(Hypothesis(value=hypotheses[0][1:],
                                                   score=hyp_scores[0].item()))

        completed_hypotheses.sort(key=lambda hyp: hyp.score, reverse=True)

        return completed_hypotheses


    def forward_pass(self, src_sent):
        src_sents_var = self.vocab.src.to_input_tensor([src_sent], self.device)

        src_encodings, dec_init_vec = self.encode(src_sents_var, [len(src_sent)])
        src_encodings_att_linear = self.att_projection(src_encodings)

        h_tm1 = dec_init_vec
        att_tm1 = torch.zeros(1, self.hidden_size, device=self.device)

        eos_id = self.vocab.tgt['</s>']

        hypotheses = [['<s>']]
        hyp_scores = torch.zeros(len(hypotheses), dtype=torch.float, device=self.device)
        completed_hypotheses = []

        exp_src_encodings = src_encodings.expand(hyp_num,
                                                     src_encodings.size(1),
                                                     src_encodings.size(2))
        exp_src_encodings_att_linear = src_encodings_att_linear.expand(hyp_num,
                                                                           src_encodings_att_linear.size(1),
                                                                           src_encodings_att_linear.size(2))
        y_tm1 = torch.tensor([self.vocab.tgt[hyp[-1]] for hyp in hypotheses], dtype=torch.long, device=self.device)

        



    def step(self, Ybar_t: torch.Tensor,
            dec_state: Tuple[torch.Tensor, torch.Tensor],
            enc_hiddens: torch.Tensor,
            enc_hiddens_proj: torch.Tensor,
            enc_masks: torch.Tensor) -> Tuple[Tuple, torch.Tensor, torch.Tensor]:
        """ Compute one forward step of the LSTM decoder, including the attention computation.
        """

        combined_output = None
        dec_state = self.decoder(Ybar_t, dec_state)
        dec_hidden, dec_cell = dec_state
        e_t = torch.squeeze(torch.bmm(enc_hiddens_proj, torch.unsqueeze(dec_hidden, dim=2)), dim=2)

        # Set e_t to -inf where enc_masks has 1 to ignore <pad> tokens
        if enc_masks is not None:
            e_t.data.masked_fill_(enc_masks.bool(), -float('inf'))

        alpha_t = F.softmax(e_t, dim=1) # (b, src_len)
        a_t = torch.squeeze(torch.bmm(torch.unsqueeze(alpha_t, 1), enc_hiddens), dim=1) # (b, 2h)
        U_t = torch.cat((dec_hidden, a_t), dim=1)
        V_t = self.combined_output_projection(U_t)
        O_t = self.dropout(self.activation(V_t))

        combined_output = O_t
        return dec_state, combined_output, e_t
        

    def generate_sent_masks(self, enc_hiddens: torch.Tensor, source_lengths: List[int]) -> torch.Tensor:
        """Generate sentence masks for encoder hidden states"""
        enc_masks = torch.zeros(enc_hiddens.size(0), enc_hiddens.size(1), dtype=torch.float)
        for e_id, src_len in enumerate(source_lengths):
            enc_masks[e_id, src_len:] = 1
        return enc_masks.to(self.device)

    def save(self, path):
        """ Save the odel to a file.
        """
        print('save model parameters to [%s]' % path, file=sys.stderr)

        params = {
            'args': dict(embed_size=self.model_embeddings.embed_size, hidden_size=self.hidden_size, dropout_rate=self.dropout_rate, 
                         rnn_layer=self.rnn_layer, num_layers=self.num_layers, activation=self.activation,
                         bidirectional=self.bidirectional),
            'vocab': self.vocab,
            'state_dict': self.state_dict()
        }

        torch.save(params, path)

    @staticmethod
    def load(model_path):
        """ Load the model from a file.
        """
        params = torch.load(model_path, map_location=lambda storage, loc: storage)
        args = params['args']
        model = NMT(vocab=params['vocab'], **args)
        model.load_state_dict(params['state_dict'])

        return model

class ModelEmbeddings(nn.Module): 
    """
    Class that converts input words to their embeddings.
    """
    def __init__(self, embed_size, vocab):
        """
        Init the Embedding layers. An embedding layer is a simple feedforward
        layer the maps an id to a certain size vector to better represent the word.
        Embeddings are relatively low-dimensional compared to something like one-hot
        and embeddings try to learn the semantics of the word by placing similar words
        closer together in the embedding space.
        https://developers.google.com/machine-learning/crash-course/embeddings/video-lecture#:~:text=An%20embedding%20is%20a%20relatively,like%20sparse%20vectors%20representing%20words.
        """
        super(ModelEmbeddings, self).__init__()
        self.embed_size = embed_size

        # default values
        self.source = None
        self.target = None

        src_pad_token_idx = vocab.src['<pad>']
        tgt_pad_token_idx = vocab.tgt['<pad>']

        # padding_idx is ignored by backpropagation
        self.source = nn.Embedding(len(vocab.src), self.embed_size, padding_idx=src_pad_token_idx)
        self.target = nn.Embedding(len(vocab.tgt), self.embed_size, padding_idx=tgt_pad_token_idx)




