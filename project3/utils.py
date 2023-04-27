"""utils.py
Helper functions for this project 
"""
#import sentencepiece as spm
import math
import numpy as np

def pad_sents(sentences, pad_token):
    """ Pad list of sentences according to the longest sentence in the batch.
        The paddings should be at the end of each sentence.
    """
    sentences_padded = []

    max_len = max([len(word) for word in sentences])
    for sentence in sentences:
        sentence += [pad_token] * (max_len - len(sentence))
        sentences_padded.append(sentence)

    return sentences_padded


# def read_corpus(file_path, source, vocab_size=2500):
#     """ Read file, where each sentence is dilineated by a `\n`.
#     @param file_path (str): path to file containing corpus
#     @param source (str): "tgt" or "src" indicating whether text
#         is of the source language or target language
#     @param vocab_size (int): number of unique subwords in
#         vocabulary when reading and tokenizing
#     """
#     data = []
#     sp = spm.SentencePieceProcessor()
#     sp.load('src.model')#.format(source))

#     with open(file_path, 'r', encoding='utf8') as f:
#         for line in f:
#             subword_tokens = sp.encode_as_pieces(line)
            
#             # only append <s> and </s> to the target sentence
#             if source == 'tgt':
#                 subword_tokens = ["<s>"] + subword_tokens + ["</s>"]
                
#             data.append(subword_tokens)

#     return data

def read_corpus(input_file, source, vocab_size=21000):
    
    data = []
    if source == 'translate':
        data.append(input_file.split())
    else:
        input_lines = open(input_file, encoding='utf-8').read().strip().split('\n')
    
        for sentence in input_lines:
            words = sentence.split()
            if source == 'target':
                words = ['<s>'] + words + ['</s>']
            data.append(words)

    return data


def batch_iter(data, batch_size, shuffle=False):
    """Yield batches of source and target sentences reverse sorted by length (largest to smallets)"""
    batch_num = math.ceil(len(data) / batch_size)
    index_array = list(range(len(data)))

    if shuffle:
        np.random.shuffle(index_array)

    for i in range(batch_num):
        indices = index_array[i * batch_size: (i+1) * batch_size]
        examples = [data[idx] for idx in indices]

        examples = sorted(examples, key=lambda e: len(e[0]), reverse=True)
        examples = [examples[i] for i in range(len(examples)) if len(examples[i][0]) > 0 and len(examples[i][1]) > 0]

        src_sents = [e[0] for e in examples]
        tgt_sents = [e[1] for e in examples]
        yield src_sents, tgt_sents



