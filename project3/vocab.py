"""vocab.py
Contains class to read and store the vocab list for a language
"""
import torch

from utils import pad_sents


class VocabEntry:
    """Vocabulary entry"""
    def __init__(self, word2id=None):
        if word2id:
            self.word2id = word2id
        else:
            # Dictionary to maps words to ids 
            self.word2id = dict()
            self.word2id['<pad>'] = 0   # Pad Token - variable sequence lengts
            self.word2id['<s>'] = 1     # Start Token
            self.word2id['</s>'] = 2    # End Token
            self.word2id['<unk>'] = 3   # Unknown Token - words not in vocab
        self.unk_id = self.word2id['<unk>']
        # create dictionary to map ids to words
        self.id2word = {v: k for k, v in self.word2id.items()} 

    def __getitem__(self, word):
        """Retrieve id of word, return unknown id if word not found"""
        return self.word2id.get(word, self.unk_id)

    def __contains__(self, word):
        """Check if word is in Vocab Entry"""
        return word in self.word2id

    def __len__(self):
        """Vocab length"""
        return len(self.word2id)

    def __repr__(self):
        """Return string representation of the VocabEntry object.
        Used to print the oject"""
        return f'Vocabulary[size={len(self)}]'

    def id2word(self, wid):
        """Return the word that maps to an wid (word index)"""
        self.id2word[wid]

    def add(self, word):
        """Add word to VocaEntry if it does not already exit"""
        if word not in self: # 'in self' comes from __contains__ i think
            wid = self.word2id[word] = len(self) # set wid and word2id equal to length
            self.id2word[wid] = word             # this works bc indexing starts at 0
            return wid
        else:
            return self[word]

    def words2indices(self, sents):
        """ Convert list of words or list of sentences of words
        into list or list of list of indices.
        """
        if type(sents[0]) == list: # if of sentences of words
            return [[self[w] for w in s] for s in sents]
        else: # if list of words
            return [self[w] for w in sents]
        
    def indices2words(self, word_ids):
        """Convert list of indices into words"""
        return [self.id2word[w_id] for w_id in word_ids]

    def to_input_tensor(self, sents, device):
        """Convert list of sentences into tensor with ids and necessary padding"""
        word_ids = self.words2indices(sents)
        sents_t = pad_sents(word_ids, self['<pad>'])
        sents_var = torch.tensor(sents_t, dtype=torch.long, device=device)
        return torch.t(sents_var)

    @staticmethod # allows you to call this method directly from the Class without creating and object
    def from_corpus(corpus, size, freq_cutoff=2):
        """ Given a corpus (body of text) construct a VocabEntry.
        """
        vocab_entry = VocabEntry()
        word_freq = Counter(chain(*corpus)) # chain combines many iterables into one
        valid_words = [w for w, v in word_freq.items() if v >= freq_cutoff]
        print('number of word types: {}, number of word types w/ frequency >= {}: {}'
              .format(len(word_freq), freq_cutoff, len(valid_words)))
        top_k_words = sorted(valid_words, key=lambda w: word_freq[w], reverse=True)[:size]
        for word in top_k_words:
            vocab_entry.add(word)
        return vocab_entry

    @staticmethod
    def from_subword_list(subword_list):
        vocab_entry = VocabEntry()
        for sentence in subword_list:
            for word in sentence:
                vocab_entry.add(word)
        return vocab_entry


class Vocab:
    """Vocab of source and target languages"""
    def __init__(self, src_vocab, tgt_vocab):
        self.src = src_vocab
        self.tgt = tgt_vocab

    @staticmethod
    def build(src_sents, tgt_sents):
        """Build Vocabulary"""
        print('Initialize source vocabulary...')
        src = VocabEntry.from_subword_list(src_sents)

        print('Initialize target vocabulary...')
        tgt = VocabEntry.from_subword_list(tgt_sents)

        return Vocab(src, tgt)

    def save(self, file_path):
        """ IS THIS NECESSARY?
        Save Vocab to file as JSON dump"""
        with open(file_path, 'w') as f:
            json.dump(dict(src_word2id=self.src.word2id, tgt_word2id=self.tgt.word2id), f, indent=2)

    @staticmethod
    def load(file_path):
        """Load vocabulary from JSON dump"""
        entry = json.load(open(file_path, 'r'))
        src_word2id = entry['src_word2id']
        tgt_word2id = entry['tgt_word2id']
        return Vocab(VocabEntry(src_word2id), VocabEntry(tgt_word2id))

    def __repr__(self):
        """ Representation of Vocab to be used wwen printing the object"""
        return 'Vocab(source %d words, target %d words)' % (len(self.src), len(self.tgt))



