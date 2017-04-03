import os
import torch
import collections

class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)


class Corpus(object):
    def __init__(self, path, vocab_size=10000):
        self.vocab_size = vocab_size
        self.dictionary = Dictionary()
        self.build_vocab(path,vocab_size)

        self.train = self.tokenize(os.path.join(path, 'train.txt'))
        self.valid = self.tokenize(os.path.join(path, 'valid.txt'))
        self.test = self.tokenize(os.path.join(path, 'test.txt'))

    def build_vocab(self, path, vocab_size):
        words = []
        with open(os.path.join(path, 'train.txt'), 'r', errors='ignore') as f:
            for line in f:
                words.extend(line.split())

        with open(os.path.join(path, 'valid.txt'), 'r', errors='ignore') as f:
            for line in f:
                words.extend(line.split())
        
        with open(os.path.join(path, 'test.txt'), 'r', errors='ignore') as f:
            for line in f:
                words.extend(line.split())
        
        counter = collections.Counter(words).most_common(vocab_size - 2)
        self.dictionary.add_word('<unk>')
        self.dictionary.add_word('<eos>')
        for word,_ in counter:
            self.dictionary.add_word(word)    


    def tokenize(self, path):
        """Tokenizes a text file."""
        assert os.path.exists(path)
        # Add words to the dictionary
        with open(path, 'r', errors='ignore') as f:
            tokens = 0
            for line in f:
                words = line.split() + ['<eos>']
                tokens += len(words)
                
        # Tokenize file content
        with open(path, 'r', errors='ignore') as f:
            ids = torch.LongTensor(tokens)
            token = 0
            for line in f:
                words = line.split() + ['<eos>']
                for word in words:
                    if word in self.dictionary.word2idx:
                        ids[token] = self.dictionary.word2idx[word]
                    else:
                        ids[token] = self.dictionary.word2idx['<unk>']    
                    token += 1

        return ids
