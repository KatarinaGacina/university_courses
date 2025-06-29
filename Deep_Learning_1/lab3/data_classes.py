import torch
import torch.nn as nn

import csv

import re

from torch.nn.utils.rnn import pad_sequence

class Vocab:
    def __init__(self, frequencies, special_tokens, max_size=-1, min_freq=1):
        self.max_size = max_size
        self.min_freq = min_freq

        self.special_tokens = special_tokens

        #frequencies = rijecnik frekvencija: kljuc = token, value = frekvencija
        self.frequencies = frequencies

        self.itos = dict()
        self.stoi = dict()

        filtered_dict = {key: value for key, value in frequencies.items() if value >= min_freq}
        sorted_freq_dict = sorted(filtered_dict.items(), key=lambda x: (x[1], x[0]), reverse=True)

        tokens = [key for key, _ in sorted_freq_dict]
        if (max_size != -1):
            tokens = tokens[:max_size]

        i = 0
        if special_tokens:
            self.itos[0] = "<PAD>"
            self.stoi["<PAD>"] = 0

            self.itos[1] = "<UNK>"
            self.stoi["<UNK>"] = 1

            i = 2

        for token in tokens:
            self.itos[i] = token
            self.stoi[token] = i

            i += 1

    def encode(self, text_instance):
        #text_instance = ['yet', 'the', 'act', 'is', 'still', 'charming', 'here']
        
        encoded_text = []
        for word in text_instance:
            if word in self.stoi:
                encoded_text.append(self.stoi[word])
            else:
                encoded_text.append(self.stoi["<UNK>"])

        return torch.tensor(encoded_text)
    
    def encode_word(self, word):
        #word_instance = positive
        if word in self.stoi:
            return torch.tensor(self.stoi[word])
        else:
            return torch.tensor(self.stoi["<UNK>"])
        

def vector_representations(vocab, d=300, vector_path=None):
    tokens = list(vocab.keys())
    vector_matrix = torch.randn(len(tokens), d)

    pad_index = tokens.index("<PAD>")
    vector_matrix[pad_index] = torch.zeros(1, d)

    if vector_path is not None:
        token_vectors = {}
        with open(vector_path, "r") as file:
            for line in file:
                components = line.strip().split(" ")
                
                token = components[0]
                vector = [float(x) for x in components[1:]]

                token_vectors[token] = vector

        for i, token in enumerate(tokens):
            if token in token_vectors:
                vector_matrix[i] = torch.tensor(token_vectors[token]) 
    
    return vector_matrix
    
#embedding_matrix = torch.tensor(vector_matrix)
#embedding_layer = nn.Embedding.from_pretrained(embedding_matrix, padding_idx=0, freeze=True)   

class NLPDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, data_vocab, label_vocab):
        self.tekstovi = []
        self.labels = []

        with open(dataset) as file:
            reader = csv.reader(file)
            for row in reader:
                tekst, label = row

                self.tekstovi.append(tekst)
                self.labels.append(label)

        self.data_vocab = data_vocab
        self.label_vocab = label_vocab

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        tokens = re.split(r'[-=.!?;\s/:\\]+', self.tekstovi[index])
        label = self.labels[index]

        """sample = {
            'text': self.data_vocab.encode(tokens),
            'label': self.label_vocab.encode_word(label)
        }"""

        return self.data_vocab.encode(tokens), self.label_vocab.encode_word(label)
    
def pad_collate_fn(batch, pad_index=0):
    texts, labels = zip(*batch)
    lengths = [len(text) for text in texts]

    texts = pad_sequence(texts, batch_first=True, padding_value=pad_index)

    return texts, torch.tensor(labels), torch.tensor(lengths)

def get_frequencies(dataset):
    tekst_freq = {}
    label_freq = {}

    with open(dataset) as file:
        reader = csv.reader(file)
        for row in reader:
            tekst, label = row
            tokens = re.split(r'[-=.!?;\s/:\\]+', tekst)
            
            for token in tokens:
                if token in tekst_freq:
                    tekst_freq[token] += 1
                else:
                    tekst_freq[token] = 1
            
            if label in label_freq:
                label_freq[label] += 1
            else:
                label_freq[label] = 1
    
    return tekst_freq, label_freq

tekst_freq, label_freq = get_frequencies("sst_train_raw.csv")

#print(tekst_freq)
#print(label_freq)

vocab_data = Vocab(tekst_freq, True)
vocab_label = Vocab(label_freq, False)

#print(vocab_data)
#print(vocab_label)

data_vec = vector_representations(vocab_data.stoi, 300, "sst_glove_6b_300d.txt")
#print(data_vec)
#embedding_matrix = torch.tensor(data_vec)
embedding_layer = nn.Embedding.from_pretrained(data_vec, padding_idx=0, freeze=True)   

train_dataset = NLPDataset("sst_train_raw.csv", vocab_data, vocab_label)
#print(train_dataset)