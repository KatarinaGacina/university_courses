import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import random

from data_classes import Vocab, get_frequencies, vector_representations, NLPDataset, pad_collate_fn

from torch.utils.data import DataLoader

from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

class Model1(nn.Module):
  def __init__(self, data_vec):
    super(Model1, self).__init__()

    self.embedding_layer = nn.Embedding.from_pretrained(data_vec, padding_idx=0, freeze=True) #freeze! 
    self.flatten = nn.Flatten()
    self.fc1 = nn.Linear(300, 150)
    
    self.relu1 = nn.ReLU()

    self.fc2 = nn.Linear(150, 150)
    
    self.relu2 = nn.ReLU()
    
    self.fc3 = nn.Linear(150, 1)

    self.reset_parameters()

  def reset_parameters(self):
    for m in self.modules():
      if isinstance(m, nn.Linear) and m is not self.fc3:
        nn.init.kaiming_normal_(m.weight.to(torch.float64), nonlinearity='relu')
        nn.init.normal_(m.bias.to(torch.float64), 0.0)
      elif (m is self.fc3):
        nn.init.xavier_normal_(m.weight.to(torch.float64))
        nn.init.constant_(m.bias.to(torch.float64), 0.0)
    
  def forward(self, x):
    #print(x.shape)
    e = self.embedding_layer(x)
    #print(e.shape)
    pooled = torch.mean(e, dim=1)
    #print(pooled.shape)
    h1 = self.fc1(pooled)
    a1 = self.relu1(h1)
    h2 = self.fc2(a1)
    a2 = self.relu2(h2)
    h3 = self.fc3(a2)

    return h3

def train(model, data, optimizer, criterion, device, args):
  
    model.train()

    """first_batch = next(iter(data))
    print("First batch:")
    print(first_batch)"""

    for batch_num, (podaci, target, _) in enumerate(data):
        podaci = podaci.to(device)
        target = target.to(device).float()

        model.zero_grad()

        logits = model(podaci).squeeze().float()
        loss = criterion(logits, target)

        loss.backward()
        #torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        optimizer.step()

def evaluate(model, data, criterion, device, args):
    model.eval()

    targets = []
    predictions = []

    loss_all = 0

    with torch.no_grad():
        for batch_num, (podaci, target, _) in enumerate(data):
            podaci = podaci.to(device)
            target = target.to(device).float()

            logits = model(podaci).squeeze().float() 
            loss = criterion(logits, target)

            loss_all += loss

            probabilities = torch.sigmoid(logits)
            prediction = (probabilities > 0.5)

            targets.extend(target.cpu().tolist())
            predictions.extend(prediction.cpu().tolist())

        accuracy = accuracy_score(targets, predictions)
        f1 = f1_score(targets, predictions)
        conf_matrix = confusion_matrix(targets, predictions)

    return loss_all, accuracy, f1, conf_matrix

seed = 15061946

def seed_worker(worker_id):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)

def main():
    train_batch = 10
    valid_batch = 32
    test_batch = 32

    lr_rate = 1e-4
    num_epochs = 5

    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)

    g = torch.Generator()
    g.manual_seed(seed)

    tekst_freq, label_freq = get_frequencies("sst_train_raw.csv")

    vocab_data = Vocab(tekst_freq, True, min_freq=1)
    vocab_label = Vocab(label_freq, False)

    train_dataset = NLPDataset("sst_train_raw.csv", vocab_data, vocab_label)
    test_dataset = NLPDataset("sst_test_raw.csv", vocab_data, vocab_label)
    valid_dataset = NLPDataset("sst_valid_raw.csv", vocab_data, vocab_label)

    #print(train_dataset[3])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #print(device)

    train_dataloader = DataLoader(dataset=train_dataset, batch_size=train_batch, shuffle=True, collate_fn=pad_collate_fn, worker_init_fn=seed_worker,
    generator=g)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=test_batch, shuffle=True, collate_fn=pad_collate_fn, worker_init_fn=seed_worker,
    generator=g)
    valid_dataloader = DataLoader(dataset=valid_dataset, batch_size=valid_batch, shuffle=True, collate_fn=pad_collate_fn, worker_init_fn=seed_worker,
    generator=g)

    data_vec = vector_representations(vocab_data.stoi, 300, "sst_glove_6b_300d.txt")
    model = Model1(data_vec).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr_rate)

    for epoch in range(num_epochs):
        train(model, train_dataloader, optimizer, criterion, device, None)

        print("Validation: epoch: " + str(epoch) + ":")
        loss, acc, f1, conf = evaluate(model, valid_dataloader, criterion, device, None)
        print("Loss: " + str(loss))
        print("Accuracy: " + str(acc))
        print("F1-score: " + str(f1))
        print("Confusion matrix: ")
        print(conf)
        print()

    print("Test: ")
    loss, acc, f1, conf = evaluate(model, test_dataloader, criterion, device, None)
    print("Loss: " + str(loss))
    print("Accuracy: " + str(acc))
    print("F1-score: " + str(f1))
    print("Confusion matrix: ")
    print(conf)
    print()

if __name__ == "__main__":
    main()