import numpy as np
import torch
import random
import torch.nn as nn
import torch.nn.functional as F

from data_classes import Vocab, get_frequencies, vector_representations, NLPDataset, pad_collate_fn

from torch.utils.data import DataLoader

from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

from torch.autograd import Variable

class Model3(nn.Module):
  def __init__(self, data_vec, num_layers, use_attention, rnn_layer):
    super(Model3, self).__init__()

    self.num_layers = num_layers
    self.rnn_type = rnn_layer
    self.attention = use_attention

    self.embedding_layer = nn.Embedding.from_pretrained(data_vec, padding_idx=0, freeze=True)

    if (rnn_layer == "lstm"):
        self.rnn1 = nn.LSTM(300, 150, num_layers=num_layers)
    elif (rnn_layer == "gru"):
        self.rnn1 = nn.GRU(300, 150, num_layers=num_layers)
    else:
       self.rnn1 = nn.RNN(300, 150, num_layers=num_layers)

    if self.attention:
        self.fc = nn.Linear(300, 1)
    else:
        self.fc = nn.Linear(150, 1)

    if (self.attention):
        self.W1 = nn.Linear(150, 150 // 2)
        self.W2 = nn.Linear(150 // 2, 1, bias=False)

    self.reset_parameters()

  def reset_parameters(self):
    for m in self.modules():
      if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight.to(torch.float64))
        if (m is self.fc) or (m is self.W1):
            nn.init.constant_(m.bias.to(torch.float64), 0.0)
      elif isinstance(m, nn.LSTM) or isinstance(m, nn.RNN) or isinstance(m, nn.GRU):
         for name, param in m.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0.0)

  def forward(self, x, lengths=None):
    batch_size  = x.shape[0]

    e = self.embedding_layer(x)
    e = torch.transpose(e, 0, 1)

    h_01 = Variable(torch.zeros(self.num_layers, batch_size, 150)).to(x.device)
       
    if (self.rnn_type == "lstm"):
        c_01 = Variable(torch.zeros(self.num_layers, batch_size, 150)).to(x.device)
        l1, (h_1, c_1) = self.rnn1(e, (h_01, c_01))
    else:
        l1, _ = self.rnn1(e, h_01)

    if (self.attention):
        #print(l1.shape)

        a = self.W2(torch.tanh(self.W1(l1)))
        #print(a.shape)
        alpha = F.softmax(a, dim=0)
        
        #print(alpha.shape)
        out_attention = torch.sum(alpha * l1, dim=0)
        #print(out_attention.shape)

        l1 = l1[-1, :, :]
        #print(l1.shape)
        combined_output =  torch.cat((l1, out_attention), dim=1)
        #print(combined_output.shape)

        output = self.fc(combined_output)

    else:
        l1 = torch.transpose(l1, 0, 1)
        l1 = l1[:, -1, :]
        output = self.fc(l1)

    return output

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
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.25)
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

seed = 7052020

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
    
    use_attention = False
    rnn_type = "lstm"

    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)

    g = torch.Generator()
    g.manual_seed(seed)

    tekst_freq, label_freq = get_frequencies("sst_train_raw.csv")

    vocab_data = Vocab(tekst_freq, True)
    vocab_label = Vocab(label_freq, False)

    train_dataset = NLPDataset("sst_train_raw.csv", vocab_data, vocab_label)
    test_dataset = NLPDataset("sst_test_raw.csv", vocab_data, vocab_label)
    valid_dataset = NLPDataset("sst_valid_raw.csv", vocab_data, vocab_label)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataloader = DataLoader(dataset=train_dataset, batch_size=train_batch, shuffle=True, collate_fn=pad_collate_fn, worker_init_fn=seed_worker,
    generator=g)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=test_batch, shuffle=True, collate_fn=pad_collate_fn, worker_init_fn=seed_worker,
    generator=g)
    valid_dataloader = DataLoader(dataset=valid_dataset, batch_size=valid_batch, shuffle=True, collate_fn=pad_collate_fn, worker_init_fn=seed_worker,
    generator=g)

    data_vec = vector_representations(vocab_data.stoi, 300, "sst_glove_6b_300d.txt")
    model = Model3(data_vec, 2, use_attention, rnn_type).to(device)

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