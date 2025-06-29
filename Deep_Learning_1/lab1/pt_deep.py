import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, average_precision_score

import data

class PTDeep(torch.nn.Module):
  def __init__(self, W_list, a, param_lambda):
    super(PTDeep, self).__init__()

    self.param_lambda = param_lambda

    self.W = torch.nn.ParameterList()
    self.b = torch.nn.ParameterList()

    self.a = a

    for i in range(len(W_list) - 1):
      self.W.append(torch.nn.Parameter(torch.randn(W_list[i], W_list[i+1]), requires_grad=True))
      self.b.append(torch.nn.Parameter(torch.randn(W_list[i+1]), requires_grad=True))

  def forward(self, X):
    h = X

    for br, (wn, bn) in enumerate(zip(self.W, self.b)):
      h = torch.mm(h, wn) + bn

      if (br < (len(self.b) - 1)):
        #h = self.a[br](h)
        h = self.a(h)

    return torch.softmax(h, dim = 1)
  
  def get_loss(self, X, Yoh_):
    reg = 0
    """for w in self.W:
      reg += torch.linalg.vector_norm(w.view(-1), ord=2)
    reg *= self.param_lambda"""
    for w in self.W:
      reg += torch.sum(torch.norm(w, 2, dim=0))
    reg *= self.param_lambda

    return (-torch.mean(Yoh_ * torch.log(self.forward(X) + 1e-13))) + reg
  
  def count_params(self):
    br = 0
    for name, p in self.named_parameters():
      print(f'Parametar {name}: shape= {p.shape}')
      br += p.numel()
    
    print(f'Ukupan broj parametara  = {br}')


def train(model, X, Yoh_, param_niter, param_delta):
  optimizer = torch.optim.SGD(model.parameters(), lr=param_delta)

  optimizer.zero_grad()
  for i in range(param_niter):
      loss = model.get_loss(X, Yoh_)
      loss.backward()
      optimizer.step()
      
      if i % 10 == 0:
          print(f'Iteracija {i}: loss= {loss.item()}')

      optimizer.zero_grad()


def eval(model, X):
  X_tensor = torch.Tensor(X)
  return model.forward(X_tensor).detach().cpu().numpy()

if __name__ == "__main__":
  np.random.seed(100)

  #x, yoh = data.sample_gmm_2d(6, 2, 10)
  x, yoh = data.sample_gmm_2d(4, 2, 40)
  
  #x, yoh = data.sample_gmm_2d(6, 3, 50)

  X = torch.FloatTensor(x)
  
  yoh_ = torch.tensor(yoh)
  num_classes = len(torch.unique(yoh_))
  Yoh_ = torch.zeros(len(yoh), num_classes)

  Yoh_.scatter_(1, yoh_.unsqueeze(1).to(torch.int64), 1)

  deepModel = PTDeep([2, 10, 2], torch.sigmoid, 0)
  train(deepModel, X, Yoh_, 1000, 0.1)

  #deepModel = PTDeep([2, 3], torch.relu, 0)
  #train(deepModel, X, Yoh_, 1000, 0.2)

  probs = eval(deepModel, X)

  Y = np.argmax(probs, axis=1)
  print(Y)

  print(data.eval_perf_multi(Y, yoh))

  most_probable = np.max(-probs, axis=1)
  ranked_values = np.argsort(-most_probable)

  ap = data.eval_AP(yoh_[ranked_values].numpy())
  #print(ap)

  print()
  deepModel.count_params()

  # iscrtaj rezultate, decizijsku plohu
  rect=(np.min(x, axis=0), np.max(x, axis=0))
  data.graph_surface(lambda x: np.argmax(eval(deepModel, x), axis=1), rect, offset=0)

  data.graph_data(x, yoh, Y, special=[])

  plt.show()