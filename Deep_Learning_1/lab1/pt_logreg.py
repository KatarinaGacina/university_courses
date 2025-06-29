import torch
import numpy as np
from sklearn.datasets import make_classification
import matplotlib.pyplot as plt
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix

import data

class PTLogreg(torch.nn.Module):
  def __init__(self, D, C, param_lambda=1e-3):
    super(PTLogreg, self).__init__()

    """Arguments:
       - D: dimensions of each datapoint 
       - C: number of classes
    """
    self.C = C
    self.D = D

    self.param_lamda = param_lambda

    # inicijalizirati parametre (koristite nn.Parameter):
    # imena mogu biti self.W, self.b
    # ...
    self.W = torch.nn.Parameter(torch.randn(self.D, self.C), requires_grad=True)
    self.b = torch.nn.Parameter(torch.zeros(1, self.C), requires_grad=True)

  def forward(self, X):
    # unaprijedni prolaz modela: izračunati vjerojatnosti
    #   koristiti: torch.mm, torch.softmax
    # ...

    return torch.softmax((torch.mm(X, self.W) + self.b), dim = 1)
  
  def get_loss(self, X, Yoh_):
    # formulacija gubitka
    #   koristiti: torch.log, torch.exp, torch.sum
    #   pripaziti na numerički preljev i podljev
    # ...

    reg = torch.linalg.vector_norm(self.W.view(-1), ord=2) * self.param_lamda
    loss = -torch.mean(Yoh_ * torch.log(self.forward(X) + 1e-13)) + reg

    return loss

def train(model, X, Yoh_, param_niter, param_delta):
  """Arguments:
     - X: model inputs [NxD], type: torch.Tensor
     - Yoh_: ground truth [NxC], type: torch.Tensor
     - param_niter: number of training iterations
     - param_delta: learning rate
  """
  
  # inicijalizacija optimizatora
  # ...

  optimizer = torch.optim.SGD(model.parameters(), lr=param_delta)

  # petlja učenja
  # ispisujte gubitak tijekom učenja
  # ...
  for i in range(param_niter):
      #Y = model.forward(X, Yoh_)
      
      loss = model.get_loss(X, Yoh_)

      loss.backward()

      optimizer.step()
      
      if i % 10 == 0:
          #print(model.W)
          #print(model.b)
          print(f'Iteracija {i}: loss= {loss.item()}')

      optimizer.zero_grad()


def eval(model, X):
  """Arguments:
     - model: type: PTLogreg
     - X: actual datapoints [NxD], type: np.array
     Returns: predicted class probabilites [NxC], type: np.array
  """
  # ulaz je potrebno pretvoriti u torch.Tensor
  # izlaze je potrebno pretvoriti u numpy.array
  # koristite torch.Tensor.detach() i torch.Tensor.numpy()
  X_tensor = torch.Tensor(X)

  return model.forward(X_tensor).detach().cpu().numpy()

if __name__ == "__main__":
  # inicijaliziraj generatore slučajnih brojeva
  np.random.seed(100)

  # instanciraj podatke X i labele Yoh_
  x, yoh = data.sample_gmm_2d(6, 3, 50)
  
  X = torch.FloatTensor(x)
  
  yoh_ = torch.tensor(yoh)
  num_classes = len(torch.unique(yoh_))
  Yoh_ = torch.zeros(len(yoh), num_classes)

  Yoh_.scatter_(1, yoh_.unsqueeze(1).to(torch.int64), 1)
  #Yoh_ = F.one_hot(yoh_, num_classes)

  # definiraj model:
  ptlr = PTLogreg(X.shape[1], Yoh_.shape[1])

  # nauči parametre (X i Yoh_ moraju biti tipa torch.Tensor):
  train(ptlr, X, Yoh_, 1000, 0.2)

  # dohvati vjerojatnosti na skupu za učenje
  probs = eval(ptlr, X)
  #print(probs)

  Y = np.argmax(probs, axis=1)
  print(Y)

  # ispiši performansu (preciznost i odziv po razredima)
  cm = confusion_matrix(yoh, Y)
  recall = np.diag(cm) / np.sum(cm, axis = 1)
  precision = np.diag(cm) / np.sum(cm, axis = 0)

  print(recall)
  print(precision)

  for c in range(num_classes):
    tn = 0
    tp = 0
    fp = 0
    fn = 0
    for yp, yt in zip(yoh, Y):
      if (yp != c and yt != c):
        tn +=1
      elif (yp != c and yt == c):
        fn += 1
      elif (yp == c and yt != c):
        fp += 1
      else:
        tp += 1
    
    print(f'Klasa {c}: precision= {tp/(tp+fp+ 1e-13)}; recall= {tp/(tp+fn+ 1e-13)}')

  # iscrtaj rezultate, decizijsku plohu
  rect = (np.min(x, axis=0), np.max(x, axis=0))
  data.graph_surface(lambda x: (np.argmax(eval(ptlr, x), axis=1)), rect, offset=0)

  data.graph_data(x, yoh, Y, special=[])

  plt.show()