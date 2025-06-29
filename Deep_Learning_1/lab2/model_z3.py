import torch
from torch import nn, optim, utils
import time
from pathlib import Path
import numpy as np
from torchvision.datasets import MNIST
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import math
import os
import skimage as ski
import skimage.io
 
class ModelZad3(nn.Module):
  def __init__(self, in_num=1, c=10):
    super(ModelZad3, self).__init__()
    
    self.conv1 = nn.Conv2d(in_channels=in_num, out_channels=16, kernel_size=5, padding=2, bias=True, dtype=torch.float64)
    self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
    self.relu1 = nn.ReLU()
    self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, padding=2, bias=True, dtype=torch.float64)
    self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
    self.relu2 = nn.ReLU()
    self.flatten1 = nn.Flatten()
    self.fc1 = nn.Linear(in_features=1568, out_features=512, bias=True, dtype=torch.float64)
    self.relu3 = nn.ReLU()
    self.logits = nn.Linear(in_features=512, out_features=c, bias=True, dtype=torch.float64)

    self.reset_parameters()

  def reset_parameters(self):
    for m in self.modules():
      if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight.to(torch.float64), mode='fan_in', nonlinearity='relu')
        nn.init.constant_(m.bias.to(torch.float64), 0)

      elif isinstance(m, nn.Linear) and m is not self.logits:
        nn.init.kaiming_normal_(m.weight.to(torch.float64), mode='fan_in', nonlinearity='relu')
        nn.init.constant_(m.bias.to(torch.float64), 0)
    
    self.logits.reset_parameters()

  def forward(self, x):
    h = self.conv1(x)
    h = self.pool1(h)
    #h = torch.relu(h)  # može i h.relu() ili nn.functional.relu(h)
    h = self.relu1(h)

    h = self.conv2(h)
    h = self.pool2(h)
    h = self.relu2(h)

    #h = h.view(h.shape[0], -1)
    h = self.flatten1(h)

    h = self.fc1(h)
    h = self.relu3(h)

    logits = self.logits(h)

    return logits
  
  def reg_loss2(self, l):
    return 0.5 * l * (np.square(np.linalg.norm(self.conv1.weight.detach().cpu().numpy().ravel(), ord=2)) + np.square(np.linalg.norm(self.conv2.weight.detach().cpu().numpy().ravel(), ord=2))
                      + np.square(np.linalg.norm(self.fc1.weight.detach().cpu().numpy().ravel(), ord=2)))
  
  def reg_loss(self, l):
    return l * (np.sum(np.square(self.conv1.weight.detach().cpu().numpy())) + np.sum(np.square(self.conv2.weight.detach().cpu().numpy())) + np.sum(np.square(self.fc1.weight.detach().cpu().numpy())))

def dense_to_one_hot(y, class_count):
    return np.eye(class_count)[y]

def draw_conv_filters(epoch, step, layer, writer):
  w = layer.weight.clone().detach().cpu().numpy()

  #print(w.shape)

  C = w.shape[1]
  num_filters = w.shape[0]
  k = w.shape[2]

  w -= w.min()
  w /= w.max()

  border = 1
  cols = 8

  rows = math.ceil(num_filters / cols)
  width = cols * k + (cols-1) * border
  height = rows * k + (rows-1) * border

  for i in range(C):
    img = np.zeros([height, width])
    for j in range(num_filters):
      r = int(j / cols) * (k + border)
      c = int(j % cols) * (k + border)
      img[r:r+k,c:c+k] = w[j,i]
    
    imgname = '%s_epoch_%02d_step_%06d_input_%03d.png' % ("conv1", epoch, step, i)
    img = (img * 255.).astype(np.uint8)

    plt.imshow(img, cmap='gray')
    plt.axis('off')
    fig = plt.gcf()

    writer.add_figure(imgname, fig)

def train(train_x, train_y, valid_x, valid_y, config):
  writer = SummaryWriter()

  epochs = config['max_epochs']
  batch_size = config['batch_size']
  lambda_reg = config['weight_decay']
  lr_rate = config['lr']

  #valid_x = torch.tensor(valid_x, dtype=torch.float64)
  #valid_y = torch.tensor(valid_y, dtype=torch.float64)

  num_batches = valid_y.shape[0] // batch_size

  dataset_eval = torch.utils.data.TensorDataset(torch.tensor(valid_x, dtype=torch.float64), torch.tensor(valid_y, dtype=torch.float64))
  dataloader_eval = utils.data.DataLoader(dataset_eval, batch_size=batch_size, shuffle=True)

  dataset = torch.utils.data.TensorDataset(torch.tensor(train_x, dtype=torch.float64), torch.tensor(train_y, dtype=torch.float64))
  dataloader = utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

  model = ModelZad3()

  weight_decay_list = [
    {'params': model.conv1.weight, 'weight_decay': lambda_reg},
    {'params': model.conv2.weight, 'weight_decay': lambda_reg},
    {'params': model.fc1.weight, 'weight_decay': lambda_reg}
  ]

  criterion = nn.CrossEntropyLoss()
  optimizer = optim.SGD(weight_decay_list, lr=lr_rate)
  #optimizer = optim.SGD(model.parameters(), lr=lr_rate)
  scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=2, gamma=0.1)

  for epoch in range(epochs):
    for i, (x, y) in enumerate(dataloader):
        output = model(x)
        loss = criterion(output, y) #+ model.reg_loss(lambda_reg)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 500 == 0:
            print(f"Batch {i}, epoch {epoch}: {loss}")

    with torch.no_grad():
      outputs = []
      labels = []

      eval_loss = 0
      for (x, y) in (dataloader_eval):
        output = model(x)

        outputs.extend(torch.argmax(output, dim=1))
        labels.extend(torch.argmax(y, dim=1))

        eval_loss += (criterion(output, y)) #+ model.reg_loss(lambda_reg))

      correct_predictions = sum(o == l for o, l in zip(outputs, labels))
      val_accuracy = correct_predictions / valid_x.shape[0]

      writer.add_scalar('Validation_loss', (eval_loss / num_batches), epoch)

      print(f"Epoch: {epoch}")
      print("Validation accuracy = %.2f" % val_accuracy)
      print("Validation loss = %.2f\n" % (eval_loss / num_batches))
      print()
    
    """with torch.no_grad():
        outputs = model(valid_x)

        val_loss = criterion(outputs, valid_y) + model.reg_loss(lambda_reg)
        writer.add_scalar('Validation_loss', val_loss, epoch)

        _, indicies = torch.max(outputs, 1)
        _, indicies_real = torch.max(valid_y, 1)
        correct_predictions = (indicies == indicies_real).sum().item()

        val_accuracy = correct_predictions / valid_x.shape[0]

        print(f"Epoch: {epoch}")
        print("Validation accuracy = %.2f" % val_accuracy)
        print("Validation loss = %.2f\n" % val_loss)
        print()"""

    writer.add_scalar('Loss', loss, epoch)
    draw_conv_filters(epoch, i*batch_size, model.conv1, writer)
    
    scheduler.step()
  
  writer.close()

  return model, criterion

def test_model(model, criterion, test_x, test_y, l, batch_size):
  num_batches = test_y.shape[0] // batch_size

  dataset_eval = torch.utils.data.TensorDataset(torch.tensor(test_x, dtype=torch.float64), torch.tensor(test_y, dtype=torch.float64))
  dataloader_eval = utils.data.DataLoader(dataset_eval, batch_size=batch_size, shuffle=True)

  with torch.no_grad():
      outputs = []
      labels = []

      eval_loss = 0
      for (x, y) in (dataloader_eval):
        output = model(x)

        outputs.extend(torch.argmax(output, dim=1))
        labels.extend(torch.argmax(y, dim=1))

        eval_loss += (criterion(output, y)) #+ model.reg_loss(l))

      correct_predictions = sum(o == l for o, l in zip(outputs, labels))
      eval_accuracy = correct_predictions / test_x.shape[0]

      print()
      print("Test accuracy = %.2f" % eval_accuracy)
      print("Test loss = %.2f\n" % (eval_loss / num_batches))

  """test_x = torch.tensor(test_x, dtype=torch.float64)
  test_y = torch.tensor(test_y, dtype=torch.float64)
  
  with torch.no_grad():
    outputs = model(test_x)

    loss = criterion(outputs, test_y) #+ model.reg_loss(l)

    _, indicies = torch.max(outputs, 1)
    _, indicies_real = torch.max(test_y, 1)
    correct_predictions = (indicies == indicies_real).sum().item()

    test_accuracy = correct_predictions / test_x.shape[0]

    print("Test accuracy = %.2f" % test_accuracy)
    print("Test loss = %.2f\n" % loss)
    print()"""

if __name__ == "__main__":
    config = {}
    config['max_epochs'] = 8
    config['batch_size'] = 50
    config['weight_decay'] = 1e-3
    config['lr'] = 1e-1

    DATA_DIR = Path(__file__).parent / 'datasets' / 'MNIST'

    ds_train, ds_test = MNIST(DATA_DIR, train=True, download=True), MNIST(DATA_DIR, train=False)
    train_x = ds_train.data.reshape([-1, 1, 28, 28]).numpy().astype(np.float) / 255
    train_y = ds_train.targets.numpy()
    train_x, valid_x = train_x[:55000], train_x[55000:]
    train_y, valid_y = train_y[:55000], train_y[55000:]
    test_x = ds_test.data.reshape([-1, 1, 28, 28]).numpy().astype(np.float) / 255
    test_y = ds_test.targets.numpy()
    train_mean = train_x.mean()
    train_x, valid_x, test_x = (x - train_mean for x in (train_x, valid_x, test_x))
    train_y, valid_y, test_y = (dense_to_one_hot(y, 10) for y in (train_y, valid_y, test_y))

    model, criterion = train(train_x, train_y, valid_x, valid_y, config)

    test_model(model, criterion, test_x, test_y, config['weight_decay'], config['batch_size'])