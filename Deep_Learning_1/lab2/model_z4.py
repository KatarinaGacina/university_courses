import torch
from torch import nn, optim, utils
import numpy as np
import matplotlib.pyplot as plt
import math
import os
import skimage as ski
import skimage.io
import pickle
from sklearn.metrics import confusion_matrix
 
class ModelZad4(nn.Module):
  def __init__(self, in_num=3, c=10):
    super(ModelZad4, self).__init__()
    
    self.conv1 = nn.Conv2d(in_channels=in_num, out_channels=16, kernel_size=5, padding=0, bias=True, dtype=torch.float64)
    self.relu1 = nn.ReLU()
    self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2)
    self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, padding=0, bias=True, dtype=torch.float64)
    self.relu2 = nn.ReLU()
    self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2)
    self.flatten1 = nn.Flatten()
    self.fc1 = nn.Linear(in_features=512, out_features=256, bias=True, dtype=torch.float64)
    self.relu3 = nn.ReLU()
    self.fc2 = nn.Linear(in_features=256, out_features=128, bias=True, dtype=torch.float64)
    self.relu4 = nn.ReLU()
    self.logits = nn.Linear(in_features=128, out_features=c, bias=True, dtype=torch.float64)

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
    h = self.relu1(h)
    #h = torch.relu(h)  # može i h.relu() ili nn.functional.relu(h)
    h = self.pool1(h)

    h = self.conv2(h)
    h = self.relu2(h)
    h = self.pool2(h)

    #h = h.view(h.shape[0], -1)
    h = self.flatten1(h)

    h = self.fc1(h)
    h = self.relu3(h)

    h = self.fc2(h)
    h = self.relu4(h)

    logits = self.logits(h)

    return logits
  
  def reg_loss(self, l):
    return 0.5 * l * (np.square(np.linalg.norm(self.conv1.weight.detach().cpu().numpy().ravel(), ord=2)) + np.square(np.linalg.norm(self.conv2.weight.detach().cpu().numpy().ravel(), ord=2))
                      + np.square(np.linalg.norm(self.fc1.weight.detach().cpu().numpy().ravel(), ord=2)) + np.square(np.linalg.norm(self.fc2.weight.detach().cpu().numpy().ravel(), ord=2)))

def draw_conv_filters(epoch, step, weights, save_dir):
    w = weights.copy()
    num_filters = w.shape[0]
    num_channels = w.shape[1]
    k = w.shape[2]
    assert w.shape[3] == w.shape[2]
    w = w.transpose(2, 3, 1, 0)
    w -= w.min()
    w /= w.max()
    border = 1
    cols = 8
    rows = math.ceil(num_filters / cols)
    width = cols * k + (cols-1) * border
    height = rows * k + (rows-1) * border
    img = np.zeros([height, width, num_channels])
    for i in range(num_filters):
      r = int(i / cols) * (k + border)
      c = int(i % cols) * (k + border)
      img[r:r+k,c:c+k,:] = w[:,:,:,i]
    filename = 'epoch_%02d_step_%06d.png' % (epoch, step)
    img = (img * 255.).astype(np.uint8)
    ski.io.imsave(os.path.join(save_dir, filename), img)

def draw_image(img, mean, std):
    img = img.transpose(1, 2, 0)
    img *= std
    img += mean
    img = img.astype(np.uint8)
    ski.io.imshow(img)
    #ski.io.show()

def plot_training_progress(save_dir, data):
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16,8))

    linewidth = 2
    legend_size = 10
    train_color = 'm'
    val_color = 'c'

    num_points = len(data['train_loss'])
    x_data = np.linspace(1, num_points, num_points)
    ax1.set_title('Cross-entropy loss')
    ax1.plot(x_data, data['train_loss'], marker='o', color=train_color,
              linewidth=linewidth, linestyle='-', label='train')
    ax1.plot(x_data, data['valid_loss'], marker='o', color=val_color,
              linewidth=linewidth, linestyle='-', label='validation')
    ax1.legend(loc='upper right', fontsize=legend_size)
    ax2.set_title('Average class accuracy')
    ax2.plot(x_data, data['train_acc'], marker='o', color=train_color,
              linewidth=linewidth, linestyle='-', label='train')
    ax2.plot(x_data, data['valid_acc'], marker='o', color=val_color,
              linewidth=linewidth, linestyle='-', label='validation')
    ax2.legend(loc='upper left', fontsize=legend_size)
    ax3.set_title('Learning rate')
    ax3.plot(x_data, data['lr'], marker='o', color=train_color,
              linewidth=linewidth, linestyle='-', label='learning_rate')
    ax3.legend(loc='upper left', fontsize=legend_size)

    save_path = os.path.join(save_dir, 'training_plot.png')
    print('Plotting in: ', save_path)
    plt.savefig(save_path)

def dense_to_one_hot(y, class_count):
    return np.eye(class_count)[y]

def show_worst_classified(model, criterion, x, y, mean, std, num, dir):
    x_t = torch.tensor(x, dtype=torch.float64)
    y_t = torch.tensor(dense_to_one_hot(y, 10))

    with open(dir + '/batches.meta', 'rb') as f:
      cifar_meta = pickle.load(f, encoding='bytes')
    cifar_class_names = [label.decode('utf-8') for label in cifar_meta[b'label_names']]

    with torch.no_grad():
      output = model(x_t)
      outputs = torch.argmax(output, dim=1)

      losses = []
      for i in range(len(y)):
        loss = criterion(output[i], y_t[i])
        losses.append(loss.item())
      
      cm = confusion_matrix(outputs, y)
      class_accuracy = np.diag(cm)/ cm.sum(axis=1)

      top_indices = np.argsort(-np.array(losses))[:num]
      top_classes = np.argsort(-class_accuracy)[:3]
      print(f"Top 3 classes: {top_classes}")


      """fig, axs = plt.subplots(4, 5)
      for i, ax in enumerate(axs.flat):
        draw_image(x[top_indices[i]], mean, std)
        ax.set_title(cifar_class_names[outputs[top_indices[i]]])
      plt.tight_layout()
      plt.show()"""

      fig, axs = plt.subplots(4, 5)
      plt.subplots_adjust(wspace=1, hspace=1)
      for i in range(len(top_indices)):
        plt.subplot(4, 5, i+1)
        draw_image(x[top_indices[i]], mean, std)
        plt.title(cifar_class_names[outputs[top_indices[i]]] + "; " + str(cifar_class_names[y[top_indices[i]]]))

      plt.show()

      """fig, axs = plt.subplots(4, 5, figsize=(10, 8))
      plt.subplots_adjust(wspace=0.5, hspace=0.5)
      for i in range(4):
        for j in range(5):
          ax = axs[i, j]
          draw_image(x[top_indices[i]], mean, std)
          ax.set_title(cifar_class_names[outputs[top_indices[i]]])
      plt.setp(axs, xticks=[], yticks=[])
      plt.show()"""

    return top_classes
       

def evaluate(model, criterion, x, y, batch_size, l):
  print("Evaluating...")
  assert y.shape[0] % batch_size == 0
  num_batches = y.shape[0] // batch_size

  dataset = torch.utils.data.TensorDataset(torch.tensor(x, dtype=torch.float64), torch.tensor(dense_to_one_hot(y, 10), dtype=torch.float64))
  dataloader = utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

  with torch.no_grad():
    outputs = []
    labels = []

    eval_loss = 0
    for (x, y) in (dataloader):
      output = model(x)

      outputs.extend(torch.argmax(output, dim=1))
      labels.extend(torch.argmax(y, dim=1))

      eval_loss += (criterion(output, y)) # + model.reg_loss(l))

    cm = confusion_matrix(outputs, labels)
    recall = np.diag(cm) / np.sum(cm, axis = 1)
    precision = np.diag(cm) / np.sum(cm, axis = 0)
    accuracy = np.sum(np.diag(cm)) / np.sum(cm)

    print(f"Confusion matrix:\n {cm}")
    print(f"precision:\n {precision}")
    print(f"Recall:\n {recall}")

    print("accuracy = %.2f" % accuracy)
    print("loss = %.2f\n" % (eval_loss / num_batches))
    print()

    return (eval_loss / num_batches), accuracy

def train(train_x, train_y, valid_x, valid_y, config, save_dir):
  plot_data = {}

  epochs = config['max_epochs']
  batch_size = config['batch_size']
  lr_rate = config['lr']
  l = config['weight_decay']

  dataset = torch.utils.data.TensorDataset(torch.tensor(train_x, dtype=torch.float64), torch.tensor(dense_to_one_hot(train_y, 10), dtype=torch.float64))
  dataloader = utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

  model = ModelZad4()

  weight_decay_list = [
    {'params': model.conv1.weight, 'weight_decay': l},
    {'params': model.conv2.weight, 'weight_decay': l},
    {'params': model.fc1.weight, 'weight_decay': l}
  ]

  criterion = nn.CrossEntropyLoss()
  #optimizer = optim.SGD(model.parameters(), lr=lr_rate)
  optimizer = optim.SGD(weight_decay_list, lr=lr_rate)
  scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.95)

  train_losses = []
  train_accuraccies = []
  
  valid_losses = []
  valid_accuracies = []

  lr_rates = []

  for epoch in range(epochs):
    print(f"Epoch {epoch} training...")
    for i, (x, y) in enumerate(dataloader):
        output = model(x)
        loss = criterion(output, y) #+ model.reg_loss(l)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i%200 == 0:
          draw_conv_filters(epoch, i, model.conv1.weight.detach().cpu().numpy(), SAVE_DIR)

    model.eval()
    train_loss, train_accuracy = evaluate(model, criterion, train_x, train_y, batch_size, l)
    val_loss, val_accuracy = evaluate(model, criterion, valid_x, valid_y, batch_size, l)
    model.train()

    train_losses.append(train_loss.item())
    train_accuraccies.append(train_accuracy.item())
    valid_losses.append(val_loss.item())
    valid_accuracies.append(val_accuracy.item())

    lr_rates.append(optimizer.param_groups[0]['lr'])
    
    scheduler.step()

  plot_data['train_loss'] = train_losses
  plot_data['valid_loss'] = valid_losses
  plot_data['train_acc'] = train_accuraccies
  plot_data['valid_acc'] = valid_accuracies
  plot_data['lr'] = lr_rates

  plot_training_progress(SAVE_DIR, plot_data)

  """plt.subplot(2,2,1)
  plt.plot(train_losses, label="train")
  plt.plot(valid_losses, label="valid")
  plt.title("Loss")

  plt.subplot(2,2,2)
  plt.plot(train_accuraccies, label="train")
  plt.plot(valid_accuracies, label="valid")
  plt.title("Accuracy")

  plt.subplot(2,2,3)
  plt.plot(lr_rates)
  plt.title("Learning rate")

  plt.show()"""

  return model, criterion

def shuffle_data(data_x, data_y):
  indices = np.arange(data_x.shape[0])
  np.random.shuffle(indices)
  shuffled_data_x = np.ascontiguousarray(data_x[indices])
  shuffled_data_y = np.ascontiguousarray(data_y[indices])
  return shuffled_data_x, shuffled_data_y

def unpickle(file):
  fo = open(file, 'rb')
  dict = pickle.load(fo, encoding='latin1')
  fo.close()
  return dict

if __name__ == "__main__":
    config = {}
    config['max_epochs'] = 50
    config['batch_size'] = 50
    config['lr'] = 1e-1
    config['weight_decay'] = 1e-3

    DATA_DIR = './datasets/CIFAR/data/'
    SAVE_DIR = './out3/model2/'

    img_height = 32
    img_width = 32
    num_channels = 3
    num_classes = 10

    train_x = np.ndarray((0, img_height * img_width * num_channels), dtype=np.float32)
    train_y = []
    for i in range(1, 6):
      subset = unpickle(os.path.join(DATA_DIR, 'data_batch_%d' % i))
      train_x = np.vstack((train_x, subset['data']))
      train_y += subset['labels']
    train_x = train_x.reshape((-1, num_channels, img_height, img_width)).transpose(0, 2, 3, 1)
    train_y = np.array(train_y, dtype=np.int32)

    subset = unpickle(os.path.join(DATA_DIR, 'test_batch'))
    test_x = subset['data'].reshape((-1, num_channels, img_height, img_width)).transpose(0, 2, 3, 1).astype(np.float32)
    test_y = np.array(subset['labels'], dtype=np.int32)

    valid_size = 5000
    train_x, train_y = shuffle_data(train_x, train_y)
    valid_x = train_x[:valid_size, ...]
    valid_y = train_y[:valid_size, ...]
    train_x = train_x[valid_size:, ...]
    train_y = train_y[valid_size:, ...]
    data_mean = train_x.mean((0, 1, 2))
    data_std = train_x.std((0, 1, 2))

    train_x = (train_x - data_mean) / data_std
    valid_x = (valid_x - data_mean) / data_std
    test_x = (test_x - data_mean) / data_std

    train_x = train_x.transpose(0, 3, 1, 2)
    valid_x = valid_x.transpose(0, 3, 1, 2)
    test_x = test_x.transpose(0, 3, 1, 2)

    model, criterion = train(train_x, train_y, valid_x, valid_y, config, SAVE_DIR)
    
    print("Test:")
    evaluate(model, criterion, test_x, test_y, config['batch_size'], config['weight_decay'])

    show_worst_classified(model, criterion, test_x, test_y, data_mean, data_std, 20, DATA_DIR)