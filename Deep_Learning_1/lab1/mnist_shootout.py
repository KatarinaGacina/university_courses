import torch
import torchvision

import numpy as np
import matplotlib.pyplot as plt

from sklearn.svm import SVC
from sklearn import metrics

import data

import pt_deep

def refractor(x, y, device):    
    N = x.shape[0]
    D = x.shape[1] * x.shape[2]
    C = y.max().add_(1).item()

    X = x.view(N, -1)

    Y = torch.zeros(y.shape[0], C)
    Y.scatter_(1, y.unsqueeze(1).to(torch.int64), 1)

    X = X.to(device)
    Y = Y.to(device)

    return X, Y

def train2(model, X, Yoh_, param_niter, param_delta):
    optimizer = torch.optim.SGD(model.parameters(), lr=param_delta)

    loss_list = []
    for i in range(param_niter):
        loss = model.get_loss(X, Yoh_)

        loss.backward()
        optimizer.step()
        
        if i % 100 == 0:
            print(f'Iteracija {i}: loss= {loss.item()}')
        loss_list.append(loss.detach().cpu().numpy())

        optimizer.zero_grad()

    return loss_list

def train3(model, X_train, Yoh_train, X_val, Yoh_val, param_niter, param_delta):
    optimizer = torch.optim.SGD(model.parameters(), lr=param_delta)

    min_loss = None
    best_W = None
    best_b = None

    br = 0

    loss_list = []
    for i in range(param_niter):
        loss = model.get_loss(X_train, Yoh_train)

        loss.backward()
        optimizer.step()
        
        if i % 10 == 0:
            print(f'Iteracija {i}: loss= {loss.item()}')
        
        optimizer.zero_grad()

        loss_list.append(loss.detach().cpu().numpy())
        
        with torch.no_grad():
            val_loss = model.get_loss(X_val, Yoh_val)
            
        if min_loss is None or min_loss > val_loss:
            min_loss = val_loss

            best_W = model.W
            best_b = model.b

            br = 0
        else:
            br += 1

            if br >= 10:
                model.W = best_W
                model.b = best_b
                break

    return loss_list, best_W, best_b

def train_mb(model, X_train, Yoh_train, param_niter, param_delta, num_batches):
    optimizer = torch.optim.SGD(model.parameters(), lr=param_delta)
    
    batch_size = y_train.shape[0] // num_batches

    for i in range(param_niter):
        indices = list(range(y_train.shape[0]))
        np.random.shuffle(indices)

        X_train = X_train[indices]
        Yoh_train = Yoh_train[indices]

        for x, y in zip(torch.split(X_train, batch_size), torch.split(Yoh_train, batch_size)):
            loss = model.get_loss(x, y)

            loss.backward()
            optimizer.step()
            
            if i % 10 == 0:
                print(f'Iteracija {i}: loss= {loss.item()}')
            
            optimizer.zero_grad()

def train56(model, X_train, Yoh_train, param_niter, num_batches):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    gamma = 1 - 1e-4
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
    
    batch_size = y_train.shape[0] // num_batches

    for i in range(param_niter):
        indices = list(range(y_train.shape[0]))
        np.random.shuffle(indices)

        X_train = X_train[indices]
        Yoh_train = Yoh_train[indices]

        for x, y in zip(torch.split(X_train, batch_size), torch.split(Yoh_train, batch_size)):
            loss = model.get_loss(x, y)

            loss.backward()
            optimizer.step()
            
            if i % 10 == 0:
                print(f'Iteracija {i}: loss= {loss.item()}')
            
            optimizer.zero_grad()
        
        scheduler.step()

def eval(model, X, Y):
    print(data.eval_perf_multi(np.argmax(pt_deep.eval(model, X), axis=1), Y))

def zad1(x_train, y_train, device):
    X_train, Y_train = refractor(x_train, y_train, device)

    model = pt_deep.PTDeep([784, 10], torch.relu, 1)
    model.to(device)

    pt_deep.train(model, X_train, Y_train, 10000, 0.1)

    digits_w = model.W[0].detach().cpu()
    digits = digits_w.transpose(0, 1).reshape((-1, 28, 28))

    for i, w in enumerate(digits.numpy()):
        #print(w)
        plt.subplot(2, 5, i+1)
        plt.imshow(w, cmap='gray')
        plt.axis("off")
        plt.title(f' {i}')

    plt.show()

def zad2(x_train, x_test, y_train, y_test, device):
    lambda_par = 0

    X_train, Y_train = refractor(x_train, y_train, device)
    X_test, _ = refractor(x_test, y_test, device)

    model1 = pt_deep.PTDeep([784, 10], torch.relu, lambda_par)
    model1.to(device)

    loss1 = train2(model1, X_train, Y_train, 2000, 0.1)

    model2 = pt_deep.PTDeep([784, 100, 10], torch.relu, lambda_par)
    model2.to(device)

    loss2 = train2(model2, X_train, Y_train, 2000, 0.1)

    model3 = pt_deep.PTDeep([784, 100, 100, 10], torch.relu, lambda_par)
    model3.to(device)

    loss3 = train2(model3, X_train, Y_train, 2000, 0.1)

    model4 = pt_deep.PTDeep([784, 100, 100, 100, 10], torch.relu, lambda_par)
    model4.to(device)

    loss4 = train2(model4, X_train, Y_train, 2000, 0.1)

    print("Train dataset:")
    eval(model1, X_train, y_train)
    eval(model2, X_train, y_train)
    eval(model3, X_train, y_train)
    eval(model4, X_train, y_train)

    print("Test dataset:")
    eval(model1, X_test, y_test)
    eval(model2, X_test, y_test)
    eval(model3, X_test, y_test)
    eval(model4, X_test, y_test)

    plt.plot(loss1, label='model1_loss')
    plt.plot(loss2, label='model2_loss')
    plt.plot(loss3, label='model3_loss')
    plt.plot(loss4, label='model4_loss')

    plt.legend()
    plt.show()

def zad3(x_train, x_test, y_train, y_test, device):
    indices = list(range(y_train.shape[0]))
    np.random.shuffle(indices)

    n = len(indices) // 5
    
    x_val = x_train[indices][(len(indices) - n):]
    x_train = x_train[indices][:(len(indices) - n)]

    y_val = y_train[indices][(len(indices) - n):]
    y_train = y_train[indices][:(len(indices) - n)]

    model = pt_deep.PTDeep([784, 10], torch.relu, 1e-4)
    model.to(device)

    X_train, Y_train = refractor(x_train, y_train, device)
    X_val, Y_val = refractor(x_val, y_val, device)
    
    train3(model, X_train, Y_train, X_val, Y_val, 2000, 0.1)

    X_test, _ = refractor(x_test, y_test, device)

    print("Test dataset:")
    eval(model, X_test, y_test)

def zad4(x_train, x_test, y_train, y_test, device):
    model = pt_deep.PTDeep([784, 10], torch.relu, 1e-4)
    model.to(device)

    X_train, Y_train = refractor(x_train, y_train, device)
    
    train_mb(model, X_train, Y_train, 2000, 0.1, 10)

    X_test, _ = refractor(x_test, y_test, device)

    print("Test dataset:")
    eval(model, X_test, y_test)

def zad56(x_train, x_test, y_train, y_test, device):
    model = pt_deep.PTDeep([784, 10], torch.relu, 1e-4)
    model.to(device)

    X_train, Y_train = refractor(x_train, y_train, device)
    
    train56(model, X_train, Y_train, 2000, 10)

    X_test, _ = refractor(x_test, y_test, device)

    print("Test dataset:")
    eval(model, X_test, y_test)

def zad7(x_train, y_train):
    model = pt_deep.PTDeep([784, 100, 100, 10], torch.relu, 1e-4)

    X_train, Y_train = refractor(x_train, y_train, "cpu")
    
    print(model.get_loss(X_train, Y_train))

def zad8(x_train, x_test, y_train, y_test):
    X_train, _ = refractor(x_train, y_train, "cpu")
    X_train = X_train.numpy()

    X_test, _ = refractor(x_test, y_test, "cpu")
    X_test = X_test.numpy()

    svm_linear = SVC(kernel="linear", decision_function_shape="ovo").fit(X_train, y_train.numpy())

    y_linear = svm_linear.predict(X_test)
    print(data.eval_perf_multi(y_linear, y_test))

    print()

    svm_rbf = SVC(kernel="rbf", decision_function_shape="ovo").fit(X_train, y_train)
    
    y_rbf = svm_rbf.predict(X_test)
    print(data.eval_perf_multi(y_rbf, y_test))

    print()

    print(metrics.classification_report(y_test, y_linear))
    print()
    print(metrics.classification_report(y_test, y_rbf))


if __name__=="__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset_root = './data'  # change this to your preference
    mnist_train = torchvision.datasets.MNIST(dataset_root, train=True, download=True)
    mnist_test = torchvision.datasets.MNIST(dataset_root, train=False, download=True)

    x_train, y_train = mnist_train.data, mnist_train.targets
    x_test, y_test = mnist_test.data, mnist_test.targets
    x_train, x_test = x_train.float().div_(255.0), x_test.float().div_(255.0)

    #zad1(x_train, y_train, device)

    zad2(x_train, x_test, y_train, y_test, device)

    #zad3(x_train, x_test, y_train, y_test, device)

    #zad4(x_train, x_test, y_train, y_test, device)

    #zad56(x_train, x_test, y_train, y_test, device)

    #zad7(x_train, y_train)

    #zad8(x_train, x_test, y_train, y_test)



