import torch
import torch.nn as nn
import torch.optim as optim


## Definicija računskog grafa
# podaci i parametri, inicijalizacija parametara
a = torch.randn(1, requires_grad=True)
b = torch.randn(1, requires_grad=True)

#X = torch.tensor([1, 2])
#Y = torch.tensor([3, 5])

X = torch.tensor([1, 2, 0])
Y = torch.tensor([3, 5, 2])

# optimizacijski postupak: gradijentni spust
optimizer = optim.SGD([a, b], lr=0.1)

for i in range(100):
    # afin regresijski model
    Y_ = a*X + b

    diff = (Y-Y_)

    # kvadratni gubitak
    loss = torch.mean(diff**2) #sum -> mean

    # računanje gradijenata
    loss.backward()

    # korak optimizacije
    optimizer.step()
    
    a_grad = (-2) * torch.mean(X * (Y - ((a * X) + b)))
    b_grad = (-2) * torch.mean(Y - ((a * X) + b))

    print(f'a_gradijent:{a.grad}, b_gradijent:{b.grad}')
    print(f'at_gradijent:{a_grad}, bt_gradijent:{b_grad}')

    # Postavljanje gradijenata na nulu
    optimizer.zero_grad()

    print(f'step: {i}, loss:{loss}, Y_:{Y_}, a:{a}, b {b}')
    print()