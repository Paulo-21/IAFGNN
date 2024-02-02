import torch

x = torch.tensor(3., requires_grad=True)
y = torch.tensor(3., requires_grad=True)

def f(x, y) :
    return x**2+y**2

lr = 0.1
#o = res.grad()
for _ in range(200):
    res = f(x, y)
    res.backward()
    with torch.no_grad():
        x -= lr*x.grad
        y -= lr*y.grad
    x.grad.zero_()
    y.grad.zero_()

print(x)
print(y)
res = f(x, y)
print(res)