import torch
from torch import nn
import numpy as np
from matplotlib import pyplot as plt
from visdom import Visdom

def mseLoss(pred, target):
    return torch.square(pred-y).sum()/pred.shape[0]

'''生成训练和测试数据'''
def generate_data():
    w = np.array([1.2])
    b = np.array([4.])
    x = np.linspace(-100, 100, 50)
    y = w * x + b
    error = np.random.randn(y.shape[0]) * 5
    return x,y+error
device =  torch.device('cuda:0')
epoch = 500
lr = 1e-4
net = nn.Linear(1,1)
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(net.parameters(), lr=lr, weight_decay=1e-5)
viz = Visdom()
viz.line([0.], [0.], win='train_loss', opts=dict(title='train_loss',
                                                 legend=['train']))
x ,y = generate_data()
x, y = torch.tensor(x).unsqueeze(1), torch.tensor(y)

net.to(device)
criterion.to(device)
x, y = x.to(device).float(), y.to(device).float()



for i in range(epoch):
    pred = net(x)
    pred = pred.squeeze(1)
    loss = mseLoss(pred, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print('epoch:{} loss:{}'.format(i, loss.item()))
    viz.line([loss.item()], [i], win='train_loss', update='append')

x_test = torch.linspace(-100, 100, 50).reshape(-1, 1).to(device)
y_test = net(x_test)

x_test, y_test = x_test.squeeze(1).cpu(), y_test.squeeze(1).detach().cpu().numpy()
x, y = x.squeeze(1).cpu(), y.cpu()
plt.scatter(x, y, marker='*')
plt.plot(x_test, y_test)
plt.show()