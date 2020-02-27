import torch
from torch_geometric.data import Data, DataLoader
import csv
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

data_list = []
bulk_data = []

with open('BTC-ETH-XLM-CVC_5s.csv', 'r') as f:
    reader = csv.reader(f)
    bulk_data = list(reader)

labels = bulk_data.pop(0)

# All-to-all, undirected connections between four assets - can modify in the future for n assets
edge_index = torch.tensor([[0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3],
                           [1, 2, 3, 0, 2, 3, 0, 1, 3, 0, 1, 2]], dtype=torch.long)


period = 3  # 3*5s = 15s period
max_index = len(bulk_data)-len(bulk_data)%period  # Don't overshoot
data = np.array(bulk_data[:max_index]).astype(float)
batch_size = 32
look_ahead = 2  # Steps to look ahead, will become the label

# Fill this with data loaders - still not sure how to save this
loaders = []

def pos_vel_next(last, current, look_ahead):

    x = []
    y = []

    last = [float(i) for i in last]
    current = [float(i) for i in current]
    look_ahead = [float(i) for i in look_ahead]

    for i, curr in enumerate(current):
        pos_i = curr
        vel_i = curr - last[i]
        label = look_ahead[i]
        x.append([pos_i])
        y.append([label])


    x = torch.tensor(x, dtype=torch.float)
    y = torch.tensor(y, dtype=torch.float)

    return x, y

'''
# Multiplex periods of > 1 second
for i in range(period):

    train_episode = []
    val_episode = []

    # Filter data s.t. only rows corresponding to that period remain
    period_indices = [j for j in range(i, len(data), period)]
    tempdata = data[period_indices]

    # Swap col/rows for easier access 
    tempdata = list(tempdata.transpose())

    # Normalize by row

    for row in range(len(tempdata) - look_ahead):

        x, y = pos_vel_next(tempdata[row - 1], tempdata[row], tempdata[row + look_ahead])

        data_list.append(Data(x=x, y=y, edge_index=edge_index))


    loader = DataLoader(data_list, batch_size=batch_size)

    #torch.save(data_list, "{}_p{}_la{}.pt".format("_".join(labels), period, look_ahead))

    loaders.append(loader)

    data_list = []
'''

print("max 0 ", np.amax(data, 0))
data = data/np.amax(data, 0)
print("max 0 ", np.amax(data, 0))

for i in range(5):
    print(data[i])
    print(data[len(data) - i - 1])


for row in range(len(data) - look_ahead):
    x, y = pos_vel_next(data[row - 1], data[row], data[row + look_ahead])
    data_list.append(Data(x=x, y=y, edge_index=edge_index))


print(len(loaders))


from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = GCNConv(1, 4)
        self.conv2 = GCNConv(4, 1)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        return x


from tqdm import tqdm
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Net().to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

train_test_split = 0.8
#data = data[0:int(len(data)/10)]
train_data = data_list[0:int(len(data_list)*train_test_split)]
test_data = data_list[int(len(data_list)*train_test_split):]


model.train()

for epoch in range(1):
    for example in tqdm(train_data):
        #print("Data: ", data)


        optimizer.zero_grad()
        out = model(example)

        loss = F.mse_loss(out, example.y)
        #print("Out =    ", out.squeeze(1))
        #print("Data.y = ", data.y.squeeze(1))
        #print("Loss = ", loss)
        loss.backward()
        optimizer.step()














