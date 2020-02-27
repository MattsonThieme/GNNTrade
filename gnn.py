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
look_ahead = 60  # Steps to look ahead, will become the label

# Fill this with data loaders - still not sure how to save this
loaders = []

def pos_vel_next(last, current, last_look_ahead, look_ahead):

    x = []
    y = []

    last = [float(i) for i in last]
    current = [float(i) for i in current]
    look_ahead = [float(i) for i in look_ahead]
    last_look_ahead = [float(i) for i in last_look_ahead]

    for i, curr in enumerate(current):
        pos_i = curr
        vel_i = curr - last[i]
        pos_ahead = look_ahead[i]
        vel_ahead = pos_ahead - last_look_ahead[i]
        x.append([pos_i, vel_i])
        y.append([pos_ahead, vel_ahead])


    x = torch.tensor(x, dtype=torch.float)
    y = torch.tensor(y, dtype=torch.float)

    return x, y

# Normalize data by row
data = data/np.amax(data, 0)

# Create list of Data objects
for row in range(len(data) - look_ahead):
    x, y = pos_vel_next(data[row - 1], data[row], data[row + look_ahead - 1], data[row + look_ahead])
    data_list.append(Data(x=x, y=y, edge_index=edge_index))


from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

# Simple 2 layer GCN
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = GCNConv(2, 4)
        self.conv2 = GCNConv(4, 2)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, 0.0)#training=self.training)
        x = self.conv2(x, edge_index)

        return x


from tqdm import tqdm
import random
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Net().to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

train_test_split = 0.8


#data = data[0:int(len(data)/10)]
train_data = data_list[0:int(len(data_list)*train_test_split)]
test_data = data_list[int(len(data_list)*train_test_split):]


train_data = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_data = DataLoader(test_data, batch_size=batch_size, shuffle=True)



model.train()

for epoch in range(2):
    loss_track = 0
    for batch in tqdm(train_data):

        optimizer.zero_grad()
        out = model(batch)

        loss = F.mse_loss(out, batch.y)
        loss_track += loss.item()
        #print("Out =    ", out.squeeze(1))
        #print("Data.y = ", data.y.squeeze(1))
        #print("Loss = ", loss)
        loss.backward()
        optimizer.step()

    print("Avg loss: ", loss_track/len(train_data))

model.eval()

loss = 0

final_out = None
final_batch = None

for batch in tqdm(test_data):
        out = model(batch)
        loss += F.mse_loss(out, batch.y)
        final_out = out
        final_batch = batch.y

print("Final loss: ", loss/len(train_data))



print(final_out)
print(final_batch)

'''
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

'''












