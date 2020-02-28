# TODO

# Randomize data_list
# Check if we're predicting the right velocity and if we did indeed move up or down in the final states
# Organize/cleanup
# Save models


import torch
from torch_geometric.data import Data, DataLoader
import csv
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GatedGraphConv
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from tqdm import tqdm
import random

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

#bulk_data = bulk_data[:int(len(bulk_data)/5)]

max_index = len(bulk_data)-len(bulk_data)%period  # Don't overshoot
data = np.array(bulk_data[:max_index]).astype(float)
batch_size = 1024
look_ahead = 12  # Steps to look ahead, will become the label
num_epochs = 15
train_test_split = 0.8

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

from torch.nn import Parameter as Param
from torch_geometric.nn.conv import MessagePassing

# Simple 2 layer GCN
class GCN(MessagePassing):
    def __init__(self):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(2, 4)
        self.conv2 = GCNConv(4, 2)
        #self.ggc1 = GatedGraphConv(2,2)

    def forward(self, data):

        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, 0.0)
        x = self.conv2(x, edge_index)
        #x = self.ggc1(x, edge_index)
        return x

from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
from torch_geometric.utils import remove_self_loops, add_self_loops

class SAGEConv(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(SAGEConv, self).__init__(aggr='add') #  "Max" aggregation.
        self.lin1 = torch.nn.Linear(in_channels, out_channels)
        self.act1 = torch.nn.ReLU()
        #self.lin2 = torch.nn.Linear(in_channels, out_channels)
        #self.act2 = torch.nn.ReLU()
        self.update_lin = torch.nn.Linear(in_channels + out_channels, in_channels, bias=True)
        self.update_act = torch.nn.ReLU()
        
    def forward(self, data):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]

        x, edge_index = data.x, data.edge_index
        num_nodes = data.num_nodes
        
        edge_index, _ = remove_self_loops(edge_index)
        edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)
        
        
        return self.propagate(edge_index, size=(num_nodes, num_nodes), x=x)

    def message(self, x_j):
        # x_j has shape [E, in_channels]

        x_j = self.lin1(x_j)
        x_j = self.act1(x_j)
        #x_j = self.lin2(x_j)
        #x_j = self.act2(x_j)
        
        return x_j

    def update(self, aggr_out, x):
        # aggr_out has shape [N, out_channels]

        new_embedding = torch.cat([aggr_out, x], dim=1)
        
        new_embedding = self.update_lin(new_embedding)
        new_embedding = self.update_act(new_embedding)
        
        return new_embedding


model = SAGEConv(2,2).to(device)
#model = GCN().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-5)#5e-4)

train_data = data_list[0:int(len(data_list)*train_test_split)]
test_data = data_list[int(len(data_list)*train_test_split):]

train_data = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_data = DataLoader(test_data, batch_size=batch_size, shuffle=True)


model.train()

for epoch in range(num_epochs):
    loss_track = 0
    for batch in tqdm(train_data):

        optimizer.zero_grad()
        out = model(batch)
        loss = F.mse_loss(out, batch.y)
        #loss = nn.l1_loss(out, batch.y)
        loss_track += loss.item()
        loss.backward()
        optimizer.step()

    print("Epoch {}/{}, Avg loss: {}".format(epoch, num_epochs, loss_track/len(train_data)))

model.eval()

loss_track = 0

final_out = None
final_batch_label = None
final_batch_initial = None

for batch in tqdm(test_data):
        out = model(batch)
        loss_track += F.mse_loss(out, batch.y).item()
        final_out = out
        final_batch_label = batch.y
        final_batch_initial = batch.x

dir_correct = 0
dir_wrong = 0

# Calculate correct moves
for i, x in enumerate(final_batch_label):

    initial = final_batch_initial[i].data[0]
    final = x.data[0]
    pred = final_out[i].data[0]
    actual_shift = 100*(final - initial)/final
    predicted_shift = 100*(pred - initial)/final
    price_dir = final - initial
    pred_dir = pred - initial

    if predicted_shift < 0 and actual_shift < 0:
        dir_correct += 1
    if predicted_shift > 0 and actual_shift > 0:
        dir_correct += 1
    else:
        dir_wrong += 1


    print("{} to {} shifted {}% | pred off by {}%".format(initial, final, actual_shift, predicted_shift))

print("Correct direction ACC: ", dir_correct/(dir_correct + dir_wrong))

#print(final_out)
#print(final_batch)

