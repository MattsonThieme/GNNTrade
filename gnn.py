# TODO

# Randomize data_list
# Check if we're predicting the right velocity and if we did indeed move up or down in the final states
# Organize/cleanup
# Save models

# Don't count zero shifts negatively
# Keep track of accuracy for each asset individually



############

# Am I predicting one value for all four assets?!

###########

# Make message passing operation an LSTM
# Make each LSTM it's own network. See https://github.com/ethanfetaya/NRI/blob/master/modules.py line 460




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


period = 6  # 6*5s = 30s period

# Scale down dataset for prototyping
#bulk_data = bulk_data[:int(len(bulk_data)/10)]

max_index = len(bulk_data)-len(bulk_data)%period  # Don't overshoot
data = np.array(bulk_data[:max_index]).astype(float)
batch_size = 1024
look_ahead = 60  # Steps to look ahead - 30 minutes
num_epochs = 15
train_test_split = 0.8
look_back = 60  # Steps to look back - 1 hr
label_size = 1  # Only want to predict

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

def gen_data(history, look_ahead):
    x = []
    y = []
    for i in range(len(history[0])):
        x.append(history[:,i])
        y.append([look_ahead[i]])



    x = torch.tensor(x, dtype=torch.float)
    y = torch.tensor(y, dtype=torch.float)

    return x, y


def multiplex(data, period):
    max_index = len(data)-len(data)%period  # Don't overshoot
    data = np.array(data[:max_index])

    datasets = []

    # Multiplex periods of > 1 second
    for i in range(period):
        # Filter data s.t. only rows corresponding to that period remain
        period_indices = [j for j in range(i, len(data), period)]
        tempdata = data[period_indices]
        datasets.append(tempdata)


    return datasets

# Normalize data by row
data = data/np.amax(data, 0)

datasets = multiplex(data, period)

# Create list of Data objects
for data in datasets:
    for row in range(look_back, len(data) - look_ahead):
        #x, y = pos_vel_next(data[row - 1], data[row], data[row + look_ahead - 1], data[row + look_ahead])
        x, y = gen_data(data[row-look_back:row], data[row+look_ahead])
        #print("SHAPE ", y.shape, y)    

        data_list.append(Data(x=x, y=y, edge_index=edge_index))

from torch.nn import Parameter as Param
from torch_geometric.nn.conv import MessagePassing

# Simple 2 layer GCN
class GCN(MessagePassing):
    def __init__(self):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(2, 4)
        self.conv2 = GCNConv(4, 2)

    def forward(self, data):

        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, 0.0)
        x = self.conv2(x, edge_index)
        return x

from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
from torch_geometric.utils import remove_self_loops, add_self_loops


# Try changing the update function to be an LSTM, and the massage function to be still an MLP
# Need to change parser to include x price steps back, reasonable
class SAGEConv(MessagePassing):
    def __init__(self, in_channels, out_channels, message_out, input_dim, hidden_dim, n_layers):
        super(SAGEConv, self).__init__(aggr='add') #  "Max" aggregation.
        self.lin1 = torch.nn.Linear(in_channels, 128)
        self.LLn1 = nn.LayerNorm(128)
        self.lin2 = torch.nn.Linear(128, message_out)
        self.act1 = torch.nn.ReLU()
        self.act2 = torch.nn.ReLU()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim  # Note: output_dim for LSTM = hidden_dim for now
        self.n_layers = n_layers
        self.lstm_layer = nn.LSTM(self.input_dim, self.hidden_dim, self.n_layers, batch_first=True)
        #self.lstm_layer = nn.GRU(self.input_dim, self.hidden_dim, self.n_layers, batch_first=True)

        self.batch_size = 1
        self.seq_len = 1

        self.inp = torch.randn(self.batch_size, self.seq_len, self.input_dim)
        self.hidden_state = torch.randn(self.n_layers, self.batch_size, self.hidden_dim)
        self.cell_state = torch.randn(self.n_layers, self.batch_size, self.hidden_dim)
        self.hidden = (self.hidden_state, self.cell_state)

        
        #self.update_lin = torch.nn.Linear(in_channels + out_channels, 1, bias=True)
        self.update_lin1 = torch.nn.Linear(hidden_dim + message_out, 128, bias=False)
        self.ULn1 = nn.LayerNorm(128)
        self.update_act1 = torch.nn.ReLU()
        self.update_lin2 = torch.nn.Linear(128, out_channels, bias=False)
        self.update_act2 = torch.nn.ReLU()
    
    def init_hidden(self):
        self.inp = torch.randn(self.batch_size, self.seq_len, self.input_dim)
        self.hidden_state = torch.randn(self.n_layers, self.batch_size, self.hidden_dim)
        self.cell_state = torch.randn(self.n_layers, self.batch_size, self.hidden_dim)
        self.hidden = (self.hidden_state, self.cell_state)

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

        x_j = self.act1(self.LLn1(self.lin1(x_j)))
        x_j = self.act2(self.lin2(x_j))
        
        return x_j

    def update(self, aggr_out, x):

        #self.hidden_state.detach_()
        #self.cell_state.detach_()

        new_embedding, self.hidden = self.lstm_layer(x.unsqueeze(0), self.hidden)

        # Detach hidden and cell states from the computation graph after each batch, otherwise training slows to a halt
        #self.hidden_state.detach_()
        #self.cell_state.detach_()

        new_embedding = torch.cat([aggr_out, new_embedding.squeeze(0)], dim=1).unsqueeze(0)

        new_embedding = self.update_act1(self.ULn1(self.update_lin1(new_embedding)))
        new_embedding = self.update_act2(self.update_lin2(new_embedding))
        
        return new_embedding.squeeze(0)

#model = SAGEConv(2,2).to(device)

message_out = 8  # Size of the output of messages
input_dim = look_back  # Input dim to the LSTM
output_dim = label_size  # Output of final linear embedding update
hidden_dim = 16  # Hidden dim of LSTM
num_layers = 1  # Number of LSTM layers

model = SAGEConv(look_back, output_dim, message_out, input_dim, hidden_dim, num_layers).to(device)
#model = GCN().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-5)

train_data = data_list[0:int(len(data_list)*train_test_split)]
test_data = data_list[int(len(data_list)*train_test_split):]

train_data = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=10)
test_data = DataLoader(test_data, batch_size=batch_size, shuffle=True, num_workers=10)


model.train()
from torch.autograd import Variable
for epoch in range(num_epochs):
    loss_track = 0
    if epoch == 3:
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0015, weight_decay=5e-5)
    if epoch == 9:
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-5)
    for batch in tqdm(train_data):
        
        model.init_hidden()
        optimizer.zero_grad()
        out = model(batch)
        loss = F.mse_loss(out, batch.y)
        
        loss.backward(retain_graph=True)
        loss_track += loss.item()
        optimizer.step()
        #model.hidden_state.detach_()
        #model.cell_state.detach_()
        

    print("Epoch {}/{}, Avg loss: {}".format(epoch+1, num_epochs, loss_track/len(train_data)))

model.eval()

loss_track = 0

final_out = None
final_batch_label = None
final_batch_initial = None

dir_correct = 0
dir_wrong = 0

# Track accuracies for individual assets
assets = [[],[],[],[]]
asset_names = ['BTC','ETH','XLM','CVC']

for batch in tqdm(test_data):
    out = model(batch)
    loss_track += F.mse_loss(out, batch.y).item()
    final_out = out
    final_batch_label = batch.y
    final_batch_initial = batch.x

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
            assets[i%len(assets)].append(1)
        if predicted_shift > 0 and actual_shift > 0:
            dir_correct += 1
            assets[i%len(assets)].append(1)
        else:
            dir_wrong += 1
            assets[i%len(assets)].append(0)


    #print("{} to {} shifted {}% | pred off by {}%".format(initial, final, actual_shift, predicted_shift))

print("Correct direction ACC: ", dir_correct/(dir_correct + dir_wrong))

for i, scores in enumerate(assets):
    print("{}: {}% correct".format(asset_names[i], sum(scores)/len(scores)))


