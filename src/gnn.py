# TODO
# Make each LSTM it's own network. See https://github.com/ethanfetaya/NRI/blob/master/modules.py line 460

import torch
from torch_geometric.data import Data, DataLoader
import csv
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GatedGraphConv
from torch.nn import Parameter as Param
#from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
from torch_geometric.utils import remove_self_loops, add_self_loops

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from tqdm import tqdm
import random
import configuration

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Execute(object):
    def __init__(self):

        data_list = []
        bulk_data = []

        # Load data
        with open('../data/{}'.format(configuration.filename), 'r') as f:
            reader = csv.reader(f)
            bulk_data = list(reader)

        labels = bulk_data.pop(0)

        # All-to-all, undirected connections between four assets - can modify in the future for n assets
        edge_index = torch.tensor(configuration.edge_index, dtype=torch.long)
        num_edges = edge_index.shape[1]
        period = int(configuration.period/configuration.sampling_rate)

        # Scale down dataset for prototyping
        bulk_data = bulk_data[:int(len(bulk_data)/configuration.scale_raw_data)]
        max_index = len(bulk_data)-len(bulk_data)%period  # Don't overshoot
        data = np.array(bulk_data[:max_index]).astype(float)

        batch_size = configuration.batch_size
        look_ahead = int(configuration.look_ahead*60/configuration.period)  # Steps to look ahead - 30 minutes
        self.num_epochs = configuration.num_epochs
        train_test_split = configuration.train_test_split
        look_back = int(configuration.look_back*60/configuration.period)  # Steps to look back
        label_size = configuration.label_size  # Only want to predict one value per node (future value)

        # Fill this with data loaders - still not sure how to save this
        loaders = []

        # Normalize data by row (by asset)
        data = data/np.amax(data, 0)
        datasets = multiplex(data, period)

        # Create list of Data objects
        for data in datasets:
            for row in range(look_back, len(data) - look_ahead):
                #x, y = pos_vel_next(data[row - 1], data[row], data[row + look_ahead - 1], data[row + look_ahead])
                x, y = gen_data(data[row-look_back:row], data[row+look_ahead])   

                data_list.append(Data(x=x, y=y, edge_index=edge_index))


        message_out = configuration.message_out  # Size of the output of messages
        input_dim = look_back  # Input dim to the LSTM
        output_dim = label_size  # Output of final linear embedding update
        hidden_dim = configuration.hidden_dim  # Hidden dim of LSTM
        num_layers = configuration.num_layers  # Number of LSTM layers

        self.model = MPNN(look_back, output_dim, message_out, input_dim, hidden_dim, num_layers, num_edges).to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.003, weight_decay=5e-5)

        train_data = data_list[0:int(len(data_list)*train_test_split)]
        test_data = data_list[int(len(data_list)*train_test_split):]

        self.train_data = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=10)
        self.test_data = DataLoader(test_data, batch_size=batch_size, shuffle=True, num_workers=10)

    def train(self):

        self.model.train()
        for epoch in range(self.num_epochs):
            loss_track = 0
            #if epoch == 3:
                #optimizer = torch.optim.Adam(model.parameters(), lr=0.0015, weight_decay=5e-5)
            #if epoch == 9:
                #optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-5)
            for batch in tqdm(self.train_data):
                
                self.model.init_hidden()
                self.optimizer.zero_grad()
                out = self.model(batch)
                loss = F.mse_loss(out, batch.y)
                
                loss.backward(retain_graph=True)
                loss_track += loss.item()
                self.optimizer.step()

            torch.save(self.model.state_dict(),"GNN_{}.pt".format("-".join(configuration.asset_names)))
                
            print("Epoch {}/{}, Avg loss: {}".format(epoch+1, self.num_epochs, loss_track/len(self.train_data)))

    def test(self):

        self.model.eval()

        loss_track = 0

        final_out = None
        final_batch_label = None
        final_batch_initial = None

        dir_correct = 0
        dir_wrong = 0

        # Track accuracies for individual assets
        assets = [[] for i in range(len(configuration.asset_names))]
        asset_names = configuration.asset_names

        for batch in tqdm(self.test_data):
            out = self.model(batch)
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
                if predicted_shift >= 0 and actual_shift >= 0:
                    dir_correct += 1
                    assets[i%len(assets)].append(1)
                else:
                    dir_wrong += 1
                    assets[i%len(assets)].append(0)


        print("Correct direction ACC: ", dir_correct/(dir_correct + dir_wrong))

        for i, scores in enumerate(assets):
            print("{}: {}% correct".format(asset_names[i], sum(scores)/len(scores)))


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

class MPNN(MessagePassing):
    def __init__(self, in_channels, out_channels, message_out, input_dim, hidden_dim, n_layers, num_edges):#, flow='target_to_source'):
        super(MPNN, self).__init__(aggr='add') #  "Max" aggregation.

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim  # Note: output_dim for LSTM = hidden_dim for now
        self.n_layers = n_layers
        self.lstm_layer = nn.LSTM(self.input_dim, self.hidden_dim, self.n_layers, batch_first=True)
        
        # Messages - MLP
        self.lin1 = torch.nn.Linear(in_channels, 128)
        self.LLn1 = nn.LayerNorm(128)
        self.lin2 = torch.nn.Linear(128, message_out)
        self.act1 = torch.nn.ReLU()
        self.act2 = torch.nn.ReLU()

        # Messages - LSTM
        self.lstm_message = nn.LSTM(self.input_dim, self.hidden_dim, self.n_layers, batch_first=True)
        self.l1_message = torch.nn.Linear(self.hidden_dim, message_out, bias=False)

        # Unique LSTMs for each interaction type - IN PROGRESS
        self.msg_lstm_list = nn.ModuleList([nn.LSTM(self.input_dim, self.hidden_dim, self.n_layers, batch_first=True) for _ in range(num_edges)])

        # Always assume this - batching handles batch updates
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

        x, edge_index = data.x, data.edge_index
        num_nodes = data.num_nodes
        
        # Not sure if these two are necessary
        edge_index, _ = remove_self_loops(edge_index)
        edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)
        
        return self.propagate(edge_index, size=(num_nodes, num_nodes), x=x)


    # Need to update this and maybe __collect__ manually
    def propagate(self, edge_index, size=None, **kwargs):
        """The initial call to start propagating messages.

        Args:
            edge_index (Tensor): The indices of a general (sparse) assignment
                matrix with shape :obj:`[N, M]` (can be directed or
                undirected).
            size (list or tuple, optional): The size :obj:`[N, M]` of the
                assignment matrix. If set to :obj:`None`, the size will be
                automatically inferred and assumed to be quadratic.
                (default: :obj:`None`)
            **kwargs: Any additional data which is needed to construct and
                aggregate messages, and to update node embeddings.
        """

        size = [None, None] if size is None else size
        size = [size, size] if isinstance(size, int) else size
        size = size.tolist() if torch.is_tensor(size) else size
        size = list(size) if isinstance(size, tuple) else size
        assert isinstance(size, list)
        assert len(size) == 2

        kwargs = self.__collect__(edge_index, size, kwargs)

        msg_kwargs = self.__distribute__(self.__msg_params__, kwargs)
        out = self.message(**msg_kwargs)

        aggr_kwargs = self.__distribute__(self.__aggr_params__, kwargs)
        out = self.aggregate(out, **aggr_kwargs)

        update_kwargs = self.__distribute__(self.__update_params__, kwargs)
        out = self.update(out, **update_kwargs)

        return out

    def message(self, x_j):

        if configuration.message_type == 'mlp':

            x_j = self.act1(self.LLn1(self.lin1(x_j)))
            x_j = self.act2(self.lin2(x_j))

        if configuration.message_type == 'lstm':

            self.init_hidden()
            x_j, self.hidden = self.lstm_message(x_j.unsqueeze(0), self.hidden)
            x_j = self.act1(x_j)
            x_j = self.act2(self.l1_message(x_j.squeeze(0)))

        return x_j

    def update(self, aggr_out, x):

        new_embedding, self.hidden = self.lstm_layer(x.unsqueeze(0), self.hidden)
        new_embedding = torch.cat([aggr_out, new_embedding.squeeze(0)], dim=1).unsqueeze(0)
        new_embedding = self.update_act1(self.ULn1(self.update_lin1(new_embedding)))
        new_embedding = self.update_act2(self.update_lin2(new_embedding))
        
        return new_embedding.squeeze(0)

beast = Execute()
beast.train()
beast.test()

