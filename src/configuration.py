
# Data parameters
asset_names = ['BTC','ETH','XLM','CVC']
filename = 'BTC-ETH-XLM-CVC_5s.csv'
sampling_rate = 5          # Sampling rate of the raw data

# Training parameters
period = 30                # Period to feed into network (seconds)
batch_size = 1024          # Batch size
look_ahead = 30            # Minutes to look ahead
num_epochs = 1             # Number of epochs
train_test_split = 0.8     # Train/test split
look_back = 60             # Minutes to look back
label_size = 1             # Out put dimension - default to predicting one value per node (future asset price)

# Modify connections (default to fully connected)
edge_index = [[0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3],
              [1, 2, 3, 0, 2, 3, 0, 1, 3, 0, 1, 2]]

scale_raw_data = 1         # Scale down raw data for prototyping (len(dataset) -> len(dataset)/scale_raw_data)

# Network parameters
message_out = 8            # Message embedding size
hidden_dim = 16            # Hidden dim of the update LSTM
num_layers = 1             # Number of LSTM layers in update network
message_type = 'lstm'      # Network type for message massing
