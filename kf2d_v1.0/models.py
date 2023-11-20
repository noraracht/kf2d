import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Bidirectional recurrent neural network (many-to-one)
class BiRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(BiRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, num_classes)  # 2 for bidirection

    def forward(self, x):
        # Set initial states
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(device)  # 2 for bidirection
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(device)

        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size*2)

        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        #print(out)
        return out

# Linear network with one hidden layer
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        #self.celu = nn.CELU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        #out = self.celu(out)
        out = self.fc2(out)
        
        return out


class NeuralNetClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, embedding_size, num_classes):
        super(NeuralNetClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, embedding_size)
        self.fc3 = nn.Linear(embedding_size, num_classes)
        self.relu = nn.ReLU()
        # self.celu = nn.CELU()


    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        # out = self.celu(out)
        out_mse = self.fc2(out)
        out = self.relu(out_mse)
        out = F.log_softmax(self.fc3(out), dim=1)
        #print(out.size())

        return out_mse, out


class NeuralNetClassifierForked(nn.Module):
    def __init__(self, input_size, hidden_size, embedding_size, num_classes):
        super(NeuralNetClassifierForked, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, embedding_size)
        self.fc3 = nn.Linear(hidden_size, num_classes)
        self.relu = nn.ReLU()
        # self.celu = nn.CELU()


    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        # out = self.celu(out)

        out_mse = self.fc2(out)

        #out = self.relu(out_mse)
        out = F.log_softmax(self.fc3(out), dim=1)
        #print(out.size())

        return out_mse, out



class NeuralNetClassifierOnly(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNetClassifierOnly, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, num_classes)
        self.relu = nn.ReLU()
        # self.celu = nn.CELU()


    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = F.log_softmax(self.fc3(out), dim=1)
        #print(out.size())

        return out



class NeuralNetClassifierTrans(nn.Module):
    def __init__(self, input_size, hidden_size, embedding_size, num_classes):
        super(NeuralNetClassifierTrans, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, embedding_size)

        encoder_layer = nn.TransformerEncoderLayer(d_model=embedding_size, nhead=16, batch_first=True, dropout=0.0)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)

        self.fc3 = nn.Linear(embedding_size, num_classes)
        self.relu = nn.ReLU()
        # self.celu = nn.CELU()


    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        # out = self.celu(out)
        out = self.fc2(out)

        out_mse = out

        out = out.unsqueeze(0)
        out = self.transformer_encoder(out)
        out = out.squeeze(0)

        out_trans = out

        #print(out.shape)

        #out = self.relu(out)
        out = F.log_softmax(self.fc3(out), dim=1)
        #print(out.size())


        return out_mse, out_trans, out



# Linear network with two hidden layers
class NeuralNet_2layer(nn.Module):
    def __init__(self, input_size, hidden_size_fc1, hidden_size_fc2, embedding_size):
        # Encoder Network
        super(NeuralNet_2layer, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size_fc1)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size_fc1, hidden_size_fc2)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size_fc2, embedding_size)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu1(out)
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.fc3(out)
        return out



# Linear network with two hidden layers
class CNN_network(nn.Module):
    def __init__(self, input_size, hidden_size_fc1, embedding_size):
        # Encoder Network
        super(CNN_network, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=input_size, out_channels=input_size, kernel_size=1,
                               stride=1, padding=0, bias=True)
        self.sigmoid1 = nn.Sigmoid()

        # self.conv2 = nn.Conv1d(in_channels=2*input_size, out_channels=4 * input_size, kernel_size=1,
        #                        stride=1, padding=0,bias=True)
        # self.sigmoid2 = nn.Sigmoid()

        self.fc1 = nn.Linear(input_size, hidden_size_fc1)
        self.celu1 = nn.CELU()
        self.fc2 = nn.Linear(hidden_size_fc1, embedding_size)


    def forward(self, x):

        x = torch.unsqueeze(x, dim=-1) #[16, 8192] -> [16, 8192, 1]

        out = self.conv1(x)
        out = self.sigmoid1(out)
        # out = self.conv2(out)
        # out = self.sigmoid2(out)

        out = torch.squeeze(out, -1)

        out = self.fc1(out)
        out = self.celu1(out)
        out = self.fc2(out)

        return out


# Linear network with two hidden layers
class CNN_network_2(nn.Module):
    def __init__(self, input_size, hidden_size_fc1, embedding_size):
        # Encoder Network
        super(CNN_network_2, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=input_size, out_channels=2*input_size, kernel_size=1,
                               stride=1, padding=0, bias=True)
        self.sigmoid1 = nn.Sigmoid()

        self.conv2 = nn.Conv1d(in_channels=2*input_size, out_channels=2 * input_size, kernel_size=1,
                               stride=1, padding=0,bias=True)
        self.sigmoid2 = nn.Sigmoid()


        self.fc1 = nn.Linear(2*input_size, hidden_size_fc1)
        self.celu1 = nn.CELU()
        self.fc2 = nn.Linear(hidden_size_fc1, embedding_size)


    def forward(self, x):

        x = torch.unsqueeze(x, dim=-1) #[16, 8192] -> [16, 8192, 1]

        out = self.conv1(x)
        out = self.sigmoid1(out)
        out = self.conv2(out)
        out = self.sigmoid2(out)

        out = torch.squeeze(out, -1)

        out = self.fc1(out)
        out = self.celu1(out)
        out = self.fc2(out)

        return out


# Linear network with three hidden layers
class NeuralNet_3layer(nn.Module):
    def __init__(self, input_size, hidden_size_fc1, hidden_size_fc2,
                 hidden_size_fc3, embedding_size):
        # Encoder Network
        super(NeuralNet_3layer, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size_fc1)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size_fc1, hidden_size_fc2)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size_fc2, hidden_size_fc3)
        self.relu3 = nn.ReLU()
        self.fc4 = nn.Linear(hidden_size_fc3, embedding_size)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu1(out)
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.fc3(out)
        out = self.relu3(out)
        out = self.fc4(out)
        return out


# Linear network with four hidden layers
class NeuralNet_4layer(nn.Module):
    def __init__(self, input_size, hidden_size_fc1, hidden_size_fc2,
                 hidden_size_fc3, hidden_size_fc4, embedding_size):
        # Encoder Network
        super(NeuralNet_4layer, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size_fc1)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size_fc1, hidden_size_fc2)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size_fc2, hidden_size_fc3)
        self.relu3 = nn.ReLU()
        self.fc4 = nn.Linear(hidden_size_fc3, hidden_size_fc4)
        self.relu4 = nn.ReLU()
        self.fc5 = nn.Linear(hidden_size_fc4, embedding_size)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu1(out)
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.fc3(out)
        out = self.relu3(out)
        out = self.fc4(out)
        out = self.relu4(out)
        out = self.fc5(out)
        return out


# Linear network with two hidden layers w dropout layers
class NeuralNet_2l_drop(nn.Module):
    def __init__(self, input_size, hidden_size_fc1, hidden_size_fc2, embedding_size):
        # Encoder Network
        super(NeuralNet_2l_drop, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size_fc1)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size_fc1, hidden_size_fc2)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size_fc2, embedding_size)

        # Dropout layer (p=0.2)
        self.fc1_dropout = nn.Dropout(p=0.2)
        self.fc2_dropout = nn.Dropout(p=0.2)
        #self.dropout = nn.Dropout(0.2)


    def forward(self, x):
        out = self.fc1(x)
        out = self.relu1(out)
        out = self.fc1_dropout(out)    # add dropout layer
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.fc2_dropout(out)    # add dropout layer
        out = self.fc3(out)
        return out


# Linear network with two hidden layers with batchnorm layers
class NeuralNet_2l_bn(nn.Module):
    def __init__(self, input_size, hidden_size_fc1, hidden_size_fc2, embedding_size):
        # Encoder Network
        super(NeuralNet_2l_bn, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size_fc1)
        self.bn1 = nn.BatchNorm1d(hidden_size_fc1)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size_fc1, hidden_size_fc2)
        self.bn2 = nn.BatchNorm1d(hidden_size_fc2)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size_fc2, embedding_size)


    def forward(self, x):
        out = self.fc1(x)
        out = self.bn1(out)    # add batchnorm layer
        out = self.relu1(out)
        out = self.fc2(out)
        out = self.bn2(out)    # add batchnorm layer
        out = self.relu2(out)
        out = self.fc3(out)
        return out