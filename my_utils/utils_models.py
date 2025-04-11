import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as dsets
from torch.autograd import Variable
import torch.nn.functional as F

class linear_model(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(linear_model, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        
    def forward(self, x):
        out = x.view(-1, self.num_flat_features(x))
        out = self.linear(out)
        return out
        
    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
    
class mlp_model(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(mlp_model, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.BN = torch.nn.BatchNorm1d(input_dim)
        def init_weights(m):
            if type(m) == nn.Linear:
                nn.init.normal_(m.weight) #nn.init.xavier_uniform_(m.weight) # nn.init.uniform_(m.weight) #nn.init.normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    def forward(self, x):
        out = x.view(-1, self.num_flat_features(x))
        out = self.BN(out)
        out = self.fc1(out)
        out = self.relu1(out)
        out = self.fc2(out)
        out = self.relu1(out)
        out = self.fc2(out)
        out = self.relu1(out)
        out = self.fc2(out)
        out = self.relu1(out)
        out = self.fc3(out)
        return out

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features



class Cnn(nn.Module):
    def __init__(self, input_channels, n_outputs, dropout_rate):
        self.dropout_rate = dropout_rate
        super(Cnn, self).__init__()

        self.c1 = nn.Conv2d(input_channels, 128, kernel_size=3, stride=1, padding=1)
        self.c2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.c3 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.c4 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.c5 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.c6 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.c7 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=0)
        self.c8 = nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=0)
        self.c9 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=0)
        self.l_c1 = nn.Linear(128, n_outputs)
        self.bn1 = nn.BatchNorm2d(128)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm2d(256)
        self.bn6 = nn.BatchNorm2d(256)
        self.bn7 = nn.BatchNorm2d(512)
        self.bn8 = nn.BatchNorm2d(256)
        self.bn9 = nn.BatchNorm2d(128)

    def forward(self, x):
        x = self.c1(x)
        x = self.bn1(x)
        x = F.leaky_relu(x, negative_slope=0.01)
        x = self.c2(x)
        x = self.bn2(x)
        x = F.leaky_relu(x, negative_slope=0.01)
        x = self.c3(x)
        x = self.bn3(x)
        x = F.leaky_relu(x, negative_slope=0.01)
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.dropout2d(x, p=self.dropout_rate)

        x = self.c4(x)
        x = self.bn4(x)
        x = F.leaky_relu(x, negative_slope=0.01)
        x = self.c5(x)
        x = self.bn5(x)
        x = F.leaky_relu(x, negative_slope=0.01)
        x = self.c6(x)
        x = self.bn6(x)
        x = F.leaky_relu(x, negative_slope=0.01)
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.dropout2d(x, p=self.dropout_rate)

        x = self.c7(x)
        x = self.bn7(x)
        x = F.leaky_relu(x, negative_slope=0.01)
        x = self.c8(x)
        x = self.bn8(x)
        x = F.leaky_relu(x, negative_slope=0.01)
        x = self.c9(x)
        x = self.bn9(x)
        x = F.leaky_relu(x, negative_slope=0.01)
        x = F.avg_pool2d(x, kernel_size=x.data.shape[2])

        x = x.view(x.size(0), x.size(1))
        x = self.l_c1(x)

        return x

