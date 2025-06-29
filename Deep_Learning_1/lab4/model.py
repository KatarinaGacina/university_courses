import torch
import torch.nn as nn
import torch.nn.functional as F

class _BNReluConv(nn.Sequential):
    def __init__(self, num_maps_in, num_maps_out, k=3, bias=True):
        super(_BNReluConv, self).__init__()
        # YOUR CODE HERE
        self.append(nn.BatchNorm2d(num_maps_in))
        self.append(nn.ReLU())
        self.append(nn.Conv2d(num_maps_in, num_maps_out, kernel_size=k, bias=bias))

class SimpleMetricEmbedding(nn.Module):
    def __init__(self, input_channels, emb_size=32):
        super().__init__()
        self.emb_size = emb_size
        # YOUR CODE HERE

        self.input_channels = input_channels

        self.conv1 = _BNReluConv(input_channels, emb_size)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.conv2 = _BNReluConv(emb_size, emb_size)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.conv3 = _BNReluConv(emb_size, emb_size)
        self.avg_pool = nn.AvgPool2d(kernel_size=2)
    
    def forward(self, x):
        c1 = self.conv1(x)
        p1 = self.pool1(c1)
        c2 = self.conv2(p1)
        p2 = self.pool2(c2)
        c3 = self.conv3(p2)
        p3 = self.avg_pool(c3)
        
        return p3

    def get_features(self, img):
        # Returns tensor with dimensions BATCH_SIZE, EMB_SIZE
        # YOUR CODE HERE
        x = self.forward(img)
        x = x.reshape(x.shape[0], x.shape[1])
        return x

    def loss(self, anchor, positive, negative):
        margin = 1
        p = 2

        a_x = self.get_features(anchor)
        p_x = self.get_features(positive)
        n_x = self.get_features(negative)

        # YOUR CODE HERE
        distance_positive = torch.norm(a_x - p_x, p=p, dim=1)
        distance_negative = torch.norm(a_x - n_x, p=p, dim=1)

        loss_all = distance_positive - distance_negative + margin

        loss = torch.max(loss_all, torch.tensor(0.0).to(anchor.device))
        
        return torch.mean(loss)
    

class IdentityModel(nn.Module):
    def __init__(self):
        super(IdentityModel, self).__init__()

    def get_features(self, img):
        # YOUR CODE HERE
        feats = img.flatten(start_dim=1)
        return feats