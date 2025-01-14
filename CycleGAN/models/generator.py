import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np 
import itertools

# create a sphere in 3D space with n_points points
def create_sphere(n_points):
    # create points on a sphere
    points = []
    phi = np.pi * (3. - np.sqrt(5.))            # golden angle in radians

    for i in range(n_points):
        y = 1 - (i / float(n_points - 1)) * 2
        radius = np.sqrt(1 - y * y)
        theta = phi * i

        x = np.cos(theta) * radius
        z = np.sin(theta) * radius

        points.append([x, y, z])

    return np.array(points)

# create gaussian noise in 3D space space with n_points points
def create_gaussian(n_points):
    return np.random.normal(size=(n_points, 3))

# define knn function
def knn(x, k):
    batch_size = x.size(0)
    num_points = x.size(2)

    inner = -2*torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
 
    idx = pairwise_distance.topk(k=k, dim=-1)[1]                    # (batch_size, num_points, k)

    if idx.get_device() == -1:
        idx_base = torch.arange(0, batch_size).view(-1, 1, 1)*num_points
    else:
        idx_base = torch.arange(0, batch_size, device=idx.get_device()).view(-1, 1, 1)*num_points
    idx = idx + idx_base
    idx = idx.view(-1)

    return idx

# define local covariance function
def local_cov(pts, idx):
    batch_size = pts.size(0)
    num_points = pts.size(2)
    pts = pts.view(batch_size, -1, num_points)                      # (batch_size, 3, num_points)
 
    _, num_dims, _ = pts.size()

    x = pts.transpose(2, 1).contiguous()                            # (batch_size, num_points, 3)
    x = x.view(batch_size*num_points, -1)[idx, :]                   # (batch_size*num_points*2, 3)
    x = x.view(batch_size, num_points, -1, num_dims)                # (batch_size, num_points, k, 3)

    x = torch.matmul(x[:,:,0].unsqueeze(3), x[:,:,1].unsqueeze(2))  # (batch_size, num_points, 3, 1) * (batch_size, num_points, 1, 3) -> (batch_size, num_points, 3, 3)
    x = x.view(batch_size, num_points, 9).transpose(2, 1)           # (batch_size, 9, num_points)

    x = torch.cat((pts, x), dim=1)                                  # (batch_size, 12, num_points)

    return x

# define local maxpool function
def local_maxpool(x, idx):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
 
    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()                      # (batch_size, num_points, num_dims)
    x = x.view(batch_size*num_points, -1)[idx, :]           # (batch_size*n, num_dims) -> (batch_size*n*k, num_dims)
    x = x.view(batch_size, num_points, -1, num_dims)        # (batch_size, num_points, k, num_dims)
    x, _ = torch.max(x, dim=2)                              # (batch_size, num_points, num_dims)

    return x

# define graph feature function
def get_graph_feature(x, k=3, idx=None):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)                              # (batch_size, num_dims, num_points)
    if idx is None:
        idx = knn(x, k=k)                                               # (batch_size, num_points, k)
 
    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()                                  # (batch_size, num_points, num_dims)
    feature = x.view(batch_size*num_points, -1)[idx, :]                 # (batch_size*n, num_dims) -> (batch_size*n*k, num_dims)
    feature = feature.view(batch_size, num_points, k, num_dims)         # (batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)  # (batch_size, num_points, k, num_dims)
    
    feature = torch.cat((feature-x, x), dim=3).permute(0, 3, 1, 2)      # (batch_size, num_points, k, 2*num_dims) -> (batch_size, 2*num_dims, num_points, k)
  
    return feature                                                      # (batch_size, 2*num_dims, num_points, k)

class ChamferLoss(nn.Module):
    def __init__(self):
        super(ChamferLoss, self).__init__()
        self.use_cuda = torch.cuda.is_available()

    def batch_pairwise_dist(self, x, y): 
        xx = x.pow(2).sum(dim=-1)
        yy = y.pow(2).sum(dim=-1)
        zz = torch.bmm(x, y.transpose(2, 1))
        rx = xx.unsqueeze(1).expand_as(zz.transpose(2, 1))
        ry = yy.unsqueeze(1).expand_as(zz)
        P = (rx.transpose(2, 1) + ry - 2 * zz)
        return P
    
    def forward(self, preds, gts):
        P = self.batch_pairwise_dist(gts, preds)        # compute pairwise distance between preds and gts
        mins, _ = torch.min(P, 1)                       # find the nearest gt point for each pred point
        loss_1 = torch.mean(mins)                       # take mean across got <-> pred loss
        mins, _ = torch.min(P, 2)                       # find the nearest pred point for each gt point
        loss_2 = torch.mean(mins)                       # take mean across pred <-> gt loss
        return (loss_1 + loss_2)

class FoldNet_Encoder(nn.Module): ## FoldingNet Encoder
    def __init__(self):
        super(FoldNet_Encoder, self).__init__()
        self.k = 16 # number of nearest neighbors
        self.n = 2048   # input point cloud size
        self.mlp1 = nn.Sequential(
            nn.Conv1d(12, 64, 1),
            nn.ReLU(),
            nn.Conv1d(64, 64, 1),
            nn.ReLU(),
            nn.Conv1d(64, 64, 1),
            nn.ReLU(),
        )
        self.linear1 = nn.Linear(64, 64)
        self.conv1 = nn.Conv1d(64, 128, 1)
        self.linear2 = nn.Linear(128, 128)
        self.conv2 = nn.Conv1d(128, 1024, 1)
        self.mlp2 = nn.Sequential(
            nn.Conv1d(1024, 512, 1),
            nn.ReLU(),
            nn.Conv1d(512, 512, 1),
        )

    def graph_layer(self, x, idx):           
        x = local_maxpool(x, idx)    
        x = self.linear1(x)  
        x = x.transpose(2, 1)                                     
        x = F.relu(self.conv1(x))                            
        x = local_maxpool(x, idx)  
        x = self.linear2(x) 
        x = x.transpose(2, 1)                                   
        x = self.conv2(x)                       
        return x

    def forward(self, pts):
        pts = pts.transpose(2, 1)               # (batch_size, 3, num_points)
        idx = knn(pts, k=self.k)          # (batch_size, num_points, k)
        x = local_cov(pts, idx)           # (batch_size, 3, num_points) -> (batch_size, 12, num_points])            
        x = self.mlp1(x)                        # (batch_size, 12, num_points) -> (batch_size, 64, num_points])
        x = self.graph_layer(x, idx)            # (batch_size, 64, num_points) -> (batch_size, 1024, num_points)
        x = torch.max(x, 2, keepdim=True)[0]    # (batch_size, 1024, num_points) -> (batch_size, 1024, 1)
        x = self.mlp2(x)                        # (batch_size, 1024, 1) -> (batch_size, feat_dims, 1)
        feat = x.transpose(2,1)                 # (batch_size, feat_dims, 1) -> (batch_size, 1, feat_dims)
        return feat                             # (batch_size, 1, feat_dims)


class FoldNet_Decoder(nn.Module):
    def __init__(self):
        super(FoldNet_Decoder, self).__init__()
        self.x1 = -1 * 100
        self.x2 = 1 * 100
        self.p = 45

        self.shape = 'sphere'
        self.m = self.p**2 if self.shape == 'plane' else 2048
        self.meshgrid = [[self.x1, self.x2, self.p], [self.x1, self.x2, self.p]]
        self.sphere = create_sphere(self.m)       # define sphere point cloud
        self.gaussian = create_gaussian(self.m)   # define gaussian point cloud
        if self.shape == 'plane':
            self.folding1 = nn.Sequential(
                nn.Conv1d(512+2, 512, 1),
                nn.ReLU(),
                nn.Conv1d(512, 512, 1),
                nn.ReLU(),
                nn.Conv1d(512, 3, 1),
            )
        else:
            self.folding1 = nn.Sequential(
                nn.Conv1d(512+3, 512, 1),
                nn.ReLU(),
                nn.Conv1d(512, 512, 1),
                nn.ReLU(),
                nn.Conv1d(512, 3, 1),
            )  
        self.folding2 = nn.Sequential(
            nn.Conv1d(512+3, 512, 1),
            nn.ReLU(),
            nn.Conv1d(512, 512, 1),
            nn.ReLU(),
            nn.Conv1d(512, 3, 1),
        )

    def build_grid(self, batch_size):
        if self.shape == 'plane':
            x = np.linspace(*self.meshgrid[0])
            y = np.linspace(*self.meshgrid[1])
            points = np.array(list(itertools.product(x, y)))
        elif self.shape == 'sphere':
            points = self.sphere * 100 # scale up to 1000
        elif self.shape == 'gaussian':
            points = self.gaussian * 100 # scale up to 1000
        points = np.repeat(points[np.newaxis, ...], repeats=batch_size, axis=0)
        points = torch.tensor(points)
        return points.float()

    def forward(self, x):
        x_new = x.transpose(1, 2).repeat(1, 1, self.m)      # (batch_size, feat_dims, num_points)
        points = self.build_grid(x_new.shape[0]).transpose(1, 2)  # (batch_size, 2, num_points) or (batch_size, 3, num_points)
        if x.get_device() != -1:
            points = points.cuda(x_new.get_device())
        cat1 = torch.cat((x_new, points), dim=1)            # (batch_size, feat_dims+2, num_points) or (batch_size, feat_dims+3, num_points)
        folding_result1 = self.folding1(cat1)           # (batch_size, 3, num_points)
        cat2 = torch.cat((x_new, folding_result1), dim=1)   # (batch_size, 515, num_points)
        folding_result2 = self.folding2(cat2)           # (batch_size, 3, num_points)
        return folding_result2.transpose(1,2)          # (batch_size, num_points ,3)
    
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.encoder = FoldNet_Encoder()                # Define encoder
        self.decoder = FoldNet_Decoder()                # Define decoder
        self.loss = ChamferLoss()                       # Define internal loss function

    def forward(self, input):
        codeword = self.encoder(input)
        output = self.decoder(codeword)
        return output

    def get_parameters(self):
        return list(self.encoder.parameters()) + list(self.decoder.parameters())