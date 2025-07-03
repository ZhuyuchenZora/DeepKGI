import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch_geometric.nn import GCNConv, GATConv, max_pool


class GNN_cell(torch.nn.Module):
    def __init__(self, num_feature, layer_cell, dim_cell, cluster_predefine):
        super().__init__()
        self.num_feature = num_feature
        self.layer_cell = layer_cell
        self.dim_cell = dim_cell
        self.cluster_predefine = cluster_predefine
        self.final_node = len(self.cluster_predefine[self.layer_cell - 1].unique())
        self.convs_cell = torch.nn.ModuleList()
        self.bns_cell = torch.nn.ModuleList()
        self.multi_head = torch.nn.ModuleList()
        
        self.linear_attn = nn.Linear(self.dim_cell,1,bias=False)
        # self.linear_attn1 = nn.Linear(self.dim_cell,1,bias=False)
        # self.link = nn.Linear(self.dim_cell*8,self.dim_cell,bias=False)
        self.link0 = nn.Linear(self.dim_cell,self.dim_cell,bias=False)       
        self.linkl = nn.Linear(self.dim_cell,self.dim_cell,bias=False)
        # self.link2 = nn.Linear(self.dim_cell,self.dim_cell,bias=False)
        self.link_ = nn.Linear(self.dim_cell,self.dim_cell,bias=False)
        self.bn_ = torch.nn.BatchNorm1d(self.dim_cell, affine=False)

        self.linear_0 = nn.Linear(self.dim_cell*4,4)

        # for i in range(8):
        #    self.multi_head.append(nn.Linear(self.dim_cell*4,4))

        # self.activations = torch.nn.ModuleList()
        # self.linear_attn = nn.Linear(self.dim_cell,1,bias=False) 
        # self.Weight=nn.Parameter(torch.tensor([0.5,0.2,0.2,0.1]))

        for i in range(self.layer_cell):
            if i:
                conv = GATConv(self.dim_cell, self.dim_cell)
            else:
                conv = GATConv(self.num_feature, self.dim_cell)
            bn = torch.nn.BatchNorm1d(self.dim_cell, affine=False)  # True or False
            # activation = nn.PReLU(self.dim_cell)

            self.convs_cell.append(conv)
            self.bns_cell.append(bn)
            # self.activations.append(activation)

    def forward(self, cell):
        # print("================")
        # print(cell.edge_index)

        Edge = cell.edge_index
        edge_index = {0:Edge[:,0:2268],1:Edge[:,2268:9140],2:Edge[:,9140:12228],3:Edge[:,12228:15378]}
        # edge_index = {0:Edge[:,0:4536],1:Edge[:,4536:5378],2:Edge[:,5378:9914],3:Edge[:,9914:16214]}
        # print(edge_index[1].size())
        # print("Edge",edge_index[0].size(),edge_index[1].size(),edge_index[2].size())
        r=0
        y = cell.x
        batch = cell.batch
        # z = cell.num_graphs
        # print("Cell",cell)
        # for j in [0]:
          # print("xunhuan",j)
          # Edge = cell.edge_index
          # cell.edge_index={0:Edge[:,0:1214],1:Edge[:,1214:7410],2:Edge[:,7410:8624]}
        w=[0,0,0,0]

        ALL=[]

        for j in range(4):
          # print("循环",j)
          for i in range(self.layer_cell):
            # print("i:================",i)
            # print(cell.edge_index[i][i])
            # cell.edge_index = edge_index[j]
            # print(edge_index[j].size())
            if i==0:
               cell.edge_index=edge_index[j].contiguous()
               cell.x = F.relu(self.convs_cell[i](cell.x,cell.edge_index))
            else:
               cell.x = F.relu(self.convs_cell[i](cell.x,cell.edge_index))
            # print(cell.x.size())
            # cell.x = F.relu(self.convs_cell[i](cell.x, edge_index[j]))
            # print("y",y.size())
            num_node = int(cell.x.size(0) / cell.num_graphs)
            #print(self.cluster_predefine)
            cluster = torch.cat([self.cluster_predefine[i] + j * num_node for j in range(cell.num_graphs)])
            # cell_m.x=cell.x 
            # cell_m.edge_index=cell.edge_index[i]
            # print("1===",cell)
            # print("======",cell.edge_index.size())
            # cell.edge_index = edge_index[j].contiguous()
            # cell.edge_index = edge_index[j].contiguous()
            # print("CELL",cell,edge_index[j].size())
            cell = max_pool(cluster, cell, transform=None)
            # print("i_cell",cell.x.size())
            cell.x = self.bns_cell[i](cell.x)
            
          # r+=w[j]*cell.x
          # ALL=torch.stack((ALL,cell.x),dim=0)
          
          if j==0:
            # ALL = torch.unsqueeze(cell.x,dim=0)
            ALL = cell.x.unsqueeze(0)
            # print(ALL.size())
          else:
            #print(ALL.size(),cell.x.size())
            ALL = torch.cat((ALL,cell.x.unsqueeze(0)),dim=0)
          

          cell.x=y
          cell.batch=batch
          
          # print("y",y.size())
          # print("cell.x.size",cell.x.size())
          # cell.edge_index=edge_index[j+1]
          # print("num_graphs")
          # print(cell.num_graphs)
          # cell.num_graphs=z
    
        # ALL=torch.tensor(ALL)
        # B = torch.tensor([])
        #### Multi-View
        C=[]
        for k in range(1):
          B=[]
          for i in range(4):
            if i==0:
              B=self.linear_attn(ALL[i])
              # B=self.multi_head[k](ALL[i]) 
              # B = torch.mean(ALL[i],dim=0,keepdim=True).t()
            else:
              B=torch.cat((B,self.linear_attn(ALL[i])),dim=1)
              # B=torch.cat((B,torch.mean(ALL[i],dim=0,keepdim=True).t()),dim=1)

          # print(B.size())
          B = B.mm(torch.diag(1/torch.norm(B,p=2,dim=0)))
          Attn_map = F.softmax(F.leaky_relu(torch.mm(B.t(), B),0.1),dim=-1)
          ALL=ALL.permute(1,2,0)
          # print("ALL",ALL.size())
          Attn_map=Attn_map.expand(ALL.size(0),4,4)
          # print("Attn",Attn_map.size())
          ALL=torch.bmm(ALL,Attn_map).permute(2,0,1)
          if k==0:
              C=ALL
          else:
              C = torch.cat((C,ALL),dim=2)
              #for i in range(3):
               # print("i+",C[i].size())
                # C[i]=torch.cat((C[i],ALL[i]),dim=1)
        # print("=====",C.size())
        
        # View-level attention
        l = ALL.size()[1]
        E = torch.reshape(ALL,(l,-1))
        E = self.linear_0(E)
        """
        for i in range(8):
            if i==0:
               EE=self.multi_head[i](E)
            else:
               EE=torch.cat((EE,self.multi_head[i](E)),dim=0)
        """
        w = F.softmax(torch.mean(E,dim=0),dim=0)
    
        """
        for i in range(4):
            w[i] = torch.norm(self.linear_0(ALL[i]),p=2).pow(2)
        w = torch.tensor(w)
        w = F.softmax(w,dim=0)
        """

        for i in range(4):
              # temp = self.link(C[i])
              #temp = F.relu(self.linkl(C[i]))
              # r+=w[i]*temp
              r+=w[i]*ALL[i]

        # print(r.size())
        r = self.link0(r)
        r = F.selu(r)
        r = self.linkl(r)
        # r = F.leaky_relu(self.linkl(r),0.1)
        r = F.selu(r)
        # r = self.link2(r)
        # r = F.selu(r)
        r = self.link_(r)
        # C=C.reshape(-1,self.dim_cell*3)
        # r=self.link(C)
        # r = self.bn_(r)
        # node_representation = cell.x.reshape(-1, self.final_node * self.dim_cell)
        
        node_representation = r.reshape(-1, self.final_node * self.dim_cell)
        # print(node_representation.size())

        return node_representation
    
    def grad_cam(self, cell):
    
        Edge = cell.edge_index
        edge_index = {0:Edge[:,0:1214],1:Edge[:,1214:7410],2:Edge[:,7410:8624]}
        # print(edge_index[0].size(),edge_index[1].size(),edge_index[2].size())
        r=0
        y = cell.x
        for j in range(4):
          for i in range(self.layer_cell):
            cell.x = F.relu(self.convs_cell[i](cell.x, edge_index[j]))
            if i == 0:
                cell_node = cell.x
                cell_node.retain_grad()
            num_node = int(cell.x.size(0) / cell.num_graphs)
            cluster = torch.cat([self.cluster_predefine[i] + j * num_node for j in range(cell.num_graphs)])
            cell = max_pool(cluster, cell, transform=None)
            cell.x = self.bns_cell[i](cell.x)
          r+=cell.x
          cell.x=y
        node_representation = r.reshape(-1, self.final_node * self.dim_cell)

        return cell_node, node_representation
