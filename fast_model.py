from abc import ABC, abstractmethod
from torch import nn, cat, squeeze, softmax, Tensor, flatten
from torch.nn import functional as F
from math import sqrt


class QModel(nn.Module):
    """A base class for Fastchem models
    embed = [(n_vocab, len_vec, param.requires_grad),...]
        The QDataset reports any categorical values it has to encode and whether 
        or not to train the embedding or fix it as a onehot
        and then serves up the values to be encoded as the x_cat component
        of the __getitem__ method.
    
    self.embeddings = embedding_layer() method checks the QDatasets embed 
    requirements and creates a list of embedding layers as appropriate"""
    def __init__(self, embed=[]):
        super().__init__()
        #self.embeddings = self.embedding_layer(embed)
        #self.layers = nn.ModuleList()
        
    def embedding_layer(self, embed):
        if len(embed) == 0:
            return None
        else:
            embeddings = [nn.Embedding(voc, vec, padding_idx=None).to('cuda:0') for voc, vec, _ in embed]
            for i, e in enumerate(embed):
                param = embeddings[i].weight
                param.requires_grad = e[2]
            return embeddings

    def forward(self, x_con, x_cat):
        """check for categorical and/or continuous inputs, get the embeddings and  
        concat as appropriate, feed to model.  
        x_cat = list of torch cuda tensors which are the embedding indices
        x_con = torch cuda tensor of concatenated continous feature vectors"""
        if len(x_cat) != 0:
            emb = []
            for i in range(len(x_cat)):
                out = self.embeddings[i](x_cat[i])
                emb.append(flatten(out, start_dim=1))
            emb = cat(emb, dim=1)
            if x_con.shape[1] != 0:
                x = cat([x_con, emb], dim=1)
            else:  
                x = emb    
        else:
            x = x_con
        
        for l in self.layers:
            x = l(x)
        return x
        
    def adapt(self, shape):
        """for adapting a dataset shape[0] to a saved model shape[1]"""
        # freeze the layers
        for param in self.parameters(): 
            param.requires_grad = False
        # prepend a trainable adaptor layer    
        for l in self.ffunit(shape[0], shape[1], 0.2)[::-1]:
            self.layers.insert(0, l)
            
    def ffunit(self, D_in, D_out, drop):
        ffu = []
        ffu.append(nn.BatchNorm1d(D_in))
        ffu.append(nn.Linear(D_in, D_out))
        ffu.append(nn.SELU())
        ffu.append(nn.Dropout(drop))
        return ffu
    
class FFNet(QModel):
    
    model_config = {}
    model_config['simple'] = {'shape': [('D_in',1),(1,1),(1,1/2),(1/2,'D_out')], 
                              'dropout': [.2, .3, .1]}
    model_config['funnel'] = {'shape': [('D_in',1),(1,1/2),(1/2,1/2),(1/2,1/4),(1/4,1/4),(1/4,'D_out')], 
                              'dropout': [.1, .2, .3, .2, .1]}

    def __init__(self, model_name='funnel', D_in=0, H=0, D_out=0, embed=[]):
        super().__init__()
        
        config = FFNet.model_config[model_name]
        layers = []
        layers.append(self.ffunit(D_in, int(config['shape'][0][1]*H), config['dropout'][0]))
        for i, s in enumerate(config['shape'][1:-1]):
            layers.append(self.ffunit(int(s[0]*H), int(s[1]*H), config['dropout'][i]))
        layers.append([nn.Linear(int(config['shape'][-1][0]*H), D_out)])
        self.layers = [l for ffu in layers for l in ffu] # flatten
        self.layers = nn.ModuleList(self.layers)  
    
        self.embeddings = self.embedding_layer(embed)
        
class MAB(nn.Module):
    """Multihead Attention Block"""
    def __init__(self, dim_Q, dim_K, dim_V, num_heads, ln=False):
        super(MAB, self).__init__()
        self.dim_V = dim_V
        self.num_heads = num_heads
        self.fc_q = nn.Linear(dim_Q, dim_V)
        self.fc_k = nn.Linear(dim_K, dim_V)
        self.fc_v = nn.Linear(dim_K, dim_V)
        if ln:
            self.ln0 = nn.LayerNorm(dim_V)
            self.ln1 = nn.LayerNorm(dim_V)
        self.fc_o = nn.Linear(dim_V, dim_V)

    def forward(self, Q, K):
        Q = self.fc_q(Q)
        K, V = self.fc_k(K), self.fc_v(K)

        dim_split = self.dim_V // self.num_heads
        Q_ = torch.cat(Q.split(dim_split, 2), 0)
        K_ = torch.cat(K.split(dim_split, 2), 0)
        V_ = torch.cat(V.split(dim_split, 2), 0)

        A = torch.softmax(Q_.bmm(K_.transpose(1,2))/math.sqrt(self.dim_V), 2)
        O = torch.cat((Q_ + A.bmm(V_)).split(Q.size(0), 0), 2)
        O = O if getattr(self, 'ln0', None) is None else self.ln0(O)
        O = O + F.relu(self.fc_o(O))
        O = O if getattr(self, 'ln1', None) is None else self.ln1(O)
        return O

class SAB(nn.Module):
    """Set Attention Block"""
    def __init__(self, dim_in, dim_out, num_heads, ln=False):
        super().__init__()
        self.mab = MAB(dim_in, dim_in, dim_out, num_heads, ln=ln)

    def forward(self, X):
        return self.mab(X, X)

class ISAB(nn.Module):
    """Induced Set Attention Block"""
    def __init__(self, dim_in, dim_out, num_heads, num_inds, ln=False):
        super().__init__()
        self.I = nn.Parameter(torch.Tensor(1, num_inds, dim_out))
        nn.init.xavier_uniform_(self.I)
        self.mab0 = MAB(dim_out, dim_in, dim_out, num_heads, ln=ln)
        self.mab1 = MAB(dim_in, dim_out, dim_out, num_heads, ln=ln)

    def forward(self, X):
        H = self.mab0(self.I.repeat(X.size(0), 1, 1), X)
        return self.mab1(X, H)

class PMA(nn.Module):
    """Pooling by Multihead Attention"""
    def __init__(self, dim, num_heads, num_seeds, ln=False):
        super().__init__()
        self.S = nn.Parameter(torch.Tensor(1, num_seeds, dim))
        nn.init.xavier_uniform_(self.S)
        self.mab = MAB(dim, dim, dim, num_heads, ln=ln)

    def forward(self, X):
        return self.mab(self.S.repeat(X.size(0), 1, 1), X)
    
    
class SetTransformer(QModel):
    """https://github.com/juho-lee/set_transformer"""
    def __init__(self, dim_input, num_outputs, dim_output,
            num_inds=32, dim_hidden=128, num_heads=4, ln=False, embed=[]):
        Super().__init__()
        self.embeddings = self.embedding_layer(embed)
        
        self.enc = nn.Sequential(
                ISAB(dim_input, dim_hidden, num_heads, num_inds, ln=ln),
                ISAB(dim_hidden, dim_hidden, num_heads, num_inds, ln=ln))
        self.dec = nn.Sequential(
                PMA(dim_hidden, num_heads, num_outputs, ln=ln),
                SAB(dim_hidden, dim_hidden, num_heads, ln=ln),
                SAB(dim_hidden, dim_hidden, num_heads, ln=ln),
                nn.Linear(dim_hidden, dim_output))
        
    def forward(self, X):
        return self.dec(self.enc(X))