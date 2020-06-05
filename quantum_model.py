from torch import nn, cat, squeeze
from torch.nn import functional as F

class FFNet(nn.Module):
    
    model_config = {}
    model_config['funnel'] = {'shape': [('D_in', 1), (1, 1/2), (1/2, 1/2), (1/2, 1/4), (1/4, 1/4), (1/4, 'D_out')], 
                                'dropout': [.2, .2, .4, .2, .1]}
    model_config['straight'] = {'shape': [('D_in', 1), (1, 1), (1, 1), (1, 1), (1, 1/4), (1/4, 'D_out')], 
                                'dropout': [.2, .4, .4, .4, .1]}
    model_config['bottle'] = {'shape': [('D_in', 1), (1, 1), (1, 1/2), (1/2, 1), (1, 1), (1, 1/4), (1/4, 'D_out')], 
                              'dropout': [.2, .4, .2, .4, .4, .1]}
    model_config['simple'] = {'shape': [('D_in', 1), (1, 1), (1, 1/2), (1/2, 'D_out')], 
                              'dropout': [.2, .4, .1]}
    model_config['deep'] = {'shape': [('D_in', 1), (1, 1), (1, 1), (1, 1/2), (1/2, 1/2), (1/2, 1/2), (1/2, 1/2), 
                                      (1/2, 1/4), (1/4, 'D_out')], 
                             'dropout': [.2, .4, .4, .2, .4, .4, .4, .2, .1]}
    model_config['narrow'] = {'shape': [('D_in', 1), (1, 1), (1, 1/4), (1/4, 1/2), (1/2, 1/2), (1/2, 1/4), (1/4, 'D_out')], 
                              'dropout': [.2, .4, .1, .3, .4, .1]}
    
    def __init__(self, model_name='funnel', D_in=0, H=0, D_out=0, embeddings=[]):
        super().__init__()
        self.embeddings = [nn.Embedding(voc, vec).to('cuda:0') for voc, vec, _ in embeddings]
        for i, e in enumerate(embeddings):
            param = self.embeddings[i].weight
            param.requires_grad = e[2]
            
        config = FFNet.model_config[model_name]
        layers = []
        layers.append(self.ffunit(D_in, int(config['shape'][0][1]*H), config['dropout'][0]))
        for i, s in enumerate(config['shape'][1:-1]):
            layers.append(self.ffunit(int(s[0]*H), int(s[1]*H), config['dropout'][i]))
        layers.append([nn.Linear(int(config['shape'][-1][0]*H), D_out)])
        self.layers = [l for ffu in layers for l in ffu] # flatten
        self.layers = nn.ModuleList(self.layers)  
        
    def ffunit(self, D_in, D_out, drop):
        ffu = []
        ffu.append(nn.BatchNorm1d(D_in))
        ffu.append(nn.Linear(D_in, D_out))
        ffu.append(nn.SELU())
        ffu.append(nn.Dropout(drop))
        return ffu
 
    def forward(self, x_con, x_cat):
        # check for categorical and/or continuous inputs, get the embeddings and  
        # concat as appropriate, feed to model
        if len(x_cat) != 0:
            emb = []
            for i in range(x_cat.shape[1]):
                out = self.embeddings[i](x_cat[:,i])
                emb.append(out)
            emb = cat(emb, dim=1)
            if len(x_con) != 0:
                x = cat([x_con, emb], dim=1)
            else:  
                x = emb    
        else:
            x = x_con
            
        for l in self.layers:
            x = l(x)
        return x
    
    def adapt(self, shape):
        # for adapting a dataset (shape[0]) to a saved model and weights (shape[1])
        # freeze the layers
        for param in self.parameters(): 
            param.requires_grad = False
        # prepend a trainable adaptor layer    
        for l in self.ffunit(shape[0], shape[1], 0.2)[::-1]:
            self.layers.insert(0, l)
   
        
            
    
