from torch import nn, cat, squeeze
from torch.nn import functional as F

class FFNet(nn.Module):
    
    model_config = {}
    model_config['funnel'] = {'shape': [('D_in',1),(1,1/2),(1/2,1/2),(1/2,1/4),(1/4,1/4),(1/4,'D_out')], 
                              'dropout': [.2, .2, .4, .2, .1]}
    model_config['deep'] = {'shape': [('D_in',1),(1,1/2),(1/2,1/2),(1/2,1/4),(1/4,1/4),(1/4,1/16),
                                      (1/16,1/64),(1/64,'D_out')], 
                              'dropout': [.2, .2, .4, .2, .4, .2, .1]}
    
    
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
        # for adapting a dataset (shape[0]) to a saved model and weights (shape[1])
        # freeze the layers
        for param in self.parameters(): 
            param.requires_grad = False
        # prepend a trainable adaptor layer    
        for l in self.ffunit(shape[0], shape[1], 0.2)[::-1]:
            self.layers.insert(0, l)
   
        
class EncoderLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers=1, drop_prob=0):
        super(EncoderLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.n_layers = n_layers

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, n_layers, dropout=drop_prob, batch_first=True)

    def forward(self, inputs, hidden):
        # Embed input words
        embedded = self.embedding(inputs)
        # Pass the embedded word vectors into LSTM and return all outputs
        output, hidden = self.lstm(embedded, hidden)
        return output, hidden

    def init_hidden(self, batch_size=1):
        return (torch.zeros(self.n_layers, batch_size, self.hidden_size, device=device),
                torch.zeros(self.n_layers, batch_size, self.hidden_size, device=device))
           
    
