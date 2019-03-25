import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import numpy as np

"""
    u_embedding: Embedding for center word.
    v_embedding: Embedding for neighbor words.
"""


class VCWEModel(nn.Module):

    def __init__(self, emb_size, emb_dimension, wordid2charid, char_size):
        super(VCWEModel, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.wordid2charid = wordid2charid
        self.emb_size = emb_size
        self.emb_dimension = emb_dimension
        self.u_embeddings = nn.Embedding(emb_size, emb_dimension, sparse=False)
        self.v_embeddings = nn.Embedding(emb_size, emb_dimension, sparse=False)
        self.char_embeddings = nn.Embedding(char_size, emb_dimension, sparse=False)
        self.cnn_model = CNNModel(32,32,self.emb_dimension)
        self.lstm_model = LSTMModel(self.emb_dimension)

        initrange = 1.0 / self.emb_dimension
        init.uniform_(self.u_embeddings.weight.data, -initrange, initrange)
        init.uniform_(self.v_embeddings.weight.data, -initrange, initrange)
        init.uniform_(self.char_embeddings.weight.data, -initrange, initrange)
        

    def forward(self, pos_u, pos_v, neg_v, img_data):
        img_emb = self.cnn_model.forward(img_data)
        emb_u = self.u_embeddings(pos_u)      
        emb_v = self.v_embeddings(pos_v).mean(dim=1)
        emb_neg_v = self.v_embeddings(neg_v)
        pos_v=pos_v.view(-1).cpu()
        temp = self.wordid2charid[pos_v].reshape(1,-1)
        temp = torch.from_numpy(temp).to(self.device).long()
        lstm_input = img_emb[temp.reshape(1, -1)].view(len(pos_v), -1, self.emb_dimension)
        del temp
        lstm_input = torch.transpose(lstm_input, 0, 1)          #  self.data.maxwordlength, batch_size, embedding_dim
        emb_char_v = self.lstm_model.forward(lstm_input, len(pos_v))
        emb_char_v = emb_char_v.view(pos_u.size(0),-1,self.emb_dimension)
        emb_char_v = torch.mean(emb_char_v,dim=1)

        pos_neg_v=neg_v.view(-1).cpu()
        temp = self.wordid2charid[pos_neg_v].reshape(1,-1)
        temp = torch.from_numpy(temp).to(self.device).long()
        lstm_input2 = img_emb[temp.reshape(1, -1)].view(len(pos_neg_v), -1, self.emb_dimension)
        del temp
        lstm_input2 = torch.transpose(lstm_input2, 0, 1)
        emb_neg_char_v = self.lstm_model.forward(lstm_input2, len(pos_neg_v))
        emb_neg_char_v = emb_neg_char_v.view(pos_u.size(0),-1,self.emb_dimension)

        c_score = torch.sum(torch.mul(emb_u, emb_char_v), dim=1)
        c_score = torch.clamp(c_score, max=10, min=-10)
        c_score = -F.logsigmoid(c_score)

        neg_c_score = torch.bmm(emb_neg_char_v, emb_u.unsqueeze(2)).squeeze()
        neg_c_score = torch.clamp(neg_c_score, max=10, min=-10)
        neg_c_score = -torch.sum(F.logsigmoid(-neg_c_score), dim=1)

        score = torch.sum(torch.mul(emb_u, emb_v), dim=1)
        score = torch.clamp(score, max=10, min=-10)
        score = -F.logsigmoid(score)

        neg_score = torch.bmm(emb_neg_v, emb_u.unsqueeze(2)).squeeze()
        neg_score = torch.clamp(neg_score, max=10, min=-10)
        neg_score = -torch.sum(F.logsigmoid(-neg_score), dim=1)

        return torch.mean(c_score + neg_c_score + score + neg_score)

    def save_embedding(self, id2word, file_name):
        embedding = self.u_embeddings.weight.cpu().data.numpy()
        with open(file_name, 'w', encoding="utf-8") as f:
            f.write('%d %d\n' % (len(id2word), self.emb_dimension))
            for wid, w in id2word.items():
                e = ' '.join(map(lambda x: str(x), embedding[wid]))
                f.write('%s %s\n' % (w, e))


class LSTMModel(nn.Module):

    def __init__(self, emb_dimension, d_a=128):
        super(LSTMModel, self).__init__()
        self.emb_dimension = emb_dimension
        self.hidden_dim = emb_dimension
        self.lstm = nn.LSTM(input_size=self.emb_dimension, hidden_size=self.hidden_dim, num_layers=1, bidirectional=True)
        initrange = 1.0 / self.emb_dimension
        init.uniform_(self.lstm.all_weights[0][0], -initrange, initrange)
        init.uniform_(self.lstm.all_weights[0][1], -initrange, initrange)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.linear_first = torch.nn.Linear(2*self.hidden_dim, d_a)
        self.linear_first.bias.data.fill_(0)
        self.linear_second = torch.nn.Linear(d_a, 1)
        self.linear_second.bias.data.fill_(0)
        self.linear_third = torch.nn.Linear(2*self.hidden_dim, self.emb_dimension)
        self.linear_third.bias.data.fill_(0)

    def init_hidden(self, batch_size):
        # first is the hidden h
        # second is the cell c
        return (torch.zeros(2, batch_size, self.hidden_dim).to(self.device),
                torch.zeros(2, batch_size, self.hidden_dim).to(self.device))

    def forward(self, input, batch_size):                 
        self.hidden = self.init_hidden(batch_size)
        lstm_out, self.hidden = self.lstm(input, self.hidden)             
        a=self.linear_first(lstm_out)             
        a=torch.tanh(a)
        a=self.linear_second(a)           
        a=F.softmax(a,dim=0)             
        a=a.expand(5,batch_size,2*self.hidden_dim)
        y=(a*lstm_out).sum(dim=0)    
        y=self.linear_third(y)  
        return y

class CNNModel(nn.Module):
    def __init__(self, output1, output2, emb_dimension):    #output1=32 output2=32
        super(CNNModel, self).__init__()
        self.emb_dimension = emb_dimension
        self.conv1 = nn.Conv2d(1, output1, (3, 3))
        self.conv2 = nn.Conv2d(output1, output2, (3, 3))
        self.hidden2result = nn.Linear(output2*64, emb_dimension)    
        self.bn1 = nn.BatchNorm2d(output1)
        self.bn2 = nn.BatchNorm2d(output2)
        self.bn3 = nn.BatchNorm1d(emb_dimension)
        initrange = 1.0 / self.emb_dimension
        initrange1 = 1e-4
        initrange2 = 1e-2
        init.uniform_(self.conv1.weight.data, -initrange1, initrange1)
        init.uniform_(self.conv2.weight.data, -initrange2, initrange2)
        init.uniform_(self.hidden2result.weight.data, -initrange, initrange)

    def forward(self, x):  
        x = self.conv1(x)
        x = F.max_pool2d(self.bn1(x), 2)
        x = self.conv2(x)
        x = F.max_pool2d(self.bn2(x), 2)
        x = x.view(x.size()[0], -1)
        x = self.hidden2result(x)
        x = F.relu(self.bn3(x))
        return x                              