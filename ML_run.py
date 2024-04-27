import pandas as pd
from torch.utils.data import DataLoader
import utils
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch import nn
import numpy as np
from sklearn.metrics import roc_auc_score


class GAFM( torch.nn.Module ):

    def __init__( self,n_entitys, n_users, n_items, dim):

        super( GAFM, self ).__init__( )

        self.items = nn.Embedding( n_items, dim, max_norm = 1 )
        self.users_df = nn.Embedding( n_entitys, dim, max_norm = 1 )
        self.users_positive = nn.Embedding( n_users, dim, max_norm = 1 )
        self.users_negative = nn.Embedding( n_users, dim, max_norm = 1 )

        #
        self.query = nn.Linear(dim, dim)
        self.key1 = nn.Linear(dim , dim)
        self.value1 = nn.Linear(dim , dim)
        self.key2 = nn.Linear(dim, dim)
        self.value2 = nn.Linear(dim, dim)

        self.f1 = nn.Linear(dim, 100)
        self.f2 = nn.Linear(100, 1)
        self.f3 = nn.Linear(dim, 100)
        self.f4 = nn.Linear(100, 1)

        self.fc1 = nn.Linear(dim * 2, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 1)
        self.relu = nn.ReLU()

    #FM聚合
    def FMaggregator( self, feature_embs ):
        square_of_sum = torch.sum( feature_embs, dim = 1 )**2
        sum_of_square = torch.sum( feature_embs**2, dim = 1 )
        output = square_of_sum - sum_of_square
        return output
    def __getEmbeddingByNeibourIndex( self, orginal_indexes, nbIndexs, aggEmbeddings ):
        new_embs = []
        for v in orginal_indexes:
            embs = aggEmbeddings[ torch.squeeze( torch.LongTensor( nbIndexs.loc[v].values )) ]

            new_embs.append( torch.unsqueeze( embs, dim = 0 ) )

        return torch.cat( new_embs, dim = 0 )


    def attentionPositive_Negative(self, users_positive, users_negative, users_df):
        Q = self.query(users_df)
        users_positive_k = self.key1(users_positive)
        users_positive_v = self.value1(users_positive)
        users_negative_k = self.key2(users_negative)
        users_negative_v = self.value2(users_negative)

        score1 = users_positive_k*Q
        score2 = users_negative_k*Q

        score1 = self.f1(score1)
        score1 = self.relu(score1)
        score1 = torch.sigmoid(self.f2(score1))
        score2 = self.f3(score2)
        score2 = self.relu(score2)
        score2 = torch.sigmoid(self.f4(score2))

        user1 = score1 *users_positive_v
        user2 = score2 *users_negative_v
        user = user1 + user2
        return user
    def gnnForward( self, adj_lists):
        n_hop = 1
        for df in adj_lists:
            if n_hop == 1:
                entity_embs = self.users_df( torch.LongTensor( df.values ) )
            else:
                entity_embs = self.__getEmbeddingByNeibourIndex( df.values, neighborIndexs, aggEmbeddings )
            target_embs = self.users_df( torch.LongTensor( df.index) )
            aggEmbeddings = self.FMaggregator( entity_embs )
            if n_hop < len( adj_lists ):
                neighborIndexs = pd.DataFrame( range( len( df.index ) ), index = df.index )
            aggEmbeddings =   aggEmbeddings + target_embs
            n_hop +=1
        return aggEmbeddings

    def gnnForwardpositive(self, adj_lists):
        n_hop = 1
        for df in adj_lists:
            # print(df)
            if n_hop == 1:
                entity_embs = self.users_positive(torch.LongTensor(df.values-30))
            else:
                entity_embs = self.__getEmbeddingByNeibourIndex(df.values, neighborIndexs, aggEmbeddings)
            weights = []
            for i in range(len(df.index)):
                if df.index[i] == df.values[i][0]== df.values[i][1]== df.values[i][2]== df.values[i][3]== df.values[i][4]:
                    weights.append([0,0,0,0,0])
                else:
                    weights.append([G2.edges[(df.values[i][0], df.index[i])]['weight'],G2.edges[(df.values[i][1], df.index[i])]['weight'],G2.edges[(df.values[i][2], df.index[i])]['weight'],G2.edges[(df.values[i][3], df.index[i])]['weight'],G2.edges[(df.values[i][4], df.index[i])]['weight']])
            weights = torch.Tensor(weights)
            weights = weights.view(len(df.index),len(df.values[0]),1)
            target_embs = self.users_positive(torch.LongTensor(df.index-30))
            aggEmbeddings = weights* entity_embs
            aggEmbeddings = torch.sum(aggEmbeddings,dim=1)
            aggEmbeddings = aggEmbeddings + target_embs
            if n_hop < len(adj_lists):
                neighborIndexs = pd.DataFrame(range(len(df.index)), index=df.index)
            n_hop += 1
        return aggEmbeddings
    def gnnForwardnegative(self, adj_lists):
        n_hop = 1
        for df in adj_lists:
            if n_hop == 1:
                entity_embs = self.users_negative(torch.LongTensor(df.values - 30))
            else:
                entity_embs = self.__getEmbeddingByNeibourIndex(df.values, neighborIndexs, aggEmbeddings)
            weights = []
            for i in range(len(df.index)):
                if df.index[i] == df.values[i][0]== df.values[i][1]== df.values[i][2]== df.values[i][3]== df.values[i][4]:
                    weights.append([0,0,0,0,0])
                else:
                    weights.append([G3.edges[(df.values[i][0], df.index[i])]['weight'],
                                G3.edges[(df.values[i][1], df.index[i])]['weight'],
                                G3.edges[(df.values[i][2], df.index[i])]['weight'],
                                G3.edges[(df.values[i][3], df.index[i])]['weight'],
                                G3.edges[(df.values[i][4], df.index[i])]['weight']])

            # print(weights)
            weights = torch.Tensor(weights)
            weights = weights.view(len(df.index), len(df.values[0]), 1)
            # print(weights)
            # print(weights.shape)
            # exit(0)
            target_embs = self.users_negative(torch.LongTensor(df.index - 30))
            # print("target_embs:", target_embs.shape)
            # print("entity_embs:", entity_embs.shape)
            aggEmbeddings = weights * entity_embs
            aggEmbeddings = torch.sum(aggEmbeddings, dim=1)
            aggEmbeddings = aggEmbeddings + target_embs
            # print("aggEmbeddings:", aggEmbeddings.shape)
            if n_hop < len(adj_lists):
                neighborIndexs = pd.DataFrame(range(len(df.index)), index=df.index)
            n_hop += 1
        return aggEmbeddings
    def forward( self,u, i, adj_lists_G1,adj_lists_G2,adj_lists_G3):
        # [batch_size, dim]
        items = self.items(i)
        users_df = self.gnnForward(adj_lists_G1)
        users_positive = self.gnnForwardpositive(adj_lists_G2)
        users_negative = self.gnnForwardnegative(adj_lists_G3)
        users=self.attentionPositive_Negative(users_positive,users_negative,users_df)
        uv = torch.cat((users, items), dim=1)
        uv = self.fc1(uv)
        uv = self.relu(uv)
        uv = self.fc2(uv)
        uv = self.relu(uv)
        uv = F.dropout(uv)
        uv = self.fc3(uv)
        uv = torch.squeeze(uv)
        logit = torch.sigmoid(uv)
        return logit

@torch.no_grad()
def doEva(net,d):
    net.eval()
    criterion = torch.nn.BCELoss()
    d = torch.LongTensor(d)
    u, i, r = d[:, 0], d[:, 1], d[:, 2]
    rr = r.float()
    u_index = u.detach().numpy()
    adj_lists_G1 = utils.graphSage4RecAdjType(G1, u_index, [3])
    adj_lists_G2 = utils.graphSage4RecAdjType2(G2, u_index)
    adj_lists_G3 = utils.graphSage4RecAdjType2(G3, u_index)
    out = net(u, i, adj_lists_G1, adj_lists_G2,adj_lists_G3)
    loss = criterion(out,rr)
    y = np.array([i for i in out])
    y_pred = np.array([1 if i >= 0.5 else 0 for i in out])
    y_true = r.detach().numpy()
    auc_score = roc_auc_score(y_true, y)

    return auc_score, loss
def train(epoch=50, batchSize=128, lr=0.0005, eva_per_epochs=1):

    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.AdamW(net.parameters(), lr=lr)

    for e in range(epoch):
        net.train()
        all_lose = 0
        for u, i, r in tqdm(DataLoader(train_data, batch_size=batchSize, shuffle=True)):
            r = torch.FloatTensor(r.detach().numpy())
            optimizer.zero_grad()
            u_index = u.detach().numpy()
            adj_lists_G1 = utils.graphSage4RecAdjType(G1, u_index, [3])
            adj_lists_G2 = utils.graphSage4RecAdjType2(G2, u_index)
            adj_lists_G3 = utils.graphSage4RecAdjType2(G3, u_index)
            logits = net(u,i,adj_lists_G1,adj_lists_G2,adj_lists_G3)

            loss = criterion(logits,r)
            all_lose+=loss
            loss.backward()
            optimizer.step()

        print('epoch {}, avg_loss = {:.4f}'.format(e, all_lose / (len(train_data) // batchSize)))
        model1_pth = f"ML_Model/model{e}.pth"
        torch.save(net.state_dict(), model1_pth)
        if e % eva_per_epochs == 0:
            auc_score,loss = doEva(net, val_data)
            print(
                'val:  auc_score{:.4f} | loss{:.4f} '.format(
                    auc_score, loss))

def test():
    model_path = f'ML_Model/model4.pth'
    net.load_state_dict(torch.load(model_path))
    auc_score, loss = doEva(net, test_data)
    print(
        'test:  auc_score{:.4f} | loss{:.4f} '.format(
            auc_score, loss))

if __name__ == '__main__':
    print('Reading triplet data...')
    user_set, item_set, train_data = utils.getdata(
        "Data/Movielens 1M/train_data.csv")
    _, _, test_data = utils.getdata("Data/Movielens 1M/test_data.csv")
    _, _, val_data = utils.getdata("Data/Movielens 1M/val_data.csv")

    print('Constructing graph...')
    # Constructing the user-attribute graph G1
    entitys, pairs = utils.readGraphData('Data/Movielens 1M/userinfo.csv')
    G1 = utils.get_graph(pairs)

    # Constructing the Positive sentiment implicit social graph G2
    G2,empty_Positive = utils.get_Positive_graph(train_data, user_set)
    #"empty_Positive" refers to the isolated set of users without positive sentiment similar friends.
    print(empty_Positive)
    # Constructing the Negative sentiment implicit social graph G3
    G3,empty_Negative = utils.get_Negative_graph(train_data, user_set)
    # "empty_Negative" refers to the isolated set of users without negative sentiment similar friends.
    print(empty_Negative)
    print("Model Training...")
    net = GAFM(max(user_set) + 1,len(user_set), max(item_set) + 1, 128)
    train()
    test()