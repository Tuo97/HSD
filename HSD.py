import torch
import torch.nn as nn
import torch.nn.functional as F
from manifold.hyperboloid import Hyperboloid
import numpy as np

class Aggregator_hyp(nn.Module):
    def __init__(self, A_in, dropout, manifold):
        super(Aggregator_hyp, self).__init__()

        self.dropout = nn.Dropout(dropout)
        self.graph = A_in
        self.manifold = manifold

    def forward(self, ego_embeddings):
        ego_embeddings = torch.sparse.mm(self.graph, ego_embeddings)
        # norm_ego_embeddings = self.manifold.minkowski_norm(ego_embeddings)
        # ego_embeddings = ego_embeddings / norm_ego_embeddings
        return ego_embeddings

class HSD(nn.Module):
    def __init__(self, config, args, image_feats, text_feats):
        super(HSD, self).__init__()
        self.min_norm = 1e-6
        self.max_norm = 1e6

        self.n_users = config['n_users']
        self.n_items = config['n_items']

        self.embed_dim = args.embed_size + 1
        self.n_layers = args.n_layers

        self.channel_weight = args.channel_weight
        self.kl_weight = args.kl_weight
        self.tau = args.tau

        self.dropout = args.dropout
        self.l2_weight = args.l2_weight
        self.batch_size = args.batch_size

        self.set_curvature(args)

        self.item_emb_img = nn.Embedding.from_pretrained(torch.Tensor(image_feats), freeze=True)
        self.item_emb_txt = nn.Embedding.from_pretrained(torch.Tensor(text_feats), freeze=True)

        self.user_emb_id = nn.init.xavier_uniform_(torch.rand((self.n_users, self.embed_dim), requires_grad=True)).cuda()
        self.item_emb_id = nn.init.xavier_uniform_(torch.rand((self.n_items, self.embed_dim), requires_grad=True)).cuda()
        self.user_emb_img = nn.init.xavier_uniform_(torch.rand((self.n_users, self.embed_dim), requires_grad=True)).cuda()
        self.user_emb_txt = nn.init.xavier_uniform_(torch.rand((self.n_users, self.embed_dim), requires_grad=True)).cuda()

        self.trans_i = nn.Linear(4096, self.embed_dim)
        self.trans_t = nn.Linear(384, self.embed_dim)

        self.all_h_list = config['all_h_list']
        self.all_t_list = config['all_t_list']
        self.all_v_list = config['all_v_list']

        self.all_h_list = torch.LongTensor(self.all_h_list).cuda()
        self.all_t_list = torch.LongTensor(self.all_t_list).cuda()
        self.indice = torch.stack([self.all_h_list, self.all_t_list], dim=0)
        self.A_values = torch.tensor(self.all_v_list).view(-1).cuda()
        self.A_values = self.A_values.float()

        self.A_in = torch.sparse.FloatTensor(self.indice, self.A_values, [self.n_users+self.n_items, self.n_users+self.n_items])

        self.manifold = Hyperboloid()
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax()

        self.aggregator_layers_id = nn.ModuleList()
        self.aggregator_layers_img = nn.ModuleList()
        self.aggregator_layers_txt = nn.ModuleList()

        for k in range(self.n_layers):
            self.aggregator_layers_id.append(Aggregator_hyp(self.A_in, self.dropout, self.manifold))
            self.aggregator_layers_img.append(Aggregator_hyp(self.A_in, self.dropout, self.manifold))
            self.aggregator_layers_txt.append(Aggregator_hyp(self.A_in, self.dropout, self.manifold))

    def set_curvature(self, args):
        self.c = torch.tensor([args.curvature], dtype=torch.float).cuda()
      
    def get_ego_embeddings(self, a, b):
        ego_emb = torch.cat([a, b], dim=0)
        return ego_emb

    def inference(self):
        item_emb_img = self.trans_i(self.item_emb_img.weight)
        item_emb_txt = self.trans_t(self.item_emb_txt.weight)

        ego_embeddings_id = self.get_ego_embeddings(self.user_emb_id, self.item_emb_id)
        ego_embeddings_img = self.get_ego_embeddings(self.user_emb_img, item_emb_img)
        ego_embeddings_txt = self.get_ego_embeddings(self.user_emb_txt, item_emb_txt)

        ego_embeddings_id = self.manifold.proj_tan0(ego_embeddings_id, self.c)
        ego_embeddings_id = self.manifold.expmap0(ego_embeddings_id, self.c)

        ego_embeddings_img = self.manifold.proj_tan0(ego_embeddings_img, self.c)
        ego_embeddings_img = self.manifold.expmap0(ego_embeddings_img, self.c)

        ego_embeddings_txt = self.manifold.proj_tan0(ego_embeddings_txt, self.c)
        ego_embeddings_txt = self.manifold.expmap0(ego_embeddings_txt, self.c)

        layer_embeddings_id = []
        layer_embeddings_img = []
        layer_embeddings_txt = []

        for i in range(self.n_layers):
            # Channel-wise graph convolution...
            ego_embeddings_id = self.aggregator_layers_id[i](ego_embeddings_id)
            ego_embeddings_img = self.aggregator_layers_img[i](ego_embeddings_img)
            ego_embeddings_txt = self.aggregator_layers_txt[i](ego_embeddings_txt)

            layer_embeddings_id.append(ego_embeddings_id)
            layer_embeddings_img.append(ego_embeddings_img)
            layer_embeddings_txt.append(ego_embeddings_txt)

        all_embeddings_id = torch.stack(layer_embeddings_id, dim=1)
        all_embeddings_id = torch.mean(all_embeddings_id, dim=1)
        self.all_embeddings_id_1 = self.manifold.logmap0(all_embeddings_id, self.c)

        all_embeddings_img = torch.stack(layer_embeddings_img, dim=1)
        all_embeddings_img = torch.mean(all_embeddings_img, dim=1)
        self.all_embeddings_img_1 = self.manifold.logmap0(all_embeddings_img, self.c)
        self.all_embeddings_img_1 = 0.15 * self.all_embeddings_img_1 

        all_embeddings_txt = torch.stack(layer_embeddings_txt, dim=1)
        all_embeddings_txt = torch.mean(all_embeddings_txt, dim=1)
        self.all_embeddings_txt_1 = self.manifold.logmap0(all_embeddings_txt, self.c)
        self.all_embeddings_txt_1 = 0.75 * self.all_embeddings_txt_1

        return self.all_embeddings_id_1,  self.all_embeddings_img_1, self.all_embeddings_txt_1

    def forward(self, user_ids, item_pos_ids, item_neg_ids):
        """
                user_ids:       (cf_batch_size)
                item_pos_ids:   (cf_batch_size)
                item_neg_ids:   (cf_batch_size)
        """
        all_embeddings_id, all_embeddings_img, all_embeddings_txt = self.inference()

        user_embeddings_id, item_embeddings_id = torch.split(all_embeddings_id, [self.n_users, self.n_items])
        user_embeddings_img, item_embeddings_img = torch.split(all_embeddings_img, [self.n_users, self.n_items])
        user_embeddings_txt, item_embeddings_txt = torch.split(all_embeddings_txt, [self.n_users, self.n_items])

        user_ids = torch.LongTensor(user_ids).cuda()
        item_pos_ids = torch.LongTensor(item_pos_ids).cuda()
        item_neg_ids = torch.LongTensor(item_neg_ids).cuda()

        pos_user_embeddings_id = user_embeddings_id[user_ids]
        pos_item_embeddings_id = item_embeddings_id[item_pos_ids]
        neg_item_embeddings_id = item_embeddings_id[item_neg_ids]

        pos_user_embeddings_img = user_embeddings_img[user_ids]
        pos_item_embeddings_img = item_embeddings_img[item_pos_ids]
        neg_item_embeddings_img = item_embeddings_img[item_neg_ids]

        pos_user_embeddings_txt = user_embeddings_txt[user_ids]
        pos_item_embeddings_txt = item_embeddings_txt[item_pos_ids]
        neg_item_embeddings_txt = item_embeddings_txt[item_neg_ids]

        # Equation (13)
        pos_score_id = torch.sum(pos_user_embeddings_id * pos_item_embeddings_id, dim=1)  # (cf_batch_size)
        neg_score_id = torch.sum(pos_user_embeddings_id * neg_item_embeddings_id, dim=1)  # (cf_batch_size)

        pos_score_img = torch.sum(pos_user_embeddings_img * pos_item_embeddings_img, dim=1)  # (cf_batch_size)
        neg_score_img = torch.sum(pos_user_embeddings_img * neg_item_embeddings_img, dim=1)  # (cf_batch_size)

        pos_score_txt = torch.sum(pos_user_embeddings_txt * pos_item_embeddings_txt, dim=1)  # (cf_batch_size)
        neg_score_txt = torch.sum(pos_user_embeddings_txt * neg_item_embeddings_txt, dim=1)  # (cf_batch_size)

        pos_score_fusion = pos_score_id + pos_score_img + pos_score_txt
        neg_score_fusion = neg_score_id + neg_score_img + neg_score_txt

        # Equation (14)
        cf_loss = (-1.0) * F.logsigmoid(pos_score_fusion - neg_score_fusion)
        cf_loss = torch.mean(cf_loss)

        # Equation (15)
        cf_loss_id = (-1.0) * F.logsigmoid(pos_score_id - neg_score_id)
        cf_loss_id = torch.mean(cf_loss_id)

        cf_loss_img = (-1.0) * F.logsigmoid(pos_score_img - neg_score_img)
        cf_loss_img = torch.mean(cf_loss_img)

        cf_loss_txt = (-1.0) * F.logsigmoid(pos_score_txt - neg_score_txt)
        cf_loss_txt = torch.mean(cf_loss_txt)

        channel_loss = cf_loss_id + cf_loss_img + cf_loss_txt

        # Equation (16)
        score_id = torch.stack([pos_score_id, neg_score_id], dim=1) / self.tau
        score_img = torch.stack([pos_score_img, neg_score_img], dim=1) / self.tau
        score_txt = torch.stack([pos_score_txt, neg_score_txt], dim=1) / self.tau
        score_fusion = torch.stack([pos_score_fusion, neg_score_fusion], dim=1) / self.tau

        score_id = self.softmax(score_id)
        score_img = self.softmax(score_img)
        score_txt = self.softmax(score_txt)
        score_fusion = self.softmax(score_fusion)

        kl_loss_id = torch.mul(score_fusion, torch.log(score_fusion / score_id))
        kl_loss_img = torch.mul(score_fusion, torch.log(score_fusion / score_img))
        kl_loss_txt = torch.mul(score_fusion, torch.log(score_fusion / score_txt))

        kl_loss_id = torch.mean((torch.sum(kl_loss_id, dim=1)), dim=0)
        kl_loss_img = torch.mean((torch.sum(kl_loss_img, dim=1)), dim=0)
        kl_loss_txt = torch.mean((torch.sum(kl_loss_txt, dim=1)), dim=0)

        kl_loss = kl_loss_id + kl_loss_img + kl_loss_txt

        loss = cf_loss + self.channel_weight * channel_loss + self.kl_weight * kl_loss

        return loss

    def predict(self, user_ids):
        user_ids = torch.LongTensor(user_ids).cuda()

        user_embeddings_id, item_embeddings_id = torch.split(self.all_embeddings_id_1, [self.n_users, self.n_items])
        pos_user_embeddings_id = user_embeddings_id[user_ids]
        cf_score_id = torch.matmul(pos_user_embeddings_id, item_embeddings_id.t())

        user_embeddings_img, item_embeddings_img = torch.split(self.all_embeddings_img_1, [self.n_users, self.n_items])
        pos_user_embeddings_img = user_embeddings_img[user_ids]
        cf_score_img = torch.matmul(pos_user_embeddings_img, item_embeddings_img.t())

        user_embeddings_txt, item_embeddings_txt = torch.split(self.all_embeddings_txt_1, [self.n_users, self.n_items])
        pos_user_embeddings_txt = user_embeddings_txt[user_ids]
        cf_score_txt = torch.matmul(pos_user_embeddings_txt, item_embeddings_txt.t())

        cf_score = cf_score_id + cf_score_img + cf_score_txt

        return cf_score

    def get_embeddings(self):
        user_embeddings_id, item_embeddings_id = torch.split(self.all_embeddings_id_1, [self.n_users, self.n_items])
        user_embeddings_img, item_embeddings_img = torch.split(self.all_embeddings_img_1, [self.n_users, self.n_items])
        user_embeddings_txt, item_embeddings_txt = torch.split(self.all_embeddings_txt_1, [self.n_users, self.n_items])

        return user_embeddings_id, item_embeddings_id, user_embeddings_txt, item_embeddings_txt, user_embeddings_img, item_embeddings_img

















