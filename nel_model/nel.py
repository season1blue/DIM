import torch
import torch.nn as nn
import numpy as np
# from torch.nn import TripletMarginLoss
# from word_level import WordLevel
# from phrase_level import PhraseLevel
# from sent_level import SentLevel
# from gated_fuse import GatedFusion
# from recursive_encoder import RecursiveEncoder
from circle_loss import CircleLoss
from triplet_loss import TripletMarginLoss, NpairLoss
from info_nce import InfoNCE
from interaction import BertLayer, CLIPEncoderLayer, Generator
from torch import functional as F


class GatedFusion(nn.Module):
    def __init__(self, args):
        super(GatedFusion, self).__init__()
        self.hidden_size = args.hidden_size
        self.dropout = args.dropout

        self.lin_seq_att = nn.Sequential(
            nn.LayerNorm(self.hidden_size),
            nn.Dropout(self.dropout)
        )

        self.lin_extra_att = nn.Sequential(
            nn.LayerNorm(self.hidden_size),
            nn.Dropout(self.dropout)
        )

        self.gate = nn.Sequential(
            nn.Linear(self.hidden_size, 1),
            nn.Softmax(dim=0),
        )

        self.filtration_gate = nn.Sequential(
            nn.Linear(self.hidden_size * 2, self.hidden_size),
            nn.ReLU()
        )

    def forward(self, mention, vision, text):
        seq_att = self.lin_seq_att(mention)
        extra_att = self.lin_extra_att(vision + text)

        # attn_gate = self.gate(torch.cat([seq_att.unsqueeze(0), extra_att.unsqueeze(0)], dim=0)).squeeze()
        # fusion = (seq_att - extra_att) * attn_gate[0].unsqueeze(-1) + extra_att
        # out = self.filtration_gate(torch.cat([seq_att, fusion[0].unsqueeze(1)], dim=-1))
        # out = out.max(dim=1)[0]
        out = seq_att + extra_att

        return out

def Contrastive_loss(out_1, out_2, batch_size, temperature=0.5):
    out = torch.cat([out_1, out_2], dim=0)  # [2*B, D]
    sim_matrix = torch.exp(torch.mm(out, out.t().contiguous()) / temperature)  # [2*B, 2*B]
    '''
    torch.mm是矩阵乘法，a*b是对应位置上的数相除，维度和a，b一样
    '''
    mask = (torch.ones_like(sim_matrix) - torch.eye(2 * batch_size, device=sim_matrix.device)).bool()
    '''
    torch.eye生成对角线上为1，其他为0的矩阵
    torch.eye(3)
    tensor([[ 1.,  0.,  0.],
            [ 0.,  1.,  0.],
            [ 0.,  0.,  1.]])
    '''
    # [2*B, 2*B-1]
    sim_matrix = sim_matrix.masked_select(mask).view(2 * batch_size, -1)

    # compute loss
    pos_sim = torch.exp(torch.sum(out_1 * out_2, dim=-1) / temperature)
    # [2*B]
    pos_sim = torch.cat([pos_sim, pos_sim], dim=0)
    loss = (- torch.log(pos_sim / (sim_matrix.sum(dim=-1) - pos_sim))).mean()
    return loss

class ClipLoss(nn.Module):
    def __init__(self, args):
        super(ClipLoss, self).__init__()
        self.device = args.device

        self.loss_img = nn.CrossEntropyLoss()
        self.loss_txt = nn.CrossEntropyLoss()
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def forward(self, image_features, text_features, batch_size):
        image_features = image_features.squeeze(1)
        text_features = text_features.squeeze(1)

        logit_scale = self.logit_scale.exp()
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)

        # cosine similarity as logits
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logits_per_image.t()

        ground_truth = torch.arange(batch_size, dtype=torch.long, device=self.device)
        total_loss = (self.loss_img(logits_per_image, ground_truth) + self.loss_txt(logits_per_text, ground_truth)) / 2

        return total_loss


class NELModel(nn.Module):
    def __init__(self, args, text_config=None, vision_config=None):
        super(NELModel, self).__init__()
        self.hidden_size = args.hidden_size
        self.dropout = args.dropout
        self.output_size = args.output_size
        self.seq_len = args.max_sent_length
        self.text_feat_size = args.text_feat_size
        self.img_feat_size = args.img_feat_size
        self.feat_cate = args.feat_cate.lower()

        self.lambda_c = args.lambda_c
        self.lambda_t = args.lambda_t


        self.split_trans = nn.Sequential(
            nn.Linear(self.img_feat_size, self.hidden_size),
            nn.ReLU(),
            nn.LayerNorm(self.hidden_size),
            nn.Dropout(self.dropout)
        )

        self.img_trans = nn.Sequential(
            nn.Linear(self.img_feat_size, self.hidden_size),
            nn.ReLU(),
            nn.LayerNorm(self.hidden_size),
            nn.Dropout(self.dropout)
        )
        self.text_trans = nn.Sequential(
            nn.Linear(self.text_feat_size, self.hidden_size),
            nn.ReLU(),
            nn.LayerNorm(self.hidden_size),
            nn.Dropout(self.dropout)
        )

        # Dimension reduction
        self.pedia_out_trans = nn.Sequential(
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_size, self.output_size),
        )
        self.mention_mix_trans = nn.Sequential(
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_size * 2, self.hidden_size),
        )

        self.img_att = nn.MultiheadAttention(self.hidden_size, args.nheaders, batch_first=True)
        self.text_att = nn.MultiheadAttention(self.hidden_size, args.nheaders, batch_first=True)

        self.gate = GatedFusion(args)
        # circle loss
        self.loss_function = args.loss_function
        self.loss_margin = args.loss_margin
        self.sim = args.similarity
        self.loss = NpairLoss(args)
        self.clip_loss = ClipLoss(args)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.text_encoder = BertLayer(text_config)
        self.vision_encoder = CLIPEncoderLayer(vision_config)

        self.text_decoder = Generator(text_config, 1)
        self.vision_decoder = Generator(vision_config, 1)

        self.att_method = "encoder"
        self.align_method = "clip"

    def forward(self, model_type, mention=None, text=None, total=None, segement=None, profile=None, scene=None, pos_feats=None, neg_feats=None):
        """
            ------------------------------------------
            Args:
                text: tensor: (batch_size, max_seq_len, text_feat_size), the output of bert hidden size
                img: float tensor: (batch_size, ..., img_feat_size), image features - resnet
                bert_mask: tensor: (batch_size, max_seq_len)
                pos_feats(optional): (batch_size, n_pos, output_size)
                neg_feats(optional): (batch_size, n_neg, output_size)
            Returns:
        """
        batch_size = mention.size(0)

        mention_trans = self.text_trans(mention)
        text_trans = self.text_trans(text) 
        profile_trans = self.text_trans(profile).max(dim=1)[0].unsqueeze(1)

        
        segement_trans = self.img_trans(segement)
        total_trans = self.img_trans(total)
        # print(mention_trans.size(), text_trans.size(), total_trans.size(), segement_trans.size())  # torch.Size([128, 1, 512]) torch.Size([128, 1, 512]) torch.Size([128, 1, 512]) torch.Size([128, 11, 512])
        # print(profile_trans.size())  # 128, 1, 512

        if self.att_method == "att":
            text_att, _ = self.img_att(mention_trans, text_trans, text_trans)
            vision_att, _ = self.img_att(mention_trans, segement_trans, segement_trans)
            profile_att, _ = self.text_att(mention_trans, profile_trans, profile_trans)
        else:
            text_att, _ = self.img_att(mention_trans, text_trans, text_trans)
            vision_att, _ = self.img_att(mention_trans, segement_trans, segement_trans)
            profile_att, _ = self.text_att(mention_trans, profile_trans, profile_trans)

            text_att = self.text_decoder(text_att)
            vision_att = self.vision_decoder(vision_att)

            

        query =  mention_trans + text_att + vision_att
        query = self.pedia_out_trans(query).squeeze(1)  # 128, 512


        # 注意这里的维度，如果不满足TripletMarginLoss的维度设置，会存在broadcast现象，导致性能大幅下降 全都要是 [bsz*hs] query 128, 512        pos 128, 1, 512     neg 128, 1, 512
        triplet_loss = self.loss(query, pos_feats.squeeze(1), neg_feats.squeeze(1))
        loss = triplet_loss

        return loss, query

    def trans(self, x):
        return x
        # return self.text_trans(x)
