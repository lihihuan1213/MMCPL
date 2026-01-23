import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from utils.DirectedGCN import GCN
from SiameseNet import SiameseNet
from utils.cl_loss import hard_info_nce, IntraModalContraLoss
from MVAE import MVAE, mave_loss
import csv

class Net(nn.Module):
    def __init__(self, adj_out, adj_in, img_emb, text_emb, resource_emb, skill_resource_dict):
        super(Net, self).__init__()
        self.adj_out = adj_out  
        self.adj_in = adj_in       
        self.image_emb = img_emb 
        self.text_emb = text_emb
        self.resource_emb = resource_emb 
        self.skill_resource_dict = skill_resource_dict

        self.text_imgAtt = CrossModalAttention(embed_dim=512, num_heads=8, dropout=0.5)
        self.img_textAtt = CrossModalAttention(embed_dim=512, num_heads=8, dropout=0.5)
        
        self.text_loss = IntraModalContraLoss(512,0.07)
        self.image_loss = IntraModalContraLoss(512,0.07)

        self.att_net = CrossAttention(256, 256, 256)
        self.siamese_net = SiameseNet(256)

        self.t1 = nn.Linear(512,512)
        self.t2 = nn.Linear(512,256)

        self.i1 = nn.Linear(512,512)
        self.i2 = nn.Linear(512,256)
        
        self.gcn1 = GCN(nfeat=768, nhid=256, nclass=256)
        self.gcn2 = GCN(nfeat=768, nhid=256, nclass=256) 
        self.w1 = nn.Linear(256, 256)
        self.w2 = nn.Linear(256, 256)

        self.gate = nn.Linear(512,1)

        self.reliner1 = nn.Linear(512,256)
        self.reliner2 = nn.Linear(512,256)

     
    def forward(self, c1, c2):
        text_aug, _ = self.text_imgAtt(self.text_emb, self.image_emb)
        image_aug, _ = self.img_textAtt(self.image_emb, self.text_emb)

        loss_intra = self.text_loss(text_aug) + self.image_loss(image_aug)

        text_embed = F.relu(text_aug + self.text_emb)
        image_embed = F.relu(image_aug + self.image_emb)

        t1 = F.relu(self.t1(text_embed))
        h_text = self.t2(t1)

        i1 = F.relu(self.i1(image_embed))
        h_image = self.i2(i1)

        cl_loss = hard_info_nce(h_text, h_image)

        the_ta = torch.sigmoid(self.gate(torch.cat([h_text, h_image], dim=1)))
        h_final = the_ta * h_text + (1-the_ta) * h_image


        X_out = self.gcn1(self.resource_emb, self.adj_out)
        X_in = self.gcn2(self.resource_emb, self.adj_in)
        X_N = F.relu(self.w1(X_out) + self.w2(X_in))

        skill_repr1 = h_final[c1]
        skill_repr2 = h_final[c2]
        resource_repr1 = self.get_averaged_embeddings(c1, X_N, self.skill_resource_dict)
        resource_repr2 = self.get_averaged_embeddings(c2, X_N, self.skill_resource_dict)
        context_repr1, _ = self.att_net(skill_repr1, resource_repr1)
        context_repr2, _ = self.att_net(skill_repr2, resource_repr2)

        c1_repr = self.reliner1(torch.cat([skill_repr1, context_repr1], dim=1))
        c2_repr = self.reliner2(torch.cat([skill_repr2, context_repr2], dim=1))

        out = torch.sigmoid(self.siamese_net(c1_repr, c2_repr))

        return out, cl_loss, loss_intra


    def get_averaged_embeddings(self, skill_ids, resources_embed, skill_resource_dict):
        pooled_embeddings = []
        embed_dim = resources_embed.shape[1]
        device = resources_embed.device

        for skill_id in skill_ids:
            if skill_id in skill_resource_dict and len(skill_resource_dict[skill_id]) > 0:
                resource_ids = skill_resource_dict[skill_id]
                resource_ids_tensor = torch.tensor(resource_ids, device=device,dtype=torch.long)
                related_embeds = resources_embed[resource_ids_tensor]

                pooled = torch.mean(related_embeds, dim=0)
            else:
                pooled = torch.zeros(embed_dim, device=device)

            pooled_embeddings.append(pooled)

        return torch.stack(pooled_embeddings, dim=0)


class CrossAttention(nn.Module):
    def __init__(self, concept_feat_dim, resource_feat_dim, hidden_dim, num_heads=4): 
        super(CrossAttention, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        assert self.head_dim * num_heads == self.hidden_dim, "hidden_dim must be equal to num_heads * head_dim"
        self.concept_proj = nn.Linear(concept_feat_dim, hidden_dim)
       
        self.resource_key_proj = nn.Linear(resource_feat_dim, hidden_dim)
        self.resource_value_proj = nn.Linear(resource_feat_dim, hidden_dim)

        self.out_proj = nn.Linear(hidden_dim, concept_feat_dim)
        self.norm = nn.LayerNorm(concept_feat_dim)
        self.dropout = nn.Dropout(0.3)

    def forward(self, concept_feats, resource_feats, mask=None):
        batch_size = concept_feats.size(0)
        if resource_feats.dim() == 2:
            resource_feats = resource_feats.unsqueeze(1)

        num_resources = resource_feats.size(1)
        query = self.concept_proj(concept_feats).unsqueeze(1)

        key = self.resource_key_proj(resource_feats)
        value = self.resource_value_proj(resource_feats)

        query = query.view(batch_size, 1, self.num_heads, self.head_dim).transpose(1, 2)
        key = key.view(batch_size, num_resources, self.num_heads, self.head_dim).transpose(1, 2)
        value = value.view(batch_size, num_resources, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(query, key.transpose(-2,-1)) / (self.head_dim ** 0.5)

        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(1)
            scores = scores.masked_fill(mask == 0, -1e9)

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        context = torch.matmul(attn_weights, value)
        context = context.transpose(1, 2).contiguous().view(batch_size, 1, self.hidden_dim)
        context = context.squeeze(1)

        context = self.out_proj(context)
        context = self.dropout(context)
        context_features = self.norm(concept_feats + context)
        return context_features, attn_weights


class CrossModalAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int = 300, 
        num_heads: int = 6,      
        dropout: float = 0.1,    
        bias: bool = True      
    ):
        super(CrossModalAttention, self).__init__()
        assert embed_dim % num_heads == 0, f"embed_dim({embed_dim}) must be divisible by num_heads({num_heads})"
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias) 
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            bias=bias,
            batch_first=False
        )

        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        q_emb: torch.Tensor,    
        kv_emb: torch.Tensor,  
        attn_mask: torch.Tensor = None,
        key_padding_mask: torch.Tensor = None
    ):

        is_2d = False
        if q_emb.dim() == 2:
            is_2d = True
            q_emb = q_emb.unsqueeze(0)  
            kv_emb = kv_emb.unsqueeze(0)  
        batch_size, seq_len, _ = q_emb.shape

        q = self.q_proj(q_emb)
        k = self.k_proj(kv_emb)
        v = self.v_proj(kv_emb)

        q = q.transpose(0, 1) 
        k = k.transpose(0, 1) 
        v = v.transpose(0, 1) 

        attn_output, attn_weights = self.multihead_attn(
            query=q,
            key=k,
            value=v,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask
        )

        attn_output = attn_output.transpose(0, 1)
        enhanced_emb = self.dropout(self.out_proj(attn_output))

        if is_2d:
            enhanced_emb = enhanced_emb.squeeze(0)
            attn_weights = attn_weights.squeeze(0) 
        return enhanced_emb, attn_weights

