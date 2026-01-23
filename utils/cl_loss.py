import torch
import torch.nn as nn
import torch.nn.functional as F

def l2norm(x):
    # L2 Normalization
    return F.normalize(x, dim=-1, eps=1e-8)


# Modal inter-modal contrastive loss
def hard_info_nce(text, image, temp=0.07, neg_weight=0.5):
    text = l2norm(text)
    image = l2norm(image)
    sim = text @ image.t() /temp
    B = sim.size(0)
    labels = torch.arange(B, device=text.device) 

    loss_t2i = F.cross_entropy(sim, labels)
    loss_i2t = F.cross_entropy(sim.t(), labels)
    loss_base = (loss_i2t + loss_t2i) / 2

    eye = torch.eye(B, device=text.device, dtype=torch.bool)
    sim_m = sim.masked_fill(eye, -torch.inf)
    hardest_scores, _ = sim_m.max(dim=1)
    loss_hard = F.cross_entropy(
        torch.stack([sim[range(B), labels], hardest_scores], dim=1),
        torch.zeros(B, device=text.device, dtype=torch.long)
    )

    return loss_base + neg_weight * loss_hard


# Modal internal contrast loss
class IntraModalContraLoss(nn.Module):
    def __init__(self, embed_dim=512, temp=0.07, augment=True):
        super(IntraModalContraLoss, self).__init__()
        self.temp = temp
        self.augment = augment

        self.augmentor = nn.Sequential(
            nn.Dropout(0.1)
        )

    def augment_embeddings(self, embeddings): 
        if not self.augment:
            return embeddings
        
        aug1 = self.augmentor(embeddings)
        aug2 = self.augmentor(embeddings) + torch.randn_like(embeddings) * 0.01  

        # proj = nn.Linear(embed_dim, embed_dim).to(embeddings.device)
        # aug2 = F.normalize(proj(embeddings), dim=-1)
        return torch.stack([aug1, aug2], dim=1)
    
    def forward(self, embeddings):
        B, D = embeddings.shape

        aug_views = self.augment_embeddings(embeddings)

        views = aug_views.reshape(2 * B, D)  # [212, 512]

        sim_matrix = torch.matmul(views, views.T) / self.temp
        
        labels = torch.arange(B).to(embeddings.device)
        pos_mask = torch.zeros(2 * B, 2 * B, dtype=torch.bool).to(embeddings.device)
        pos_mask[torch.arange(2 * B), torch.cat([labels + B, labels])] = True
        
        self_mask = torch.eye(2 * B, dtype=torch.bool).to(embeddings.device)
        pos_mask = pos_mask & ~self_mask
        
        labels = torch.cat([labels + B, labels]) 
        loss = F.cross_entropy(sim_matrix, labels)

        return loss
        