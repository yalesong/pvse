import torch
import torch.nn as nn
import numpy as np

def cosine_sim(x, y):
  """Cosine similarity between all the image and sentence pairs. Assumes x and y are l2 normalized"""
  return x.mm(y.t())

def order_sim(x, y):
  """Order embeddings similarity measure $max(0, x-y)$"""
  YmX = (y.unsqueeze(1).expand(y.size(0), x.size(0), y.size(1)) - \
          x.unsqueeze(0).expand(y.size(0), x.size(0), y.size(1)))
  score = -YmX.clamp(min=0).pow(2).sum(2).sqrt().t()
  return score

def l2norm(x):
  """L2-normalize columns of x"""
  norm = torch.pow(x, 2).sum(dim=-1, keepdim=True).sqrt()
  return torch.div(x, norm)

def rbf(x, y, gamma):
  """RBF kernel K(x,y) """
  pdist = torch.norm(x[:, None] - y, dim=2, p=2)
  return torch.exp(-gamma * pdist)


class PVSELoss(nn.Module):

  def __init__(self, opt, reduction='mean'):
    super(PVSELoss, self).__init__()

    self.margin = opt.margin if hasattr(opt, 'margin') else 1.0
    self.num_embeds = opt.num_embeds if hasattr(opt, 'num_embeds') else 1
    self.mmd_weight = opt.mmd_weight if hasattr(opt, 'mmd_weight') else 0.
    self.div_weight = opt.div_weight if hasattr(opt, 'div_weight') else 0.
    self.sim_fn = order_sim if hasattr(opt, 'order') and opt.order else cosine_sim
    self.max_violation = opt.max_violation if hasattr(opt, 'max_violation') else False
    self.reduction = reduction

    if self.num_embeds > 1:
      self.max_pool = torch.nn.MaxPool2d(self.num_embeds)


  def diversity_loss(self, x):
    x = l2norm(x) # Columns of x MUST be l2-normalized
    gram_x = x.bmm(x.transpose(1,2))
    I = torch.autograd.Variable((torch.eye(x.size(1)) > 0.5).repeat(gram_x.size(0), 1, 1))
    if torch.cuda.is_available():
      I = I.cuda()
    gram_x.masked_fill_(I, 0.0)
    loss = torch.stack([torch.norm(g, p=2) for g in gram_x]) / (self.num_embeds**2)
    return loss.mean() if self.reduction=='mean' else loss.sum()


  def mmd_rbf_loss(self, x, y, gamma=None):
    if gamma is None:
      gamma = 1./x.size(-1)
    loss = rbf(x, x, gamma) - 2 * rbf(x, y, gamma) + rbf(y, y, gamma)
    return loss.mean() if self.reduction=='mean' else loss.sum()


  def triplet_ranking_loss(self, A, B, I, max_dim):
    loss = (self.margin + A - B).clamp(min=0.0)
    loss.masked_fill_(I, 0.0)
    if self.max_violation:
      loss = loss.max(max_dim)[0]
    return loss.mean() if self.reduction=='mean' else loss.sum()


  def forward(self, img, txt, img_r, txt_r):
    loss, losses = 0, dict()

    # compute image-sentence score matrix
    if self.num_embeds > 1:
      scores = self.sim_fn(img.view(-1, img.size(-1)), txt.view(-1, txt.size(-1)))
      scores = self.max_pool(scores.unsqueeze(0)).squeeze()
    else:
      scores = self.sim_fn(img, txt)

    diagonal = scores.diag().view(img.size(0), 1)
    d1 = diagonal.expand_as(scores)
    d2 = diagonal.t().expand_as(scores)

    mask = torch.eye(scores.size(0)) > .5
    I = torch.autograd.Variable(mask)
    if torch.cuda.is_available():
      I = I.cuda()

    # compare every diagonal score to scores in its column (image-to-text retrieval)
    i2t_loss = self.triplet_ranking_loss(scores, d1, I, 1)

    # compare every diagonal score to scores in its row (text-to-image retrieval)
    t2i_loss = self.triplet_ranking_loss(scores, d2, I, 0)

    ranking_loss = i2t_loss + t2i_loss
    loss += ranking_loss
    losses['ranking_loss'] = ranking_loss

    # diversity loss
    if self.num_embeds > 1 and self.div_weight > 0.:
      div_loss = self.diversity_loss(img_r) + self.diversity_loss(txt_r)
      loss += self.div_weight * div_loss
      losses['div_loss'] = div_loss

    # domain discrepancy loss
    if self.num_embeds > 1 and self.mmd_weight > 0.:
      mmd_loss = self.mmd_rbf_loss(img.view(-1, img.size(-1)), txt.view(-1, txt.size(-1)), gamma=0.5)
      loss += self.mmd_weight * mmd_loss
      losses['mmd_loss'] = mmd_loss

    return loss, losses
