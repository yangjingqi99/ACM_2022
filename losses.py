from __future__ import print_function
import torch
import torch.nn as nn



# clear those instances that have no positive instances to avoid training error：
class SupConLoss_clear_new(nn.Module):
    def __init__(self,opt):
        super(SupConLoss_clear_new, self).__init__()
        self.temperature = opt.ins_temp
        self.temperature_a = opt.ins_tempa

    def forward(self, features, attributes, labels):

        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        batch_size = features.shape[0]
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)
        mask_n = 1 - mask
        anchor_dot_contrast = torch.div(
            torch.matmul(features, features.T),
            self.temperature)

        aa__dot_contrast=torch.div(
            torch.matmul(attributes, attributes.T),
            self.temperature_a)

        # normalize the logits for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        aa_max, _ = torch.max(aa__dot_contrast, dim=1, keepdim=True)
        aa_logits = aa__dot_contrast - aa_max.detach()

        logits_mask = torch.scatter(# scatter_(dim, index, src):
            torch.ones_like(mask),
            1,#dim=1
            torch.arange(batch_size).view(-1, 1).to(device),#index
            0
        )
        mask = mask * logits_mask
        single_samples = (mask.sum(1) == 0).float()#

        # compute log_prob (logSoftmax)
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        exp_vvlogits = torch.exp(logits) * logits_mask
        log_vvprob = logits - torch.log(exp_vvlogits.sum(1, keepdim=True))
        exp_aalogits = torch.exp(aa_logits) * logits_mask
        log_aaprob = aa_logits - torch.log(exp_aalogits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        # invoid to devide the zero
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1)+single_samples)



        # loss
        # filter those single sample
        loss_pos = - mean_log_prob_pos*(1-single_samples)
        loss_pos = loss_pos.sum()/(loss_pos.shape[0]-single_samples.sum())

        loss_neg=torch.nn.functional.mse_loss(mask_n * log_vvprob, mask_n * log_aaprob)

        return loss_pos, loss_neg

class SupConLoss_clear_mask0(nn.Module):
    def __init__(self,opt):
        super(SupConLoss_clear_mask0, self).__init__()
        self.temperature = opt.ins_temp
        self.temperature_a = opt.ins_tempa

    def forward(self, features, attributes, labels):

        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        batch_size = features.shape[0]
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)
        # mask_t = torch.eq(mask, torch.zeros_like(mask)).float().to(device)
        mask_n = 1 - mask
        anchor_dot_contrast = torch.div(
            torch.matmul(features, features.T),
            self.temperature)

        aa__dot_contrast=torch.div(
            torch.matmul(attributes, attributes.T),
            self.temperature_a)

        # normalize the logits for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()
        aa_max, _ = torch.max(aa__dot_contrast, dim=1, keepdim=True)
        aa_logits = aa__dot_contrast - aa_max.detach()

        logits_mask = torch.scatter(# scatter_(dim, index, src):
            torch.ones_like(mask),
            1,
            torch.arange(batch_size).view(-1, 1).to(device),#index
            0
        )
        mask = mask * logits_mask
        single_samples = (mask.sum(1) == 0).float()#

        # compute log_prob (logSoftmax)
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        exp_vvlogits = torch.exp(logits) * logits_mask
        log_vvprob = logits - torch.log(exp_vvlogits.sum(1, keepdim=True))
        exp_aalogits = torch.exp(aa_logits) * logits_mask
        log_aaprob = aa_logits - torch.log(exp_aalogits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        # invoid to devide the zero
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1)+single_samples)



        # loss
        # filter those single sample
        loss_pos = - mean_log_prob_pos*(1-single_samples)
        loss_pos = loss_pos.sum()/(loss_pos.shape[0]-single_samples.sum())
        loss_neg = torch.nn.functional.mse_loss(log_vvprob, log_aaprob)

        return loss_pos, loss_neg

