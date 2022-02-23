import torch
import torch.nn as nn
import torch.nn.functional as F


class BaseMoCo(nn.Module):
    """base class for MoCo-style memory cache"""
    def __init__(self, K=65536, T=0.07):
        super(BaseMoCo, self).__init__()
        self.K = K
        self.T = T
        self.index = 0

    def _update_pointer(self, bsz):
        self.index = (self.index + bsz) % self.K

    def _update_memory(self, k, queue, k_labels, queue_labels=None):
        """
        Args:
          k: key feature
          queue: memory buffer
          labels: labels of key feature
          queue_labels: labels of memory buffer
        """
        with torch.no_grad():
            num_neg = k.shape[0]
            out_ids = torch.arange(num_neg).cuda()
            out_ids = torch.fmod(out_ids + self.index, self.K).long()
            queue.index_copy_(0, out_ids, k)

            # Updating label memory
            if k_labels is not None and queue_labels is not None:
                queue_labels.index_copy_(0, out_ids, k_labels)

    def _compute_logit(self, q, k, queue):
        """
        Args:
          q: query/anchor feature
          k: key feature
          queue: memory buffer
        """
        # pos logit
        bsz = q.shape[0]
        pos = torch.bmm(q.view(bsz, 1, -1), k.view(bsz, -1, 1))
        pos = pos.view(bsz, 1)

        # neg logit
        neg = torch.mm(queue, q.transpose(1, 0))
        neg = neg.transpose(0, 1)

        out = torch.cat((pos, neg), dim=1)
        out = torch.div(out, self.T)
        out = out.squeeze().contiguous()

        return out
    
    def _compute_loss_with_labels(self, q, k, queue, batch_labels, queue_labels, topk_labels):
        """
        Args:
          q: query/anchor feature
          k: key feature
          queue: memory buffer
          batch_labels: labels of q
          queue_labels: labels of memory buffer
        """
        # bsz = q.shape[0]
        # qsz = queue.shape[0]
        k_queue = torch.cat([k, queue], dim=0)
        k_queue_labels = torch.cat([batch_labels, queue_labels], dim=0)

        logits = torch.mm(q, k_queue.transpose(1, 0))
        logits = torch.div(logits, self.T)
        logits = torch.subtract(logits, torch.max(logits.detach(), dim=1, keepdim=True)[0])
        exp_logits = torch.exp(logits)

        positives_mask = torch.eq(batch_labels, k_queue_labels.T).float().to(q.device)
        num_positives_per_row = torch.sum(positives_mask, dim=1)

        topk_mask = torch.eq(topk_labels.unsqueeze(-1).permute(1,2,0), k_queue_labels)
        topk_mask = topk_mask.any(dim=0).float().to(q.device)
        negatives_mask = 1. - positives_mask
        negatives_mask = torch.stack([topk_mask, negatives_mask], dim=0).all(dim=0)

        denominator = torch.sum(exp_logits * negatives_mask, axis=1, keepdim=True) + torch.sum(exp_logits * positives_mask, axis=1, keepdim=True)
        log_probs = (logits - torch.log(denominator)) * positives_mask
        log_probs = torch.sum(log_probs, dim=1)
        log_probs = torch.divide(log_probs, num_positives_per_row)

        loss = -log_probs

        # Scaling by temperature -- helps normalize the loss (based on TensorFlow implementation)
        loss *= self.T

        return loss.mean()


class RGBMoCo(BaseMoCo):
    """Single Modal (e.g., RGB) MoCo-style cache"""
    def __init__(self, n_dim, K=65536, T=0.07):
        super(RGBMoCo, self).__init__(K, T)
        # create memory queue
        self.register_buffer('memory', torch.randn(K, n_dim))
        # create memory label queue
        self.register_buffer('memory_labels', torch.randn(K, 1))
        self.memory = F.normalize(self.memory)

    def forward(self, q, k, q_jig=None, all_k=None, batch_labels=None, topk_labels=None):
        """
        Args:
          q: query on current node
          k: key on current node
          q_jig: jigsaw query
          all_k: gather of feats across nodes; otherwise use q
          batch_labels: labels of the batch
          topk_labels: topk predictions from a pre-trained classifier for each image in the batch (bs, k)
        """
        bsz = q.size(0)
        k = k.detach()

        # compute logit
        queue = self.memory.clone().detach()
        
        if batch_labels is None:
            logits = self._compute_logit(q, k, queue)
        # Using labels in supervised setting (SupCon) with topk masking
        else:
            queue_labels = self.memory_labels.clone().detach()
            logits = self._compute_loss_with_labels(q, k, queue, batch_labels, queue_labels, topk_labels)

        if q_jig is not None:
            logits_jig = self._compute_logit(q_jig, k, queue)

        # set label
        labels = torch.zeros(bsz, dtype=torch.long).cuda()

        # update memory
        all_k = all_k if all_k is not None else k
        if batch_labels is None:
            self._update_memory(all_k, self.memory)
        else:
            self._update_memory(all_k, self.memory, self.memory_labels)

        self._update_pointer(all_k.size(0))

        if q_jig is not None:
            return logits, logits_jig, labels
        else:
            return logits, labels


class CMCMoCo(BaseMoCo):
    """MoCo-style memory for two modalities, e.g. in CMC"""
    def __init__(self, n_dim, K=65536, T=0.07):
        super(CMCMoCo, self).__init__(K, T)
        # create memory queue
        self.register_buffer('memory_1', torch.randn(K, n_dim))
        self.register_buffer('memory_2', torch.randn(K, n_dim))
        self.memory_1 = F.normalize(self.memory_1)
        self.memory_2 = F.normalize(self.memory_2)

    def forward(self, q1, k1, q2, k2,
                q1_jig=None, q2_jig=None,
                all_k1=None, all_k2=None):
        """
        Args:
          q1: q of modal 1
          k1: k of modal 1
          q2: q of modal 2
          k2: k of modal 2
          q1_jig: q jig of modal 1
          q2_jig: q jig of modal 2
          all_k1: gather of k1 across nodes; otherwise use k1
          all_k2: gather of k2 across nodes; otherwise use k2
        """
        bsz = q1.size(0)
        k1 = k1.detach()
        k2 = k2.detach()

        # compute logit
        queue1 = self.memory_1.clone().detach()
        queue2 = self.memory_2.clone().detach()
        logits1 = self._compute_logit(q1, k2, queue2)
        logits2 = self._compute_logit(q2, k1, queue1)
        if (q1_jig is not None) and (q2_jig is not None):
            logits1_jig = self._compute_logit(q1_jig, k2, queue2)
            logits2_jig = self._compute_logit(q2_jig, k1, queue1)

        # set label
        labels = torch.zeros(bsz, dtype=torch.long).cuda()

        # update memory
        all_k1 = all_k1 if all_k1 is not None else k1
        all_k2 = all_k2 if all_k2 is not None else k2
        assert all_k1.size(0) == all_k2.size(0)
        self._update_memory(all_k1, self.memory_1)
        self._update_memory(all_k2, self.memory_2)
        self._update_pointer(all_k1.size(0))

        if (q1_jig is not None) and (q2_jig is not None):
            return logits1, logits2, logits1_jig, logits2_jig, labels
        else:
            return logits1, logits2, labels
