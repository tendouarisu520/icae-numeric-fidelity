import torch
import torch.nn.functional as F


def sequence_logp(logits, labels):
    """Mean answer-token log P; ignore prompt and padding, include EOS."""
    targets = labels[:, 1:]
    mask = targets.ne(-100)
    if not bool(mask.any(dim=1).all()):
        raise ValueError('Each sequence needs at least one scored answer token')
    safe = targets.masked_fill(~mask, 0)
    logp = F.log_softmax(logits[:, :-1].float(), dim=-1)
    selected = logp.gather(-1, safe.unsqueeze(-1)).squeeze(-1)
    return (selected * mask).sum(-1) / mask.sum(-1)


def binding_loss(correct_logp, foil_logp, margin=.5):
    return F.softplus(margin - correct_logp + foil_logp).mean()


def hard_negative_loss(query, positive, negative, temperature=.07):
    if temperature <= 0:
        raise ValueError('Temperature must be positive')
    q = F.normalize(query.float(), dim=-1)
    p = F.normalize(positive.float(), dim=-1)
    n = F.normalize(negative.float(), dim=-1)
    logits = torch.stack(((q * p).sum(-1), (q * n).sum(-1)), dim=-1) / temperature
    return F.cross_entropy(logits, torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device))
