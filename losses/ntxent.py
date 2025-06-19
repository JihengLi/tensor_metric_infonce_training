import torch
import torch.nn.functional as F


def nt_xent_loss(q, k, temperature=0.1):
    B = q.size(0)
    q = F.normalize(q, dim=1)
    k = F.normalize(k, dim=1)

    z = torch.cat([q, k], dim=0)
    sim = torch.mm(z, z.t()) / temperature

    diag = torch.eye(2 * B, device=sim.device, dtype=torch.bool)
    sim.masked_fill_(diag, float("-inf"))

    pos_idx = (torch.arange(2 * B, device=sim.device) + B) % (2 * B)
    pos_sim = sim[torch.arange(2 * B), pos_idx]

    loss = -(pos_sim - torch.logsumexp(sim, dim=1))
    return loss.mean()
