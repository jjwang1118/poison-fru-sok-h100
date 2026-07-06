"""CPU unit test for the PSMU craft ported into CFRU.

Builds a NeuMF-MLP identical in STRUCTURE to external/CFRU/benchmark/movielens1m/
model/NCF.py, runs the exact _psmu_craft_target algorithm, and measures the
target item's exposure (ER@20-style) over the model's REAL user embeddings before
vs after the craft. If the port works, ER on real users must jump well above 0
even though the craft only ever saw SYNTHETIC users -- that is the PSMU premise
and the thing the old data-injection attack failed to do.
"""
import torch, torch.nn as nn

torch.manual_seed(0)

class NeuMF(nn.Module):
    """Same forward as NCF.py: concat(user,item) -> MLP_layers -> predict_layer (logit)."""
    def __init__(self, user_num, item_num, factor_num=16, num_layers=2):
        super().__init__()
        emb = factor_num * (2 ** (num_layers - 1))
        self.embed_user = nn.Embedding(user_num, emb)
        self.embed_item = nn.Embedding(item_num, emb)
        mods = []
        for i in range(num_layers):
            insz = factor_num * (2 ** (num_layers - i))
            mods += [nn.Dropout(0.0), nn.Linear(insz, insz // 2), nn.ReLU()]
        self.MLP_layers = nn.Sequential(*mods)
        self.predict_layer = nn.Linear(factor_num, 1)
        nn.init.normal_(self.embed_user.weight, std=0.01)
        nn.init.normal_(self.embed_item.weight, std=0.01)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight); m.bias.data.zero_()

    def get_score(self, data):
        u = self.embed_user(data[0]); it = self.embed_item(data[1])
        return self.predict_layer(self.MLP_layers(torch.cat((u, it), -1))).view(-1)


# ---- EXACT craft algorithm copied from CFRU.Client._psmu_craft_target ----------
def psmu_craft_target(model, target_item, option):
    opt = option
    s          = int(opt.get('psmu_s', 8))
    user_steps = int(opt.get('psmu_user_steps', 30))
    item_steps = int(opt.get('psmu_item_steps', 20))
    lr         = float(opt.get('psmu_lr', 0.1))
    comp_k     = int(opt.get('psmu_comp_k', 40))
    std        = float(opt.get('psmu_std', 0.1))
    scale      = float(opt.get('psmu_scale', 5.0))
    device = model.embed_item.weight.device
    num_item, dim = model.embed_item.weight.shape
    MLP, PRED = model.MLP_layers, model.predict_layer
    saved_rg = [p.requires_grad for p in model.parameters()]
    for p in model.parameters(): p.requires_grad_(False)
    item_w = model.embed_item.weight.data
    server_target = item_w[target_item].clone()
    def score(u_emb, item_embs):
        u = u_emb.unsqueeze(0).expand(item_embs.shape[0], -1)
        return PRED(MLP(torch.cat([u, item_embs], dim=-1))).view(-1)
    with torch.no_grad():
        dist = torch.norm(item_w - server_target.unsqueeze(0), dim=1)
        dist[target_item] = float('inf')
        comp_idx = torch.topk(dist, k=min(comp_k, num_item - 1), largest=False).indices
        comp_embs = item_w[comp_idx].clone()
    target_emb = server_target.clone().requires_grad_(True)
    for _ in range(s):
        cluster = torch.randint(0, num_item, (comp_k,), device=device)
        neg_idx = torch.randint(0, num_item, (comp_k * 4,), device=device)
        pos_embs, neg_embs = item_w[cluster].clone(), item_w[neg_idx].clone()
        labels = torch.cat([torch.ones(pos_embs.shape[0], device=device),
                            torch.zeros(neg_embs.shape[0], device=device)])
        u = (torch.randn(dim, device=device) * std).requires_grad_(True)
        for _ in range(user_steps):
            logits = score(u, torch.cat([pos_embs, neg_embs], dim=0))
            loss_u = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels)
            gu, = torch.autograd.grad(loss_u, u)
            u = (u - lr * gu).detach().requires_grad_(True)
        u = u.detach()
        for _ in range(item_steps):
            pos = score(u, target_emb.unsqueeze(0))
            neg = score(u, comp_embs)
            loss = -torch.nn.functional.logsigmoid(pos - neg).mean()
            gt, = torch.autograd.grad(loss, target_emb)
            target_emb = (target_emb - lr * gt).detach().requires_grad_(True)
    new_target = server_target + scale * (target_emb.detach() - server_target)
    model.embed_item.weight.data[target_item] = new_target.to(item_w.dtype)
    for p, rg in zip(model.parameters(), saved_rg): p.requires_grad_(rg)
    return float(torch.norm(new_target - server_target))


def er_at_k(model, target, k=20, n_users=None):
    """Fraction of real users whose top-k over ALL items contains the target."""
    nu = model.embed_user.weight.shape[0] if n_users is None else n_users
    ni = model.embed_item.weight.shape[0]
    items = torch.arange(ni)
    hit = 0
    with torch.no_grad():
        for uidx in range(nu):
            u = torch.full((ni,), uidx).long()
            scores = model.get_score([u, items])
            topk = torch.topk(scores, k).indices.tolist()
            hit += int(target in topk)
    return hit / nu


def mean_percentile(model, target):
    """Average rank-percentile of the target over real users (1.0 = always rank-1)."""
    ni = model.embed_item.weight.shape[0]
    items = torch.arange(ni)
    ps = []
    with torch.no_grad():
        for uidx in range(model.embed_user.weight.shape[0]):
            u = torch.full((ni,), uidx).long()
            scores = model.get_score([u, items])
            rank = (scores > scores[target]).sum().item()  # 0 = best
            ps.append(1.0 - rank / (ni - 1))
    return sum(ps) / len(ps)


if __name__ == "__main__":
    USER, ITEM = 120, 400
    model = NeuMF(USER, ITEM)
    # lightly train so real users have structure (a few random positives each)
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    for _ in range(150):
        u = torch.randint(0, USER, (256,)); pos = torch.randint(0, ITEM, (256,))
        neg = torch.randint(0, ITEM, (256,))
        s_pos = model.get_score([u, pos]); s_neg = model.get_score([u, neg])
        loss = -torch.nn.functional.logsigmoid(s_pos - s_neg).mean()
        opt.zero_grad(); loss.backward(); opt.step()

    target = 377  # arbitrary cold target
    k = 20
    er0 = er_at_k(model, target, k); pc0 = mean_percentile(model, target)
    print(f"BEFORE craft:  ER@{k}={er0:.3f}  mean_percentile={pc0:.3f}")

    for scale in (1.0, 5.0, 10.0):
        m = NeuMF(USER, ITEM); m.load_state_dict(model.state_dict())  # fresh copy each scale
        shift = psmu_craft_target(m, target, {'psmu_scale': scale, 'psmu_s': 8})
        er1 = er_at_k(m, target, k); pc1 = mean_percentile(m, target)
        print(f"scale={scale:>4}:  |shift|={shift:6.3f}  ER@{k}={er1:.3f}  "
              f"mean_percentile={pc1:.3f}   (was ER={er0:.3f})")

    print("\nPASS if ER@K rises from ~0 toward 1.0 as scale increases "
          "(craft on SYNTHETIC users generalizes to REAL users -> hits the ER path).")
