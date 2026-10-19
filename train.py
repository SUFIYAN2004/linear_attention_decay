
import argparse, json, math, time
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F

p = argparse.ArgumentParser()
p.add_argument("--mixer", required=True, choices=["softmax", "decay", "idecay", "reset"])
p.add_argument("--seed", type=int, default=0)
p.add_argument("--ctx", type=int, default=1024)
p.add_argument("--d", type=int, default=640)
p.add_argument("--layers", type=int, default=8)
p.add_argument("--heads", type=int, default=10)
p.add_argument("--tokens", type=float, default=60e6)
p.add_argument("--batch_tokens", type=int, default=65536)
p.add_argument("--micro", type=int, default=8)
p.add_argument("--lr", type=float, default=6e-4)
p.add_argument("--data", default="/kaggle/working/data")
p.add_argument("--out", default="/kaggle/working/results.jsonl")
a = p.parse_args()
torch.manual_seed(a.seed)
V = int(open(f"{a.data}/vocab.txt").read())

def chunked_linear_attn(q, k, v, loga, C):
    """S_t = g_t*S_{t-1} + k_t v_t^T ; o_t = q_t S_t / (q_t z_t); loga = log g_t <= 0"""
    B, T, H, dh = q.shape; N = T // C
    ve = torch.cat([v, torch.ones_like(v[..., :1])], -1)
    r = lambda t: t.reshape(B, N, C, *t.shape[2:])
    q, k, ve, la = r(q), r(k), r(ve), r(loga)
    b = la.cumsum(2); bh = b.permute(0, 1, 3, 2)
    diff = bh[..., :, None] - bh[..., None, :]
    mask = torch.tril(torch.ones(C, C, dtype=torch.bool, device=q.device))
    D = torch.exp(diff.masked_fill(~mask, float("-inf")))
    intra = torch.einsum("bnhij,bnjhe->bnihe", torch.einsum("bnihd,bnjhd->bnhij", q, k) * D, ve)
    bC = b[:, :, -1]
    kw = k * torch.exp(bC[:, :, None] - b)[..., None]
    kv = torch.einsum("bnjhd,bnjhe->bnhde", kw, ve)
    gC = torch.exp(bC)
    S, starts = torch.zeros_like(kv[:, 0]), []
    for n in range(N):
        starts.append(S); S = S * gC[:, n, :, None, None] + kv[:, n]
    inter = torch.einsum("bnihd,bnhde->bnihe", q * torch.exp(b)[..., None], torch.stack(starts, 1))
    y = (intra + inter).reshape(B, T, H, dh + 1)
    return y[..., :dh] / (y[..., dh:] + 1e-6)

class RMSNorm(nn.Module):
    def __init__(s, d):
        super().__init__(); s.w = nn.Parameter(torch.ones(d))
    def forward(s, x):
        xf = x.float(); return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + 1e-6)) * s.w

class Mixer(nn.Module):
    def __init__(s, d, h, kind, chunk=64):
        super().__init__()
        s.d, s.h, s.dh, s.kind, s.C = d, h, d // h, kind, chunk
        s.qkv = nn.Linear(d, 3 * d, bias=False); s.g = nn.Linear(d, d); s.o = nn.Linear(d, d, bias=False)
        if kind == "decay": s.a = nn.Parameter(torch.zeros(h))
        if kind in ("idecay", "reset"): s.dec = nn.Linear(d, h)
        if kind == "reset": s.r = nn.Parameter(torch.zeros(h))      # learned reset strength per head
    def init_decay(s):
        g0 = torch.linspace(0.9, 0.999, s.h); init = torch.log(g0 / (1 - g0))
        with torch.no_grad():
            if s.kind == "decay": s.a.copy_(init)
            if s.kind in ("idecay", "reset"): s.dec.weight.zero_(); s.dec.bias.copy_(init)
    def forward(s, x, bnd):
        B, T, _ = x.shape
        gate = torch.sigmoid(s.g(x))
        q, k, v = s.qkv(x).view(B, T, 3, s.h, s.dh).unbind(2)
        if s.kind == "softmax":
            y = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=True)
            return s.o(y.transpose(1, 2).reshape(B, T, s.d) * gate)
        with torch.autocast("cuda", enabled=False):
            q, k, v = F.elu(q.float()) + 1, F.elu(k.float()) + 1, v.float()
            if s.kind == "decay":
                loga = F.logsigmoid(s.a).view(1, 1, -1).expand(B, T, -1)
            elif s.kind == "idecay":
                loga = F.logsigmoid(s.dec(x.float()))
            else:   # reset: extra forgetting right after a sentence/line boundary
                loga = F.logsigmoid(s.dec(x.float())) - bnd.float() * F.softplus(s.r)
            s.last_g = loga.detach().exp().mean((0, 1))
            y = chunked_linear_attn(q, k, v, loga, s.C)
        return s.o(y.reshape(B, T, s.d) * gate)

class Block(nn.Module):
    def __init__(s, d, h, kind):
        super().__init__()
        s.n1, s.n2, s.mix = RMSNorm(d), RMSNorm(d), Mixer(d, h, kind)
        s.fc, s.pr = nn.Linear(d, 4 * d), nn.Linear(4 * d, d)
    def forward(s, x, bnd):
        x = x + s.mix(s.n1(x), bnd)
        return x + s.pr(F.gelu(s.fc(s.n2(x))))

class LM(nn.Module):
    def __init__(s, V, d, L, h, ctx, kind):
        super().__init__()
        s.tok, s.pos = nn.Embedding(V, d), nn.Embedding(ctx, d)
        s.blocks = nn.ModuleList(Block(d, h, kind) for _ in range(L)); s.nf = RMSNorm(d)
        bt = torch.zeros(V); bt[torch.from_numpy(np.load(f"{a.data}/boundary.npy"))] = 1.0
        s.register_buffer("bt", bt, persistent=False)
        for m in s.modules():
            if isinstance(m, (nn.Linear, nn.Embedding)):
                nn.init.normal_(m.weight, 0, 0.02)
                if isinstance(m, nn.Linear) and m.bias is not None: nn.init.zeros_(m.bias)
        for n, w in s.named_parameters():
            if n.endswith("mix.o.weight") or n.endswith("pr.weight"): nn.init.normal_(w, 0, 0.02 / math.sqrt(2 * L))
        for m in s.modules():
            if isinstance(m, Mixer): m.init_decay()
    def forward(s, ids):
        x = s.tok(ids) + s.pos(torch.arange(ids.shape[1], device=ids.device))
        b = s.bt[ids]; b = torch.cat([torch.zeros_like(b[:, :1]), b[:, :-1]], 1).unsqueeze(-1)  # reset applies AFTER the boundary token
        for blk in s.blocks: x = blk(x, b)
        return F.linear(s.nf(x), s.tok.weight)

model = LM(V, a.d, a.layers, a.heads, a.ctx, a.mixer).cuda()
nparams = sum(w.numel() for w in model.parameters())
print(f"{a.mixer} seed={a.seed} params={nparams:,}", flush=True)

tr = np.memmap(f"{a.data}/train.bin", dtype=np.uint16, mode="r")
va = np.memmap(f"{a.data}/val.bin", dtype=np.uint16, mode="r")
rng = np.random.default_rng(a.seed)
def batch(n):
    ix = rng.integers(0, len(tr) - a.ctx - 1, n)
    x = np.stack([tr[i:i + a.ctx] for i in ix]).astype(np.int64)
    y = np.stack([tr[i + 1:i + 1 + a.ctx] for i in ix]).astype(np.int64)
    return torch.from_numpy(x).cuda(), torch.from_numpy(y).cuda()
nv = min(256, (len(va) - 1) // a.ctx)
xv = torch.from_numpy(np.stack([va[i * a.ctx:(i + 1) * a.ctx] for i in range(nv)]).astype(np.int64))
yv = torch.from_numpy(np.stack([va[i * a.ctx + 1:(i + 1) * a.ctx + 1] for i in range(nv)]).astype(np.int64))

@torch.no_grad()
def evaluate():
    model.eval(); tot = torch.zeros(a.ctx, device="cuda")
    for i in range(0, nv, 8):
        x, y = xv[i:i + 8].cuda(), yv[i:i + 8].cuda()
        with torch.autocast("cuda", dtype=torch.float16): lg = model(x)
        tot += F.cross_entropy(lg.float().transpose(1, 2), y, reduction="none").sum(0)
    model.train(); return (tot / nv).cpu().numpy()

steps = max(1, int(a.tokens // a.batch_tokens)); accum = a.batch_tokens // (a.micro * a.ctx)
dec = [w for w in model.parameters() if w.ndim >= 2]; nodec = [w for w in model.parameters() if w.ndim < 2]
opt = torch.optim.AdamW([{"params": dec, "weight_decay": 0.1}, {"params": nodec, "weight_decay": 0.0}],
                        lr=a.lr, betas=(0.9, 0.95), fused=True)
scaler = torch.amp.GradScaler("cuda"); warm = max(20, steps // 20)
def lr_at(s):
    if s < warm: return a.lr * (s + 1) / warm
    pr = (s - warm) / max(1, steps - warm)
    return a.lr * (0.1 + 0.45 * (1 + math.cos(math.pi * pr)))

t0, curve, bad = time.time(), [], False
for step in range(steps):
    for g in opt.param_groups: g["lr"] = lr_at(step)
    tl = 0.0
    for _ in range(accum):
        x, y = batch(a.micro)
        with torch.autocast("cuda", dtype=torch.float16): lg = model(x)
        loss = F.cross_entropy(lg.float().view(-1, V), y.view(-1)) / accum
        scaler.scale(loss).backward(); tl += loss.item()
    if not math.isfinite(tl): bad = True; print("NaN loss, stopping", flush=True); break
    scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
    if step % 50 == 0: print(f"step {step}/{steps} loss {tl:.3f} {(time.time()-t0)/60:.1f} min", flush=True)
    if (step + 1) % 200 == 0: curve.append((step + 1, float(evaluate().mean()))); print("  val", curve[-1], flush=True)

per = evaluate()
gam = [m.last_g.cpu().numpy().round(3).tolist() for m in model.modules() if isinstance(m, Mixer) and hasattr(m, "last_g")]
rst = [torch.nn.functional.softplus(m.r).detach().cpu().numpy().round(2).tolist() for m in model.modules()
       if isinstance(m, Mixer) and m.kind == "reset"]
res = dict(mixer=a.mixer, seed=a.seed, params=nparams, steps=steps, nan=bad,
           val_loss=float(per.mean()), ppl=float(np.exp(per.mean())),
           l_early=float(per[:128].mean()), l_mid=float(per[128:512].mean()), l_late=float(per[512:].mean()),
           tok_per_s=steps * a.batch_tokens / (time.time() - t0),
           mem_gb=torch.cuda.max_memory_allocated() / 1e9, curve=curve, decay=gam, reset_strength=rst)
open(a.out, "a").write(json.dumps(res) + "\n"); print(res, flush=True)
