from math import pi, log
from functools import wraps

import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, repeat

__all__ = ['PerceiverIO']

# helpers

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def cache_fn(f):
    cache = None
    @wraps(f)
    def cached_fn(*args, _cache = True, **kwargs):
        if not _cache:
            return f(*args, **kwargs)
        nonlocal cache
        if cache is not None:
            return cache
        cache = f(*args, **kwargs)
        return cache
    return cached_fn

def fourier_encode(x, max_freq, num_bands = 4, base = 2):
    x = x.unsqueeze(-1)
    device, dtype, orig_x = x.device, x.dtype, x

    scales = torch.logspace(0., log(max_freq / 2) / log(base), num_bands, base = base, device = device, dtype = dtype)
    scales = scales[(*((None,) * (len(x.shape) - 1)), Ellipsis)]

    x = x * scales * pi
    x = torch.cat([x.sin(), x.cos()], dim=-1)
    x = torch.cat((x, orig_x), dim = -1)
    return x

# helper classes

class PreNorm(nn.Module):
    def __init__(self, dim, fn, context_dim = None):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)
        self.norm_context = nn.LayerNorm(context_dim) if exists(context_dim) else None

    def forward(self, x, **kwargs):
        x = self.norm(x)

        if exists(self.norm_context):
            context = kwargs['context']
            normed_context = self.norm_context(context)
            kwargs.update(context = normed_context)

        return self.fn(x, **kwargs)

class GEGLU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim = -1)
        return x * F.gelu(gates)

class FeedForward(nn.Module):
    def __init__(self, dim, mult = 4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult * 2),
            GEGLU(),
            nn.Linear(dim * mult, dim)
        )

    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, query_dim, context_dim = None, heads = 8, dim_head = 64):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)
        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(context_dim, inner_dim * 2, bias = False)
        self.record = nn.Identity()
        self.to_out = nn.Linear(inner_dim, query_dim)

    def forward(self, x, context = None, mask = None):
        h = self.heads

        q = self.to_q(x)
        context = default(context, x)
        k, v = self.to_kv(context).chunk(2, dim = -1)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h = h), (q, k, v))

        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale

        if exists(mask):
            mask = rearrange(mask, 'b ... -> b (...)')
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b j -> (b h) () j', h = h)
            sim.masked_fill_(~mask, max_neg_value)

        # attention, what we cannot get enough of
        attn = sim.softmax(dim = -1)
        attn = self.record(attn)
        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h = h)
        return self.to_out(out)

# main class

class PerceiverIO(nn.Module):
    def __init__(
        self,
        *,
        depth,
        dim,
        domains={'naturalimages': {'num_latents':128, 'fourier_encode_data':True, 'num_freq_bands':6,'max_freq':10.0,'freq_base':2,'input_channels':3,'input_axis':2}},
        tasks={'naturalimage_classification': {'num_queries':128, 'queries_dim':32}},
        datasets={'cifar10': {'num_classes':10, 'task':'naturalimage_classification'}},
        contexts = [('naturalimages','naturalimage_classification','cifar10')],
        latent_dim = 256,
        cross_heads = 1,
        latent_heads = 8,
        cross_dim_head = 64,
        latent_dim_head = 64,
        weight_tie_layers = False,
        self_per_cross_attn = 1
    ):
        super().__init__()
        self.domains = domains
        self.tasks = tasks
        self.datasets = datasets
        self.contexts = [('naturalimages','naturalimage_classification','cifar10')]

        self.latents = {key: nn.Parameter(torch.randn((dim['num_latents'], latent_dim))) for key, dim in domains.items()}
        self.queries = {key: nn.Parameter(torch.randn((dim['num_queries'], dim['queries_dim']))) for key, dim in tasks.items()}
        
        self.decoder_cross_attn = {key: PreNorm(dim['queries_dim'], Attention(dim['queries_dim'], latent_dim, heads = cross_heads, dim_head = cross_dim_head), context_dim = latent_dim) for key, dim in tasks.items()}
        self.to_logits = {key: (nn.Linear(tasks[val['task']]['queries_dim'], val['num_classes']) if val['num_classes'] > 0 else nn.Identity()) for key, val in datasets.items()}


        get_cross_attn = lambda: PreNorm(latent_dim, Attention(latent_dim, dim, heads = cross_heads, dim_head = cross_dim_head), context_dim = dim)
        get_cross_ff = lambda: PreNorm(latent_dim, FeedForward(latent_dim))
        get_latent_attn = lambda: PreNorm(latent_dim, Attention(latent_dim, heads = latent_heads, dim_head = latent_dim_head))
        get_latent_ff = lambda: PreNorm(latent_dim, FeedForward(latent_dim))

        get_cross_attn, get_cross_ff, get_latent_attn, get_latent_ff = map(cache_fn, (get_cross_attn, get_cross_ff, get_latent_attn, get_latent_ff))

        self.layers = nn.ModuleList([])
        for i in range(depth):
            should_cache = i > 0 and weight_tie_layers
            cache_args = {'_cache': should_cache}

            self_attns = nn.ModuleList([])

            for _ in range(self_per_cross_attn):
                self_attns.append(nn.ModuleList([
                    get_latent_attn(**cache_args),
                    get_latent_ff(**cache_args)
                ]))

            self.layers.append(nn.ModuleList([
                get_cross_attn(**cache_args),
                get_cross_ff(**cache_args),
                self_attns
            ]))



    def forward(
        self,
        inputs,
    ):
        outputs = []
        if not isinstance(inputs, list): 
            inputs = [inputs]

        for i, (domain, task, dataset) in enumerate(self.contexts):
            data = inputs[i]
            b, *_, device = *data.shape, data.device
            
            fourier_encode_data = self.domains[domain]['fourier_encode_data']
            input_axis = self.domains[domain]['input_axis']
            num_freq_bands = self.domains[domain]['num_freq_bands']
            max_freq = self.domains[domain]['max_freq']
            freq_base = self.domains[domain]['freq_base']
            
            if fourier_encode_data: # TODO Instead of this, do patch embedding and have a max seq length like in the LM implemtantation https://github.com/lucidrains/perceiver-pytorch/blob/3b70ebee00c66f15b38c5980f4275f744a433895/perceiver_pytorch/perceiver_io.py#L187
                data  = rearrange(data, 'b c w h -> b w h c')
                b, *axis, _, device = *data.shape, data.device
                fourier_channels = (input_axis * ((num_freq_bands * 2) + 1)) if fourier_encode_data else 0

                axis_pos = list(map(lambda size: torch.linspace(-1., 1., steps = size, device = device), axis))
                pos = torch.stack(torch.meshgrid(*axis_pos), dim = -1)
                enc_pos = fourier_encode(pos, max_freq, num_freq_bands, base = freq_base)
                enc_pos = rearrange(enc_pos, '... n d -> ... (n d)')
                enc_pos = repeat(enc_pos, '... -> b ...', b = b)

                data = torch.cat((data, enc_pos), dim = -1)
                # concat to channels of data and flatten axis

                data = rearrange(data, 'b ... d -> b (...) d')


            latents = self.latents[domain].to(device)
            queries = self.queries[task].to(device)

            decoder_cross_attn = self.decoder_cross_attn[task].to(device)
            to_logits = self.to_logits[dataset].to(device)

            l = repeat(latents, 'n d -> b n d', b = b)
            q = repeat(queries, 'n d -> b n d', b = b)

            # layers
            for cross_attn, cross_ff, self_attns in self.layers:
                l = cross_attn(l, context = data) + l
                l = cross_ff(l) + l

                for self_attn, self_ff in self_attns:
                    l = self_attn(l) + l
                    l = self_ff(l) + l

            # cross attend from decoder queries to latents
            decoded = decoder_cross_attn(q, context = l)

            # Mean the attention over the decoded data
            if fourier_encode_data:
                decoded = decoded.mean(-2)

            # final linear out
            outputs.append(to_logits(decoded))

        if len(outputs) == 1:
            return outputs[0]

        return outputs 