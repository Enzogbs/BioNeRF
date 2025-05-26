import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.distributions import Categorical, Normal, MixtureSameFamily
from torch.quasirandom import SobolEngine

# Hierarchical sampling (section 5.2)
def sample_pdf(bins, weights, N_samples, det=False, pytest=False):
    # Get pdf
    weights = weights + 1e-5 # prevent nans
    pdf = weights / torch.sum(weights, -1, keepdim=True)
    cdf = torch.cumsum(pdf, -1)
    cdf = torch.cat([torch.zeros_like(cdf[...,:1]), cdf], -1)  # (batch, len(bins))

    # Take uniform samples
    if det:
        u = torch.linspace(0., 1., steps=N_samples)
        u = u.expand(list(cdf.shape[:-1]) + [N_samples])
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [N_samples])

    # Pytest, overwrite u with numpy's fixed random numbers
    if pytest:
        np.random.seed(0)
        new_shape = list(cdf.shape[:-1]) + [N_samples]
        if det:
            u = np.linspace(0., 1., N_samples)
            u = np.broadcast_to(u, new_shape)
        else:
            u = np.random.rand(*new_shape)
        u = torch.Tensor(u)

    # Invert CDF
    u = u.contiguous()
    inds = torch.searchsorted(cdf, u, right=True)
    below = torch.max(torch.zeros_like(inds-1), inds-1)
    above = torch.min((cdf.shape[-1]-1) * torch.ones_like(inds), inds)
    inds_g = torch.stack([below, above], -1)  # (batch, N_samples, 2)

    # cdf_g = tf.gather(cdf, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
    # bins_g = tf.gather(bins, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

    denom = (cdf_g[...,1]-cdf_g[...,0])
    denom = torch.where(denom<1e-5, torch.ones_like(denom), denom)
    t = (u-cdf_g[...,0])/denom
    samples = bins_g[...,0] + t * (bins_g[...,1]-bins_g[...,0])

    return samples

def l0_sample_pdf(bins, weights, N_samples, det=False, pytest=False):
    # Maxblur
    weights_pad = torch.cat([weights[..., :1], weights, weights[..., -1:]], axis=-1)
    weights_max = torch.maximum(weights_pad[..., :-1], weights_pad[..., 1:])
    weights = 0.5 * (weights_max[..., :-1] + weights_max[..., 1:])
    weights = weights + 1e-5 # prevent nans

    # Get integral
    integral = (weights[..., 1:] - weights[..., :-1])/(torch.log(weights[..., 1:]/weights[..., :-1]) + 1e-6)

    # Get pdf
    pdf = weights / torch.sum(weights, -1, keepdim=True)
    cdf = torch.cumsum(pdf, -1)
    cdf = torch.cat([torch.zeros_like(cdf[...,:1]), cdf], -1)  # (batch, len(bins))

    # Take uniform samples
    if det:
        u = torch.linspace(0., 1., steps=N_samples)
        u = u.expand(list(cdf.shape[:-1]) + [N_samples])
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [N_samples])

    # Pytest, overwrite u with numpy's fixed random numbers
    if pytest:
        np.random.seed(0)
        new_shape = list(cdf.shape[:-1]) + [N_samples]
        if det:
            u = np.linspace(0., 1., N_samples)
            u = np.broadcast_to(u, new_shape)
        else:
            u = np.random.rand(*new_shape)
        u = torch.Tensor(u)

    # Invert CDF
    u = u.contiguous()
    inds = torch.searchsorted(cdf, u, right=True)
    below = torch.max(torch.zeros_like(inds-1), inds-1)
    above = torch.min((cdf.shape[-1]-1) * torch.ones_like(inds), inds)
    inds_g = torch.stack([below, above], -1)  # (batch, N_samples, 2)

    # cdf_g = tf.gather(cdf, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
    # bins_g = tf.gather(bins, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

    # Find Roots
    residual = u-cdf_g[...,0]
    rhs = residual * torch.sum(integral, dim = -1, keepdim = True)
    weights_g = torch.gather(weights.unsqueeze(1).expand(matched_shape), 2, inds_g)
    denom = torch.log(weights_g[..., 1]/weights_g[..., 0]) + 1e-6
    t = torch.log1p(rhs*denom/weights_g[..., 0]) / denom

    samples = bins_g[...,0] + t * (bins_g[...,1]-bins_g[...,0])

    return samples

def sample_pdf_gmm(bins, weights, N_samples, K=4, det=False):
    """
    Replace for sample_pdf using Gaussian Mixture Model sampling.
    Args:
        bins: [N_rays, n_bins] bin centers (z_vals_mid)
        weights: [N_rays, n_bins] corresponding weights from coarse model
        N_samples: int, number of samples per ray
        K: int, number of GMM components
        det: if True, use deterministic sampling
    Returns:
        z_samples: [N_rays, N_samples] resampled points
    """

    weights = weights + 1e-5  # prevent division by zero
    weights = weights / torch.sum(weights, dim=-1, keepdim=True)

    N_rays, n_bins = bins.shape

    # Use soft k-means to get means per ray
    bin_centers = bins  # [N_rays, n_bins]

    # Create K fixed Gaussian means along the ray (alternatively use learnable means)
    gmm_means = torch.linspace(0., 1., K, device=bins.device).unsqueeze(0).expand(N_rays, K)
    gmm_means = torch.lerp(bins[:, :1], bins[:, -1:], gmm_means)  # scale means to each ray range

    gmm_stds = 0.05 * (bins[:, -1:] - bins[:, :1])  # [N_rays, 1], constant std dev per ray
    gmm_stds = gmm_stds.expand(N_rays, K)

    # Uniform weights for GMM components, or fit based on actual weights (optional)
    cat = Categorical(probs=torch.ones(N_rays, K, device=bins.device) / K)
    comp = Normal(gmm_means, gmm_stds)
    gmm = MixtureSameFamily(cat, comp)

    # Sample
    z_samples = gmm.sample((N_samples,))  # [N_samples, N_rays]
    z_samples = z_samples.transpose(0, 1)  # [N_rays, N_samples]

    if det:
        # Optionally sort for stratification
        z_samples, _ = torch.sort(z_samples, dim=-1)

    return z_samples


def sample_pdf_transport(bins, weights, N_samples, det=False, pytest=False):
    """
    Fast 1D Optimal Transport sampler (NeRF-style).
    Replaces inverse-CDF sampling using a batched linear interpolation method.

    Args:
        bins:      [N_rays, N_bins] - depths of bin centers (z_vals_mid)
        weights:   [N_rays, N_bins] - importance weights from coarse model
        N_samples: int - number of samples to draw
        det:       bool - whether to use deterministic (stratified) sampling
        pytest:    bool - reproducible sampling for unit tests

    Returns:
        z_samples: [N_rays, N_samples] - new sampled depths per ray
    """

    # Normalize weights to get PDF
    weights = weights + 1e-5  # prevent NaNs
    pdf = weights / torch.sum(weights, dim=-1, keepdim=True)

    # Compute the CDF
    cdf = torch.cumsum(pdf, dim=-1)
    cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], dim=-1)  # [N_rays, N_bins + 1]

    # Prepare uniform samples
    N_rays, N_bins = weights.shape

    if det:
        u = torch.linspace(0., 1., steps=N_samples, device=bins.device)
        u = u.expand(N_rays, N_samples)
    else:
        if pytest:
            import numpy as np
            np.random.seed(0)
            u = np.random.rand(N_rays, N_samples)
            u = torch.tensor(u, dtype=torch.float32, device=bins.device)
        else:
            u = torch.rand(N_rays, N_samples, device=bins.device)

    # Invert the CDF
    inds = torch.searchsorted(cdf, u, right=True)

    below = torch.clamp(inds - 1, min=0)
    above = torch.clamp(inds, max=N_bins)

    matched_shape = [N_rays, N_samples]

    cdf_g0 = torch.gather(cdf, 1, below)
    cdf_g1 = torch.gather(cdf, 1, above)

    bins_g0 = torch.gather(bins, 1, below)
    bins_g1 = torch.gather(bins, 1, above)

    denom = cdf_g1 - cdf_g0
    denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
    t = (u - cdf_g0) / denom

    samples = bins_g0 + t * (bins_g1 - bins_g0)

    return samples


def sample_adaptive_kernel_mix_v2(bins, weights, N_samples, det=False, kernel_type="gaussian", pytest=False):
    """
    Adaptive Kernel Mix Sampler with configurable kernels.
    Supported kernels: 'gaussian', 'epanechnikov', 'triangular', 'uniform'
    """
    weights = weights + 1e-5
    weights /= weights.sum(dim=-1, keepdim=True)

    N_rays, N_bins = bins.shape
    bandwidth = (bins[:, 1:] - bins[:, :-1]).mean(dim=-1, keepdim=True)
    bandwidth = bandwidth.clamp(min=1e-5)

    centers = torch.multinomial(weights, N_samples, replacement=True)
    z_c = torch.gather(bins, 1, centers)

    if det:
        eps = torch.linspace(-0.5, 0.5, N_samples, device=bins.device).unsqueeze(0).expand(N_rays, -1)
    else:
        if kernel_type == "gaussian":
            eps = torch.randn(N_rays, N_samples, device=bins.device)
        elif kernel_type == "epanechnikov":
            u = torch.rand(N_rays, N_samples, device=bins.device)
            eps = torch.sqrt(1 - u)
            signs = torch.randint(0, 2, eps.shape, device=bins.device).float() * 2 - 1
            eps *= signs
        elif kernel_type == "triangular":
            u = torch.rand(N_rays, N_samples, device=bins.device)
            eps = torch.sqrt(u)
            signs = torch.randint(0, 2, eps.shape, device=bins.device).float() * 2 - 1
            eps *= signs
        elif kernel_type == "uniform":
            eps = torch.rand(N_rays, N_samples, device=bins.device) * 2 - 1
        else:
            raise ValueError(f"Unsupported kernel_type: {kernel_type}")

    samples = z_c + eps * bandwidth


    return samples.clamp(bins.min(), bins.max())

def sample_adaptive_kernel_mix_v3(
    bins, weights, 
    N_samples_max=64, 
    N_samples_min=8, 
    det=False, 
    kernel_type="gaussian", 
    uncertainty_threshold=0.01,
    uncertainty_type="variance" # could be variance
):
    """
    Fast adaptive sampler with:
    - Entropy-based sample skipping
    - Entropy-based adaptive N_samples per ray
    - Kernel-based perturbation (Gaussian, Epanechnikov, etc.)
    """
    device = bins.device
    eps = 1e-5

    # Normalize weights
    weights = weights + eps
    weights = weights / weights.sum(dim=-1, keepdim=True)  # [N_rays, N_bins]
    N_rays, N_bins = weights.shape

    if uncertainty_type == "entropy":
        uncertainty = -(weights * weights.clamp(min=1e-6).log()).sum(dim=-1)  # [N_rays]
    elif uncertainty_type == "variance":
        idxs = torch.arange(N_bins, device=device).float().unsqueeze(0)  # [1, N_bins]
        mean = (weights * idxs).sum(dim=-1)
        mean2 = (weights * idxs**2).sum(dim=-1)
        uncertainty = mean2 - mean**2  # [N_rays]
    else:
        raise ValueError(f"Unknown uncertainty_type: {uncertainty_type}")

    uncertainty_norm = uncertainty / (uncertainty.max() + eps)

    # Sample skipping mask (binary)
    skip_mask = (uncertainty_norm < uncertainty_threshold)  # [N_rays]

    # Adaptive N_samples per ray
    N_samples_per_ray = (
        N_samples_min + uncertainty_norm * (N_samples_max - N_samples_min)
    ).clamp(min=N_samples_min).round().long()  # [N_rays]

    # For batch efficiency, we take max(N_samples_per_ray) and mask the rest
    N_samples = N_samples_per_ray.max().item()  # scalar
    bandwidth = (bins[:, 1:] - bins[:, :-1]).mean(dim=-1, keepdim=True).clamp(min=1e-5)

    # Sample centers from weights
    centers = torch.multinomial(weights, N_samples, replacement=True)  # [N_rays, N_samples]
    z_c = torch.gather(bins, 1, centers)

    # Generate perturbations
    if det:
        eps_samples = torch.linspace(-0.5, 0.5, N_samples, device=device).unsqueeze(0).expand(N_rays, -1)
    else:
        if kernel_type == "gaussian":
            eps_samples = torch.randn(N_rays, N_samples, device=device)
        elif kernel_type == "epanechnikov":
            u = torch.rand(N_rays, N_samples, device=device)
            eps_samples = torch.sqrt(1 - u)
            signs = torch.randint(0, 2, u.shape, device=device).float() * 2 - 1
            eps_samples *= signs
        elif kernel_type == "triangular":
            u = torch.rand(N_rays, N_samples, device=device)
            eps_samples = torch.sqrt(u)
            signs = torch.randint(0, 2, u.shape, device=device).float() * 2 - 1
            eps_samples *= signs
        elif kernel_type == "uniform":
            eps_samples = torch.rand(N_rays, N_samples, device=device) * 2 - 1
        else:
            raise ValueError(f"Unknown kernel_type: {kernel_type}")

    # Apply bandwidth scaling
    samples = z_c + eps_samples * bandwidth

    # Mask samples based on adaptive N_samples
    ray_ids = torch.arange(N_rays, device=device).unsqueeze(-1)
    sample_ids = torch.arange(N_samples, device=device).unsqueeze(0)
    mask = sample_ids < N_samples_per_ray.unsqueeze(1)  # [N_rays, N_samples]

    # Apply skipping mask
    samples[~mask] = 0.0  # Zero out unused samples
    samples[skip_mask] = 0.0  # Zero out skipped rays

    return samples, N_samples_per_ray, skip_mask


import torch
import torch.nn as nn
import torch.nn.functional as F

class BinScorerMLP(nn.Module):
    def __init__(self, num_bins):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_bins, 64),
            nn.ReLU(),
            nn.Linear(64, num_bins)
        )

    def forward(self, weights):
        return self.net(weights)  # Output: importance scores (logits)


def sample_hybrid_akm_ultra(bins, weights, N_samples, scorer_model=None, det=True, device=None):
    """
    Ultra-fast adaptive kernel-based sampler with learned bin importance.
    Args:
        bins:      [N_rays, N_bins]
        weights:   [N_rays, N_bins]
        N_samples: int
        scorer_model: optional nn.Module to predict bin importance
        det: deterministic offsets (vs random)
        device: for internal tensor creation

    Returns:
        samples: [N_rays, N_samples]
    """
    device = device or bins.device
    N_rays, N_bins = bins.shape

    # Normalize weights
    weights = weights + 1e-5
    weights = weights / torch.sum(weights, dim=-1, keepdim=True)

    # Predict importance scores (optionally learned)
    if scorer_model is not None:
        with torch.no_grad():
            scores = scorer_model(weights)  # [N_rays, N_bins]
    else:
        scores = weights

    # Get top-K bins (faster than full sort)
    k = min(N_bins, N_samples)
    _, topk_inds = torch.topk(scores, k=k, dim=-1)  # [N_rays, k]
    topk_bins = torch.gather(bins, 1, topk_inds)    # [N_rays, k]

    # Expand to N_samples
    repeat = (N_samples + k - 1) // k
    centers = topk_bins.repeat(1, repeat)[:, :N_samples]  # [N_rays, N_samples]

    # Estimate shared std
    avg_spacing = (bins[:, 1:] - bins[:, :-1]).mean(dim=-1, keepdim=True)
    stds = avg_spacing.expand(N_rays, N_samples)

    # Deterministic stratified sampling offsets
    if det:
        offsets = torch.linspace(-0.5, 0.5, N_samples, device=device).unsqueeze(0)
        noise = offsets.expand(N_rays, -1)
    else:
        noise = torch.randn(N_rays, N_samples, device=device)

    samples = centers + noise * stds

    # Clamp to valid depth range
    samples = samples.clamp(min=bins.min(), max=bins.max())
    return samples


def kernel_tilted_sample(bins, weights, N_samples, kernel='gaussian', det=False, sigma=0.1):
    """
    K-TOSS: Kernel-Tilted Ray Sampling without CDFs.
    
    Args:
        bins (Tensor): Bins representing the range for sampling.
        weights (Tensor): Importance weights for each bin.
        N_samples (int): Number of samples to generate.
        kernel (str): The type of kernel ('gaussian' or 'uniform').
        det (bool): If True, samples deterministically (uniform).
        sigma (float): Standard deviation for Gaussian kernel (if kernel='gaussian').
        
    Returns:
        samples (Tensor): Sampled ray positions.
    """
    weights = weights + 1e-5  # Prevent NaNs

    # Normalize weights (kernel adjustment)
    weights = weights / torch.sum(weights, -1, keepdim=True)
    
    # Apply kernel function
    if kernel == 'gaussian':
        # Example: Gaussian kernel to weight bins (you can replace this with any other kernel)
        kernel_weights = torch.exp(-0.5 * (weights - 0.5)**2 / sigma**2)  # Adjust sigma for sharpness
        kernel_weights = kernel_weights / torch.sum(kernel_weights, -1, keepdim=True)  # Normalize

    elif kernel == 'uniform':
        kernel_weights = torch.ones_like(weights)  # Uniform kernel: equal weight for all bins
    
    # Create sample indices based on kernel weights
    pdf = kernel_weights / torch.sum(kernel_weights, -1, keepdim=True)  # Normalized pdf from kernel
    cdf = torch.cumsum(pdf, -1)  # CDF from kernel weights
    
    # Take uniform samples in [0, 1] space
    if det:
        u = torch.linspace(0., 1., steps=N_samples)
        u = u.expand(list(cdf.shape[:-1]) + [N_samples])
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [N_samples])

    # Invert CDF using the kernel's CDF (equivalent to inverse transform sampling)
    u = u.contiguous()
    inds = torch.searchsorted(cdf, u, right=True)
    below = torch.max(torch.zeros_like(inds-1), inds-1)
    above = torch.min((cdf.shape[-1]-1) * torch.ones_like(inds), inds)
    inds_g = torch.stack([below, above], -1)  # (batch, N_samples, 2)

    # Gather CDF and bins based on indices
    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

    denom = (cdf_g[...,1]-cdf_g[...,0])
    denom = torch.where(denom<1e-5, torch.ones_like(denom), denom)
    t = (u - cdf_g[..., 0]) / denom
    samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])

    return samples

def sample_wavelet_hierarchical(bins, weights, N_samples, det=False, levels=3):
    """
    Wavelet-based hierarchical sampling for multi-scale importance
    
    Args:
        bins: [N_rays, N_bins] depth bins
        weights: [N_rays, N_bins] importance weights
        N_samples: number of samples to generate
        det: deterministic sampling
        levels: number of wavelet decomposition levels
    
    Returns:
        samples: [N_rays, N_samples] sampled depths
    """
    device = bins.device
    N_rays, N_bins = bins.shape
    
    # Normalize weights
    weights = weights + 1e-5
    weights = weights / torch.sum(weights, dim=-1, keepdim=True)
    
    # Multi-resolution analysis using simple averaging (Haar-like)
    scales = []
    current_weights = weights
    
    for level in range(levels):
        if current_weights.shape[1] < 2:
            break
            
        # Downsample by averaging pairs
        if current_weights.shape[1] % 2 == 1:
            current_weights = current_weights[:, :-1]  # Remove last element if odd
            
        downsampled = (current_weights[:, ::2] + current_weights[:, 1::2]) / 2
        scales.append(downsampled)
        current_weights = downsampled
    
    # Allocate samples across scales
    samples_per_scale = [N_samples // (2**i) for i in range(len(scales))]
    samples_per_scale[0] += N_samples - sum(samples_per_scale)  # Remainder to finest scale
    
    all_samples = []
    
    for i, (scale_weights, n_samples) in enumerate(zip(scales, samples_per_scale)):
        if n_samples == 0:
            continue
            
        # Create bins for current scale
        scale_factor = 2**i
        scale_bins = bins[:, ::scale_factor]
        
        if scale_bins.shape[1] < scale_weights.shape[1]:
            scale_weights = scale_weights[:, :scale_bins.shape[1]]
        
        # Sample from current scale
        cdf = torch.cumsum(scale_weights, dim=-1)
        cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], dim=-1)
        
        if det:
            u = torch.linspace(0., 1., steps=n_samples, device=device)
            u = u.expand(N_rays, n_samples)
        else:
            u = torch.rand(N_rays, n_samples, device=device)
        
        # Invert CDF
        inds = torch.searchsorted(cdf, u, right=True)
        below = torch.clamp(inds - 1, min=0)
        above = torch.clamp(inds, max=cdf.shape[-1] - 1)
        
        bins_g0 = torch.gather(scale_bins, 1, below)
        bins_g1 = torch.gather(scale_bins, 1, above)
        cdf_g0 = torch.gather(cdf, 1, below)
        cdf_g1 = torch.gather(cdf, 1, above)
        
        denom = cdf_g1 - cdf_g0
        denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
        t = (u - cdf_g0) / denom
        
        scale_samples = bins_g0 + t * (bins_g1 - bins_g0)
        all_samples.append(scale_samples)
    
    # Combine all samples
    if all_samples:
        combined_samples = torch.cat(all_samples, dim=1)
        # Sort to maintain order
        combined_samples, _ = torch.sort(combined_samples, dim=1)
        return combined_samples
    else:
        return torch.zeros(N_rays, N_samples, device=device)


# Benchmark comparison function
def benchmark_samplers(bins, weights, N_samples, n_trials=10):
    """
    Benchmark different sampling methods
    """
    import time
    
    samplers = {
        'original': lambda: sample_pdf_fast(bins, weights, N_samples),
        'wavelet': lambda: sample_wavelet_hierarchical(bins, weights, N_samples),
        'block_sparse': lambda: sample_block_sparse_transport(bins, weights, N_samples),
        'entropy_adaptive': lambda: sample_entropy_adaptive(bins, weights, N_samples)[0],
        'fourier': lambda: sample_fourier_importance(bins, weights, N_samples),
    }
    
    results = {}
    
    for name, sampler in samplers.items():
        times = []
        for _ in range(n_trials):
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            start = time.time()
            
            _ = sampler()
            
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            end = time.time()
            times.append(end - start)
        
        results[name] = {
            'mean_time': np.mean(times),
            'std_time': np.std(times),
            'min_time': np.min(times)
        }
    
    return results

import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import numpy as np

def visualize_sampler_dict(z_vals_mid, weights, samples_dict, ray_idx=0, title=None, resolution=512):
    """
    Visualize coarse PDF and smooth KDEs of sample distributions.

    Args:
        z_vals_mid: (B, N_bins)
        weights: (B, N_bins)
        samples_dict: {sampler_name: samples_tensor (N_samples,)}
        ray_idx: which ray to visualize
        title: optional plot title
        resolution: number of points in KDE line plot
    """
    z_vals = z_vals_mid[ray_idx].detach().cpu().numpy()
    weights_ray = weights[ray_idx].detach().cpu().numpy()
    
    # Normalize weights to PDF
    pdf = weights_ray + 1e-5
    pdf = pdf / pdf.sum()

    # Plot coarse PDF (scaled)
    plt.figure(figsize=(10, 5))
    plt.plot(z_vals, pdf * 5, label="Coarse PDF ×5", color='blue', linewidth=2)

    # Prepare x-axis range
    x_min, x_max = z_vals.min(), z_vals.max()
    x_grid = np.linspace(x_min, x_max, resolution)

    # Plot KDE for each sampler
    for name, samples in samples_dict.items():
        samples_np = samples.detach().cpu().numpy()

        if len(samples_np) > 1:
            kde = gaussian_kde(samples_np.reshape(1, -1), bw_method="scott")
            kde_vals = kde(x_grid)
            plt.plot(x_grid, kde_vals, label=name)

    plt.xlabel("Depth / Distance along Ray")
    plt.ylabel("Density (KDE)")
    plt.title(title or f"Ray {ray_idx}: Coarse PDF vs KDEs of Samplers")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

import matplotlib.pyplot as plt
import numpy as np

def visualize_coarse_pdf_and_sampler_cdfs(z_vals_mid, weights, samples_dict, ray_idx=0, title=None):
    """
    Plot coarse PDF (from z_vals_mid and weights) and CDFs of sampled points from multiple samplers.

    Args:
        z_vals_mid: (B, N_bins) tensor - coarse bin midpoints
        weights: (B, N_bins) tensor - coarse weights (unnormalized)
        samples_dict: dict of {sampler_name: samples_tensor (B, N_samples)}
        ray_idx: which ray to visualize
        title: optional plot title
    """

    z_vals_mid_np = z_vals_mid[ray_idx].detach().cpu().numpy()
    weights_np = weights[ray_idx].detach().cpu().numpy()

    # Normalize coarse weights to get PDF
    coarse_pdf = weights_np + 1e-5
    coarse_pdf /= coarse_pdf.sum()

    plt.figure(figsize=(10, 6))

    # Plot coarse PDF
    plt.plot(z_vals_mid_np, coarse_pdf, label="Coarse PDF", color='blue', linewidth=2)

    # Plot CDF for each sampler
    for name, samples in samples_dict.items():
        samples_np = samples[ray_idx].detach().cpu().numpy()
        # Sort samples for CDF plot
        samples_sorted = np.sort(samples_np)
        cdf_vals = np.arange(1, len(samples_sorted) + 1) / len(samples_sorted)

        plt.plot(samples_sorted, cdf_vals, label=f"{name} CDF", linestyle='--', linewidth=2)

    plt.xlabel("Depth / Distance along Ray")
    plt.ylabel("Density / CDF")
    plt.title(title or f"Ray {ray_idx}: Coarse PDF and Sampler CDFs")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def visualize_all_pdfs(z_vals_mid, weights, samples_dict, ray_idx=0, title=None, bins=64):
    """
    Plot coarse PDF and PDFs of each sampler's samples for a given ray.

    Args:
        z_vals_mid: (B, N_bins) tensor - midpoints of coarse bins
        weights: (B, N_bins) tensor - coarse weights (unnormalized)
        samples_dict: dict of {sampler_name: samples_tensor (B, N_samples)}
        ray_idx: which ray to visualize
        title: optional plot title
        bins: number of bins for histograms
    """
    z_vals_mid_np = z_vals_mid[ray_idx].detach().cpu().numpy()
    weights_np = weights[ray_idx].detach().cpu().numpy()

    # Normalize coarse weights to get PDF
    coarse_pdf = weights_np + 1e-5
    coarse_pdf /= coarse_pdf.sum()

    plt.figure(figsize=(10, 6))

    # Plot coarse PDF (line plot)
    plt.plot(z_vals_mid_np, coarse_pdf, label="Coarse PDF", color='blue', linewidth=2)

    # Plot PDF for each sampler (KDE if possible, else histogram)
    for name, samples in samples_dict.items():
        samples_np = samples[ray_idx].detach().cpu().numpy()

        if len(samples_np) > 1:
            try:
                # KDE for smooth PDF estimate
                kde = gaussian_kde(samples_np, bw_method='scott')
                x_grid = np.linspace(samples_np.min(), samples_np.max(), 200)
                density = kde(x_grid)
                density /= density.sum()  # Normalize for comparison
                plt.plot(x_grid, density, label=f"{name} KDE", linestyle='--', linewidth=2)
            except Exception as e:
                # If KDE fails, fallback to histogram
                hist, bin_edges = np.histogram(samples_np, bins=bins, density=True)
                bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
                plt.plot(bin_centers, hist, label=f"{name} Hist", linestyle='--')
        else:
            # Too few samples for KDE or histogram, plot samples as points on x-axis
            plt.scatter(samples_np, np.zeros_like(samples_np), label=f"{name} samples", marker='x')

    plt.xlabel("Depth / Distance along Ray")
    plt.ylabel("PDF")
    plt.title(title or f"Ray {ray_idx}: Coarse and Sampled PDFs")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_coarse_pdf_and_strip(z_vals_mid, weights, samples_dict, ray_idx=0, title=None):
    """
    Plot coarse PDF as line plot on top,
    and samples from multiple samplers as strip plots below.
    
    Args:
        z_vals_mid: (B, N_bins) torch tensor of bin midpoints
        weights: (B, N_bins) torch tensor of weights (unnormalized)
        samples_dict: dict of {sampler_name: samples_tensor (B, N_samples)}
        ray_idx: int, index of ray to plot
        title: str, optional title for the figure
    """
    # Prepare coarse PDF data
    z_vals = z_vals_mid[ray_idx].detach().cpu().numpy()
    w = weights[ray_idx].detach().cpu().numpy()
    pdf = w + 1e-5
    pdf /= pdf.sum()

    # Setup figure with two subplots: PDF on top, strip plot below
    fig, (ax_pdf, ax_strip) = plt.subplots(2, 1, figsize=(12, 6), gridspec_kw={'height_ratios': [2, 1]}, sharex=True)
    
    # Plot coarse PDF on top
    ax_pdf.plot(z_vals, pdf, color='blue', lw=2, label='Coarse PDF (weights)')
    ax_pdf.set_ylabel("PDF")
    ax_pdf.set_title(title or f"Ray {ray_idx}: Coarse PDF and Samples")
    ax_pdf.grid(True)
    ax_pdf.legend()

    # Prepare strip plot
    for i, (name, samples) in enumerate(samples_dict.items()):
        samples_np = samples[ray_idx].detach().cpu().numpy()
        y = np.ones_like(samples_np) * i
        ax_strip.scatter(samples_np, y, label=name, alpha=0.6, edgecolors='k', s=50)

    # Strip plot formatting
    ax_strip.set_yticks(range(len(samples_dict)))
    ax_strip.set_yticklabels(list(samples_dict.keys()))
    ax_strip.set_xlabel("Depth / Distance along Ray")
    ax_strip.set_ylabel("Sampler")
    ax_strip.grid(True)
    ax_strip.legend(loc='upper right')

    plt.tight_layout()
    plt.show()