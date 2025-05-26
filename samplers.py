import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.distributions import Categorical, Normal, MixtureSameFamily
from torch.quasirandom import SobolEngine
import matplotlib.pyplot as plt

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
