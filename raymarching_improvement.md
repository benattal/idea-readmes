# Path Replay Backpropagation for Radiance Field Rendering

This document explains how gradients with respect to density ($\sigma$) and emitted color ($c$) are efficiently computed at each sample point along a ray using **path replay backpropagation**. This technique enables memory-efficient differentiable volume rendering for NeRF-style radiance field reconstruction.

## Overview

Path replay backpropagation, introduced by [Vicini et al. (2021)](https://rgl.epfl.ch/publications/Vicini2021PathReplay), allows us to compute gradients with **constant memory** and **linear time** complexity by "replaying" the forward computation during the backward pass—reusing the same sample points without storing large intermediate buffers.

---

## 1. Forward Pass: Computing Total Radiance $L$

For a ray $\mathbf{r}(t) = \mathbf{o} + t\mathbf{d}$, we sample $N$ points along the ray and compute the rendered color using the discrete volume rendering equation:

$$
L = \hat{C}(\mathbf{r}) = \sum_{i=1}^{N} T_i \, \alpha_i \, c_i
$$

where:
- $\sigma_i$ = density at point $i$
- $c_i$ = emitted color (RGB) at point $i$  
- $\delta_i = t_{i+1} - t_i$ = distance between adjacent samples
- $\alpha_i = 1 - \exp(-\sigma_i \delta_i)$ = opacity at point $i$
- $T_i = \exp\left(-\sum_{j=1}^{i-1} \sigma_j \delta_j\right) = \prod_{j=1}^{i-1}(1-\alpha_j)$ = transmittance (probability ray reaches point $i$)

The weight $w_i = T_i \alpha_i$ represents the contribution of point $i$ to the final pixel color.

---

## 2. Loss and Gradient $\delta L$

Given a reference image, we compute a loss function (typically MSE):

$$
\mathcal{L} = \| L - L_{\text{ref}} \|^2
$$

The gradient of the loss with respect to the rendered pixel value is:

$$
\delta L = \frac{\partial \mathcal{L}}{\partial L} = 2(L - L_{\text{ref}})
$$

This $\delta L$ is the "adjoint radiance" that we propagate backward through the rendering equation.

---

## 3. Deriving Per-Point Gradients

### Gradient with Respect to Color $c_i$

From the rendering equation $L = \sum_i T_i \alpha_i c_i$, the color $c_i$ only appears linearly in term $i$:

$$
\boxed{\frac{\partial L}{\partial c_i} = T_i \alpha_i = w_i}
$$

### Gradient with Respect to Density $\sigma_i$

The density $\sigma_i$ affects the rendering in two ways:

1. **Direct effect on $\alpha_i$**: Since $\alpha_i = 1 - e^{-\sigma_i \delta_i}$, we have $\frac{\partial \alpha_i}{\partial \sigma_i} = \delta_i (1-\alpha_i)$

2. **Indirect effect on $T_j$ for $j > i$**: Since $T_j = \prod_{k<j}(1-\alpha_k)$, we have $\frac{\partial T_j}{\partial \sigma_i} = -T_j \delta_i$

Combining both effects:

$$
\frac{\partial L}{\partial \sigma_i} = \underbrace{T_i \delta_i (1-\alpha_i) c_i}_{\text{direct}} - \underbrace{\delta_i \sum_{j>i} T_j \alpha_j c_j}_{\text{indirect}}
$$

Define $L_{\text{remaining}}^{(i)} = \sum_{j \geq i} w_j c_j$ as the remaining radiance at step $i$. Then $\sum_{j>i} w_j c_j = L_{\text{remaining}}^{(i)} - w_i c_i$, and:

$$
\boxed{\frac{\partial L}{\partial \sigma_i} = \delta_i \left( T_i c_i - L_{\text{remaining}}^{(i)} \right)}
$$

---

## 4. Expressing Gradients via Constructed Scalar $g$

The key to path replay is expressing these gradients as derivatives of constructed expressions. Given detached (gradient-stopped) quantities $\bar{x}$, we construct scalar expressions $g$ such that $\frac{\partial g}{\partial \sigma}$ and $\frac{\partial g}{\partial c}$ give us the desired gradients to accumulate into the parameters.

### Notation

- $\bar{x}$ denotes a **detached** value (no gradient flows through it)
- $x$ (unbarred) denotes a **tracked** value (gradients enabled)

### Gradient Expression for Color

We want $\frac{\partial \mathcal{L}}{\partial c_i} = \delta L \cdot T_i \alpha_i$. Construct:

$$
\boxed{g_c = \bar{\delta L} \cdot \bar{T} \cdot \bar{\alpha} \cdot c}
$$

Then:
$$
\frac{\partial g_c}{\partial c} = \bar{\delta L} \cdot \bar{T} \cdot \bar{\alpha} = \delta L \cdot w \quad \checkmark
$$

### Gradient Expression for Density

We want $\frac{\partial \mathcal{L}}{\partial \sigma_i} = \delta L \cdot \delta_i (T_i c_i - L_{\text{remaining}})$.

Using the optical thickness $\tau = \sigma \delta$, note that:
- $\alpha = 1 - e^{-\tau}$, so $-\log(1-\alpha) = \tau = \sigma \delta$
- $\frac{\partial \tau}{\partial \sigma} = \delta$

Construct:

$$
\boxed{g_\sigma = \bar{\delta L} \cdot \left( \bar{T} \cdot \bar{c} - \bar{L}_{\text{remaining}} \right) \cdot \tau}
$$

where $\tau = \sigma \cdot \bar{\delta}$ (only $\sigma$ is tracked). Then:

$$
\frac{\partial g_\sigma}{\partial \sigma} = \bar{\delta L} \cdot \left( \bar{T} \cdot \bar{c} - \bar{L}_{\text{remaining}} \right) \cdot \bar{\delta} = \delta L \cdot \delta \cdot (T c - L_{\text{remaining}}) \quad \checkmark
$$

### Combined Expression

In practice, we evaluate both gradients together. With $e = c$ (emission) tracked:

$$
\boxed{g = \bar{\delta L} \cdot \bar{T} \cdot \bar{\alpha} \cdot e \;+\; \bar{\delta L} \cdot \left( \bar{T} \cdot \bar{e} - \bar{L} \right) \cdot \sigma \bar{\delta}}
$$

This gives correct gradients for both $c$ (from the first term) and $\sigma$ (from the second term).

---

## 5. Path Replay Algorithm

### Key Insight

Instead of storing all intermediate values, we:
1. Initialize $L$ to the forward-pass result
2. March through the volume, progressively **subtracting** each point's contribution
3. At each step, construct $g$ using current (detached) values of $T$, $\alpha$, $L$, and accumulate $\frac{\partial g}{\partial \sigma}$, $\frac{\partial g}{\partial c}$ into the parameter gradients

### Algorithm

**Given**: Total radiance $L$ from forward pass, adjoint $\delta L$

**Initialize**:
- $L \leftarrow L_{\text{forward}}$ (will be decremented)
- $T \leftarrow 1$

**For each point $i = 1, \ldots, N$**:

1. **Query scene** (with gradients enabled via `dr.resume_grad`):
   - $\sigma, c \leftarrow \text{query}(p_i)$

2. **Compute local quantities** (tracked):
   - $\alpha = 1 - \exp(-\sigma \cdot \bar{\delta})$
   - $e = c$ (emission)

3. **Construct gradient expression**:
   $$g = \bar{\delta L} \cdot \bar{T} \cdot \bar{\alpha} \cdot e + \bar{\delta L} \cdot (\bar{T} \cdot \bar{e} - \bar{L}) \cdot \sigma \bar{\delta}$$

4. **Accumulate gradients**: $\frac{\partial \mathcal{L}}{\partial \sigma} \mathrel{+}= \frac{\partial g}{\partial \sigma}$, $\frac{\partial \mathcal{L}}{\partial c} \mathrel{+}= \frac{\partial g}{\partial c}$

5. **Update running values** (all detached for next iteration):
   - $L \leftarrow L - \bar{T} \cdot \bar{\alpha} \cdot \bar{e}$
   - $T \leftarrow T \cdot (1 - \bar{\alpha})$

---

## 6. Pseudocode

```python
def backward(ray, δL, L_forward, params):
    """
    Compute gradients using path replay backpropagation.
    
    Args:
        ray: Camera ray (origin, direction)
        δL: Gradient of loss w.r.t. rendered pixel (adjoint)
        L_forward: Total radiance from forward pass
        params: Scene parameters (density grid, color grid)
    """
    # Initialize running values (these stay detached)
    L = L_forward  # Remaining radiance, will decrease
    T = 1.0        # Running transmittance
    
    # Intersect ray with volume bounds
    t_near, t_far = intersect_bounds(ray, volume_bbox)
    t = t_near
    
    # March through volume (same points as forward pass!)
    while t < t_far:
        p = ray.origin + t * ray.direction
        δt = step_size
        
        # Detach running values for use in gradient expression
        T_detach = detach(T)
        L_detach = detach(L)
        δL_detach = detach(δL)
        δt_detach = detach(δt)

        # Query scene WITH gradient tracking
        σ = sample_density(p, params)       # tracked
        e = sample_color(p, ray.direction, params)  # tracked (emission)

        # Compute opacity (depends on tracked σ)
        α = 1.0 - exp(-σ * δt_detach)
        α_detach = detach(α)
        e_detach = detach(e)

        # Construct gradient expression g such that:
        #   ∂g/∂e = δL · T · α       (color gradient)
        #   ∂g/∂σ = δL · δt · (Tc - L)  (density gradient)

        g_color = δL_detach * T_detach * α_detach * e
        g_density = δL_detach * (T_detach * e_detach - L_detach) * σ * δt_detach

        g = g_color + g_density

        # Accumulate gradients into scene parameters:
        #   ∂𝓛/∂σ += ∂g/∂σ
        #   ∂𝓛/∂c += ∂g/∂c
        backward(g, params)

        # Update running values for next iteration (all detached)
        w = T_detach * α_detach
        L = L - w * e_detach
        T = T * (1.0 - α_detach)
        
        t += δt
```

---

## 7. Why the Detached Formulation Works

The expression $g$ is carefully constructed so that:

| Gradient needed | Expression term | Derivative |
|-----------------|-----------------|------------|
| $\frac{\partial \mathcal{L}}{\partial c}$ | $\bar{\delta L} \cdot \bar{T} \cdot \bar{\alpha} \cdot c$ | $\bar{\delta L} \cdot \bar{T} \cdot \bar{\alpha}$ |
| $\frac{\partial \mathcal{L}}{\partial \sigma}$ | $\bar{\delta L} \cdot (\bar{T}\bar{c} - \bar{L}) \cdot \sigma\bar{\delta}$ | $\bar{\delta L} \cdot (\bar{T}\bar{c} - \bar{L}) \cdot \bar{\delta}$ |

By detaching everything except the variable we're differentiating with respect to, we isolate each gradient contribution. Taking $\frac{\partial g}{\partial \sigma}$ and $\frac{\partial g}{\partial c}$ then propagates these gradients through the computational graph to the underlying parameters (grid values, network weights, etc.).

---

## 8. Key Properties

| Property | Value |
|----------|-------|
| **Memory** | O(1) — constant, independent of ray length |
| **Time** | O(N) — linear in number of samples |
| **Bias** | Unbiased gradient estimates |
| **Requirements** | Deterministic sampling (same points in forward/backward) |

---

## 9. Summary

The path replay algorithm avoids O(N) memory by:

1. **Not building AD graph across iterations**: Each loop iteration is independent
2. **Reconstructing values on-the-fly**: $L_{\text{remaining}}$ gives us the "future" contribution
3. **Using detached values**: Constructing $g$ with $\bar{T}$, $\bar{\alpha}$, $\bar{L}$ isolates gradients

The core identity enabling this is:

$$
\frac{\partial \mathcal{L}}{\partial \sigma_i} = \nabla_{\sigma}\left[ \bar{\delta L} \cdot (\bar{T}\bar{c} - \bar{L}) \cdot \sigma\bar{\delta} \right]
$$

where $\bar{L}$ is the **remaining** radiance (progressively updated), not the total.

---

## References

1. **Vicini, D., Speierer, S., & Jakob, W.** (2021). *Path Replay Backpropagation: Differentiating Light Paths using Constant Memory and Linear Time*. ACM Trans. Graph. (SIGGRAPH), 40(4).

2. **Mildenhall, B., et al.** (2020). *NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis*. ECCV.

3. **Nimier-David, M., et al.** (2020). *Radiative Backpropagation: An Adjoint Method for Lightning-Fast Differentiable Rendering*. ACM Trans. Graph. (SIGGRAPH), 39(4).

4. **Mitsuba 3 Documentation**: [Radiance Field Reconstruction Tutorial](https://mitsuba.readthedocs.io/en/stable/src/inverse_rendering/radiance_field_reconstruction.html)