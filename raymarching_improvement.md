# Path Replay Backpropagation for Radiance Field Rendering

Path replay backpropagation ([Vicini et al. 2021](https://rgl.epfl.ch/publications/Vicini2021PathReplay)) computes gradients with **O(1) memory** by replaying the forward pass during backpropagation.

---

## 1. Forward Pass

For ray **r**(t) = **o** + t**d** with N samples, the rendered color is:

$$
L = \sum_{i=1}^{N} T_i \alpha_i c_i
$$

where:
- $\alpha_i = 1 - \exp(-\sigma_i \delta_i)$ — opacity
- $T_i = \prod_{j=1}^{i-1}(1-\alpha_j)$ — transmittance
- $w_i = T_i \alpha_i$ — sample weight

---

## 2. Adjoint

For any differentiable loss $J(L)$, compute the adjoint via autodiff:

$$
\partial L = \frac{\partial J}{\partial L}
$$

---

## 3. Per-Point Gradients

**Color**: $c_i$ appears linearly, so:

$$
\frac{\partial L}{\partial c_i} = T_i \alpha_i = w_i
$$

**Density**: $\sigma_i$ affects both $\alpha_i$ (direct) and $T_j$ for j > i (indirect):

$$
\frac{\partial L}{\partial \sigma_i} = \delta_i ( T_i c_i - L_i )
$$

where $L_i = \sum_{j \geq i} w_j c_j$ is the remaining radiance at step i.

---

## 4. Gradient Expression

Construct a scalar $g$ whose partials give the desired gradients. Overline denotes detached (stop-gradient) values.

$$
g = \overline{\partial L} \cdot \bar{T} \cdot \bar{\alpha} \cdot c + \overline{\partial L} \cdot ( \bar{T} \cdot \bar{c} - \bar{L} ) \cdot \sigma \cdot \bar{\delta}
$$

$$
\frac{\partial g}{\partial c} = \overline{\partial L} \cdot \bar{T} \cdot \bar{\alpha} = \partial L \cdot w_i
$$

$$
\frac{\partial g}{\partial \sigma} = \overline{\partial L} \cdot (\bar{T}\bar{c} - \bar{L}) \cdot \bar{\delta} = \partial L \cdot \delta \cdot (Tc - L_i)
$$

---

## 5. Algorithm

```
L = L_forward
T = 1
∂L = ∂J/∂L                               # adjoint (from autodiff)

for each point i:
    σ, c = query_scene(p_i)              # tracked
    α = 1 - exp(-σ * δ)              # α depends on σ

    # Construct gradient expression (sg = stop gradient)
    g = sg(∂L * T * α) * c + sg(∂L * (T * c - L) * δ) * σ

    # Accumulate gradients
    ∂J/∂c += ∂g/∂c
    ∂J/∂σ += ∂g/∂σ

    # Update running values
    L = L - sg(T * α * c)
    T = T * sg(1 - α)
```

---

## 6. Properties

| Property | Value |
|----------|-------|
| **Memory** | O(1) |
| **Time** | O(N) |
| **Requirement** | Deterministic sampling |

---

## References

1. Vicini, D., Speierer, S., & Jakob, W. (2021). *Path Replay Backpropagation*. SIGGRAPH.
2. Mildenhall, B., et al. (2020). *NeRF*. ECCV.
