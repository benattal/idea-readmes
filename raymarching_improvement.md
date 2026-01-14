# Path Replay Backpropagation for Radiance Field Rendering

Path replay backpropagation ([Vicini et al. 2021](https://rgl.epfl.ch/publications/Vicini2021PathReplay)) computes gradients with **O(1) memory** by replaying the forward pass during backpropagation.

---

## 1. Forward Pass

For ray $\mathbf{r}(t) = \mathbf{o} + t\mathbf{d}$ with $N$ samples:

$$
L = \sum_{i=1}^{N} T_i \, \alpha_i \, c_i
$$

where:
- $\alpha_i = 1 - \exp(-\sigma_i \delta_i)$ (opacity)
- $T_i = \prod_{j=1}^{i-1}(1-\alpha_j)$ (transmittance)
- $w_i = T_i \alpha_i$ (sample weight)

---

## 2. Adjoint $\delta L$

For any differentiable loss $\mathcal{L}(L)$, compute the adjoint via autodiff:

$$
\delta L = \frac{\partial \mathcal{L}}{\partial L}
$$

---

## 3. Per-Point Gradients

**Color**: $c_i$ appears linearly, so:
$$
\frac{\partial L}{\partial c_i} = T_i \alpha_i = w_i
$$

**Density**: $\sigma_i$ affects both $\alpha_i$ (direct) and $T_j$ for $j > i$ (indirect):
$$
\frac{\partial L}{\partial \sigma_i} = \delta_i \left( T_i c_i - L_{\text{remaining}}^{(i)} \right)
$$

where $L_{\text{remaining}}^{(i)} = \sum_{j \geq i} w_j c_j$.

---

## 4. Gradient Expression $g$

Construct a scalar $g$ such that its partials give the desired gradients. Using $\bar{x}$ for detached values:

$$
\boxed{g = \bar{\delta L} \cdot \bar{T} \cdot \bar{\alpha} \cdot c \;+\; \bar{\delta L} \cdot \left( \bar{T} \cdot \bar{c} - \bar{L} \right) \cdot \sigma \bar{\delta}}
$$

Then:
- $\frac{\partial g}{\partial c} = \bar{\delta L} \cdot \bar{T} \cdot \bar{\alpha} = \delta L \cdot w_i$
- $\frac{\partial g}{\partial \sigma} = \bar{\delta L} \cdot (\bar{T}\bar{c} - \bar{L}) \cdot \bar{\delta} = \delta L \cdot \delta \cdot (Tc - L_{\text{remaining}})$

---

## 5. Algorithm

**Initialize**: $L \leftarrow L_{\text{forward}}$, $T \leftarrow 1$

**For each point $i$**:

1. Query $\sigma, c$ from scene (tracked)
2. Compute $\alpha = 1 - \exp(-\sigma \cdot \bar{\delta})$
3. Construct: $g = \bar{\delta L} \cdot \bar{T} \cdot \bar{\alpha} \cdot c + \bar{\delta L} \cdot (\bar{T} \cdot \bar{c} - \bar{L}) \cdot \sigma \bar{\delta}$
4. Accumulate: $\frac{\partial \mathcal{L}}{\partial \sigma} \mathrel{+}= \frac{\partial g}{\partial \sigma}$, $\quad \frac{\partial \mathcal{L}}{\partial c} \mathrel{+}= \frac{\partial g}{\partial c}$
5. Update: $L \leftarrow L - \bar{w} \cdot \bar{c}$, $\quad T \leftarrow T \cdot (1 - \bar{\alpha})$

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
