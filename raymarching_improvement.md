# Path Replay Backpropagation for Radiance Field Rendering

Path replay backpropagation ([Vicini et al. 2021](https://rgl.epfl.ch/publications/Vicini2021PathReplay)) computes gradients with **O(1) memory** by replaying the forward pass during backpropagation.

---

## 1. Forward Pass

For ray $\mathbf{r}(t) = \mathbf{o} + t\mathbf{d}$ with $N$ samples, the rendered color is:

$$
C = \sum_{i=1}^{N} T_i \, \alpha_i \, c_i
$$

where:
- $\alpha_i = 1 - \exp(-\sigma_i \delta_i)$ (opacity)
- $T_i = \prod_{j=1}^{i-1}(1-\alpha_j)$ (transmittance)
- $w_i = T_i \alpha_i$ (sample weight)

---

## 2. Adjoint $\delta C$

For any differentiable loss $\mathcal{L}(C)$, compute the adjoint via autodiff:

$$
\delta C = \frac{\partial \mathcal{L}}{\partial C}
$$

---

## 3. Per-Point Gradients

**Color**: $c_i$ appears linearly, so:
$$
\frac{\partial C}{\partial c_i} = T_i \alpha_i = w_i
$$

**Density**: $\sigma_i$ affects both $\alpha_i$ (direct) and $T_j$ for $j > i$ (indirect):
$$
\frac{\partial C}{\partial \sigma_i} = \delta_i \left( T_i c_i - C_{\text{rem}}^{(i)} \right)
$$

where $C_{\text{rem}}^{(i)} = \sum_{j \geq i} w_j c_j$ is the remaining radiance.

---

## 4. Gradient Expression $g$

Construct a scalar $g$ such that its partials give the desired gradients. Using $\bar{x}$ for detached values:

$$
\boxed{g = \bar{\delta C} \cdot \bar{T} \cdot \bar{\alpha} \cdot c \;+\; \bar{\delta C} \cdot \left( \bar{T} \cdot \bar{c} - \bar{C} \right) \cdot \sigma \bar{\delta}}
$$

Then:
- $\frac{\partial g}{\partial c} = \bar{\delta C} \cdot \bar{T} \cdot \bar{\alpha} = \delta C \cdot w_i$
- $\frac{\partial g}{\partial \sigma} = \bar{\delta C} \cdot (\bar{T}\bar{c} - \bar{C}) \cdot \bar{\delta} = \delta C \cdot \delta \cdot (Tc - C_{\text{rem}})$

---

## 5. Algorithm

**Initialize**: $C \leftarrow C_{\text{forward}}$, $T \leftarrow 1$

**For each point $i$**:

1. Query $\sigma, c$ from scene (tracked)
2. Compute $\alpha = 1 - \exp(-\sigma \cdot \bar{\delta})$
3. Construct: $g = \bar{\delta C} \cdot \bar{T} \cdot \bar{\alpha} \cdot c + \bar{\delta C} \cdot (\bar{T} \cdot \bar{c} - \bar{C}) \cdot \sigma \bar{\delta}$
4. Accumulate: $\frac{\partial \mathcal{L}}{\partial \sigma} \mathrel{+}= \frac{\partial g}{\partial \sigma}$, $\quad \frac{\partial \mathcal{L}}{\partial c} \mathrel{+}= \frac{\partial g}{\partial c}$
5. Update: $C \leftarrow C - \bar{w} \cdot \bar{c}$, $\quad T \leftarrow T \cdot (1 - \bar{\alpha})$

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
