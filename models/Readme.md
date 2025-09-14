# 🔬 ConvNeXt: A Modern Convolutional Network

In our solution, we used **ConvNeXt**, a convolutional neural network re-designed with modern best practices and inspired by Vision Transformers (ViTs).  
ConvNeXt keeps the efficiency and inductive biases of CNNs, while borrowing design ideas from Transformers (patchify stem, large kernels, LayerNorm, inverted bottlenecks).


## 🖼️ How ConvNeXt Works on an Input Image

Let’s consider an **input image of size** `224 × 224 × 3` (RGB).  
ConvNeXt processes it in four main stages:

1. **Input**:  
   Shape = `(N, 3, 224, 224)`  

2. **Patchify Stem** (`Conv2d` with kernel=4, stride=4):  
   - Reduces spatial resolution: `224 → 56`  
   - Expands channels: `3 → 96`  (Number of output channels here are defined by the user: number of filters in the convolution layer)
   - Output shape = `(N, 96, 56, 56)`  

   *Formula*:  
  ```math
   H_{out} = \left\lfloor \frac{H - K}{S} \right\rfloor + 1
   ```
   For `H = 224, K = 4, S = 4` → `56`.

3. **Stage Stack (×4)**:  
   Each stage contains several **ConvNeXt blocks**.  
   Between stages, a downsampling layer halves resolution and doubles channels.

   - **Stage 1**: 3 blocks, channels = 96 → `(N, 96, 56, 56)`  
   - **Stage 2**: Downsample + 3 blocks, channels = 192 → `(N, 192, 28, 28)`  
   - **Stage 3**: Downsample + 9 blocks, channels = 384 → `(N, 384, 14, 14)`  
   - **Stage 4**: Downsample + 3 blocks, channels = 768 → `(N, 768, 7, 7)`  

4. **Head**:  
   - Global average pooling: `(N, 768, 7, 7) → (N, 768)`  
   - Fully connected classifier: `(N, 768) → (N, #classes)`  

---

## 🧩 Inside a ConvNeXt Block

A **ConvNeXt block** transforms `(N, C, H, W)` into `(N, C, H, W)` with residual learning:

```
Input (N, C, H, W)
├─ Depthwise Conv (7×7, groups=C) → (N, C, H, W)
├─ Permute → (N, H, W, C)
├─ LayerNorm
├─ Linear (C → 4C)
├─ GELU
├─ Linear (4C → C)
├─ LayerScale (γ ∈ ℝ^C, learnable, ~1e-6 init)
├─ Permute back → (N, C, H, W)
├─ Residual connection (+ DropPath)
Output (N, C, H, W)
```


✅ **Depthwise conv** = captures spatial context with large receptive field  
✅ **MLP (1×1 linear layers)** = mixes channel information efficiently  
✅ **LayerNorm & GELU** = borrowed from Transformers for better training  
✅ **Residual + DropPath** = stabilizes deep training  

---

### 🔢 Example with Numbers

Suppose input to **Stage 2 block** is `(N, 192, 28, 28)`.

1. **Depthwise Conv (7×7)**  
   - Applies a spatial kernel per channel (no mixing across channels).  
   - Output shape = `(N, 192, 28, 28)`

   *Operation per channel \(c\)*:  
   ```math
   y_c(i,j) = \sum_{u=-3}^{3}\sum_{v=-3}^{3} W_c(u,v) \cdot x_c(i+u, j+v)
   ```

2. **LayerNorm (channel-wise)**  
   - Normalize each channel across `(H, W)` spatial positions.  
   - Keeps shape `(N, 28, 28, 192)` in NHWC format.

   *Formula*:  
   ```math
   \hat{x}_{c} = \frac{x_c - \mu_c}{\sqrt{\sigma_c^2 + \epsilon}}
   ```

3. **Linear Expansion (C → 4C)**  
   - `(192 → 768)`  
   - Shape = `(N, 28, 28, 768)`

4. **GELU Activation**  
   ```math
   \text{GELU}(x) = 0.5x \left( 1 + \tanh\!\left(\sqrt{\tfrac{2}{\pi}} (x + 0.044715x^3)\right)\right)
   ```

5. **Linear Projection (4C → C)**  
   - `(768 → 192)`  
   - Shape = `(N, 28, 28, 192)`

6. **LayerScale**  
   - Multiply output by learnable γ ∈ ℝ^192 (initial ~1e-6).  
   - Helps stabilize deep training.

7. **Residual + DropPath**  
   - Add input `(N, 28, 28, 192)` to output.  
   - DropPath stochastically drops residual branches during training.

✅ Final output shape remains `(N, 192, 28, 28)`.

---

## 📊 Size Summary (ConvNeXt-Tiny Example)

| Stage   | Resolution | Channels | Blocks |
|---------|------------|----------|--------|
| Input   | 224×224    | 3        | –      |
| Stem    | 56×56      | 96       | –      |
| Stage 1 | 56×56      | 96       | 3      |
| Stage 2 | 28×28      | 192      | 3      |
| Stage 3 | 14×14      | 384      | 9      |
| Stage 4 | 7×7        | 768      | 3      |
| Head    | 1×1        | 768      | –      |

---

## 📊 Why ConvNeXt?

- Matches or outperforms Vision Transformers while staying purely convolutional  
- Scales efficiently across different model sizes (Tiny → Large)  
- Simple, unified design that combines the best of CNNs and Transformers  

---

## 📖 Reference

ConvNeXt was introduced in:  
> **A ConvNet for the 2020s**  
> Zhuang Liu, Hanzi Mao, Chao-Yuan Wu, Christoph Feichtenhofer, Trevor Darrell, Saining Xie  
> CVPR 2022 — [Paper](https://arxiv.org/abs/2201.03545)

