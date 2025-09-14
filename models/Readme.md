# 🔬 ConvNeXt: A Modern Convolutional Network

In our solution, we used **ConvNeXt**, a convolutional neural network re-designed with modern best practices and inspired by Vision Transformers (ViTs).  
ConvNeXt keeps the efficiency and inductive biases of CNNs, while borrowing design ideas from Transformers (patchify stem, large kernels, LayerNorm, inverted bottlenecks).


## 🖼️ How ConvNeXt Works on an Input Image

1. **Input**: RGB image (e.g. `224×224×3`)  
2. **Patchify Stem**: A `4×4` convolution with stride 4 splits the image into non-overlapping patches (like ViTs), reducing spatial size (224 → 56).  
3. **Stages (×4)**:  
   - Each stage contains multiple **ConvNeXt blocks**  
   - Between stages, downsampling halves resolution and doubles channels  
   - Typical channel sizes: `[96, 192, 384, 768]`  
4. **Head**: Global average pooling → Linear classifier → Predictions  


## 🧩 Inside a ConvNeXt Block

Each block follows a simple but powerful design:
```
Input (N, C, H, W)
├─ Depthwise Conv (7×7) → spatial mixing
├─ LayerNorm
├─ Linear (C → 4C) + GELU + Linear (4C → C) → channel mixing
├─ LayerScale (tiny learnable γ)
├─ Residual connection (+ DropPath)
Output (N, C, H, W)
```


✅ **Depthwise conv** = captures spatial context with large receptive field  
✅ **MLP (1×1 linear layers)** = mixes channel information efficiently  
✅ **LayerNorm & GELU** = borrowed from Transformers for better training  
✅ **Residual + DropPath** = stabilizes deep training  

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

