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

---

## 🧩 Inside a ConvNeXt Block

Each block follows a simple but powerful design:

