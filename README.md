# awesome-papers

## Novel 3D Shape Representation
### [CVPR25] [3D Student Splatting and Scooping](https://arxiv.org/abs/2503.10148)
This work proposes Student-t mixture model with both positive and negative components, optimized via principled SGHMC sampling. It achieves higher-fidelity and more arameter-efficient novel-view synthesis than 3DGS.

### [CVPR25] [3DCS:3D Convex Splatting: Radiance Field Rendering with 3D Smooth Convexes](https://arxiv.org/html/2411.14974v2)
This work replaces Gaussians with smooth convexes to deliver sharper geometry, higher fidelity, and lower memory use while maintaining real-time rendering.

### [arxiv] [Triangle Splatting for Real-Time Radiance Field Rendering](https://arxiv.org/abs/2505.19175)
This work employs discrete, smooth triangles as 3D representation, integrating the traditional graphics pipeline with radiance-field rendering to achieve impressive results.

### [CVPR25] [Sparse Voxels Rasterization: Real-time High-fidelity Radiance Field Rendering](https://arxiv.org/abs/2412.04459)
This work incorporates a rasterization process on adaptive sparse voxels, achieving state-of-the-art comparable novel-view synthesis results, and compatible with grid-based 3D processing techniques.

### [CVPR2025] [Deformable Radial Kernel Splatting](https://openaccess.thecvf.com/content/CVPR2025/papers/Huang_Deformable_Radial_Kernel_Splatting_CVPR_2025_paper.pdf)
The Deformable Radial Kernel (DRK) extends Gaussian splatting by introducing learnable radial bases with adjustable angles and scales, enabling more flexible 3D scene representation, improved rendering quality, and reduced primitive count for efficient rasterization.

## 3D Generation

### [arxiv] [Ultra3D: Efficient and High-Fidelity 3D Generation with Part Attention](https://arxiv.org/abs/2507.17745)
This paper first leverages the compact VecSet representation to efficiently generate a coarse object layout in the first stage, then refine per-voxel latent features in the second stage with Part Attention, a geometry-aware localized attention mechanism that restricts attention computation within semantically consistent part regions.

## VAE Architectures

### [ICLR2025] [Deep Compression Autoencoder for Efficient High-Resolution Diffusion Models](https://arxiv.org/abs/2410.10733)
The Deep Compression Autoencoder (DC-AE) improves high-resolution diffusion models by using Residual Autoencoding and Decoupled High-Resolution Adaptation to achieve higher compression ratios and faster inference without sacrificing accuracy.

## LLM Finetuning

### [ICLR2024Oral] [LongLoRA: Efficient Fine-tuning of Long-Context Large Language Models](https://arxiv.org/abs/2309.12307)
LongLoRA efficiently extends the context sizes of pre-trained large language models by using shifted sparse local attention for fine-tuning and combining it with parameter-efficient LoRA, achieving significant computation savings and strong empirical results with minimal code changes.
