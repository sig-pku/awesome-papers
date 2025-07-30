## Awesome Papers

Hi there!👋
We are from the Spatial Intelligence Group (SIG) at Peking University.
This is a collection of awesome papers discussed in our seminars.
We have organized them into various categories to provide an effective starting point for your literature surveys and research exploration.
Enjoy your reading!


### Novel 3D Shape Representation
-  [CVPR25] [3D Student Splatting and Scooping](https://arxiv.org/abs/2503.10148)

    This work proposes Student-t mixture model with both positive and negative components, optimized via principled SGHMC sampling. It achieves higher-fidelity and more parameter-efficient novel-view synthesis than 3DGS.

- [CVPR25] [3DCS:3D Convex Splatting: Radiance Field Rendering with 3D Smooth Convexes](https://arxiv.org/html/2411.14974v2)

  This work replaces Gaussians with smooth convexes to deliver sharper geometry, higher fidelity, and lower memory use while maintaining real-time rendering.

- [arxiv] [Triangle Splatting for Real-Time Radiance Field Rendering](https://arxiv.org/abs/2505.19175) ✨


  This work employs discrete, smooth triangles as 3D representation, integrating the traditional graphics pipeline with radiance-field rendering to achieve impressive results.

- [CVPR25] [Sparse Voxels Rasterization: Real-time High-fidelity Radiance Field Rendering](https://arxiv.org/abs/2412.04459)

  This work incorporates a rasterization process on adaptive sparse voxels, achieving state-of-the-art comparable novel-view synthesis results, and compatible with grid-based 3D processing techniques.


- [CVPR2025] [Deformable Radial Kernel Splatting](https://openaccess.thecvf.com/content/CVPR2025/papers/Huang_Deformable_Radial_Kernel_Splatting_CVPR_2025_paper.pdf)

  The Deformable Radial Kernel (DRK) extends Gaussian splatting by introducing learnable radial bases with adjustable angles and scales, enabling more flexible 3D scene representation, improved rendering quality, and reduced primitive count for efficient rasterization.

### 3D Generation

- [arxiv] [Ultra3D: Efficient and High-Fidelity 3D Generation with Part Attention](https://arxiv.org/abs/2507.17745)

  This paper first leverages the compact VecSet representation to efficiently generate a coarse object layout in the first stage, then refine per-voxel latent features in the second stage with Part Attention, a geometry-aware localized attention mechanism that restricts attention computation within semantically consistent part regions.

### VAE Architectures

- [ICLR2025] [Deep Compression Autoencoder for Efficient High-Resolution Diffusion Models](https://arxiv.org/abs/2410.10733)

  The Deep Compression Autoencoder (DC-AE) improves high-resolution diffusion models by using Residual Autoencoding and Decoupled High-Resolution Adaptation to achieve higher compression ratios and faster inference without sacrificing accuracy.

### LLM Finetuning

- [ICLR2024Oral] [LongLoRA: Efficient Fine-tuning of Long-Context Large Language Models](https://arxiv.org/abs/2309.12307) ✨

  LongLoRA efficiently extends the context sizes of pre-trained large language models by using shifted sparse local attention for fine-tuning and combining it with parameter-efficient LoRA, achieving significant computation savings and strong empirical results with minimal code changes.

### Human Motion Generation
- [TPAMI23] [Human motion generation: A survey](https://arxiv.org/abs/2307.10894) ✨

  This survey delivers the first comprehensive overview of conditional human motion generation, systematically reviewing text-, audio-, and scene-conditioned methods, datasets, metrics, and future challenges.

- [TOG15] [SMPL: a skinned multi-person linear model](https://dl.acm.org/doi/10.1145/2816795.2818013) ✨

  SMPL is a data-driven, blend-skin-compatible body model that jointly learns identity and pose-dependent shape variation from aligned 3D meshes, outperforming prior models while fitting standard graphics pipelines.

- [ICLR23] [Human Motion Diffsion Model](https://arxiv.org/abs/2209.14916)

  MDM is a lightweight, transformer-based diffusion model that directly predicts human motion samples—rather than noise—delivering state-of-the-art, controllable generation from text or action prompts with minimal compute.

- [arxiv] [GENMO: A GENeralist Model for Human MOtion](https://arxiv.org/abs/2505.01425)

  GENMO unifies motion estimation and generation into one diffusion model by treating estimation as constrained generation, leveraging mixed-modal data to improve both tasks in a single, flexible framework.

- [arxiv] [Sketch2Anim: Towards Transferring Sketch Storyboards into 3D Animation](https://arxiv.org/abs/2504.19189)

  Sketch2Anim pioneers direct 2D-to-3D storyboard animation through a dual-module diffusion model that maps user sketches to 3D keyposes and trajectories, yielding editable, high-quality motions without expert labor.

