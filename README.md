## Awesome Papers

Hi there!👋
We are from the Spatial Intelligence Group (SIG) at Peking University.
This is a collection of awesome papers discussed in our seminars.
We have organized them into various categories to provide an effective starting point for your literature surveys and research exploration.
Enjoy your reading!

### Part-Aware 3D Generation
-  [SIGGRAPH25] [BANG: Dividing 3D Assets via Generative Exploded Dynamics](https://arxiv.org/abs/2507.21493) ✨

   BANG is a generative 3D design framework that enables intuitive, part-level decomposition and manipulation of 3D objects through smooth exploded views and user-guided controls, enhancing creativity and practical workflows.

- [Arxiv25] [PartCrafter: Structured 3D Mesh Generation via Compositional Latent Diffusion Transformers](https://arxiv.org/abs/2506.05573)
  PartCrafter is a unified 3D generative model that synthesizes structured, part-aware 3D meshes from a single image by introducing a compositional latent space based on 3DShape2VecSet and hierarchical attention, enabling end-to-end generation of semantically meaningful object parts。

- [ICCV23] [DiffFacto: Controllable Part-Based 3D Point Cloud Generation with Cross Diffusion](https://arxiv.org/abs/2305.01921)
  DiffFacto is a generative model for controllable part-based point cloud generation that factorizes part style and configuration distributions and uses a cross-diffusion network to generate coherent shapes with fine-grained part-level control.

- [CVPR25Highlight] [PartGen: Part-level 3D Generation and Reconstruction with Multi-View Diffusion Models](https://arxiv.org/abs/2412.18608)
  PartGen addresses the lack of structure in 3D assets by formulating 3D generation, part segmentation and completion as multi-view diffusion processes, enabling the generation of coherent, editable part-based 3D objects from text, images, or unstructured 3D inputs.

- [AAAI22] [EditVAE: Unsupervised Parts-Aware Controllable 3D Point Cloud Shape Generation](https://arxiv.org/abs/2110.06679)
  This paper proposes an unsupervised, parts-aware point cloud generation method by modifying a VAE to jointly model shapes and their schematic part-based structure, enabling disentangled part representations and spatially coherent editing without requiring pre-segmented data.

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

- [SIGGRAPH15] [SMPL: a skinned multi-person linear model](https://dl.acm.org/doi/10.1145/2816795.2818013) ✨

  SMPL is a data-driven, blend-skin-compatible body model that jointly learns identity and pose-dependent shape variation from aligned 3D meshes, outperforming prior models while fitting standard graphics pipelines.

- [ICLR23] [Human Motion Diffsion Model](https://arxiv.org/abs/2209.14916)

  MDM is a lightweight, transformer-based diffusion model that directly predicts human motion samples—rather than noise—delivering state-of-the-art, controllable generation from text or action prompts with minimal compute.

- [arxiv] [GENMO: A GENeralist Model for Human MOtion](https://arxiv.org/abs/2505.01425)

  GENMO unifies motion estimation and generation into one diffusion model by treating estimation as constrained generation, leveraging mixed-modal data to improve both tasks in a single, flexible framework.

- [arxiv] [Sketch2Anim: Towards Transferring Sketch Storyboards into 3D Animation](https://arxiv.org/abs/2504.19189)

  Sketch2Anim pioneers direct 2D-to-3D storyboard animation through a dual-module diffusion model that maps user sketches to 3D keyposes and trajectories, yielding editable, high-quality motions without expert labor.

### Programmatic 3D Modeling

- [arxiv] [LL3M: Large Language 3D Modelers](https://threedle.github.io/ll3m/)

   LL3M is a multi-agent system that leverages pretrained large language models (LLMs) to generate 3D assets by writing interpretable Python code in Blender.

- [arxiv] [MeshCoder: LLM-Powered Structured Mesh Code Generation from Point Clouds](https://daibingquan.github.io/MeshCoder)
  
  MeshCoder is a novel framework that reconstructs complex 3D objects from point clouds into editable Blender Python scripts.

- [CVPR2023] [Infinite Photorealistic Worlds using Procedural Generation](https://infinigen.org/)

  Infinigen is a procedural generator of photorealistic 3D scenes of the natural world. Infinigen is entirely procedural: every asset, from shape to texture, is generated from scratch via randomized mathematical rules, using no external source and allowing infinite variation and composition.

- [CVPR2024] [Infinigen Indoors: Photorealistic Indoor Scenes using Procedural Generation](https://infinigen.org/)

   Infinigen Indoors is a Blender-based procedural generator of photorealistic indoor scenes. It builds upon the existing Infinigen system, which focuses on natural scenes, but expands its coverage to indoor scenes by introducing a diverse library of procedural indoor assets, including furniture, architecture elements, appliances, and other day-to-day objects.

- [Scene Synthesizer](https://scene-synthesizer.github.io/) 
  
  A Python package to easily create scenes for robot manipulation tasks.

- [CVPR2025 Highlight] [The Scene Language: Representing Scenes with Programs, Words, and Embeddings](https://ai.stanford.edu/~yzzhang/projects/scene-language/)

  Scene Language is a visual scene representation that concisely and precisely describes the structure, semantics, and identity of visual scenes. It represents a scene with three key components: a program that specifies the hierarchical and relational structure of entities in the scene, words in natural language that summarize the semantic class of each entity, and embeddings that capture the visual identity of each entity. 

- [arxiv] [ShapeLib: Designing a library of programmatic 3D shape abstractions with Large Language Models](https://arxiv.org/abs/2502.08884)

  ShapeLib is a method that leverages the priors of LLMs to design libraries of programmatic 3D shape abstractions. The system accepts two forms of design intent: text descriptions of functions to include in the library and a seed set of exemplar shapes, and then discovers abstractions that match this design intent with a guided LLM workflow

### Layout / Scene Generation

- [CVPR2024] [Holodeck: Language Guided Generation of 3D Embodied AI Environments](https://yueyang1996.github.io/holodeck/)

  Holodeck is a system that generates 3D environments to match a user-supplied prompt fully automatedly. It leverages a large language model (GPT-4) for common sense knowledge about what the scene might look like and uses a large collection of 3D assets from Objaverse to populate the scene with diverse objects.

- [arxiv] [HOLODECK 2.0: Vision-Language-Guided 3D World Generation with Editing](https://arxiv.org/abs/2508.05899)

   HOLODECK 2.0 is an advanced vision-language-guided framework for 3D world generation with support for interactive scene editing based on human feedback. It leverages vision-language models (VLMs) to identify and parse the objects required in a scene and generates corresponding high-quality assets via state-of-the-art 3D generative models. It then iteratively applies spatial constraints derived from the VLMs to achieve semantically coherent and physically plausible layouts.

- [CVPR2025] [LayoutVLM: Differentiable Optimization of 3D Layout via Vision-Language Models](https://ai.stanford.edu/~sunfanyun/layoutvlm/)

   LayoutVLM is a framework and scene layout representation that exploits the semantic knowledge of Vision-Language Models (VLMs) and supports differentiable optimization to ensure physical plausibility. LayoutVLM employs VLMs to generate two mutually reinforcing representations from visually marked images, and a self-consistent decoding process to improve VLMs spatial planning.

### Diffusion LLMs

- [NeurIPS 2021] [Structured Denoising Diffusion Models in Discrete State-Spaces](https://arxiv.org/abs/2107.03006)

  This paper introduces Discrete Denoising Diffusion Probabilistic Models (D3PMs), which adapt diffusion models for discrete data by using more flexible corruption processes and a novel loss function.
  
- [arxiv] [Large Language Diffusion Models](https://arxiv.org/abs/2502.09992)

  This paper introduces LLaDA, a diffusion-based language model that challenges the dominance of autoregressive models by demonstrating competitive performance in in-context learning, instruction following, and even surpassing GPT-4o in specific tasks, suggesting that key LLM capabilities are not exclusive to autoregressive architectures.
  
- [arxiv] [Fast-dLLM: Training-free Acceleration of Diffusion LLM by Enabling KV Cache and Parallel Decoding](https://arxiv.org/abs/2505.22618)

   This paper closes the performance gap between Diffusion and autoregressive LLMs by introducing a novel block-wise KV Cache and a confidence-aware parallel decoding strategy, which together achieve a massive throughput improvement with minimal accuracy loss.

- [arxiv] [Reinforcing the Diffusion Chain of Lateral Thought with Diffusion Language Models](https://arxiv.org/abs/2505.10446)

   This paper introduces the Diffusion Chain of Lateral Thought (DCoLT), a novel framework that uses reinforcement learning to optimize the entire non-linear, bidirectional reasoning trajectory within diffusion language models, significantly boosting their performance on complex math and code generation tasks.
