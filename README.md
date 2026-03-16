<div align="center">

# OmniForcing: Unleashing Real-time Joint Audio-Visual Generation

<img src="static/images/teaser_2.png" width="100%">

[Yaofeng Su]()\*¹ᐟ², [Yuming Li]()\*³, [Zeyue Xue]()¹ᐟ⁴, [Jie Huang]()¹, [Siming Fu]()¹, [Haoran Li]()¹, [Ying Li]()³, [Zezhong Qian]()³, [Haoyang Huang]()¹, [Nan Duan]()¹

\*Equal contribution &ensp;|&ensp; ¹JD Explore Academy &ensp; ²Fudan University &ensp; ³Peking University &ensp; ⁴The University of Hong Kong

<a href="https://arxiv.org/abs/2603.11647"><img src="https://img.shields.io/badge/arXiv-2603.11647-b31b1b.svg" alt="arXiv"></a>
<a href="https://omniforcing.com"><img src="https://img.shields.io/badge/Project-Page-blue.svg" alt="Project Page"></a>

</div>

**OmniForcing** is the first framework to distill an offline, bidirectional joint audio-visual diffusion model into a **real-time streaming autoregressive generator**. Built on top of LTX-2 (14B video + 5B audio), OmniForcing achieves **~25 FPS** streaming on a single GPU with a Time-To-First-Chunk of only **~0.7s** — a **~35× speedup** over the teacher — while maintaining visual and acoustic fidelity on par with the bidirectional teacher model.


## 📰 News

- **[Coming Soon]** Code and datasets will be open-sourced within two weeks. Stay tuned!
- **[2026/03]** Paper released on [arXiv 2603.11647](https://arxiv.org/abs/2603.11647). Project page is live at [OmniForcing.com](https://omniforcing.com).



## 🔧 Method Overview

<div align="center">
<img src="static/images/method.png" width="100%">
</div>

OmniForcing employs a **three-stage distillation pipeline** to progressively transform the bidirectional teacher into a causal streaming engine:

- **Stage I — Bidirectional DMD:** Distribution Matching Distillation compresses the multi-step diffusion sampling into few-step denoising, while preserving the original global attention.

- **Stage II — Causal ODE Regression:** The model is equipped with our **Asymmetric Block-Causal Mask** and trained via ODE trajectory regression to adapt to causal attention. An **Audio Sink Token** mechanism with **Identity RoPE** is introduced to resolve the Softmax collapse and gradient explosion caused by extreme audio token sparsity.

- **Stage III — Joint Self-Forcing DMD:** The model autoregressively unrolls its own generations during training, enabling it to dynamically self-correct cumulative cross-modal errors from exposure bias.

At inference time, a **Modality-Independent Rolling KV-Cache** reduces per-step context complexity to O(L) and enables concurrent execution of the video and audio streams, achieving real-time synchronized generation.

## 📊 Results & Demos

### Main Results on JavisBench
 
<div align="center">
<table>
<thead>
<tr>
<th>Model</th><th>Size</th><th>FVD ↓</th><th>FAD ↓</th><th>CLIP ↑</th><th>AV-IB ↑</th><th>DeSync ↓</th><th>Runtime ↓</th>
</tr>
</thead>
<tbody>
<tr><td>MMAudio</td><td>0.1B</td><td>–</td><td>6.1</td><td>–</td><td>0.198</td><td>0.849</td><td>15s</td></tr>
<tr><td>JavisDiT++</td><td>2.1B</td><td>141.5</td><td>5.5</td><td>0.316</td><td>0.198</td><td>0.832</td><td>10s</td></tr>
<tr><td>UniVerse-1</td><td>6.4B</td><td>194.2</td><td>8.7</td><td>0.309</td><td>0.104</td><td>0.929</td><td>13s</td></tr>
<tr><td>LTX-2 (Teacher)</td><td>19B</td><td><b>125.4</b></td><td><b>4.6</b></td><td>0.318</td><td><b>0.318</b></td><td><b>0.384</b></td><td>197s</td></tr>
<tr><td><b>OmniForcing (Ours)</b></td><td>19B</td><td>137.2</td><td>5.7</td><td><b>0.322</b></td><td>0.269</td><td>0.392</td><td><b>5.7s</b></td></tr>
</tbody>
</table>
</div>
 
### Distillation Fidelity (VBench)
 
<div align="center">
<table>
<thead>
<tr>
<th>Model</th><th>Aesthetic ↑</th><th>Imaging ↑</th><th>Motion Smooth. ↑</th><th>Subject Consist. ↑</th><th>TTFC ↓</th><th>FPS ↑</th>
</tr>
</thead>
<tbody>
<tr><td>LTX-2 (Teacher)</td><td>0.569</td><td>0.574</td><td>0.993</td><td>0.945</td><td>197.0s</td><td>–</td></tr>
<tr><td><b>OmniForcing</b></td><td><b>0.595</b></td><td><b>0.594</b></td><td><b>0.995</b></td><td><b>0.955</b></td><td><b>0.7s</b></td><td><b>25</b></td></tr>
</tbody>
</table>
</div>
 
### Demo Gallery
 
<div align="center">
<a href="https://omniforcing.com"><img src="static/images/fig4_1.png" width="100%"></a>
<p><sub>Click the image above to watch audio-visual demos on our Project Page.</sub></p>
</div>
 
> 🔊 **Audio matters!** OmniForcing generates synchronized audio and video — visit our [Project Page](https://omniforcing.com) for playable demos and side-by-side comparisons with the LTX-2 teacher.
 
 
## 🚀 Getting Started

> Coming Soon — code, model weights, and inference scripts will be released within two weeks.

<!--
### Installation
```bash
# TODO
```

### Inference
```bash
# TODO
```

### Training
```bash
# TODO
```
-->


## 📝 Citation

If you find OmniForcing useful in your research, please consider citing:

```bibtex
@article{su2026omniforcing,
  title   = {OmniForcing: Unleashing Real-time Joint Audio-Visual Generation},
  author  = {Su, Yaofeng and Li, Yuming and Xue, Zeyue and Huang, Jie and Fu, Siming
             and Li, Haoran and Li, Ying and Qian, Zezhong and Huang, Haoyang and Duan, Nan},
  journal = {arXiv preprint arXiv:2603.11647},
  year    = {2026}
}
```

## 🙏 Acknowledgements

OmniForcing builds upon several outstanding works. We thank the authors of [LTX-2](https://github.com/Lightricks/LTX-2), [Self-Forcing]([https://github.com/desaixie/self-forcing](https://github.com/guandeh17/Self-Forcing)), [CausVid](https://github.com/tianweiy/CausVid), and [DMD](https://github.com/tianweiy/DMD2) for their pioneering contributions. The project page template is adapted from [Nerfies](https://nerfies.github.io/).
