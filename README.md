# Multi-stream HiFi-GAN with Data-Driven Waveform Decomposition and NSF

> **Note**: This is an unofficial implementation. The base HifiGAN code from [IAHispano/Applio](https://github.com/IAHispano/Applio) was used as a foundation for this project.

This repository contains a PyTorch implementation of a modified HiFi-GAN generator, incorporating the **multi-stream architecture** and **sub-pixel convolution upsampling** proposed in the paper "[MULTI-STREAM HIFI-GAN WITH DATA-DRIVEN WAVEFORM DECOMPOSITION](https://ast-astrec.nict.go.jp/release/preprints/preprint_asru_2021_okamoto.pdf)" by Okamoto et al. (ASRU 2021). This implementation aims to address the limitations of fixed multi-band approaches in vocoders by employing a **data-driven waveform decomposition**.  It builds upon the Neural Source-Filter (NSF) variant of HiFi-GAN for enhanced audio quality and F0 control.

**Context and Motivation (based on the Multi-stream HiFi-GAN Paper):**

The original HiFi-GAN vocoder achieves real-time, high-fidelity speech synthesis.  To further improve inference speed while maintaining quality, a multi-band structure was introduced to HiFi-GAN (similar to Multi-band MelGAN). However, training a multi-band HiFi-GAN with a *fixed* multi-band decomposition proved challenging due to the strong constraints imposed by the fixed structure.

The **Multi-stream HiFi-GAN paper** proposes an alternative: replacing the fixed synthesis filter in a multi-band approach with a **trainable convolutional layer**. This allows the model to learn a **data-driven waveform decomposition**, optimizing how to combine multiple waveform streams for improved synthesis quality and efficiency.

**Abstract of the Original HiFi-GAN Paper (for general context):**

> Several recent work on speech synthesis have employed generative adversarial networks (GANs) to produce raw waveforms. Although such methods improve the sampling efficiency and memory usage, their sample quality has not yet reached that of autoregressive and flow-based generative models. In this work, we propose HiFi-GAN, which achieves both efficient and high-fidelity speech synthesis. As speech audio consists of sinusoidal signals with various periods, we demonstrate that modeling periodic patterns of an audio is crucial for enhancing sample quality. A subjective human evaluation (mean opinion score, MOS) of a single speaker dataset indicates that our proposed method demonstrates similarity to human quality while generating 22.05 kHz high-fidelity audio 167.9 times faster than real-time on a single V100 GPU. We further show the generality of HiFi-GAN to the mel-spectrogram inversion of unseen speakers and end-to-end speech synthesis. Finally, a small footprint version of HiFi-GAN generates samples 13.4 times faster than real-time on CPU with comparable quality to an autoregressive counterpart.

**Key Modifications (Multi-stream HiFi-GAN NSF):**

This implementation incorporates the following key modifications to the standard HiFi-GAN NSF architecture, drawing from the Multi-stream HiFi-GAN paper:

*   **Multi-stream Architecture:**  The generator produces multiple *streams* of waveforms in parallel. These streams are then combined by a trainable convolutional layer (`stream_combiner`). This replaces the fixed synthesis filter of traditional multi-band approaches, enabling the model to learn an **optimal, data-driven decomposition** of the speech waveform.

*   **Sub-pixel Convolution Upsampling:**  Replaces traditional transposed convolutions in the upsampling layers with sub-pixel convolutions (`nn.Conv1d` followed by `PixelShuffle1D`). Sub-pixel convolution is known to **reduce checkerboard artifacts** common in transposed convolution, leading to **cleaner and higher fidelity** audio generation.

*   **Neural Source-Filter (NSF) Integration:**  This implementation utilizes a Neural Source-Filter (NSF) model as the base HiFi-GAN generator.  NSF explicitly models the source and filter components of speech, taking the fundamental frequency (F0) sequence as an input alongside the mel-spectrogram. This allows for **better control over pitch and potentially improves audio quality**, especially for voiced speech segments.  *Note: While NSF enhances this implementation, the core innovation from the Multi-stream HiFi-GAN paper lies in the multi-stream architecture and data-driven decomposition, which are independent of the choice of NSF as the base generator.*


**Benefits:**

*   **Improved Inference Speed:** The use of sub-pixel convolution for upsampling is generally computationally more efficient than transposed convolutions, resulting in faster waveform generation.
*   **Enhanced Audio Quality:** The combination of the multi-stream architecture, data-driven waveform decomposition, and sub-pixel convolution contributes to potentially **higher perceptual audio quality** compared to standard HiFi-GAN and fixed multi-band approaches. The trainable stream combiner allows for a more flexible and optimized recombination of waveform components.
*   **Data-Driven Waveform Decomposition:**  The model **learns the optimal way to decompose and recombine the waveform** into streams during training. This data-driven approach is more flexible and effective than predefined, fixed filter banks used in traditional multi-band methods.
*   **Explicit F0 Control (via NSF):**  Leveraging the Neural Source-Filter (NSF) allows for **direct manipulation and control over the fundamental frequency (F0)** of the synthesized speech, a significant advantage for various speech synthesis applications.