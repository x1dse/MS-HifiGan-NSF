# Multi-stream HiFi-GAN NSF: Efficient and High-Fidelity Speech Synthesis with Data-Driven Waveform Decomposition

This repository contains a PyTorch implementation of a modified HiFi-GAN generator, incorporating the **multi-stream architecture** and **sub-pixel convolution upsampling** proposed in the paper "[MULTI-STREAM HIFI-GAN WITH DATA-DRIVEN WAVEFORM DECOMPOSITION](https://ast-astrec.nict.go.jp/release/preprints/preprint_asru_2021_okamoto.pdf)" by Okamoto et al. (ASRU 2021).  This implementation builds upon the Neural Source-Filter (NSF) variant of HiFi-GAN, further enhancing its capabilities.

**Abstract of the Original HiFi-GAN Paper (for context):**

> Several recent work on speech synthesis have employed generative adversarial networks (GANs) to produce raw waveforms. Although such methods improve the sampling efficiency and memory usage, their sample quality has not yet reached that of autoregressive and flow-based generative models. In this work, we propose HiFi-GAN, which achieves both efficient and high-fidelity speech synthesis. As speech audio consists of sinusoidal signals with various periods, we demonstrate that modeling periodic patterns of an audio is crucial for enhancing sample quality. A subjective human evaluation (mean opinion score, MOS) of a single speaker dataset indicates that our proposed method demonstrates similarity to human quality while generating 22.05 kHz high-fidelity audio 167.9 times faster than real-time on a single V100 GPU. We further show the generality of HiFi-GAN to the mel-spectrogram inversion of unseen speakers and end-to-end speech synthesis. Finally, a small footprint version of HiFi-GAN generates samples 13.4 times faster than real-time on CPU with comparable quality to an autoregressive counterpart.

**Key Modifications (Multi-stream HiFi-GAN NSF):**

This implementation incorporates the following key modifications to the standard HiFi-GAN NSF architecture, as described in the Okamoto et al. paper:

*   **Multi-stream Architecture:**  Instead of generating a single waveform directly, the generator produces multiple *streams* of waveforms. These streams are then combined using a trainable convolutional layer (`stream_combiner`). This allows the model to learn a data-driven decomposition of the waveform, improving synthesis quality and/or efficiency.  This replaces the fixed synthesis filter bank used in traditional multi-band approaches.

*   **Sub-pixel Convolution Upsampling:**  Traditional transposed convolutions (`ConvTranspose1d`) are replaced with sub-pixel convolutions (`nn.Conv1d` followed by `nn.PixelShuffle`).  Sub-pixel convolution has been shown to reduce artifacts in generated audio, leading to higher fidelity.

*   **Neural Source-Filter (NSF) Integration:** The generator uses a Neural Source-Filter model, meaning it takes the fundamental frequency (F0) sequence as an explicit input, in addition to the mel-spectrogram, to add a signal for noise.

**Benefits:**

*   **Improved Inference Speed:** The sub-pixel convolution upsampling is generally more efficient than transposed convolution, leading to faster waveform generation.
*   **Enhanced Audio Quality:** The multi-stream architecture and sub-pixel convolution, combined, allow the model to learn more complex and detailed representations of the audio signal, potentially leading to improved perceptual quality (as demonstrated in the paper).
*   **Data-Driven Decomposition:**  The model learns the optimal way to decompose the waveform into streams, rather than relying on a predefined fixed filter bank.
* **Explicit control over F0** The use of NSF, allow a better manipulation over the audio
