### MB2C: Multimodal Bidirectional Cycle Consistency for Learning Robust Visual Neural Representations
### Decoding human visual representations from brain activity data is a challenging but arguably essential task with an understanding of the real world and the human visual system. However, decoding semantically similar visual representations from brain recordings is difficult, especially for electroencephalography (EEG), which has excellent temporal resolution but suffers from spatial precision. Prevailing methods mainly focus on matching brain activity data with corresponding stimuli-responses using contrastive learning. They rely on massive and high-quality paired data and omit semantically aligned modalities distributed in distinct regions of the latent space. This paper proposes a novel Multimodal Bidirectional Cycle Consistency (MB2C) framework for learning robust visual neural representations. Specifically, we utilize dual-GAN
to generate modality-related features and inversely translate back to the corresponding semantic latent space to close the modality gap and guarantee that embeddings from different modalities with similar semantics are in the same region of representation space. We perform zero-shot tasks on the ThingsEEG dataset and EEG classification and image reconstruction tasks on the EEGCVPR40 dataset, achieving state-of-the-art performance compared to other baselines.

### Acknowledgment
We thank the following repos providing helpful components/functions in our work.
[BraVL](https://www.bing.com/ck/a?!&&p=e234990784f11875JmltdHM9MTcxODA2NDAwMCZpZ3VpZD0yOTVjNDMxYi1mYjBlLTZhMGEtMWZjNi01MmFiZmE2ODZiNTMmaW5zaWQ9NTE5OQ&ptn=3&ver=2&hsh=3&fclid=295c431b-fb0e-6a0a-1fc6-52abfa686b53&psq=bravl&u=a1aHR0cHM6Ly9naXRodWIuY29tL0NoYW5nZGVEdS9CcmFWTA&ntb=1)
[NICE](https://github.com/eeyhsong/NICE-EEG)
[ATM](https://github.com/dongyangli-del/EEG_Image_decode)

