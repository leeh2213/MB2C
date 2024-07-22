### MB2C: Multimodal Bidirectional Cycle Consistency for Learning Robust Visual Neural Representations
### This paper has been accepted by **ACM MM2024(poster)**, more code details will be released soon!
### Abstract
Decoding human visual representations from brain activity data is a challenging but arguably essential task with an understanding of the real world and the human visual system. However, decoding semantically similar visual representations from brain recordings is difficult, especially for electroencephalography (EEG), which has excellent temporal resolution but suffers from spatial precision. Prevailing methods mainly focus on matching brain activity data with corresponding stimuli-responses using contrastive learning. They rely on massive and high-quality paired data and omit semantically aligned modalities distributed in distinct regions of the latent space. This paper proposes a novel Multimodal Bidirectional Cycle Consistency (MB2C) framework for learning robust visual neural representations. Specifically, we utilize dual-GAN to generate modality-related features and inversely translate back to the corresponding semantic latent space to close the modality gap and guarantee that embeddings from different modalities with similar semantics are in the same region of representation space. We perform zero-shot tasks on the ThingsEEG dataset and EEG classification and image reconstruction tasks on the EEGCVPR40 dataset, achieving state-of-the-art performance compared to other baselines. 
![framework](https://github.com/user-attachments/assets/1bc30602-b99b-487b-84c8-22510dfe5723)
