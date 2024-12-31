# Project Title / Research Title

![project] ![research]



- <b>Project Lead(s) / Mentor(s)</b>
    1. Name (talk forum profile link)
    2. Name (talk forum profile link)
- <b>Contributor(s)</b>
    1. Name (talk forum profile link)
    2. Name (talk forum profile link)

<b>Useful Links </b>

- GitHub : <project_url>
- Talk Forum : <talk_forum_link>

---

## Summary

two - three paragraphs about your project.

## Description

### Overview
Microscopic imaging is indispensable for investigating the structural and functional properties of materials at micro and nano scales. Despite its critical role, the process often suffers from resolution limitations, resulting in prolonged acquisition times and higher costs for obtaining high-resolution (HR) images. Super-resolution techniques, such as SRGAN, ESRGAN, and SwinIR, have been developed to address these challenges by enhancing low-resolution (LR) images. However, these methods often overlook the preservation of structural integrity in image components, which is crucial for downstream quantitative analyses.

This research project proposes a novel approach to integrate structural information into super-resolution techniques. By leveraging the capabilities of conditional generative adversarial networks (GANs) and optimizing them with a structure-informed convex loss function, the objective is to enhance the quality of super-resolution images. This enhancement aims to facilitate more accurate and realistic downstream scientific analyses.


### Project phases

#### Phase 1: Data Collection and Preprocessing </b>
<b> Objective: Collect and preprocess a comprehensive set of scanning electron microscopy (SEM) images, including the SEM dataset, Hierarchical dataset, Majority dataset, and 100% dataset.
<b> Outcome: A clean and well-annotated dataset ready for training and evaluation purposes.

#### Phase 2: Evaluation of State-of-the-Art Models
<b> Objective:</b> Conduct experiments using existing super-resolution models (SRGAN, ESRGAN, VDSR, SPSR, dSRVAE) on the collected datasets to establish baseline performance metrics.
<b> Outcome:</b> Experimental results providing insights into the strengths and limitations of current models.

#### Phase 3.1: Development of the SINSR Model
##### 1. Model Architecture Design
<b> Generator Design:</b> The generator model comprises advanced super-resolution residual blocks and attention blocks, coupled with multi-scale processing to enhance feature extraction capabilities.
<b> Discriminator Design:</b> The discriminator employs a combination of residual blocks and PatchGAN-style convolutions to effectively differentiate between real and generated images.

##### 2. Integration of Structure Informed Loss Function
<b>Loss Components:</b>
<b>Adversarial Loss:</b> Utilizes binary cross-entropy to train the GAN, ensuring realistic image generation.
<b>Perceptual Loss:</b> Employs a pre-trained VGG19 model to compare high-level features between the ground truth and generated images.
<b>Gradient Loss:</b> Computes the difference in Sobel edges between the ground truth and generated images to maintain edge information.
<b>Second-Order Gradient Loss:</b> Extends the gradient loss by considering second-order gradients, further enhancing edge preservation.
<b>Total Variation Loss:</b> Encourages spatial smoothness in the generated images.
<b>Structural Similarity Loss:</b> Measures the structural similarity between ground truth and generated images.

<b>Total Loss Calculation:</b> Combines all the above loss components into a single, comprehensive loss function for training.

#### Phase 4: Model Training and Evaluation
<b>Objective:</b> Train the SINSR model using the structure-informed loss function and evaluate its performance on the collected dataset.
<b>Outcome:</b> Detailed performance metrics, comparisons with baseline models, and qualitative analyses of the generated HR images.

#### Phase 5: Advanced Model Development
<b>Objective:</b> Integrate additional architectural advancements, such as incorporating a Gradient Branch alongside the SR branch, operating in parallel.
<b>Outcome:</b> An enhanced SINSR model architecture that significantly improves performance in maintaining structural integrity in HR images.

#### Phase 6: Final Evaluation and Analysis
<b>Objective:</b> Conduct a comprehensive evaluation of the final SINSR model, incorporating both quantitative metrics and qualitative analyses.
##### 1. Quantitative metrics
<b>1. PSNR</b>
<b>2. SSIM</b>
<b>3. LPIPS</b>

##### 2. Quantitative metrics
Outcome: A fully developed and evaluated SINSR model, ready for application in downstream scientific analyses.

- Diagrams
- Approches

## More references

1. Reference
2. Link

---

### License

Apache License 2.0

### Code of Conduct

Please read our [code of conduct document here](https://github.com/aaivu/aaivu-introduction/blob/master/docs/code_of_conduct.md).

[project]: https://img.shields.io/badge/-Project-blue
[research]: https://img.shields.io/badge/-Research-yellowgreen
