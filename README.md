
<!DOCTYPE html>
<html>
<head>
    <title>Structure-informed super resolution technique for scientific imaging</title>
</head>
<body>
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
<h1 style="font-family: 'Times New Roman', Times, serif; font-size: 30px; color: blue;">
    <b>Summary</b>
</h1>

<p style="font-family: Arial, sans-serif; font-size: 16px; color: black;">
    Microscopic imaging is essential for examining materials at micro and nano scales, yet it often faces challenges with resolution limitations, leading to longer acquisition times and increased costs for high-resolution images. Traditional super-resolution techniques like SRGAN, ESRGAN, and SwinIR enhance low-resolution images but frequently fail to maintain the structural integrity of the image components, which is crucial for accurate quantitative analysis. This project addresses this limitation by integrating structural information into super-resolution techniques, leveraging conditional generative adversarial networks (GANs) and optimizing them with a structure-informed convex loss function.
</p>

<p style="font-family: Arial, sans-serif; font-size: 16px; color: black;">
    The project involves multiple phases, starting with the collection and preprocessing of an extensive set of scanning electron microscopy (SEM) images. It then evaluates existing state-of-the-art super-resolution models to establish baseline performance metrics. The core of the research is the development of the SINSR model, which features an advanced generator and discriminator architecture, coupled with a multi-component loss function designed to preserve structural integrity. The model is trained and evaluated on the collected dataset, and further architectural advancements, such as a parallel Gradient Branch, are explored to enhance performance. The final SINSR model is rigorously evaluated to ensure it significantly improves the quality of super-resolution images, facilitating more accurate scientific analyses.
</p>

<p style="font-family: Arial, sans-serif; font-size: 16px; color: black;">
    Through this innovative approach, the project aims to revolutionize microscopic imaging by producing high-resolution images that retain structural integrity, thus enabling more reliable and realistic downstream scientific analyses. The integration of conditional GANs and a structure-informed loss function represents a significant advancement in super-resolution techniques, paving the way for enhanced accuracy in various scientific and research applications.
</p>

<h1 style="font-family: 'Times New Roman', Times, serif; font-size: 30px; color: blue;">
    <b>Description</b>
</h1>

<h2 style="font-family: Arial, sans-serif; font-size: 20px; color: black;">Overview</h2>
<p style="font-family: Arial, sans-serif; font-size: 16px; color: black;">
    Microscopic imaging is indispensable for investigating the structural and functional properties of materials at micro and nano scales. Despite its critical role, the process often suffers from resolution limitations, resulting in prolonged acquisition times and higher costs for obtaining high-resolution (HR) images. Super-resolution techniques, such as SRGAN, ESRGAN, and SwinIR, have been developed to address these challenges by enhancing low-resolution (LR) images. However, these methods often overlook the preservation of structural integrity in image components, which is crucial for downstream quantitative analyses.
</p>

<p style="font-family: Arial, sans-serif; font-size: 16px; color: black;">
    This research project proposes a novel approach to integrate structural information into super-resolution techniques. By leveraging the capabilities of conditional generative adversarial networks (GANs) and optimizing them with a structure-informed convex loss function, the objective is to enhance the quality of super-resolution images. This enhancement aims to facilitate more accurate and realistic downstream scientific analyses.
</p>

<h2 style="font-family: Arial, sans-serif; font-size: 20px; color: black;">Project Phases</h2>

<h3 style="font-family: Arial, sans-serif; font-size: 18px; color: black;">Phase 1: Data Collection and Preprocessing</h3>
<p style="font-family: Arial, sans-serif; font-size: 16px; color: black;">
    <b>Objective:</b> Collect and preprocess a comprehensive set of scanning electron microscopy (SEM) images, including the SEM dataset, Hierarchical dataset, Majority dataset, and 100% dataset.
</p>
<p style="font-family: Arial, sans-serif; font-size: 16px; color: black;">
    <b>Outcome:</b> A clean and well-annotated dataset ready for training and evaluation purposes.
</p>

<h3 style="font-family: Arial, sans-serif; font-size: 18px; color: black;">Phase 2: Evaluation of State-of-the-Art Models</h3>
<p style="font-family: Arial, sans-serif; font-size: 16px; color: black;">
    <b>Objective:</b> Conduct experiments using existing super-resolution models (SRGAN, ESRGAN, VDSR, SPSR, dSRVAE) on the collected datasets to establish baseline performance metrics.
</p>
<p style="font-family: Arial, sans-serif; font-size: 16px; color: black;">
    <b>Outcome:</b> Experimental results providing insights into the strengths and limitations of current models.
</p>

<h3 style="font-family: Arial, sans-serif; font-size: 18px; color: black;">Phase 3.1: Development of the SINSR Model</h3>

<h4 style="font-family: Arial, sans-serif; font-size: 16px; color: black;">1. Model Architecture Design</h4>
<p style="font-family: Arial, sans-serif; font-size: 16px; color: black;">
    <b>Generator Design:</b> The generator model comprises advanced super-resolution residual blocks and attention blocks, coupled with multi-scale processing to enhance feature extraction capabilities.
</p>
<p style="font-family: Arial, sans-serif; font-size: 16px; color: black;">
    <b>Discriminator Design:</b> The discriminator employs a combination of residual blocks and PatchGAN-style convolutions to effectively differentiate between real and generated images.
</p>

<h4 style="font-family: Arial, sans-serif; font-size: 16px; color: black;">2. Integration of Structure Informed Loss Function</h4>
<p style="font-family: Arial, sans-serif; font-size: 16px; color: black;">
    <b>Loss Components:</b>
</p>
<ul style="font-family: Arial, sans-serif; font-size: 16px; color: black;">
    <li><b>Adversarial Loss:</b> Utilizes binary cross-entropy to train the GAN, ensuring realistic image generation.</li>
    <li><b>Perceptual Loss:</b> Employs a pre-trained VGG19 model to compare high-level features between the ground truth and generated images.</li>
    <li><b>Gradient Loss:</b> Computes the difference in Sobel edges between the ground truth and generated images to maintain edge information.</li>
    <li><b>Second-Order Gradient Loss:</b> Extends the gradient loss by considering second-order gradients, further enhancing edge preservation.</li>
    <li><b>Total Variation Loss:</b> Encourages spatial smoothness in the generated images.</li>
    <li><b>Structural Similarity Loss:</b> Measures the structural similarity between ground truth and generated images.</li>
</ul>
<p style="font-family: Arial, sans-serif; font-size: 16px; color: black;">
    <b>Total Loss Calculation:</b> Combines all the above loss components into a single, comprehensive loss function for training.
</p>

<h3 style="font-family: Arial, sans-serif; font-size: 18px; color: black;">Phase 4: Model Training and Evaluation</h3>
<p style="font-family: Arial, sans-serif; font-size: 16px; color: black;">
    <b>Objective:</b> Train the SINSR model using the structure-informed loss function and evaluate its performance on the collected dataset.
</p>
<p style="font-family: Arial, sans-serif; font-size: 16px; color: black;">
    <b>Outcome:</b> Detailed performance metrics, comparisons with baseline models, and qualitative analyses of the generated HR images.
</p>

<h3 style="font-family: Arial, sans-serif; font-size: 18px; color: black;">Phase 5: Advanced Model Development</h3>
<p style="font-family: Arial, sans-serif; font-size: 16px; color: black;">
    <b>Objective:</b> Integrate additional architectural advancements, such as incorporating a Gradient Branch alongside the SR branch, operating in parallel.
</p>
<p style="font-family: Arial, sans-serif; font-size: 16px; color: black;">
    <b>Outcome:</b> An enhanced SINSR model architecture that significantly improves performance in maintaining structural integrity in HR images.
</p>

<h3 style="font-family: Arial, sans-serif; font-size: 18px; color: black;">Phase 6: Final Evaluation and Analysis</h3>
<p style="font-family: Arial, sans-serif; font-size: 16px; color: black;">
    <b>Objective:</b> Conduct a comprehensive evaluation of the final SINSR model, incorporating both quantitative metrics and qualitative analyses.
</p>
<ul style="font-family: Arial, sans-serif; font-size: 16px; color: black;">
    <li>1. Quantitative metrics</li>
    <ul>
        <li>PSNR</li>
        <li>SSIM</li>
        <li>LPIPS</li>
    </ul>
</ul>
<ul style="font-family: Arial, sans-serif; font-size: 16px; color: black;">
    <li>Outcome: A fully developed and evaluated SINSR model, ready for application in downstream scientific analyses.</li>
</ul>

<h2 style="font-family: Arial, sans-serif; font-size: 20px; color: black;">More References</h2>
<ol style="font-family: Arial, sans-serif; font-size: 16px; color: black;">
    <li>Reference</li>
    <li>Link</li>
    <li>Dataset 1 : <a href="https://b2share.eudat.eu/records/19cc2afd23e34b92b36a1dfd0113a89f">NFFA-EUROPE - SEM Dataset</a></li>
</ol>

<hr style="border: 1px solid black;">

<h3 style="font-family: Arial, sans-serif; font-size: 18px; color: black;">License</h3>
<p style="font-family: Arial, sans-serif; font-size: 16px; color: black;">Apache License 2.0</p>

<h3 style="font-family: Arial, sans-serif; font-size: 18px; color: black;">Code of Conduct</h3>
<p style="font-family: Arial, sans-serif; font-size: 16px; color: black;">Please read our <a href="https://github.com/aaivu/aaivu-introduction/blob/master/docs/code_of_conduct.md">code of conduct document here</a>.</p>

<img src="https://img.shields.io/badge/-Project-blue" alt="Project">
<img src="https://img.shields.io/badge/-Research-yellowgreen" alt="Research">

</body>
</html>


