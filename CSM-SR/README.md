<!DOCTYPE html>
<html>
<head>
    <h1 align="center">CSM-SR:Conditional Structure-Informed Multi-Scale GAN for Scientific Image Super-Resolution</h1>
    
</head>
<body>
    
<div align="center">
    <h2>
        <a href="https://github.com/aaivu/Structure-informed-super-resolution-/tree/master/CSM-SR">
            <img src="https://img.shields.io/badge/Project-GitHub-coral" alt="Project">
        </a>
        <a href="https://drive.google.com/file/d/1nZgB0Zq6sPeGh3h5duNINIqRsGrdK2JM/view?usp=sharing">
            <img src="https://img.shields.io/badge/Survey Paper-Read%20Now-steelblue" alt="Survey Paper">
        </a>
        <a href="https://drive.google.com/file/d/1nAWR9VU8oc6-gqQ53amh0t0gOMbJzQhY/view?usp=sharing">
            <img src="https://img.shields.io/badge/Research%20Paper-Read%20Now-teal" alt="Research Paper">
        </a>
        <a href="https://github.com/aaivu/Structure-informed-super-resolution-/tree/master/State-of-art-models">
            <img src="https://img.shields.io/badge/Experiments-State of Art Models-slategray" alt="State of Art Models">
        </a>
        <a href="talk_forum_link">
            <img src="https://img.shields.io/badge/Talk Forum-Join us-lightyellow" alt="Talk Forum">
        </a>
    </h2>
</div>


<p align="justify">
The rapid advancements in scientific imaging, particularly in fields such as material science, medical imaging, and nanotechnology, have underscored the need for highly detailed images at the micro and nano scales. Traditional microscopic imaging techniques often face significant resolution limitations, leading to increased costs and insufficient detail for precise scientific analysis. Image Super-Resolution (SR) techniques have emerged as a promising solution, offering the ability to recover high-resolution images from low-resolution counterparts through advanced image processing methods. While deep learning-based SR models like <b>VDSR</b>--<a href="https://arxiv.org/abs/1511.04587">Very Deep Super-Resolution</a>--, <b>EDSR</b>--<a href="https://arxiv.org/abs/1707.02921">Enhanced Deep Residual Networks for Single Image Super-Resolution</a>--, GAN based <b>SRGAN</b>--<a href="https://arxiv.org/abs/1609.04802">Super-Resolution Generative Adversarial Network</a>--, <b>ESRGAN</b>--<a href="https://arxiv.org/abs/1809.00219">Enhanced Super-Resolution Generative Adversarial Networks</a>--, <b>SPSR</b>--<a href="https://arxiv.org/abs/2109.12530">Structure-Preserving Super Resolution with Gradient Guidance</a>--, variational autoencoder based <b>DSR-VAE</b>--<a href="https://arxiv.org/abs/2203.09445">Deep Super-Resolution Variational Autoencoder</a>--</b> and transformer based <b>SwinIR</b>--<a href="https://arxiv.org/abs/2108.10257">Swin Transformer for Image Restoration--</a>, <b>HMA-Net</b>--<a href="https://arxiv.org/pdf/2405.05001"HMANet: Hybrid Multi-Axis Aggregation Network for Image Super-Resolution--</a>, have demonstrated state-of-the-art performance in enhancing image resolution, they often fail to preserve the structural integrity crucial for accurate scientific analysis.
![Comparative Visual Images of SISR Techniques on state of art models](https://github.com/user-attachments/assets/fd30f563-7a54-4150-ad4c-ecfc790a076a)

To address this gap, the proposed approach integrates structural information using advanced conditional generative adversarial networks (cGANs) and a structure-informed convex loss function. This methodology is designed to improve both the visual quality and structural accuracy of super-resolved images. The research seeks to develop a super-resolution technique that not only enhances image quality but also preserves the structural integrity of image components, thereby facilitating more precise and realistic scientific analyses in fields such as material science and medical imaging.
</p>

<h2>CSM-SR</h2>
<pre>
<!-- Root folder for the CSM-SR project -->
|-- CSM-SR/
|   |-- README.md                        <!-- Project overview, instructions, and usage -->
|   |-- LICENSE                   <!-- Open-source or proprietary license details -->
|   |-- requirements.txt          <!-- Python dependencies -->
|   |-- setup.py                  <!-- Optional: Package setup script (if needed) -->
|   |-- configs/<!-- Configuration files for various components of the project -->
|   |   |-- dataset_config.json   <!-- Dataset configuration -->
|   |   |-- loss_function_config.json<!-- Configuration for loss functions -->
|   |   |-- model_architecture_config.json<!-- Model architecture configuration -->
|   |   |-- training_config.json<!-- Training setup configuration -->
|   |-- csm_sr_model.py       <!-- Primary model (formerly architecture_8.py) -->
|   |   |-- archived_architectures/<!-- Archived unused model architectures -->
|   |-- experiment/<!-- Generated outputs and logs -->
|   |   |-- generated_images/     <!-- Organized results (e.g., SR output images) -->
|   |   |-- model_checkpoints/    <!-- Optional: Saved model checkpoints -->
|   |   |-- logs/                 <!-- Training logs or output files -->
|   |-- loss_functions/<!-- Custom loss functions -->
|   |   |-- content_loss.py<!-- Mean squared error loss function -->
|   |   |-- structural_loss.py<!-- Custom defined loss function -->
|   |   |-- texture_loss.py
|   |-- evaluation/<!-- Evaluation scripts and utilities -->
|   |   |-- metrics.py            <!-- Evaluation metrics implementation -->
|   |   |-- results_visualizer.py <!-- Visualizations for evaluation (optional) -->
|   |-- utilities/
|   |   |-- callbacks.py          <!-- Training callbacks (e.g., early stopping) -->
|   |   |-- config_parser.py      <!-- Formerly config_loader.py -->
|   |   |-- data_preprocessor.py  <!-- Formerly data_loader.py -->
|   |   |-- lmdb_data_loader.py   <!-- Data loader for LMDB format -->
|   |-- setup.py 
|   |-- tests.py 
|   |-- train.py         <!-- Unit tests for key functionalities -->
</pre>
<h2>Installation</h2>
<ol>
<li>Clone the repository:</li>
<pre>
<code>
git clone <repository_url>
cd super-resolution
</code>
</pre>
<li>Create a virtual environment:</li>
<pre>
<code>
python -m venv myenv
source myenv/bin/activate  # On Windows use `myenv\Scripts\activate`
</code>
</pre>
<li>Install the dependencies:</li>
<pre>
<code>
pip install -r requirements.txt
</code>
</pre>
</ol>

<h2>Usage</h2>
<ol>
<li>Prepare your datasets: Ensure your datasets are placed in the paths specified in <code>train.json</code>.</li>
<li>Modify configurations: Adjust <code>model_config.json</code>, <code>train.json</code>, and <code>loss_config.json</code> as needed for your setup.</li>
<li>Run the training script:</li>
<pre>
<code>
python train.py
</code>
</pre>
</ol>

<h2>Model Architectures</h2>
<h3>Generator</h3>
<p>The generator model leverages residual blocks, attention mechanisms, and upsampling layers to enhance low-resolution images.</p>

<h3>Discriminator</h3>
<p>The discriminator model uses a hybrid of convolutional layers and feature pyramid networks to distinguish between real and generated high-resolution images.</p>

<h2>Monitoring and Evaluation</h2>
<ol>
<li>TensorBoard: TensorBoard is used for monitoring the training process. Logs are saved in the logs directory specified in <code>train.json</code>.
<pre>
<code>
tensorboard --logdir=path/to/logs
</code>
</pre>
</li>
<li>Generated Images: Generated super-resolution images are saved in the path specified in <code>train.json</code> at each epoch.</li>
</ol>


<h2>Project Phases</h2>
<p align="justify">
<b>1. Data Collection and Preprocessing:</b><br>
<b>Objective:</b> Collect and preprocess a comprehensive set of scanning electron microscopy (SEM) images, including the SEM dataset, Hierarchical dataset, Majority dataset, and 100% dataset.<br>
<div align="center">
    <img src=https://github.com/user-attachments/assets/91718064-fe3c-4658-93e3-f7bb0c4705a0 alt="Electron Microscope Images">
    <p>Figure: Electron microscope images showcasing various textures, patterns, and structures at microscopic scales.</p>
</div>
<b>Outcome:</b> A clean and well-annotated dataset ready for training and evaluation purposes.<br><br>

<b>2. Evaluation of State-of-the-Art Models:</b><br>
<b>Objective:</b> Conduct experiments using existing super-resolution models (SRGAN, ESRGAN, VDSR, SPSR, dSRVAE, SwinIR) on the collected datasets to establish baseline performance metrics.<br>
<div align="center">
    <img src="https://github.com/user-attachments/assets/fd30f563-7a54-4150-ad4c-ecfc790a076a" alt="Results of Comparison With Different State-of-art Methods on the SEM-Dataset">
    
</div>
<b>Outcome:</b> Experimental results providing insights into the strengths and limitations of current models.<br><br>

<b>3.1: Development of the Advance-Conditional Multi-Scale GAN Model:</b><br>
<b>1. Model Architecture Design</b><br>
<b>Generator Design:</b> The generator model comprises advanced super-resolution residual blocks and attention blocks, coupled with multi-scale processing to enhance feature extraction capabilities.
![The Generator Architecture of CSM-SR for image super-resolution](https://github.com/user-attachments/assets/61da11e1-b16b-4c4f-bbb0-93c65bbce398)
![Feature Conditionning Encoder Network (FCEN)](https://github.com/user-attachments/assets/a7a2a398-9655-489e-a2b7-b405c3da0671)
![Feature Processing Block](https://github.com/user-attachments/assets/ee0ccdfb-368b-4b57-8b9f-8c69296a5d8c)
<br>
<b>Discriminator Design:</b> The discriminator employs a combination of residual blocks and PatchGAN-style convolutions to effectively differentiate between real and generated images.<br>
![The Discriminator Architecture of CSM-SR](https://github.com/user-attachments/assets/0325a699-ba52-4570-b762-9e0b4eb69e00)

<b>2. Integration of Structure-Informed Loss Function</b><br>
<b>Loss Components:</b>
<ul>
    <li><b>Adversarial Loss:</b> Utilizes binary cross-entropy to train the GAN, ensuring realistic image generation.</li>
    <li><b>Perceptual Loss:</b> Employs a pre-trained VGG19 model to compare high-level features between the ground truth and generated images.</li>
    <li><b>Gradient Loss:</b> Computes the difference in Sobel edges between the ground truth and generated images to maintain edge information.</li>
    <li><b>Second-Order Gradient Loss:</b> Extends the gradient loss by considering second-order gradients, further enhancing edge preservation.</li>
    <li><b>Total Variation Loss:</b> Encourages spatial smoothness in the generated images.</li>
    <li><b>Structural Similarity Loss:</b> Measures the structural similarity between ground truth and generated images.</li>
</ul>
<b>Total Loss Calculation:</b> Combines all the above loss components into a single, comprehensive loss function for training.<br><br>

<b>4. Model Training and Evaluation</b><br>
<b>Objective:</b> Train the SINSR model using the structure-informed loss function and evaluate its performance on the collected dataset.<br>
<b>Outcome:</b> Detailed performance metrics, comparisons with baseline models, and qualitative analyses of the generated high-resolution images.<br><br>

<b>5. Advanced Model Development<b><br>
<b>Objective:</b> Integrate additional architectural advancements, such as incorporating a Gradient Branch alongside the super-resolution branch, operating in parallel.<br>
<b>Outcome:</b> An enhanced SINSR model architecture that significantly improves performance in maintaining structural integrity in high-resolution images.<br><br>

<b>6. Final Evaluation and Analysis<b><br>
<b>Objective:</b> Conduct a comprehensive evaluation of the final SINSR model, incorporating both quantitative metrics and qualitative analyses.<br>

<b>1. Quantitative Metrics<b>
<ul>
    <li>PSNR</li>
    <li>SSIM</li>
    <li>LPIPS</li>
</ul>
<b>Outcome:</b> A fully developed and evaluated SINSR model, ready for application in downstream scientific analyses.
</p>



<!--
<h3>Project Lead</h3>
<ul>
    <li>
        Dr. Uthayasanker Thayasivam
        <a href="https://github.com/github_profile_link">
            <img src="https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=github&logoColor=white" alt="GitHub" width="80" height="20"/>
        </a>
        <a href="https://linkedin.com/linkedin_profile_link">
            <img src="https://img.shields.io/badge/LinkedIn-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white" alt="LinkedIn" width="80" height="20"/>
        </a>
    </li>
</ul>

<h4>Mentor(s)</h4>
<ul>
    <li>
        Brinthan, Vithurabhiman
        <a href="https://github.com/github_profile_link">
            <img src="https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=github&logoColor=white" alt="GitHub" width="80" height="20"/>
        </a>
        <a href="https://linkedin.com/linkedin_profile_link">
            <img src="https://img.shields.io/badge/LinkedIn-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white" alt="LinkedIn" width="80" height="20"/>
        </a>
    </li>
</ul>

<h4>Contributor(s)</h4>
<ul>
    <li>
        Randika Prabashwara
        <a href="https://github.com/github_profile_link">
            <img src="https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=github&logoColor=white" alt="GitHub" width="80" height="20"/>
        </a>
        <a href="https://linkedin.com/linkedin_profile_link">
            <img src="https://img.shields.io/badge/LinkedIn-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white" alt="LinkedIn" width="80" height="20"/>
        </a>
    </li>
    <li>
        <a href="https://github.com/Gayani2001">Gayani Wickramarathna</a>
        <a href="https://linkedin.com/linkedin_profile_link">
            <img src="https://img.shields.io/badge/LinkedIn-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white" alt="LinkedIn" width="80" height="20"/>
        </a>
    </li>
    <li>
        Oshadi Perera
        <a href="https://github.com/github_profile_link">
            <img src="https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=github&logoColor=white" alt="GitHub" width="80" height="20"/>
        </a>
        <a href="https://linkedin.com/linkedin_profile_link">
            <img src="https://img.shields.io/badge/LinkedIn-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white" alt="LinkedIn" width="80" height="20"/>
        </a>
    </li>
</ul>
-->
<table>
    <tr>
        <th>🎓 Role</th>
        <th>👲 Name</th>
        <th>🔗 GitHub</th>
        <th>🔗 LinkedIn</th>
    </tr>
    <tr>
        <td>Project Lead</td>
        <td>Dr. Uthayasanker Thayasivam</td>
        <td><a href="https://github.com/github_profile_link"><img src="https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=github&logoColor=white" alt="GitHub" width="80" height="20"/></a></td>
        <td><a href="https://www.linkedin.com/in/rtuthaya/"><img src="https://img.shields.io/badge/LinkedIn-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white" alt="LinkedIn" width="80" height="20"/></a></td>
    </tr>
    <tr>
        <td>Mentor</td>
        <td>Brinthan, Vithurabhiman</td>
        <td><a href="https://github.com/github_profile_link"><img src="https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=github&logoColor=white" alt="GitHub" width="80" height="20"/></a></td>
        <td><a href="https://linkedin.com/linkedin_profile_link"><img src="https://img.shields.io/badge/LinkedIn-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white" alt="LinkedIn" width="80" height="20"/></a></td>
    </tr>
    <tr>
        <td>Contributor</td>
        <td>Randika Prabashwara</td>
        <td><a href="https://github.com/randikapra"><img src="https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=github&logoColor=white" alt="GitHub" width="80" height="20"/></a></td>
        <td><a href="https://www.linkedin.com/in/randika-prabashwara-739bba237/"><img src="https://img.shields.io/badge/LinkedIn-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white" alt="LinkedIn" width="80" height="20"/></a></td>
    </tr>
    <tr>
        <td>Contributor</td>
        <td>Gayani Wickramarathna</a></td>
        <td><a href="https://github.com/Gayani2001"><img src="https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=github&logoColor=white" alt="GitHub" width="80" height="20"/></a></td>
        <td><a href="https://www.linkedin.com/in/gwickramarathna/"><img src="https://img.shields.io/badge/LinkedIn-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white" alt="LinkedIn" width="80" height="20"/></a></td>
    </tr>
    <tr>
        <td>Contributor</td>
        <td>Oshadi Perera</td>
        <td><a href="https://github.com/Oshadi20"><img src="https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=github&logoColor=white" alt="GitHub" width="80" height="20"/></a></td>
        <td><a href="https://www.linkedin.com/in/gwickramarathna/"><img src="https://img.shields.io/badge/LinkedIn-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white" alt="LinkedIn" width="80" height="20"/></a></td>
    </tr>
</table>

<p align="center">
    <a href="https://github.com/aaivu/aaivu-introduction/blob/master/docs/code_of_conduct.md">
        <img src="https://img.shields.io/badge/Code of Conduct-Please Read-blue" alt="release"/>
    </a>
    <img src="https://img.shields.io/badge/release-v1.0.0-blue" alt="release"/>
    <a href="https://github.com/aaivu/Structure-informed-super-resolution-/blob/master/LICENSE"><img src="https://img.shields.io/badge/License-Apache License 2.0-blue" alt="Apache License 2.0"/></a>
</p>


</body>
</html>


