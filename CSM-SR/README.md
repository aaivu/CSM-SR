<!DOCTYPE html>
<html>
<head>
<h1><b>CSM-SR: Conditional Structure-Informed Multi-Scale GAN for Scientific Image Super-Resolution</b></h1>

<p>This project aims to develop a deep learning model for super-resolution image generation using a combination of convolutional neural networks (CNNs), residual blocks, attention mechanisms, and upsampling layers. The project leverages both TensorFlow and PyTorch for data management, model architecture, training, and evaluation.</p>

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
|   |-- models/<!-- Folder for model files -->
|   |   |-- csm_sr_model.py       <!-- Primary model (formerly architecture_8.py) -->
|   |   |-- archived_architectures/<!-- Archived unused model architectures -->
|       |-- architecture_1.py<!-- Experimental architecture 1 -->
|       |-- architecture_2.py<!-- Experimental architecture 2 -->
|       |-- ...
|       |-- architecture_7.py<!-- Experimental architecture 7 -->
|   |-- outputs/<!-- Generated outputs and logs -->
|   |   |-- generated_images/     <!-- Organized results (e.g., SR output images) -->
|   |   |-- model_checkpoints/    <!-- Optional: Saved model checkpoints -->
|   |   |-- logs/                 <!-- Training logs or output files -->
|   |-- loss_functions/<!-- Custom loss functions -->
|   |   |-- mse_loss.py<!-- Mean squared error loss function -->
|   |   |-- custom_loss.py<!-- Custom defined loss function -->
|   |-- evaluation/<!-- Evaluation scripts and utilities -->
|   |   |-- metrics.py            <!-- Evaluation metrics implementation -->
|   |   |-- results_visualizer.py <!-- Visualizations for evaluation (optional) -->
|   |-- utilities/
|   |   <!-- Utility scripts for various tasks -->
|   |   |-- callbacks.py          <!-- Training callbacks (e.g., early stopping) -->
|   |   |-- config_parser.py      <!-- Formerly config_loader.py -->
|   |   |-- data_preprocessor.py  <!-- Formerly data_loader.py -->
|   |   |-- lmdb_data_loader.py   <!-- Data loader for LMDB format -->
|   |-- tests/ <!-- Unit testing folder -->
|       |-- unit_tests.py         <!-- Unit tests for key functionalities -->
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

<h2>Contributions</h2>
<p>Contributions are welcome! Please feel free to submit a Pull Request or open an issue.</p>

<h2>License</h2>
<p>This project is licensed under the MIT License. See the <code>LICENSE</code> file for more details.</p>

</body>
</html>
