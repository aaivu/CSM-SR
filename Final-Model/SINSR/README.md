<!DOCTYPE html>
<html>
<head>
<style>
.custom-heading {
    font-family: "Times New Roman", Times, serif;
    font-size: 30px;
    color: blue;
}
</style>
</head>
<body>

<h1 class="custom-heading"><b>Structure-Informed Convex Loss function for Super-Resolution Image Generation Microscopic-Scientific Images</b></h1>

<p>This project aims to develop a deep learning model for super-resolution image generation using a combination of convolutional neural networks (CNNs), residual blocks, attention mechanisms, and upsampling layers. The project leverages both TensorFlow and PyTorch for data management, model architecture, training, and evaluation.</p>

<h2>Project Structure</h2>
<pre>
|-- architectures/
|   |-- architecture_1.py
|   |-- architecture_2.py
|   |-- architecture_3.py
|-- config/
|   |-- data_config.json
|   |-- loss_config.json
|   |-- model_config.json
|   |-- train.json
|-- data/  # This folder is ignored by .gitignore
|-- loss_functions/
|   |-- loss_functions.py
|-- metrics/
|   |-- evaluation_metrics.py
|-- utils/
|   |-- callbacks.py
|   |-- config_loader.py
|   |-- data_loader.py
|   |-- data_loader_lmdb.py
|   |-- model_setup.py
|   |-- train_util.py
|   |-- training_functions.py
|   |-- utilities.py
|-- test.py
|-- train.py
|-- README.md
|-- requirements.txt
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
