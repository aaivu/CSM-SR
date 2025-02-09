<p style="font-size:24px;"><b>Structure-Informed Convex Loss function for Super-Resolution Image Generation Microscopic-Scintific Images (24px)</b></p>

This project aims to develop a deep learning model for super-resolution image generation using a combination of convolutional neural networks (CNNs), residual blocks, attention mechanisms, and upsampling layers. The project leverages both TensorFlow and PyTorch for data management, model architecture, training, and evaluation.

## Project Structure
<pre> |-- architectures/
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
|-- mtrics/
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
|-- requirements.txt </pre>

### Installation
1. Clone the repository:
bash
>> git clone 
>> cd super-resolution
2. Create a virtual environment:
bash
>> python -m venv myenv
>> source myenv/bin/activate  # On Windows use `myenv\Scripts\activate`
3. Install the dependencies:
bash
>> pip install -r requirements.txt

### Usage
1. Prepare your datasets:
Ensure your datasets are placed in the paths specified in train.json.

2. Modify configurations:
Adjust model_config.json, train.json, and loss_config.json as needed for your setup.

3. Run the training script:
bash
python train.py

## Model Architectures
### Generator
The generator model leverages residual blocks, attention mechanisms, and upsampling layers to enhance low-resolution images.
### Discriminator
The discriminator model uses a hybrid of convolutional layers and feature pyramid networks to distinguish between real and generated high-resolution images.

## Monitoring and Evaluation
1. TensorBoard:
TensorBoard is used for monitoring the training process. Logs are saved in the logs directory specified in train.json.
bash
>> tensorboard --logdir=path/to/logs
2. Generated Images:
Generated super-resolution images are saved in the path specified in train.json at each epoch.

## Contributions
Contributions are welcome! Please feel free to submit a Pull Request or open an issue.

## License
This project is licensed under the MIT License. See the LICENSE file for more details.
