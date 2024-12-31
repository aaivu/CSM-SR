import numpy as np

# Path to the .npy file
file_path = '/home/oshadi/SISR-Final_Year_Project/envs/SISR-Project/State-of-art-models/SRGAN-master/model/vgg19.npy'

# Load the .npy file
npz = np.load(file_path, allow_pickle=True, encoding='latin1')

# Print the keys and shapes of the arrays
for key, val in sorted(npz.items()):
    if isinstance(val, np.ndarray):
        print(f"Key: {key}, Shape: {val.shape}, Type: {type(val)}")
    else:
        print(f"Key: {key}, Type: {type(val)}")

# Check for unexpected data types
for key, val in sorted(npz.items()):
    if not isinstance(val, np.ndarray):
        print(f"Unexpected data type found for key: {key}, Type: {type(val)}")



import numpy as np

# Path to the .npy file
file_path = '/home/oshadi/SISR-Final_Year_Project/envs/SISR-Project/State-of-art-models/SRGAN-master/model/vgg19.npy'

# Load the .npy file
npz = np.load(file_path, allow_pickle=True, encoding='latin1')

# Inspect the contents of the vgg19 key
vgg19_data = npz['vgg19']

# Print the type and contents of the vgg19 key
print(f"Type of vgg19 data: {type(vgg19_data)}")
print(f"Contents of vgg19 data: {vgg19_data}")
