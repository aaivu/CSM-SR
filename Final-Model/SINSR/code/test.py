# from tensorflow.keras.models import load_model # type: ignore
# from PIL import Image
# import numpy as np
# import os

# # Load the trained model
# model_path = '/home/oshadi/SISR-Final_Year_Project/envs/SISR-Project/Final-Model/SINSR/models/generator7_t1_7d/best_model.keras'
# model = load_model(model_path, safe_mode=False)  # Load the model with safe mode disabled

# # Load the LR image
# lr_image_path = '/home/oshadi/SISR-Final_Year_Project/envs/SISR-Project/data/DIV2K/DIV2K_train_LR_bicubic/X4/L9_00fc0a86bd4f02995acdd5b3f63401b9x4.jpg'
# lr_image = Image.open(lr_image_path)
# # lr_image = lr_image.resize((192, 256))  # Resize to match your model input shape if necessary
# lr_image_array = np.array(lr_image) / 255.0  # Normalize the image
# lr_image_array = np.expand_dims(lr_image_array, axis=0)  # Add batch dimension

# # Generate the SR image
# sr_image_array = model.predict(lr_image_array)
# sr_image_array = np.squeeze(sr_image_array, axis=0)  # Remove batch dimension
# sr_image_array = (sr_image_array * 255).astype(np.uint8)  # Denormalize the image
# sr_image = Image.fromarray(sr_image_array)

# # Save the SR image
# save_dir = '/home/oshadi/SISR-Final_Year_Project/envs/SISR-Project/Final-Model/SINSR/code/generated_images'
# if not os.path.exists(save_dir):
#     os.makedirs(save_dir)
# sr_image_path = os.path.join(save_dir, 'sr_image.png')
# sr_image.save(sr_image_path)

# print(f'Super-Resolution image saved at: {sr_image_path}')


from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import os

# Load the trained model
model_path = '/home/oshadi/SISR-Final_Year_Project/envs/SISR-Project/Final-Model/SINSR/models/generator1_t1/best_model.keras'
model = load_model(model_path, safe_mode=False)  # Load the model with safe mode disabled

# Load the LR image
lr_image_path = '/home/oshadi/SISR-Final_Year_Project/envs/SISR-Project/data/DIV2K/DIV2K_train_LR_bicubic/X4/L9_ffaa4e2ca12ad56fe542cbcdb0de6b5bx4.jpg'
lr_image = Image.open(lr_image_path)
# lr_image = lr_image.resize((192, 256))  # Resize if necessary
lr_image_array = np.array(lr_image).astype(np.float32) / 127.5 - 1  # Normalize to [-1, 1]
lr_image_array = np.expand_dims(lr_image_array, axis=0)  # Add batch dimension

# Generate the SR image
sr_image_array = model.predict(lr_image_array)
sr_image_array = np.squeeze(sr_image_array, axis=0)  # Remove batch dimension
sr_image_array = (sr_image_array + 1) * 127.5  # Denormalize from [-1, 1] to [0, 255]
sr_image_array = np.clip(sr_image_array, 0, 255).astype(np.uint8)  # Ensure values are within [0, 255]
sr_image = Image.fromarray(sr_image_array)

# Save the SR image
save_dir = '/home/oshadi/SISR-Final_Year_Project/envs/SISR-Project/Final-Model/SINSR/code/generated_images'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
sr_image_path = os.path.join(save_dir, 'sr_image.png')
sr_image.save(sr_image_path)

print(f'Super-Resolution image saved at: {sr_image_path}')
