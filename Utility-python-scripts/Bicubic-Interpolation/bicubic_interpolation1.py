
# import os
# import cv2
# import concurrent.futures
# from sklearn.model_selection import train_test_split

# # Define paths
# dataset_dir = '/home/oshadi/SISR-Final_Year_Project/envs/SISR-Project/data/Scanning-electron-microscopy-images/SEM-dataset'
# output_dir = '/home/oshadi/SISR-Final_Year_Project/envs/SISR-Project/data/DIV2K'

# # Create directories if they don't exist
# train_hr_dir = os.path.join(output_dir, 'DIV2K_train_HR')
# train_lr_dir = os.path.join(output_dir, 'DIV2K_train_LR_bicubic', 'X4')
# valid_hr_dir = os.path.join(output_dir, 'DIV2K_valid_HR')
# valid_lr_dir = os.path.join(output_dir, 'DIV2K_valid_LR_bicubic', 'X4')
# test_lr_dir = os.path.join(output_dir, 'DIV2K_test_LR_bicubic', 'X4')

# os.makedirs(train_hr_dir, exist_ok=True)
# os.makedirs(train_lr_dir, exist_ok=True)
# os.makedirs(valid_hr_dir, exist_ok=True)
# os.makedirs(valid_lr_dir, exist_ok=True)
# os.makedirs(test_lr_dir, exist_ok=True)

# print("Directories created successfully!")

# # Function to process and save images
# def process_image(file, hr_dir, lr_dir):
#     img = cv2.imread(file)
#     img_name = os.path.basename(file).split('.')[0]
    
#     # Save HR image
#     cv2.imwrite(os.path.join(hr_dir, f"{img_name}.jpg"), img)
    
#     # Save LR image
#     img_lr = cv2.resize(img, (img.shape[1] // 4, img.shape[0] // 4), interpolation=cv2.INTER_CUBIC)
#     cv2.imwrite(os.path.join(lr_dir, f"{img_name}x4.jpg"), img_lr)

# def save_images(files, hr_dir, lr_dir):
#     with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
#         futures = [executor.submit(process_image, file, hr_dir, lr_dir) for file in files]
#         concurrent.futures.wait(futures)

# # Get all category directories in the dataset directory
# category_dirs = [os.path.join(dataset_dir, d) for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))]

# # Initialize lists to hold files from all categories
# all_train_files = []
# all_valid_files = []
# all_test_files = []

# # Iterate over each category directory and split files
# for category in category_dirs:
#     # Load image file paths from current category
#     image_files = [os.path.join(category, f) for f in os.listdir(category) if f.endswith('.jpg')]
    
#     # Split dataset for current category
#     train_files, test_files = train_test_split(image_files, test_size=0.2, random_state=42)
#     train_files, valid_files = train_test_split(train_files, test_size=0.25, random_state=42)  # 0.25 * 0.8 = 0.2
    
#     # Append to overall lists
#     all_train_files.extend(train_files)
#     all_valid_files.extend(valid_files)
#     all_test_files.extend(test_files)

# # Save train, validation, and test images
# save_images(all_train_files, train_hr_dir, train_lr_dir)
# save_images(all_valid_files, valid_hr_dir, valid_lr_dir)
# save_images(all_test_files, test_lr_dir, test_lr_dir)

# print("Dataset split and images saved successfully!")





import os
import cv2
import concurrent.futures
from sklearn.model_selection import train_test_split

# Define paths
base_dir = '/home/oshadi/SISR-Final_Year_Project/envs/SISR-Project/data/Scanning-electron-microscopy-images/SEM-dataset/Fibres'  # Update this to your dataset directory
output_dir = '/home/oshadi/SISR-Final_Year_Project/envs/SISR-Project/data/DIV2K'
train_hr_dir = os.path.join(output_dir, 'DIV2K_train_HR')
train_lr_dir = os.path.join(output_dir, 'DIV2K_train_LR_bicubic', 'X4')
valid_hr_dir = os.path.join(output_dir, 'DIV2K_valid_HR')
valid_lr_dir = os.path.join(output_dir, 'DIV2K_valid_LR_bicubic', 'X4')
test_lr_dir = os.path.join(output_dir, 'DIV2K_test_LR_bicubic', 'X4')

# Create directories if they don't exist
os.makedirs(train_hr_dir, exist_ok=True)
os.makedirs(train_lr_dir, exist_ok=True)
os.makedirs(valid_hr_dir, exist_ok=True)
os.makedirs(valid_lr_dir, exist_ok=True)
os.makedirs(test_lr_dir, exist_ok=True)

print("Directories created successfully!")

# Load image file paths
image_files = [os.path.join(base_dir, f) for f in os.listdir(base_dir) if f.endswith('.jpg')]

# Split dataset
train_files, test_files = train_test_split(image_files, test_size=0.2, random_state=42)
train_files, valid_files = train_test_split(train_files, test_size=0.25, random_state=42)  # 0.25 * 0.8 = 0.2

def process_image(file, hr_dir, lr_dir):
    img = cv2.imread(file)
    img_name = os.path.basename(file).split('.')[0]
    
    # Save HR image
    cv2.imwrite(os.path.join(hr_dir, f"{img_name}.jpg"), img)
    
    # Save LR image
    img_lr = cv2.resize(img, (img.shape[1] // 4, img.shape[0] // 4), interpolation=cv2.INTER_CUBIC)
    cv2.imwrite(os.path.join(lr_dir, f"{img_name}x4.jpg"), img_lr)

def save_images(files, hr_dir, lr_dir):
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_image, file, hr_dir, lr_dir) for file in files]
        concurrent.futures.wait(futures)

# Save train, validation, and test images
save_images(train_files, train_hr_dir, train_lr_dir)
save_images(valid_files, valid_hr_dir, valid_lr_dir)
save_images(test_files, test_lr_dir, test_lr_dir)

print("Dataset split and images saved successfully!")
