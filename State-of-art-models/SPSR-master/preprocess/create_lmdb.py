# import sys
# import os.path
# import glob
# import pickle
# import lmdb
# import cv2
# from tqdm import tqdm
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# # configurations
# img_folder = os.path.join('/home/oshadi/SISR-Final_Year_Project/envs/SISR-Project/State-of-art-models/SPSR-master/data',
#                           'dataset','BSD100_sub','HR','*')  # glob matching pattern
# lmdb_save_path = os.path.join('/home/oshadi/SISR-Final_Year_Project/envs/SISR-Project/State-of-art-models/SPSR-master/data',
#                               'dataset','BSD100_sub_HR.lmdb')  # must end with .lmdb

# img_list = sorted(glob.glob(img_folder))
# dataset = []
# data_size = 0

# print('Read images...')
# pbar = tqdm(total=len(img_list))
# for i, v in tqdm(enumerate(img_list), total=len(img_list)):
#     pbar.update(1)
#     img = cv2.imread(v, cv2.IMREAD_UNCHANGED)
#     dataset.append(img)
#     data_size += img.nbytes
# env = lmdb.open(lmdb_save_path, map_size=data_size * 10)
# pbar.close()
# print('Finish reading {} images.\nWrite lmdb...'.format(len(img_list)))

# pbar = tqdm(total=len(img_list))
# with env.begin(write=True) as txn:  # txn is a Transaction object
#     for i, v in tqdm(enumerate(img_list), total=len(img_list)):
#         pbar.update(1)
#         base_name = os.path.splitext(os.path.basename(v))[0]
#         key = base_name.encode('ascii')
#         data = dataset[i]
#         if dataset[i].ndim == 2:
#             H, W = dataset[i].shape
#             C = 1
#         else:
#             H, W, C = dataset[i].shape
#         meta_key = (base_name + '.meta').encode('ascii')
#         meta = '{:d}, {:d}, {:d}'.format(H, W, C)
#         # The encode is only essential in Python 3
#         txn.put(key, data)
#         txn.put(meta_key, meta.encode('ascii'))
# pbar.close()
# print('Finish writing lmdb.')

# # create keys cache
# keys_cache_file = os.path.join(lmdb_save_path, '_keys_cache.p')
# env = lmdb.open(lmdb_save_path, readonly=True, lock=False, readahead=False, meminit=False)
# with env.begin(write=False) as txn:
#     print('Create lmdb keys cache: {}'.format(keys_cache_file))
#     keys = [key.decode('ascii') for key, _ in txn.cursor()]
#     pickle.dump(keys, open(keys_cache_file, "wb"))
# print('Finish creating lmdb keys cache.')

import sys
import os.path
import glob
import pickle
import lmdb
import cv2
from tqdm import tqdm

# configurations
img_folder = os.path.join('/home/oshadi/SISR-Final_Year_Project/envs/SISR-Project/State-of-art-models/SPSR-master/data',
                          'Tdataset', 'HR', '*')  # glob matching pattern
lmdb_save_path = os.path.join('/home/oshadi/SISR-Final_Year_Project/envs/SISR-Project/State-of-art-models/SPSR-master/data',
                              'Tdataset', 'HR.lmdb')  # must end with .lmdb

img_list = sorted(glob.glob(img_folder))
dataset = []
data_size = 0

print('Read images...')
pbar = tqdm(total=len(img_list))
for i, v in enumerate(img_list):
    img = cv2.imread(v, cv2.IMREAD_UNCHANGED)
    if img is None:
        print(f"Error reading image {v}")
        continue
    dataset.append(img)
    data_size += img.nbytes
    pbar.update(1)
pbar.close()
print('Finish reading {} images.\nWrite lmdb...'.format(len(img_list)))

env = lmdb.open(lmdb_save_path, map_size=data_size * 10)
pbar = tqdm(total=len(img_list))
with env.begin(write=True) as txn:  # txn is a Transaction object
    for i, v in enumerate(img_list):
        base_name = os.path.splitext(os.path.basename(v))[0]
        key = base_name.encode('ascii')
        data = dataset[i]
        if data.ndim == 2:
            H, W = data.shape
            C = 1
        else:
            H, W, C = data.shape
        meta_key = (base_name + '.meta').encode('ascii')
        meta = '{:d}, {:d}, {:d}'.format(H, W, C)
        txn.put(key, data)
        txn.put(meta_key, meta.encode('ascii'))
        pbar.update(1)
pbar.close()
print('Finish writing lmdb.')

# create keys cache
keys_cache_file = os.path.join(lmdb_save_path, '_keys_cache.p')
env = lmdb.open(lmdb_save_path, readonly=True, lock=False, readahead=False, meminit=False)
with env.begin(write=False) as txn:
    print('Create lmdb keys cache: {}'.format(keys_cache_file))
    keys = [key.decode('ascii') for key, _ in txn.cursor()]
    pickle.dump(keys, open(keys_cache_file, "wb"))
print('Finish creating lmdb keys cache.')
