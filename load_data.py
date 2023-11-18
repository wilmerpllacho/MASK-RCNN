import cv2
import glob
import numpy as np
import nibabel as nib
from sklearn.utils import shuffle
from skimage import exposure
np.seterr(divide='ignore', invalid='ignore')

"""
Function to verify the .nii image orientation
"""
def check_orientation(ct_image, ct_arr):
  x, y, z = nib.aff2axcodes(ct_image.affine)
  if x != 'R':
    ct_arr = nib.orientations.flip_axis(ct_arr, axis=0)
  if y != 'P':
    ct_arr = nib.orientations.flip_axis(ct_arr, axis=1)
  if z != 'S':
    ct_arr = nib.orientations.flip_axis(ct_arr, axis=2)
  return ct_arr


"""
Function to process the .nii data
    Args:   path
    Output: data values
"""
def get_data_values(path):
    data_values = []

    for value in path:
        img         = nib.load(value)
        img_arr     = img.get_fdata()
        img_aux_nii = check_orientation(img, img_arr)
        img_aux_nii = np.squeeze(img_aux_nii)
        aux_idx     = len(img_aux_nii[0][0])

        for value_aux in range(aux_idx):
            image_aux = img_aux_nii[:,:,value_aux]
            image_aux = np.rot90(image_aux, 3)
            image_aux = np.fliplr(image_aux)
            if image_aux.shape[0] != 512:
                image_aux = cv2.resize(image_aux, (512, 512))
            image_aux = (image_aux-image_aux.min())/(image_aux.max()-image_aux.min())
            image_aux = np.nan_to_num(image_aux)
            #image_aux = clahe(image_aux)
            data_values.append(image_aux)
    
    return data_values

def read_image_mask(images_path, masks_path):
    image_values = get_data_values([images_path])
    mask_values  = get_data_values([masks_path])
    return image_values, mask_values
"""
Function to read CTs and Masks
    Args:   images path and masks path
    Output: CTs and Masks values
"""
def read_dataset(images_path, masks_path):
    image_values = get_data_values(images_path)
    mask_values  = get_data_values(masks_path)
    image_values, mask_values = shuffle(image_values, mask_values, random_state=0)

    return image_values, mask_values


"""
Function to split the data into training and validation
    Output: CTs and Mask for training and validation respectively
"""
def split_dataset(img_path, mask_path, train_split):
    # Reading the dataset
    images_path         = sorted(glob.glob(img_path  + '*.nii'))
    masks_path          = sorted(glob.glob(mask_path + '*.nii'))
    img_data, mask_data = read_dataset(images_path, masks_path)

    # Training dataset
    train_image_data    = img_data [:int(len(img_data) * train_split)]
    train_mask_data     = mask_data[:int(len(mask_data)* train_split)]

    # Validation Dataset
    val_image_data      = img_data [int(len(img_data) * train_split):]
    val_mask_data       = mask_data[int(len(mask_data)* train_split):]

    return train_image_data, train_mask_data, val_image_data, val_mask_data

def generate_dataset(images,masks,transform,split_generate):
    images_generate = []
    masks_generate  = []
    for i in range(len(images)):
      generate = transform(image=images[i],mask=masks[i])
      images_generate.append(generate['image'])
      masks_generate.append(generate['mask'])
      images_generate, masks_generate = shuffle(images_generate, masks_generate, random_state=0)
    # Validation Dataset
    images_generate      = images_generate[:int(len(images_generate) * split_generate)]
    masks_generate       = masks_generate[:int(len(masks_generate)* split_generate)]
    return images_generate, masks_generate
    
def generate_dataset_mask(images,masks):
    lista = []
    contador =  1
    for i in range(len(images)):
      info = { }
      info['id']     = str(contador)
      info['image']  = images[i]
      info['mask']   = masks[i]
      lista.append(info)
      contador = contador + 1
    return lista