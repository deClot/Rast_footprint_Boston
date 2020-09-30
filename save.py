from os import listdir
import numpy as np

from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img

path_to_input = './dataset/areas/'
path_to_target = './dataset/buildings/'


batch_size = 5_000

def load_images():
    size = len(listdir(path_to_input))
    for batch_number in range(size // batch_size +1): # go throught batches
        print(batch_number)
        idx = (batch_number*batch_size, (batch_number+1)*batch_size)
        data_input, data_target = load_images_from_path(path_to_input, idx), \
                                    load_images_from_path(path_to_target, idx)  
        print(data_input.shape, data_target.shape)
        np.savez_compressed(f'./dataset/dataset_batch{batch_number+1}.npz', data_input, data_target)
        del data_input, data_target

def load_images_from_path(path, idx):
    data = list()
    for filename in listdir(path)[idx[0]:idx[1]]:
        # load and resize the image
        pixels = load_img(path + filename)
        # convert to numpy array
        pixels = img_to_array(pixels)
        data.append(pixels)
        
    return np.asarray(data)

def load_combined_images(path, name):
    size = len(listdir(path))
    for batch_number in range(size // batch_size +1): # go throught batches
        print(batch_number)
        idx = (batch_number*batch_size, (batch_number+1)*batch_size)
        data = load_images_from_path(path, idx)
        np.savez_compressed(f'./dataset/dataset_{name}_batch{batch_number+1}.npz', data)
        del data


if __name__ == '__main__':
#     load_images()
    load_combined_images('./dataset/closer/', 'closer')
    load_combined_images('./dataset/combined/', 'clombined')
    
    
