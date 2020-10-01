from os import listdir
import pickle

import geopandas as gpd # Для парсинга карты
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt # Для отрисовки
from tqdm import tqdm

from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img


DATASET_FOLDER = './dataset'
path_to_input = './dataset/areas/'
path_to_target = './dataset/buildings/'

batch_size = 2_000

def plot_dataset(x_ini):
    '''create 3 type of sample pictures to DATASET_FOLDER:
     - building footprint, 
     - area/place,
     - combine both
    x_ini - pandas Series with featers for biuldings (first 29) and features for area (others)
    '''
    
    # take data for place
    x = x_ini.iloc[29:].copy()
    area = gpd.GeoDataFrame(geometry=pd.Series(x.geometry_place))
    # take data for buiding and plot on the same axis
    x = x_ini.iloc[:29].copy()
    building = gpd.GeoDataFrame(geometry=pd.Series(x.geometry_building))
    
    dpi=100
    figsize = (256/dpi, 256/dpi)
    
    # plot places
    p = area.plot(color='000',  figsize=figsize)
    p.set_axis_off()
    plt.savefig(f'{DATASET_FOLDER}/areas/area_{x.name}.png',\
               dpi=dpi)   
    plt.close('all')
    
    # plot buildings
    b = building.plot(color='000',  figsize=figsize)
    b.set_axis_off()
    xl = p.get_xlim()
    yl = p.get_ylim()
    b.axis(xmin=xl[0],xmax=xl[1], ymin=yl[0], ymax=yl[1])
    plt.savefig(f'{DATASET_FOLDER}/buildings/building_{x.name}.png',
               dpi=dpi)   
    plt.close('all')
    
    
#     ax = plt.subplot(121)
#     ax.set_axis_off()
#     area.plot(color='000', figsize=figsize, ax=ax)
#     plt.tight_layout()
#     ax2 = plt.subplot(122)
#     ax2.set_axis_off()
#     building.plot(color='000', figsize=figsize, ax=ax2)
#     xl = ax.get_xlim()
#     yl = ax.get_ylim()
#     ax2.axis(xmin=xl[0],xmax=xl[1], ymin=yl[0], ymax=yl[1])
#     plt.tight_layout()
    
#     plt.savefig(f'{DATASET_FOLDER}/closer/both_{x.name}.png',dpi=dpi)
    

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
    print(idx)
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
#     plt.ioff()
#     plt.margins(0, 0)
    
#     df_ini = pickle.load(open('df_ini.pkl', 'br'))
#     tqdm.pandas()
#     df_ini.progress_apply(plot_dataset, axis=1)
    
    load_images()
    #load_combined_images('./dataset/closer/', 'closer')
    #load_combined_images('./dataset/combined/', 'combined')
    
    
  

    
    
