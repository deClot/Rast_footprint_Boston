import geopandas as gpd # Для парсинга карты
import pandas as pd
import numpy as np
import pickle
# import cudf

import matplotlib.pyplot as plt # Для отрисовки
from tqdm import tqdm
DATASET_FOLDER = './dataset'
figsize = (3, 3)

def plot_dataset(x_ini):
    '''create 3 type of sample pictures to DATASET_FOLDER:
     - building footprint, 
     - area/place,
     - combine both
    x_ini - pandas Series with featers for biuldings (first 29) and features for area (others)
    '''
    
    dpi=40
    # take data for place
    x = x_ini.iloc[29:].copy()
    area = gpd.GeoDataFrame(geometry=pd.Series(x.geometry_place))
    # take data for buiding and plot on the same axis
    x = x_ini.iloc[:29].copy()
    building = gpd.GeoDataFrame(geometry=pd.Series(x.geometry_building))
    
    # plot
    # area 
    ax = plt.subplot(111)
    ax.set_axis_off()
    area.plot(color='000', figsize=figsize, ax=ax)
    plt.tight_layout()
    plt.savefig(f'{DATASET_FOLDER}/areas/area_{x.name}.png',\
               dpi=dpi)   
    plt.close('all')
    
    ax2 = plt.subplot(111)
    ax2.set_axis_off()
    building.plot(color='000', figsize=figsize, ax=ax2)
    xl = ax.get_xlim()
    yl = ax.get_ylim()
    ax2.axis(xmin=xl[0],xmax=xl[1], ymin=yl[0], ymax=yl[1])
    plt.tight_layout()
    plt.savefig(f'{DATASET_FOLDER}/buildings/building_{x.name}.png',
               dpi=dpi)   
    plt.close('all')
    
    ax = plt.subplot(111)
    ax.set_axis_off()
    area.plot(color='000', figsize=figsize, ax=ax)
    building.plot(color='r', figsize=figsize, ax=ax)
    plt.tight_layout()
    plt.savefig(f'{DATASET_FOLDER}/combined/total_{x.name}.png',
               dpi=dpi,
#                 bbox_inches='tight')   
               )
    plt.close('all')   
    
    ax = plt.subplot(121)
    ax.set_axis_off()
    area.plot(color='000', figsize=figsize, ax=ax)
    plt.tight_layout()
    ax2 = plt.subplot(122)
    ax2.set_axis_off()
    building.plot(color='000', figsize=figsize, ax=ax2)
    xl = ax.get_xlim()
    yl = ax.get_ylim()
    ax2.axis(xmin=xl[0],xmax=xl[1], ymin=yl[0], ymax=yl[1])
    plt.tight_layout()
    
    plt.savefig(f'{DATASET_FOLDER}/closer/both_{x.name}.png',dpi=dpi)
    
    
if __name__ == '__main__':
    df_ini = pickle.load(open('df_ini.pkl', 'br'))
    tqdm.pandas()
    df_ini.progress_apply(plot_dataset, axis=1)
    
    