import cartopy.crs as ccrs
import cartopy.feature as cfeature
from shapely.geometry import Polygon
from shapely.geometry import Point
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras import layers, losses
from tensorflow.keras.models import Model

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from reliability.Distributions import Weibull_Distribution, Mixture_Model
from reliability.Fitters import Fit_Weibull_Mixture
from reliability.Other_functions import histogram

from scipy.spatial.distance import jensenshannon
from scipy import stats
from skimage import measure

import matplotlib.lines as mlines

import os

def sin_95(x):
    return np.round(np.sin(((2*np.pi)/95)*x), 5)

def untransform_HR(datapt):
    norm_max, norm_min = np.load('normalization_constants_HRwind.npy')
    #first, since data were normalized then sqrt'd, we need to square the data
    square = np.power(datapt, 2)
    #then do inverse of normalization 
    un_norm = square* (norm_max - norm_min) + norm_min
    return un_norm





from scipy.spatial.distance import jensenshannon

def get_js_distance(predicted_im, observed_im): # hardcode upper and lower bound, set bins to 100
    if len(predicted_im.shape) > 1 and len(observed_im.shape) == 1:
        pred_freq = np.histogram(predicted_im.flatten(), bins=np.linspace(0, 50, 100))[0]/len(predicted_im.flatten())
        obs_freq = np.histogram(observed_im, bins=np.linspace(0, 50, 100))[0]/len(observed_im)
    elif len(observed_im.shape) > 1 and len(predicted_im.shape) == 1:
        obs_freq = np.histogram(observed_im.flatten(), bins=np.linspace(0, 50, 100))[0]/len(observed_im.flatten())
        pred_freq = np.histogram(predicted_im, bins=np.linspace(0, 50, 100))[0]/len(predicted_im)
    else:
        pred_freq = np.histogram(predicted_im, bins=np.linspace(0, 50, 100))[0]/len(predicted_im)
        obs_freq = np.histogram(observed_im, bins=np.linspace(0, 50, 100))[0]/len(observed_im)
    
    '''returns Jensen-Shannon distance between two PDFs, in this case prob vectors of observed & predicted vals'''
    return jensenshannon(pred_freq, obs_freq)


##not used in this script
def js_histogram(observed, predicted):
    js = []
#     js_pred = np.array([el.flatten() for el in predicted])
#     js_obs = np.array([el.flatten() for el in observed])

    for i in range(len(predicted)): 
        js.append(get_js_distance(predicted[i], observed[i]))
        
    plt.hist(js, density=True, bins=25, range=(0,1.))
    
    
    
def parse(X, nx, ny):
    '''Parses geographic domain into 100km cells for CNN validation'''
    arr = []
    d = X.shape #assumes X is a numpy object
    dx = int(np.floor(d[0]/nx))
    dy = int(np.floor(d[1]/ny))
    for i in range(0,dx):
        for j in range(0,dy):
            x = X[(i*nx):((1+i)*nx),(j*ny):((1+j)*ny)]
            arr.append(x)
    return np.array(arr), dx, dy

def load_coords(lats, lons):
    lats = np.load(lats)
    lons = np.load(lons)
    return lats, lons

def abs_diff_percentage(truth, pred, dx, dy):
    abs_diff = np.abs((truth-pred)/truth)
    mean_diff_pct = np.array([np.mean(i) for i in abs_diff]).reshape((dx,dy))
    return mean_diff_pct

def error_plot(truth_parsed, pred_parsed, dx_pred, dy_pred, num, dname):
    error_pct = abs_diff_percentage(truth_parsed, pred_parsed, dx_pred, dy_pred)
    plt.figure(figsize=(10,5))
    plt.imshow(error_pct)
    plt.colorbar(fraction=0.3)
    plt.savefig(dname+'/error_plot{}'.format(num))
    return error_pct



def cell_hist_plotter(truth, pred, dirname):
    
    def weibull_fitter(truth_subset):
        #fit weibull to truth data
        weib_truth = Fit_Weibull_Mixture(failures= truth_subset.flatten(), show_probability_plot=False, print_results=False)
        w1 = Weibull_Distribution(alpha=weib_truth.alpha_1, beta=weib_truth.beta_1)
        w2 = Weibull_Distribution(alpha=weib_truth.alpha_2, beta=weib_truth.beta_2)
        wmix= Mixture_Model([w1,w2], [weib_truth.proportion_1, 1-weib_truth.proportion_1])
        return weib_truth, wmix
    
    centile_names = ['min_error_sample_', '25p_error_sample_', 'med_error_sample_', 
                     '75p_error_sample_', 'max_error_sample_']

    wmix_sample_array_10x5 = []
    stat_matrix = np.zeros([50,5])
    for i in range(truth.shape[0]):
        for j in range(truth.shape[1]):
            # get weibull fit first
            weib_truth, wmix = weibull_fitter(truth[i][j])
            #wmix vs truth j-s values
            wmix_samples = wmix.random_samples(len(truth[i][j].flatten()), seed=42)
            
            #JS vals
            js_wt = get_js_distance(truth[i][j], wmix_samples)
            js_tp = get_js_distance(truth[i][j], pred[i][j])
            
            ##compile JS vals and stats
            stat_matrix[5*i+j][0:2] = js_wt, js_tp
            stat_matrix[5*i+j][2:] = np.min(pred[i][j]), np.percentile(pred[i][j], 50), np.max(pred[i][j])

            #histogram plotting
            xmin = min(truth[i][j].flatten())
            xmax = max(truth[i][j].flatten())
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,5))
            ax1.hist(truth[i][j].flatten(), 
                     bins=np.linspace(xmin, xmax, 25),
                     alpha=0.5, edgecolor='black', density=True, label='truth')
            ax1.hist(pred[i][j].flatten(), 
                     bins=np.linspace(xmin, xmax, 25),
                     alpha=0.5, edgecolor='black', density=True, label='prediction')
            pdf = wmix.PDF(xvals = np.linspace(xmin, xmax, 25), label='Weibull Mixture', show_plot=False)
            ax1.plot(np.linspace(xmin, xmax, 25), pdf)
            ax1.legend(loc='upper right')
            ax1.set_xlabel('Windspeed (m/s)')
            ax1.set_title('JS Weibull/Observed: {}, JS Obs/Predicted: {}'.format(np.round(js_wt, 4), np.round(js_tp, 4)))
            ###ax2
            error_pts = np.abs(truth[i][j]-pred[i][j])/truth[i][j]
            img = ax2.imshow(np.flip(error_pts.T, axis=0), aspect='equal')
            cb = plt.colorbar(img, ax=ax2, fraction=0.2)
            cb.set_label('proportion error')
            ax2.set_title('ptwise '+centile_names[j]+'{}'.format(i+1))
            
            
            if not os.path.exists(dirname):
                os.makedirs(dirname)
            plt.savefig(dirname+centile_names[j]+'{}'.format(i+1))
            plt.show()
            
    np.save(dirname+'model_stats', stat_matrix)
    
    
def get_percentiles(truth, predictions, dname, lats, lons):
    '''Validation script that returns quantitative comparisons between subgrid wind distributions'''
    #get coordinate matrices
    lats, lons = load_coords(lats, lons)
    
    #parsecoordinates
    lats_parsed, dx_lat, dy_lat = parse(lats, 32, 32)
    lons_parsed, dx_lon, dy_lon = parse(lons, 32, 32)
    
    predicted_arr = []
    truth_arr =[]
    coarse_arr = []
    lats_arr = []
    lons_arr = []
    errors = []
    pterr_arr = []
    cent_arr = []
    
    for i in range(predictions.shape[0]):
        pred_parsed, dx_pred, dy_pred = parse(predictions[i], 32, 32)
        truth_parsed, dx_truth, dy_truth = parse(truth[i], 32, 32)
        #coarse_parsed, dxcoarse_truth, dycoarse_truth = parse(coarse[i], 4,4)
        
        #get mean absolute error at each grid cell  
        error_pct = error_plot(truth_parsed, pred_parsed, dx_pred, dy_pred, i+1, dname)
        
        #get indices for various percentiles of error
        #if len(cents)==0:
        min_err_idx = np.argmin(error_pct)
        max_err_idx = np.argmax(error_pct)
        med_err_idx = np.argsort(error_pct.flatten())[len(error_pct.flatten())//2]
        pctile_25_idx = np.argsort(error_pct.flatten())[len(error_pct.flatten())//4]
        pctile_75_idx = np.argsort(error_pct.flatten())[int(np.floor(len(error_pct.flatten())*(3/4)))]
        centiles = [min_err_idx, pctile_25_idx, med_err_idx, pctile_75_idx, max_err_idx]
        cent_arr.append(centiles) 
#         else:
#             print('using other cents')
#             centiles = cents[i]
            
        print(centiles)
        
        #subset sample cells 
        #centiles = [min_err_idx, pctile_25_idx, med_err_idx, pctile_75_idx, max_err_idx]
        pred_parsed_cent = pred_parsed[centiles]
        truth_parsed_cent = truth_parsed[centiles]
        #coarse_parsed_cent = coarse_parsed[centiles]
        lats_cent = lats_parsed[centiles]
        lons_cent = lons_parsed[centiles]
        
        errors.append(error_pct)
        predicted_arr.append(pred_parsed_cent)
        truth_arr.append(truth_parsed_cent)
        #coarse_arr.append(coarse_parsed_cent)
        lats_arr.append(lats_cent)
        lons_arr.append(lons_cent)
        
    print(cent_arr)
    #np.array(coarse_arr) arg removed in return statement
    return np.array(errors), np.array(predicted_arr), np.array(truth_arr), np.array(cent_arr), np.array(lats_arr), np.array(lons_arr), lats, lons

def error_and_centile(errors, lon_full, lat_full, cent_lon, cent_lat, dirname): 
    lonerr = measure.block_reduce(lon_full, block_size=(32,32), func=np.mean)[0:14, 0:18]
    laterr = measure.block_reduce(lat_full, block_size=(32,32), func=np.mean)[0:14, 0:18]
    
    for i in range(errors.shape[0]):
            #generate plot
        fig = plt.figure(figsize=(8, 10))
        ax1 = plt.axes(projection=ccrs.PlateCarree())
        ax1.set_extent([-110,-90,25,45],ccrs.PlateCarree())
        ax1.coastlines()
        ax1.add_feature(cfeature.STATES.with_scale('10m'))
        
        dot = mlines.Line2D([], [], color='red', marker='o', linestyle='None',
                          markersize=10, label='min error')
        tri = mlines.Line2D([], [], color='red', marker='^', linestyle='None',
                                  markersize=10, label='25th error')
        square = mlines.Line2D([], [], color='red', marker='s', linestyle='None',
                                  markersize=10, label='med err')
        star = mlines.Line2D([], [], color='red', marker='*', linestyle='None',
                                  markersize=10, label='75th error')
        plus = mlines.Line2D([], [], color='red', marker='P', linestyle='None',
                                  markersize=10, label='max error')
        
        cmap=plt.get_cmap('viridis')
        #plot contour map of error
        plt.tricontourf((lonerr-360).flatten(), laterr.flatten(), errors[i].flatten(),
                        levels=np.arange(np.min(errors[i]), np.max(errors[i]), .025),cmap=cmap)
        #plot markers for centerpoints of percentile grid cells
        plt.plot(cent_lon[i][0].mean()-360, cent_lat[i][0].mean(), color='red', marker='o', markersize=10) # min
        plt.plot(cent_lon[i][1].mean()-360, cent_lat[i][1].mean(), color='red', marker='^', markersize=10) #25th
        plt.plot(cent_lon[i][2].mean()-360, cent_lat[i][2].mean(), color='red', marker='s', markersize=10) #med
        plt.plot(cent_lon[i][3].mean()-360, cent_lat[i][3].mean(), color='red', marker='*', markersize=10) #75th
        plt.plot(cent_lon[i][4].mean()-360, cent_lat[i][4].mean(), color='red', marker='P', markersize=10) #max
        plt.legend(handles=[dot, tri, square, star, plus])
        cb1 = plt.colorbar(shrink=0.5)
        cb1.set_label('error proportion')
        plt.title('MAE error proportion sample {}'.format(i+1));
        if not os.path.exists(dirname+'error_maps'):
            os.makedirs(dirname+'error_maps')
        plt.savefig('{}/error_maps/error_sample_{}.png'.format(dirname, i+1))
        plt.close("all")
        
        
##### FOR JS VALUES ONLY ########
def js_calculator(truth, pred, dirname):
    def weibull_fitter(truth_subset):
        #fit weibull to truth data
        weib_truth = Fit_Weibull_Mixture(failures= truth_subset.flatten(), show_probability_plot=False, print_results=False)
        w1 = Weibull_Distribution(alpha=weib_truth.alpha_1, beta=weib_truth.beta_1)
        w2 = Weibull_Distribution(alpha=weib_truth.alpha_2, beta=weib_truth.beta_2)
        wmix= Mixture_Model([w1,w2], [weib_truth.proportion_1, 1-weib_truth.proportion_1])
        return weib_truth, wmix
    
    wmix_sample_array_10x5 = []
    stat_matrix = np.zeros([50,5])
    for i in range(truth.shape[0]):
        for j in range(truth.shape[1]):
            # get weibull fit first
            weib_truth, wmix = weibull_fitter(truth[i][j])
            #wmix vs truth j-s values
            wmix_samples = wmix.random_samples(len(truth[i][j].flatten()), seed=42)
            
            #JS vals
            js_wt = get_js_distance(truth[i][j], wmix_samples)
            js_tp = get_js_distance(truth[i][j], pred[i][j])
            
            ##compile JS vals and stats
            stat_matrix[5*i+j][0:2] = js_wt, js_tp
            stat_matrix[5*i+j][2:] = np.min(pred[i][j]), np.percentile(pred[i][j], 50), np.max(pred[i][j]) 
            
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    np.save(dirname+'model_stats', stat_matrix)      
