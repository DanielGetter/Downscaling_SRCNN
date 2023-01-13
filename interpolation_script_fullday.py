import numpy as np
import netCDF4 as nc
import pyinterp as pp
import os
import fnmatch

def load_nc_data(subdir, filename):
    fpath = '../Data/'+ subdir 
    df = nc.Dataset(fpath+filename)
    lons = df.variables['Longitude'][:]
    lats = df.variables['Latitude'][:]
    wind = df.variables['WINDSPD_10M'][:]
    return lons, lats, wind, df

#get list of filenames
dirlist = ['USsw20x20', 'ESahara', 'EAsia']
nc_list = []
for file in os.listdir('../Data/'+dirlist[2]): #getting netCDF files from EAsia directory
    if fnmatch.fnmatch(file, 'W10*.nc'):
        nc_list.append(file)
        

#first_ten = nc_list[0:10]
# print(len(nc_list))

# interpolation script
def idw_helper(meshx, meshy, mesh_data):
    '''Performs inverse distance weighting interpolation'''
    idw, neighbors = mesh_data.inverse_distance_weighting(
        np.vstack((meshx.ravel(), meshy.ravel())).T,
        within=False,  # Extrapolation is forbidden
        k=11,  # We are looking for at most 11 neighbors 
        radius=600000,
        num_threads=0)
    idw = idw.reshape(meshx.shape)
    return idw

def interpolate(file, granularity=3):
    '''reworked to include variable granularity input (in km) as well as compute interpolation of all timestamps available in day (96)'''
    lons, lats, wind_all, df = load_nc_data(dirlist[2]+'/',file) # loads .nc file
    df.close()
    
    X0, X1 = lons.min(), lons.max() #gets range of lat/lon
    Y0, Y1 = lats.min(), lats.max()

    latstep = (granularity*20.0)/1725. #custom stepsizes that roughly correspond to native grid data pts
    lonstep = (granularity*20.0)/1340.

    mx, my = np.meshgrid(np.arange(X0, X1+lonstep, lonstep),
                     np.arange(Y0, Y1+latstep, latstep),
                     indexing='ij')
    

    final_array = np.empty([wind_all.shape[1], mx.shape[0], mx.shape[1]])

    for time in range(wind_all.shape[1]):
        mesh_i = pp.RTree()
        mesh_i.packing(np.vstack((lons,lats)).T, wind_all[:,time])
        idw_i = idw_helper(mx,my, mesh_i)

        final_array[time, :, :] = idw_i
    return mx, my, final_array, wind_all, lons, lats



nc_list = sorted(nc_list)

def main_script(directory, gran):
    for f in range(len(nc_list)):
        mx, my, wind_all, native_wind, nat_lon, nat_lat  = interpolate(nc_list[f], gran)
        
        #plt.imshow(wind_all[0][:,::-1].T, cmap='jet')
        #plt.savefig('interpolation_debug/interp_map_{}.png'.format(nc_list[f]))
        
        #fig = plt.figure(figsize=(10, 12))

#         cmap=plt.get_cmap('jet')
#         plt.tricontourf(np.array(nat_lon)-360, np.array(nat_lat), np.array(native_wind[:,0]),
#                         levels=np.arange(0,np.max(native_wind[:,0]),1.),cmap=cmap)
#         plt.savefig('interpolation_debug/native_map_{}.png'.format(nc_list[f]))

        if f == 0:
            np.save('../Data/{}/interpolated_data/{}/lon_grid.npy'.format(dirlist[2],directory), mx)
            np.save('../Data/{}/interpolated_data/{}/lat_grid.npy'.format(dirlist[2],directory), my)
            np.save('../Data/{}/interpolated_data/{}/'.format(dirlist[2], directory)+nc_list[f][0:-9]+'interp.npy', wind_all)
            print('first file saved')
        else:
            print('other file')
            np.save('../Data/{}/interpolated_data/{}/'.format(dirlist[2], directory)+nc_list[f][0:-9]+'interp.npy', wind_all)
        
        
#main_script('MR_25km', 25)
#main_script('LR_100km', 100)
main_script('HR_3km', 3)
        
   
