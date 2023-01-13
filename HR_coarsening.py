import numpy as np
from skimage.measure import block_reduce


HR1 = np.load('../Data/USsw20x20/interpolated_data/HR_3km/all_days.npy')
HR2 = np.load('../Data/ESahara/HR_3km/all_days.npy')
HR3 = np.load('../Data/EAsia/interpolated_data/HR_3km/all_days.npy')

def coarsen(data):
    data_coarsened_MR16 = np.array([block_reduce(im, block_size=(4,4), func=np.mean) for im in data]).reshape((-1, 112, 144, 1))
    data_coarsened_MR = np.array([block_reduce(im, block_size=(8,8), func=np.mean) for im in data]).reshape((-1, 56, 72, 1))
    data_coarsened_MR2 = np.array([block_reduce(im, block_size=(16,16), func=np.mean) for im in data]).reshape((-1, 28, 36, 1))
    data_coarsened_LR = np.array([block_reduce(im, block_size=(32,32), func=np.mean) for im in data]).reshape((-1, 14, 18, 1))
    return data_coarsened_MR, data_coarsened_MR2, data_coarsened_LR, data_coarsened_MR16


coarse_MR1, coarse2_MR1, coarse_LR1, coarse_MR161 = coarsen(HR1)
np.save('../Data/USsw20x20/interpolated_data/HR_3km/all_days_coarseMR.npy', coarse_MR1)
np.save('../Data/USsw20x20/interpolated_data/HR_3km/all_days_coarseMR2.npy', coarse2_MR1)
np.save('../Data/USsw20x20/interpolated_data/HR_3km/all_days_coarseLR.npy', coarse_LR1)
np.save('../Data/USsw20x20/interpolated_data/HR_3km/all_days_coarseMR12.npy', coarse_MR161)


coarse_MR2, coarse2_MR2, coarse_LR2, coarse_MR162 = coarsen(HR2)
np.save('../Data/ESahara/HR_3km/all_days_coarseMR.npy', coarse_MR2)
np.save('../Data/ESahara/HR_3km/all_days_coarseMR2.npy', coarse2_MR2)
np.save('../Data/ESahara/HR_3km/all_days_coarseLR.npy', coarse_LR2)
np.save('../Data/ESahara/HR_3km/all_days_coarseMR12.npy', coarse_MR162)


coarse_MR, coarse_MR2, coarse_LR, coarse_MR16 = coarsen(HR3)
np.save('../Data/EAsia/interpolated_data/HR_3km/all_days_coarseMR.npy', coarse_MR)
np.save('../Data/EAsia/interpolated_data/HR_3km/all_days_coarseMR2.npy', coarse_MR2)
np.save('../Data/EAsia/interpolated_data/HR_3km/all_days_coarseLR.npy', coarse_LR)
np.save('../Data/EAsia/interpolated_data/HR_3km/all_days_coarseMR12.npy', coarse_MR16)
