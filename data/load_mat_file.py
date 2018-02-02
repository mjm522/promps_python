import os
import pickle
from  scipy.io import loadmat

data = loadmat(os.environ['ROOT_DIR'] + '/data/im_data_1d_opt.mat')['imData']


steps  = {'numElements': data[0,0][0][0][0]['steps']['numElements'],
           'actions': data[0,0][0][0][0]['steps']['actions'],
           'nextStates': data[0,0][0][0][0]['steps']['nextStates'],
           'rewards': data[0,0][0][0][0]['steps']['rewards'],
           'states': data[0,0][0][0][0]['steps']['states'],
           'timeSteps': data[0,0][0][0][0]['steps']['timeSteps'],}

data_py = {'numElements': data[0,0][0][0][0]['numElements'],
           'iterationNumber': data[0,0][0][0][0]['iterationNumber'],
           'finalRewards': data[0,0][0][0][0]['finalRewards'],
            'dt':data[0,0][0][0][0]['dt'],
            'noise_std':data[0,0][0][0][0]['noise_std'],
            'init_m':data[0,0][0][0][0]['init_m'],
            'init_std':data[0,0][0][0][0]['init_std'],
            'maxAction':data[0,0][0][0][0]['maxAction'],
            'steps':steps, }


pickle_out = open(os.environ['ROOT_DIR'] + '/data/data.pkl', "wb")
pickle.dump(data_py, pickle_out)
pickle_out.close()