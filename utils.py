import os
import random

import torch

import numpy as np

def seed_everything(seed : int, deterministic : bool = False):
    '''
    Set seeds for reproducibility
    '''

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    torch.backends.cudnn.benchmark = not deterministic
    torch.backends.cudnn.deterministic = deterministic
    torch.use_deterministic_algorithms(deterministic)
    if deterministic:
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    


### @JL I don't want a bandaid fix here, rather fix it properly...

# import lightning as pl


# ## Utils to handle newer PyTorch Lightning changes from version 0.6
# ## ==================================================================================================== ##


# def data_loader(fn):
#     """
#     Decorator to handle the deprecation of data_loader from 0.7
#     :param fn: User defined data loader function
#     :return: A wrapper for the data_loader function
#     """

#     def func_wrapper(self):
#         try: # Works for version 0.6.0
#             return pl.data_loader(fn)(self)

#         except: # Works for version > 0.6.0
#             return fn(self)

#     return func_wrapper
