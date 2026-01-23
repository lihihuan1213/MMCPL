<<<<<<< HEAD
import random
import os
import numpy as np
import torch


def set_seed(seed):
    # Set the random seed
    random.seed(seed)
    np.random.seed(seed)
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
=======
import random
import os
import numpy as np
import torch


def set_seed(seed):
    # Set the random seed
    random.seed(seed)
    np.random.seed(seed)
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
>>>>>>> 258366d2c030f8958cb244cd87866492b978986a
