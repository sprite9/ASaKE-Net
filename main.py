from framework.framework import Framework
from config.config import Configargs
import torch
import numpy as np
import random

seed = 2222
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False




config = Configargs()
fw = Framework(config)
fw.train()
#fw.test()
