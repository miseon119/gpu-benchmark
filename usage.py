import time
import datetime
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from IPython import display
from collections import OrderedDict

from pynvml import (nvmlInit,
                     nvmlDeviceGetCount, 
                     nvmlDeviceGetHandleByIndex, 
                     nvmlDeviceGetUtilizationRates,
                     nvmlDeviceGetName)
def gpu_info():
    "Returns a tuple of (GPU ID, GPU Description, GPU % Utilization)"
    nvmlInit()
    deviceCount = nvmlDeviceGetCount()
    info = []
    for i in range(0, deviceCount): 
        handle = nvmlDeviceGetHandleByIndex(i) 
        util = nvmlDeviceGetUtilizationRates(handle)
        desc = nvmlDeviceGetName(handle) 
        info.append((i, desc, util.gpu)) #['GPU %i - %s' % (i, desc)] = util.gpu
    return info

utils = []
cnt = 0
while True:
    try:
        dt = datetime.datetime.now()
        util = gpu_info()
        utils.append([dt] + [x[2] for x in util])
        # Don't plot anything on the first pass
        if len(utils) == 1:
            continue
        df = pd.DataFrame(utils, columns=['dt'] + 
                          ['GPU %i - %s' % (x[0], x[1]) for x in util]).set_index('dt')

        time.sleep(1)
        cnt = cnt + 1
        print("cnt = ", cnt)

        if cnt > 10:
            ax = df.plot();
            vals = ax.get_yticks();
            ax.set_yticklabels(['{:3.0f}%'.format(x) for x in vals]);
            ax.set_ylabel('GPU Utilization');
            ax.set_xlabel('Time');
            display.clear_output(wait=True)
            display.display(plt.gcf())
            plt.savefig('usage_sample.png')
            plt.show()
            break

    except KeyboardInterrupt:
        break

