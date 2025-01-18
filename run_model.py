import numpy as np
import argparse
import cProfile
import time
import datetime
import os
import pickle

import torch
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from torch.backends import cudnn
from torch.utils.data import DataLoader
from GraphDataSet import GraphDataSet
from gc_utils import attributes_plot, is_k_color, find_closest_kcoloring, sklearn_k_means, check_k_colorable_and_assign, plot, reassign_clusters_respecting_order
from model import GCPNet
from sklearn.preprocessing import StandardScaler
from torch_scatter import scatter_min, scatter_max

import sys


col3_files = [
                "/home/elad/Documents/kcol/tmp/json/train/1013.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m14350.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m2836.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/4443.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/2023.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m8939.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/11636.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m3832.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m6836.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m10328.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m14398.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/6193.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m3132.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/13536.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m4430.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/6143.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m9230.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/8803.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m10398.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/9233.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m4037.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/3353.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/11539.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m6937.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/9033.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/4583.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m10343.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m10321.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/1743.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m16356.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/1453.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m10372.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m9633.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m13301.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/1253.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m753.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/12436.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/4063.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/7213.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m9735.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m1932.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m1530.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m11327.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m3736.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m16359.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m4132.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m783.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/2673.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/2153.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m9939.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/12834.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/14433.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m8838.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/5703.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/9883.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m8630.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m7938.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m4930.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/9643.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m12391.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/2193.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m13364.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/4183.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/12639.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/4323.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/6683.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/8413.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m3232.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m6130.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m15398.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/3213.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/8363.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/8303.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m4437.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/3533.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/4663.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m1430.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m14341.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m10374.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/3813.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/11534.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/10636.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/10933.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m10352.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m13378.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m5631.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m4831.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m9433.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m14385.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m10399.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/9533.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/1873.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m3136.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/3093.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m13360.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m7333.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/9633.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m5431.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m4933.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m3234.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/11132.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m13334.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m10325.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/9503.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m6838.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m8535.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m16381.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m12335.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/2923.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/11837.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m3032.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/10632.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m2532.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/12136.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m5033.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/11331.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m3332.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/11731.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m12354.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m7335.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/4273.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/8723.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m16320.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m6630.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m4438.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m3532.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/6333.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/7173.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/5273.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m6432.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/12738.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m8536.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/2453.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m15378.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m10393.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m7533.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m10365.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m853.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m6731.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/4463.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/10239.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/1593.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/8823.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/7663.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/14035.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m13346.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m6931.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m6639.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m15310.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/6943.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/13333.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m10326.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/4363.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m10311.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/12236.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m13303.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m16394.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m15372.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m13329.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/8293.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/11137.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m14354.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m8038.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m7637.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/12334.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/12037.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/5753.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/3803.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m9736.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m5834.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/8163.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/7493.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/6453.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/1683.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m14379.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m16380.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m11378.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m3830.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m14314.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m4131.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/9923.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/2323.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m11342.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m12378.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m2435.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m16384.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m7930.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/5893.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/2213.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m10368.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m11391.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m15389.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m8632.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m1831.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/14437.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/11635.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m16370.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/2733.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/12431.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/3363.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/1643.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/7673.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/12836.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/5683.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/3003.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/7793.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m9638.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/4213.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m3433.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/3563.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/3773.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/10234.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m1431.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m5538.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m5739.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/2423.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m11354.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m11321.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m12383.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m5139.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m11359.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m6537.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m7036.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/7013.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m1230.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/12031.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/6693.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m10381.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/12232.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m13348.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m163.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m213.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/7863.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m9234.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/8073.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m3738.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/4453.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m12357.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m4338.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m9432.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/1663.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m5233.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/13931.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m8032.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m7133.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/4433.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m6934.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m4335.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m413.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m3037.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m353.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/5523.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m4135.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m2236.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/14135.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m653.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/3413.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m14392.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/14330.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/14230.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/8133.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m4539.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/5423.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m1036.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/3233.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m15382.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/7483.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/2403.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m2133.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m14304.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/1433.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m8737.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m8330.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m7430.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/7813.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/9863.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m12310.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/3033.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m4033.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m4531.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/4523.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m13318.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/8433.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/9473.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m7230.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/8273.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m15337.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/3713.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/5283.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/12430.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/10837.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m8039.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/8553.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m10355.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m733.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m6232.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/5993.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m7233.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m16368.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m2934.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m2939.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m12333.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/7093.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/4933.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m14394.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/13039.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m1830.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m2634.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m11325.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/3833.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m15377.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m9637.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/10134.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/11535.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m15306.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/5153.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m1731.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/9223.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m2637.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/2793.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/6823.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m9734.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m11384.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m13394.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/6973.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m453.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/1573.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/10035.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m1134.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m1734.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m973.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m9539.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/13430.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m5038.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/14332.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/3783.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m8035.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m623.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m15303.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m11393.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/3253.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m16390.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/2823.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m13356.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m16345.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m13361.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m8936.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/8733.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m4837.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/4903.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/14133.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m8435.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/6573.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/10039.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/9603.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m6933.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m793.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/10032.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m15300.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m7031.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/3323.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m11379.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m4832.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/4653.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/5373.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/10634.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/2003.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m10354.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m14335.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/9993.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/3613.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/10236.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m13322.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m2733.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/4003.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/9023.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/10337.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m10347.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m2036.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m1236.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/5843.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m3936.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m7232.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m2339.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m5636.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m9236.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m11346.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/9013.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m10377.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/14436.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m15317.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/7503.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m8432.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m7330.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/6643.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/7063.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m16325.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/2373.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/2563.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m4337.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m2233.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/7243.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m2933.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/4083.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m12320.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m2537.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m13320.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m843.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/3823.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m4030.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m1237.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/5853.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m3834.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/12632.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/4883.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m7838.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m3336.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/10435.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/14237.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m15355.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/13635.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m2639.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m6439.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/13237.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m16308.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m11332.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m11383.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/1063.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/3553.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m10322.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m7439.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/11138.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/3513.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m16329.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/11838.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m10375.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m3537.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/12734.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/6133.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m10388.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/13834.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m16305.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m2134.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/5693.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/13239.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m3034.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m14393.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m5539.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/1033.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m373.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/13836.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m6034.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m10350.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/7353.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m3838.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m15373.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m9835.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/3743.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/7073.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/14031.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/12837.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/12138.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m5237.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m7734.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m6634.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m12389.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/3853.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m15349.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m5930.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m9431.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m10341.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/6963.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m13387.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m7630.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m803.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/5033.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/4693.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m3935.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/10737.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m11319.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m8634.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m11390.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m3630.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/7573.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m7738.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m9934.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m16336.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/8283.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/14138.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m3539.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m593.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m14315.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m8433.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/13632.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/11032.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m8533.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m4639.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/5533.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m13319.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m15381.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/13730.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/2553.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/6703.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/4643.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/9873.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m15325.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m16378.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/2743.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m11315.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m9037.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m5532.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m2032.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m6039.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m5737.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/10932.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/3423.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/14132.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m16300.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m14349.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m14368.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/9813.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m1832.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/4993.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/4923.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/1303.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m10370.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m4039.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/9713.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m5938.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m14337.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/10138.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/9303.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m9237.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m15350.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m9837.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m11324.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m5333.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m6737.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/3133.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/4973.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/14335.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/5113.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/7743.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/2223.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/7943.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m12355.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m4432.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/1933.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/2473.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/12030.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m10323.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m2932.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m14300.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/8383.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/1283.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m7735.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/8373.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/4853.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m16373.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/4753.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m8134.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m2231.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/7723.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/10339.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m5732.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/8393.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m8839.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m1532.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/14039.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/6373.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/1213.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m16318.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m13323.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m5239.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/8913.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/3703.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m1633.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m2631.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/4793.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/7993.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m15375.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/4143.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m10337.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m16375.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m12358.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m15336.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/12135.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/9133.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m8232.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/11530.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m7833.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/11438.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m3732.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m10319.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m13324.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m15395.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m13381.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m16348.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/12535.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m5130.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/13932.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/8233.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m11372.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/4293.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/5313.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m15369.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/11734.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m6338.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m14356.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m4939.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m16316.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/5513.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/10130.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m4134.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m523.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m12368.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/1943.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m6334.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m15330.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m10364.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m12302.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/4573.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m15357.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/2663.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/2763.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/5713.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m1631.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/2903.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m10356.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/2013.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/7403.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m7338.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m463.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/8813.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m2332.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/11937.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/13331.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m8633.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/13530.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/8173.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/7123.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m123.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m8430.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/13739.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m2736.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/12335.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/12231.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/10330.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/11835.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m13383.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m10349.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m8938.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/5333.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/6723.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/3183.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m6435.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/5593.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m9634.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m14322.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m13308.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/12839.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m11356.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/8653.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m11320.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m4339.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m11358.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m3735.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/2623.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m13341.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m13330.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m113.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m1333.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/3943.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m3634.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m9635.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/1523.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/10832.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m16340.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m10360.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m9436.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/10036.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/14339.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m8438.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/6043.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/8613.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m9630.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/6663.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m8033.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/13935.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m3530.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/8763.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/8543.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m563.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m6433.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m9739.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m12347.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/2843.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m12326.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m5936.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m4836.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/13534.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/3653.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/6103.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m7839.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m6037.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m5336.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/5483.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/9553.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m9538.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/8453.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m12341.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/7293.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/2463.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m15390.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m10395.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m6032.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m9738.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/9963.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/2233.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/8573.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/4943.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m3335.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m8931.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m7238.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m12321.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m6235.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m6533.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m7931.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/m6834.graph.json",
                "/home/elad/Documents/kcol/tmp/json/train/14334.graph.json"
    ]


device_id = 0
if len(sys.argv) > 1:
  device_id= int(sys.argv[1])
  print("device is", device_id)
torch.cuda.set_device(device_id)

EPOC_STEP = 50
# CHECK_POINT_PATH = './checkpoints/second_fix'
CHECK_POINT_PATH = f'./checkpoints/dist_from_one_another9'
DATA_SET_PATH = '/content/pickles/pickles/'  # '/content/drive/MyDrive/project_MSC/train_jsons'  #
DEVICE =  'cuda'  # 'cuda'  #

def save_model(epoc, model, acc, loss, opt, test_acc, best):
    new_best = best<test_acc

    best = max(best, test_acc)
    dt = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    os.makedirs(CHECK_POINT_PATH, exist_ok=True)
    save_obj = {
            'epoc': epoc,
            'model': model.save_all_attributes(),
            'acc': acc[-1],
            'loss': loss[-1],
            'optimizer_state_dict': opt.state_dict(),
            'test_acc': test_acc,
            "best": best,
        }
    torch.save(save_obj, '{}/checkpoint_no_V_{}_{}.pt'.format(CHECK_POINT_PATH, dt, epoc))
    if new_best: torch.save(save_obj, '{}/best.pt'.format(CHECK_POINT_PATH))
    torch.save(save_obj, '{}/latest.pt'.format(CHECK_POINT_PATH))
    return best

def load_from_tf(gcp):
    gcp.v_normal.data = torch.tensor([-0.169884145,0.0430190973,0.091173254,-0.0339165181,0.0643557236,0.145693019,0.112589225,-0.0952830836,0.0254595,-0.0693574101,-0.0650991499,0.235404313,-0.31420821,-0.0290404037,-0.161913335,-0.09325625,0.298154235,-0.169444725,-0.207124308,0.0723744854,-0.0849481523,0.0168008488,0.00895659439,-0.0171319768,-0.127776787,-0.0971129909,-0.0536339432,0.168108433,0.177107826,0.320735186,-0.0755678415,0.139883056,-0.388966531,-0.0078522,-0.00130009966,0.143557593,0.035293255,-0.12994355,0.1157846,-0.121418417,-0.115577929,0.0780592263,-0.194125444,0.113405302,0.244302094,-0.0874284953,-0.0544838,0.0926826522,0.0209452771,0.0718942657,0.0228996184,0.298201054,0.0192331262,-0.0319460481,-0.17595163,-0.0833073,0.0334902816,0.14013885,-0.14659746,0.181580797,-0.00996331591,-0.0195714869,0.160506919,0.0497409627]).to(args.device) #torch.Tensor(loaded_dict["V_init:0"]).to(args.device)
    gcp.c_rand.data = torch.tensor([[0.569332063,0.19621861,0.936044037,0.0672274604,0.989149,0.916594744,0.754,0.431524485,0.445979536,0.333774686,0.732518792,0.822434127,0.711422324,0.753830671,0.836414278,0.209573701,0.527794242,0.3339068,0.832167804,0.6979146,0.807687044,0.690893054,0.00416331459,0.971259296,0.615243,0.69255811,0.669207,0.670641,0.85558778,0.00144830858,0.76548326,0.409540862,0.888088107,0.717633903,0.584715724,0.263450205,0.459266245,0.986697912,0.698782682,0.63641417,0.400523841,0.221628249,0.405968219,0.579900086,0.725307345,0.455515683,0.131517351,0.763612092,0.928811967,0.349458158,0.832664609,0.914531469,0.495537758,0.163773,0.827578843,0.815654,0.429762304,0.835437894,0.323074102,0.756760597,0.627905488,0.249528378,0.8888852,0.242653042]]).to(args.device)
    # return

    # loaded_dict = {item[0]: item[1] for item in np.load('./tf_data.npz', allow_pickle=True)['arr_0']}
    loaded_dict = {item: value for item, value in np.load('./original.npz', allow_pickle=True).items()}
    for k in loaded_dict:
        print(k, loaded_dict[k].shape)
    gcp.mlpV.l1.weight.data = torch.Tensor(loaded_dict["graph-coloring/V_msg_C_MLP_layer_1/kernel:0"]).to(args.device).T
    gcp.mlpV.l1.bias.data = torch.Tensor(loaded_dict["graph-coloring/V_msg_C_MLP_layer_1/bias:0"]).to(args.device)
    gcp.mlpV.l2.weight.data = torch.Tensor(loaded_dict["graph-coloring/V_msg_C_MLP_layer_2/kernel:0"]).to(args.device).T
    gcp.mlpV.l2.bias.data = torch.Tensor(loaded_dict["graph-coloring/V_msg_C_MLP_layer_2/bias:0"]).to(args.device)
    gcp.mlpV.l3.weight.data = torch.Tensor(loaded_dict["graph-coloring/V_msg_C_MLP_layer_3/kernel:0"]).to(args.device).T
    gcp.mlpV.l3.bias.data = torch.Tensor(loaded_dict["graph-coloring/V_msg_C_MLP_layer_3/bias:0"]).to(args.device)
    gcp.mlpV.l4.weight.data = torch.Tensor(loaded_dict["graph-coloring/V_msg_C_MLP_layer_4/kernel:0"]).to(args.device).T
    gcp.mlpV.l4.bias.data = torch.Tensor(loaded_dict["graph-coloring/V_msg_C_MLP_layer_4/bias:0"]).to(args.device)
    gcp.mlpC.l1.weight.data = torch.Tensor(loaded_dict["graph-coloring/C_msg_V_MLP_layer_1/kernel:0"]).to(args.device).T
    gcp.mlpC.l1.bias.data = torch.Tensor(loaded_dict["graph-coloring/C_msg_V_MLP_layer_1/bias:0"]).to(args.device)
    gcp.mlpC.l2.weight.data = torch.Tensor(loaded_dict["graph-coloring/C_msg_V_MLP_layer_2/kernel:0"]).to(args.device).T
    gcp.mlpC.l2.bias.data = torch.Tensor(loaded_dict["graph-coloring/C_msg_V_MLP_layer_2/bias:0"]).to(args.device)
    gcp.mlpC.l3.weight.data = torch.Tensor(loaded_dict["graph-coloring/C_msg_V_MLP_layer_3/kernel:0"]).to(args.device).T
    gcp.mlpC.l3.bias.data = torch.Tensor(loaded_dict["graph-coloring/C_msg_V_MLP_layer_3/bias:0"]).to(args.device)
    gcp.mlpC.l4.weight.data = torch.Tensor(loaded_dict["graph-coloring/C_msg_V_MLP_layer_4/kernel:0"]).to(args.device).T
    gcp.mlpC.l4.bias.data = torch.Tensor(loaded_dict["graph-coloring/C_msg_V_MLP_layer_4/bias:0"]).to(args.device)
    gcp.V_vote_mlp.l1.weight.data = torch.Tensor(loaded_dict["V_vote_MLP_layer_1/kernel:0"]).to(args.device).T
    gcp.V_vote_mlp.l1.bias.data = torch.Tensor(loaded_dict["V_vote_MLP_layer_1/bias:0"]).to(args.device)
    gcp.V_vote_mlp.l2.weight.data = torch.Tensor(loaded_dict["V_vote_MLP_layer_2/kernel:0"]).to(args.device).T
    gcp.V_vote_mlp.l2.bias.data = torch.Tensor(loaded_dict["V_vote_MLP_layer_2/bias:0"]).to(args.device)
    gcp.V_vote_mlp.l3.weight.data = torch.Tensor(loaded_dict["V_vote_MLP_layer_3/kernel:0"]).to(args.device).T
    gcp.V_vote_mlp.l3.bias.data = torch.Tensor(loaded_dict["V_vote_MLP_layer_3/bias:0"]).to(args.device)
    gcp.V_vote_mlp.l4.weight.data = torch.Tensor(loaded_dict["V_vote_MLP_layer_4/kernel:0"]).to(args.device).T
    gcp.V_vote_mlp.l4.bias.data = torch.Tensor(loaded_dict["V_vote_MLP_layer_4/bias:0"]).to(args.device)
    gcp.LSTM_v.fc.weight.data = torch.Tensor(loaded_dict["graph-coloring/V_cell/layer_norm_basic_lstm_cell/kernel:0"]).to(args.device).T
    gcp.LSTM_v.ln_ih.gamma.data = torch.Tensor(loaded_dict["graph-coloring/V_cell/layer_norm_basic_lstm_cell/input/gamma:0"]).to(args.device)
    gcp.LSTM_v.ln_ih.beta.data = torch.Tensor(loaded_dict["graph-coloring/V_cell/layer_norm_basic_lstm_cell/input/beta:0"]).to(args.device)
    gcp.LSTM_v.ln_ho.gamma.data = torch.Tensor(loaded_dict["graph-coloring/V_cell/layer_norm_basic_lstm_cell/output/gamma:0"]).to(args.device)
    gcp.LSTM_v.ln_ho.beta.data = torch.Tensor(loaded_dict["graph-coloring/V_cell/layer_norm_basic_lstm_cell/output/beta:0"]).to(args.device)
    gcp.LSTM_v.ln_hf.gamma.data = torch.Tensor(loaded_dict["graph-coloring/V_cell/layer_norm_basic_lstm_cell/transform/gamma:0"]).to(args.device)
    gcp.LSTM_v.ln_hf.beta.data = torch.Tensor(loaded_dict["graph-coloring/V_cell/layer_norm_basic_lstm_cell/transform/beta:0"]).to(args.device)
    gcp.LSTM_v.ln_hc.gamma.data = torch.Tensor(loaded_dict["graph-coloring/V_cell/layer_norm_basic_lstm_cell/forget/gamma:0"]).to(args.device)
    gcp.LSTM_v.ln_hc.beta.data = torch.Tensor(loaded_dict["graph-coloring/V_cell/layer_norm_basic_lstm_cell/forget/beta:0"]).to(args.device)
    gcp.LSTM_v.ln_hcy.gamma.data = torch.Tensor(loaded_dict["graph-coloring/V_cell/layer_norm_basic_lstm_cell/state/gamma:0"]).to(args.device)
    gcp.LSTM_v.ln_hcy.beta.data = torch.Tensor(loaded_dict["graph-coloring/V_cell/layer_norm_basic_lstm_cell/state/beta:0"]).to(args.device)
    gcp.LSTM_c.fc.weight.data = torch.Tensor(loaded_dict["graph-coloring/C_cell/layer_norm_basic_lstm_cell/kernel:0"]).to(args.device).T
    gcp.LSTM_c.ln_ih.gamma.data = torch.Tensor(loaded_dict["graph-coloring/C_cell/layer_norm_basic_lstm_cell/input/gamma:0"]).to(args.device)
    gcp.LSTM_c.ln_ih.beta.data = torch.Tensor(loaded_dict["graph-coloring/C_cell/layer_norm_basic_lstm_cell/input/beta:0"]).to(args.device)
    gcp.LSTM_c.ln_ho.gamma.data = torch.Tensor(loaded_dict["graph-coloring/C_cell/layer_norm_basic_lstm_cell/output/gamma:0"]).to(args.device)
    gcp.LSTM_c.ln_ho.beta.data = torch.Tensor(loaded_dict["graph-coloring/C_cell/layer_norm_basic_lstm_cell/output/beta:0"]).to(args.device)
    gcp.LSTM_c.ln_hf.gamma.data = torch.Tensor(loaded_dict["graph-coloring/C_cell/layer_norm_basic_lstm_cell/transform/gamma:0"]).to(args.device)
    gcp.LSTM_c.ln_hf.beta.data = torch.Tensor(loaded_dict["graph-coloring/C_cell/layer_norm_basic_lstm_cell/transform/beta:0"]).to(args.device)
    gcp.LSTM_c.ln_hc.gamma.data = torch.Tensor(loaded_dict["graph-coloring/C_cell/layer_norm_basic_lstm_cell/forget/gamma:0"]).to(args.device)
    gcp.LSTM_c.ln_hc.beta.data = torch.Tensor(loaded_dict["graph-coloring/C_cell/layer_norm_basic_lstm_cell/forget/beta:0"]).to(args.device)
    gcp.LSTM_c.ln_hcy.gamma.data = torch.Tensor(loaded_dict["graph-coloring/C_cell/layer_norm_basic_lstm_cell/state/gamma:0"]).to(args.device)
    gcp.LSTM_c.ln_hcy.beta.data = torch.Tensor(loaded_dict["graph-coloring/C_cell/layer_norm_basic_lstm_cell/state/beta:0"]).to(args.device)

def decompose_to_submatrices(M_vv, features, split_list):
    start_idx_i = 0
    start_idx_j = 0
    submatrices = []
    subfeatures = []
    for size_i in range(0, len(split_list), 1):
        i = split_list[size_i]
        # j = split_list[size_i+1]
        end_idx_i = start_idx_i + i
        # end_idx_j = start_idx_j + j
        # submatrix = M_vv[start_idx_i:end_idx_i, start_idx_j:end_idx_j]
        submatrix = M_vv[start_idx_i:end_idx_i, start_idx_i:end_idx_i]
        submatrices.append(submatrix)
        subfeature = features[start_idx_i:end_idx_i, :]
        subfeatures.append(subfeature)
        start_idx_i = end_idx_i
        # start_idx_j = end_idx_j
    return submatrices, subfeatures

def adjacency_loss(V_ret, M_vv, margin=1.0, device='cpu'):
    """
    Compute the loss to make embeddings of adjacent nodes far apart.

    Parameters:
    - V_ret: Tensor of node embeddings, shape (num_nodes, embedding_size)
    - M_vv: Adjacency matrix, shape (num_nodes, num_nodes)
    - margin: The minimum distance between embeddings of adjacent nodes
    - device: The device on which to perform calculations

    Returns:
    - loss: The computed adjacency loss
    """
    # num_nodes = V_ret.shape[0]
    # loss = 0.0
    #
    # # Convert M_vv to boolean for adjacency checking
    # adjacency_matrix = M_vv.to(device) > 0
    #
    # count = 0
    # for i in range(num_nodes):
    #     for j in range(i + 1, num_nodes):
    #         if adjacency_matrix[i, j]:
    #             count += 1
    #             # Calculate Euclidean distance between embeddings of adjacent nodes
    #             distance = torch.norm(V_ret[i] - V_ret[j], p=2)
    #             # Calculate loss for this pair if distance is less than the margin
    #             loss += torch.max(torch.tensor(0.0, device=device), margin - distance) ** 2
    # return loss/count if count>0 else loss


    ############# too much memory
    # Step 1: Identify adjacent node pairs
    adj_indices = torch.nonzero(M_vv > 0, as_tuple=False)

    # Step 2: Gather embeddings of adjacent nodes
    node_embeddings_i = V_ret[adj_indices[:, 0]]
    node_embeddings_j = V_ret[adj_indices[:, 1]]

    # Step 3: Calculate distances for adjacent pairs only
    # distances = torch.norm(node_embeddings_i - node_embeddings_j, dim=1, p=2)
    # distances, argmins = scatter_min(distances, adj_indices[:, 0])
    distances = (node_embeddings_i * node_embeddings_j).sum(1) ** 2
    distances, argmaxs = scatter_max(distances, adj_indices[:, 0])

    # Step 4: Compute loss
    # loss = torch.mean(
    #     torch.max(
    #         torch.zeros(size=(1,), device=device),
    #         margin - distances) ** 2)
    loss1 = torch.mean(distances)
    loss1 = torch.max(distances)
    loss1 = torch.sum(distances)
    loss2 = torch.min(torch.max(
            torch.zeros(size=(1,), device=device),
            margin - V_ret.norm(dim=1, p=2)
    ))
    loss2 = torch.sum(torch.max(
        torch.zeros(size=(1,), device=device),
        margin - V_ret.norm(dim=1, p=2)
    ))
    loss = loss1 + loss2
    return loss

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--graph_dir', default='./data_json')
    parser.add_argument('--graph_dir', default="/home/elad/Documents/kcol/GCP_project/data_json/data")
    # parser.add_argument('--graph_dir', default="/home/elad/Documents/kcol/tmp/json/test")
    #parser.add_argument('--graph_dir', default="/home/elad/Documents/kcol/tmp/json/train")
    # parser.add_argument('--graph_dir', default='/home/shellad/kcol/GNN_GCP/json_data')
    parser.add_argument('--embedding_size', type=int, default=64)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--tmax', type=int, default=30)
    parser.add_argument('--epochs', type=int, default=5300)
    parser.add_argument('--device', type=str, default='cuda')
    #parser.add_argument('--check_path', type=str, default="checkpoints/from_tensortflow2/best.pt")
    # parser.add_argument('--check_path', type=str, default="checkpoints/3col2.pt")
    # parser.add_argument('--check_path', type=str, default="checkpoints/dist_from_one_another5/checkpoint_no_V_2024_10_07_18_49_12_18.pt")
    parser.add_argument('--check_path', type=str, default=None)
    parser.add_argument('--test', type=bool, default=False)
    args = parser.parse_args()

    embedding_size = args.embedding_size
    gcp = GCPNet(embedding_size, tmax=args.tmax, device=args.device)
    gcp.to(args.device)
    # if torch.cuda.device_count() > 1:
    #     devices = [0,1,2]
    #     print("Let's use", len(devices), "GPUs!")
    #     gcp = torch.nn.DataParallel(gcp, device_ids=devices)
    if args.test:
        print("test - eval")
        gcp.eval()

    ds = GraphDataSet(
        args.graph_dir,
        batch_size=args.batch_size,
        filter=lambda f: f.split("/")[-1][3]=="3",
        # filter=lambda f: sum([f.split("/")[-1] in c3f for c3f in col3_files])>0,
        # filter=lambda g: g.split('/')[-1].split("_")[1]=="5"
    )

    test_ds = GraphDataSet(
        # os.path.join("/", *args.graph_dir.split("/")[:-1], "test"),
        # "/home/elad/Documents/kcol/tmp/json/test",
        "/home/elad/Documents/kcol/tmp/json/train",
        filter=lambda f: sum([f.split("/")[-1] in c3f for c3f in col3_files]) > 0,
        # "/home/shellad/kcol/tmp/jsons/test",
        batch_size=args.batch_size,
        # filter = lambda g: (g.split('/')[-1].split("_")[1] == "5")
    ) # GraphDataSet(args.graph_dir, batch_size=args.batch_size, limit=1000)
    test_dl = DataLoader(test_ds, batch_size=1, shuffle=True)

    # load_from_tf(gcp)

    opt = torch.optim.Adam(
        gcp.parameters(),
        #[
        #    {'params': gcp.c_init.parameters()},
        #    {'params': gcp.v_init.parameters()}
        #],
        # lr=2e-5,
        lr=2e-5,
        weight_decay=1e-10
    )
    lr_decay_factor=0.5
    lr_scheduler_patience=25
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=lr_decay_factor,patience=lr_scheduler_patience)
    critirion = torch.nn.BCEWithLogitsLoss(reduction="mean")
    l2norm_scaling = 1e-10
    global_norm_gradient_clipping_ratio = 0.5

    epoc_loss = []
    epoc_acc = []
    epoch_size = 100_000 / embedding_size #128
    accumulated_size = 0
    start_epoc = 0
    best = -1
    # load checkpoint
    if args.check_path is not None:
        print("loading:", args.check_path)
        checkpoint = torch.load(args.check_path, map_location="cpu")
        gcp.load_all_attributes(checkpoint['model'])
        # opt.load_state_dict(checkpoint['optimizer_state_dict'])
        if "best" in checkpoint:
            best = checkpoint["best"]

        epoc_acc = (checkpoint['acc'] if len(checkpoint['acc'].size()) >0 else [checkpoint['acc']] ) if hasattr(epoc_acc, '__iter__')  else [checkpoint['acc']] 
         

        start_epoch_string = args.check_path.split("/")[-1].split("_")[-1].split(".")[0]
        if "epoc" in checkpoint:
            start_epoc = checkpoint['epoc']
        if start_epoch_string.isdigit():
            start_epoc = int(start_epoch_string)

        start_epoc = checkpoint['epoc']
    gcp.to(args.device)
    #print("------------------------", args.check_path is not None, hasattr(epoc_acc, '__iter__'), epoc_acc)
    last_pred = None

    prev_train_flag = gcp.training
    if args.test: gcp.eval()
    for i in range(start_epoc, args.epochs if args.test is False else start_epoc+1):
        ds.shuffle()
        dl = DataLoader(ds, batch_size=1, shuffle=True)
        preds = []
        lables_agg = []
        plot_loss = 0
        plot_acc = 0
        t1 = time.perf_counter()
        print('Running epoc: {}'.format(i))
        for j, b in enumerate(dl):
            if j == epoch_size: break
            M_vv, labels, cn, split, M_vc = b
            labels = labels.squeeze()
            split = split.squeeze(0)
            split = [int(s) for s in split]
            M_vv = M_vv.squeeze()
            M_vc = M_vc.squeeze()

            v_size = M_vv.shape[0]
            M_vv = M_vv.to(device=args.device)
            M_vc = M_vc.to(device=args.device)

            opt.zero_grad()
            pred, means, V_ret, C_ret = gcp.forward(M_vv, M_vc, split, cn=cn)
            # l = critirion(means.to(DEVICE), torch.Tensor(labels).to(device=DEVICE))
            l = 0
            dist_l = 0
            sub_mats, sub_features = decompose_to_submatrices(M_vv, gcp.history, split_list=split)
            for sub_mat, sub_feat in zip(sub_mats, sub_features):
                dist_l += adjacency_loss(sub_feat, sub_mat, margin=1.0, device=DEVICE)
            # dist_l += adjacency_loss(gcp.history, M_vv, margin=10.0, device=DEVICE)
            l = l + dist_l/len(sub_mats)

            preds += pred.tolist() if len(pred.shape)!=0 else [pred.item()]

            lables_agg += labels.tolist() if len(labels.shape)!=0 else [labels.item()]
            plot_loss += l.detach()
            acc = ((pred.detach().cpu() > 0.5) == torch.Tensor(labels)).sum()/float(cn.shape[1])
            plot_acc += acc



            from sklearn.decomposition import PCA
            import plotly.express as px
            # if len(labels.size())>0:
            #     k = cn[0, 0].item() + (1 - int(labels[0].item()))
            # else:
            #     k = cn[0, 0].item() + (1 - int(labels.item()))
            # k = k
            # n = split[0]
            # adj = M_vv[:n, :n]
            # if k ==3:
            #    features = gcp.history[:n].clone().detach().cpu()
            #    # ass, cent = sklearn_k_means(features, k)
            #    ass = reassign_clusters_respecting_order(features, adj, k)
            #    is_k_col = is_k_color(adj, ass)
            #    close_ass = find_closest_kcoloring(ass, adj, k)
            #    markers = [int(a != c) for a, c in zip(ass, close_ass)]
            #    dist = sum(markers)
            #    print(k, acc, is_k_col, dist)
            # attributes_plot(adj, ass, features, k)
            # pca = PCA(3)
            # lower = pca.fit_transform(features.numpy())
            # x = lower[:, 0]
            # y = lower[:, 1]
            # z = lower[:, 2]
            #
            # degrees = adj.sum(axis=0)  # Assuming 'adj' is your adjacency matrix
            # fig = px.scatter(x=x, y=y, title='Scatter Plot of x vs. y nodes colored by degree',
            #                  labels={'x': 'x', 'y': 'y'},
            #                  color=degrees[:n])
            # fig.show()
            #
            # fig = px.scatter(x=x, y=y, title=f'Scatter Plot of x vs. y nodes {is_k_col} {dist}',
            #                  labels={'x': 'x', 'y': 'y'},
            #                  color=[f"{ass[i]}" for i in range(n)])
            # fig.show()
            # fig = px.scatter(x=x, y=y, title=f'Scatter Plot of x vs. y nodes closest assignment',
            #                  labels={'x': 'x', 'y': 'y'},
            #                  color=[f"{close_ass[i]}" for i in range(n)])
            # fig.show()
            # fig = px.scatter(x=x, y=y, title=f'Scatter Plot of x vs. y nodes closest assignment diff',
            #                  labels={'x': 'x', 'y': 'y'},
            #                  color=[f"{m}" for m in markers])
            # fig.show()
            # fig = px.scatter_3d(x=x, y=y, z=z, title=f'Scatter Plot of x vs. y nodes {is_k_col} {dist}',
            #                     labels={'x': 'x', 'y': 'y'},
            #                     color=[f"{ass[i]}" for i in range(n)])
            # fig.show()
            # fig = px.scatter_3d(x=x, y=y, z=z, title=f'Scatter Plot of x vs. y nodes closest assignment',
            #                     labels={'x': 'x', 'y': 'y'},
            #                     color=[f"{close_ass[i]}" for i in range(n)])
            # fig.show()
            # fig = px.scatter_3d(x=x, y=y, z=z, title=f'Scatter Plot of x vs. y nodes closest assignment diff',
            #                     labels={'x': 'x', 'y': 'y'},
            #                     color=[f"{m}" for m in markers])
            # fig.show()


            # import plotly.express as px
            # from sklearn.decomposition import PCA
            # from dataset import self_contained_min_coloring
            # # index = 0
            # # start = sum(split[:2*index])
            # # finish = start + split[2*index+2]
            # mat = M_vv[:42,:42]
            # ass, color_ = self_contained_min_coloring(mat)
            # pca = PCA(n_components=2)
            # lower_dimension_data = pca.fit_transform(V_ret.detach().cpu().numpy()[:42, :42])
            # fig = px.scatter(x=lower_dimension_data[:, 0], y=lower_dimension_data[:, 1],
            #     color=[f"{ass[c]}" for c in range(len(mat))],
            #     labels={'x': 'PC1', 'y': 'PC2'}, title='PCA Result')
            # fig.show()

            # print(cn.shape[1])
            # accumulated_size += 1 if len(pred.shape) == 0 else pred.shape[0]
            #####print(f"\t Epoch: {i} Batch: {j} Loss: {l:.4f} Acc: {acc:.4f} AVG_label:{torch.Tensor(labels).mean()} Mean_pred: {pred.mean():.4f} Mean_rounded_preb: {(pred>0.5).float().mean():.4f}")
            # print(f"\t\t lables: {labels.tolist()}\n\t\tpreds: {pred.tolist()}\n\t\t rounded_preds: {(pred>0.5).tolist()}")
            if not args.test:
                # initial_weights = {name: param.clone() for name, param in gcp.named_parameters()}

                ind_sum = 0
                # for n_i, l_value, c in zip(split[::2], labels, cn[0,::2]):
                #     end = ind_sum+n_i
                #     k_means_loss = []
                #     if l_value == 1:
                #         scaler = StandardScaler()
                #         v = V_ret[ind_sum:end].detach().cpu().numpy()
                #         km = KMeans(n_clusters=c.item(), random_state=0)
                #         v = scaler.fit_transform(v)
                #         km.fit(v)
                #         dists = km.transform(v)
                #         min_dist = dists.min(axis=1)
                #         avg_dist = min_dist.mean()
                #         k_means_loss.append(avg_dist/1000)
                #     ind_sum = end
                #     l += np.mean(k_means_loss) if len(k_means_loss) > 0 else 0
                ll = l
                # ll = l + l2norm_scaling * sum([param.norm() ** 2 for param in gcp.parameters()])
                ll.backward()
                torch.nn.utils.clip_grad_norm_(gcp.parameters(), global_norm_gradient_clipping_ratio)
                opt.step()
                # for name, param in gcp.named_parameters():
                #    if not torch.equal(initial_weights[name], param):
                #        # print(f"Weights updated for: {name}")
                #         pass
                #         break
                #     else:
                #        print(f"No weights were updated for {name}")
            print(f"\t Epoch: {i} Batch: {j} Loss: {l:.4f} Acc: {acc:.4f} AVG_label:{torch.Tensor(labels).mean()} Mean_pred: {pred.mean():.4f} Mean_rounded_preb: {(pred>0.5).float().mean():.4f}")
        t2 = time.perf_counter()
        print('Time: t2-t1={}'.format(t2-t1))
        plot_acc /= (j+1)
        plot_loss /= (j+1)
        epoc_acc.append(plot_acc)
        epoc_loss.append(plot_loss)

        # Test
        if not args.test:
            prev_train_flag_test = gcp.training
            gcp.eval()
            loss_agg = []
            with torch.no_grad():
                test_acc = 0
                for j, b in enumerate(test_dl):
                    if j == epoch_size: break
                    M_vv, labels, cn, split, M_vc = b
                    labels = labels.squeeze()
                    split = split.squeeze(0)
                    split = [int(s) for s in split]
                    M_vv = M_vv.squeeze()
                    M_vc = M_vc.squeeze()
                    v_size = M_vv.shape[0]
                    M_vv = M_vv.to(device=args.device)
                    M_vc = M_vc.to(device=args.device)
                    pred, means, V_ret, C_ret = gcp.forward(M_vv, M_vc, split, cn=cn)
                    # l = critirion(means.to(DEVICE), torch.Tensor(labels).to(device=DEVICE))
                    # loss_agg.append(l)
                    l = adjacency_loss(gcp.history, M_vv, margin=2.0, device=DEVICE)
                    test_acc += ((pred.detach().cpu() > 0.5) == torch.Tensor(labels)).sum()/float(cn.shape[1])
                test_acc /= (j+1)
            gcp.train(prev_train_flag_test)
            lr_scheduler.step(np.mean(loss_agg))
        else:
            test_acc = -1
        if best < test_acc or (i % EPOC_STEP) == 0 and not args.test:
            print('Saving model')
            best = save_model(i, gcp, epoc_acc, epoc_loss, opt, test_acc, best)
        best = save_model(i, gcp, epoc_acc, epoc_loss, opt, test_acc, best)
        print('ACC:{}\t Accuracy: {}\tLoss: {}\t Test: {}\t Max: {}'.format(sum(epoc_acc)/len(epoc_acc), plot_acc, plot_loss, test_acc, best),
              f"Average label: {sum([(l >= 0.5) for l in lables_agg]) / len(lables_agg)}",
              f"Average preds: {sum([(p >= 0.5) for p in preds]) / len(preds)}",
              f"learning rate: {opt.param_groups[0]['lr']}"
        )
        # print(
        #     #       'Accuracy: {}'.format(sum(epoc_acc)/len(epoc_acc)),
        #     #       f"Average acc: {sum([((l>=0.5) == (p>=0.5)) for l,p in zip(lables_agg, preds)]) / len(preds)}",
        #     #   f"Average label: {sum([(l >= 0.5) for l in lables_agg]) / len(lables_agg)}",
        #     #   f"Average preds: {sum([(p>=0.5) for p in preds])/len(preds)}",
        #     #       "" if last_pred is None else f"Last pred: {sum([(l>=0.5) == (p>=0.5) for l,p in zip(last_pred, preds)])/len(preds)}",
        #     )
        last_pred = preds
    gcp.train(prev_train_flag)


"""
import torch
from dataset import self_conrained_solve_csp
import numpy as np
from sklearn.decomposition import PCA
import plotly.express as px
ass = self_conrained_solve_csp(M_vv[:split[0],:split[0]], cn[0][0].item())
print(ass)
V_ret_np = V_ret.detach().cpu().numpy()
pca = PCA(n_components=3)
components = pca.fit_transform(V_ret_np[:split[0],:split[0]])
fig = px.scatter_3d(x=components[:, 0], y=components[:, 1], z=components[:, 2], color=[str(ass[i]) for i in range(split[0])], labels={'x': 'PC1', 'y': 'PC2'}, title='2D PCA Scatter Plot of V_ret')
fig.show()
"""
