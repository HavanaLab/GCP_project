import itertools
import io

import plotly.graph_objects as go
import cv2 as cv
import networkx as nx
import numpy as np
import argparse
import cProfile
import time
import datetime
import os
import pickle

import torch
from fontTools.varLib.cff import conv_to_int
from matplotlib import pyplot as plt
from pynndescent.distances import spearmanr
from pyparsing import restOfLine
from scipy.stats import spearmanr as sci_spearmanr
from sklearn.cluster import KMeans
from sympy.codegen.ast import continue_
from sympy.stats.sampling.sample_numpy import numpy
from torch.backends import cudnn
from scipy.stats import linregress
from torch.utils.data import DataLoader

from GeneratedGraphDataSet import GeneratedGraphs
from GraphDataSet import GraphDataSet
from gc_utils import attributes_plot, is_k_color, find_closest_kcoloring, sklearn_k_means, check_k_colorable_and_assign, \
    plot, reassign_clusters_respecting_order, calculate_sum_and_avg_degree_of_neighbors, count_neighbors_colors, \
    calculate_avg_distance, leave_one_out_knn, \
    initialize_nodes, classify_third, adjust_positions, plot_positions, lovasz_theta, \
    max_distance_between_adjacent_nodes, most_frequent_number, sdp_coloring, lovasz_theta_max_dist, greedyColoring, \
    greedy_coloring_least_conflicts, solve_max_3_cut_frieze_jerrum, run_max_k_cut, find_all_k_coloring, remove_cycles, \
    has_cycle, find_least_common_neighbor_color, last_change_compare, calc_dist_over_iteartions, plot_conf_over_time, \
    confidance_same_pair, confidance_pair
from graph_coloring_attributes import color_support_count, support_for_color, support_for_vertex_color_assignment
from model import GCPNet
from sklearn.preprocessing import StandardScaler
from scipy.sparse.csgraph import connected_components
import pandas as pd
import sys

import plotly.io as pio
from sklearn.decomposition import PCA
import plotly.express as px

from random_planted import create_gnp, create_planted
from utils.fs import FS
from utils.secrets import data_path
import json

pio.renderers.default = 'browser'

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
    device_id = int(sys.argv[1])
    print("device is", device_id)
torch.cuda.set_device(device_id)

EPOC_STEP = 50
# CHECK_POINT_PATH = './checkpoints/second_fix'
CHECK_POINT_PATH = f'./checkpoints/simple_dist1'
DATA_SET_PATH = '/content/pickles/pickles/'  # '/content/drive/MyDrive/project_MSC/train_jsons'  #
DEVICE = 'cpu'  # 'cuda'  #


def my_pca(X):
    mean = np.array([7.72914216e-02, 2.44451612e-01, 1.08308524e-01, 1.67811260e-01,
                     1.10065177e-01, 6.69563636e-02, 1.51939258e-01, 3.57034743e-01,
                     7.99783785e-03, 4.06010635e-02, 1.70105368e-01, 8.12189579e-02,
                     1.81952491e-02, 7.51252830e-01, 7.92243406e-02, 1.74674034e-01,
                     6.24253089e-03, 1.37604550e-01, 6.67794883e-01, 3.93953621e-02,
                     7.87022188e-02, 3.03224027e-02, 4.73975539e-01, 1.08665019e-01,
                     3.43645662e-02, 3.82529587e-01, 2.15074532e-02, 1.42164305e-01,
                     1.03629321e-01, 2.33280525e-01, 5.13885990e-02, 5.84835075e-02,
                     1.18602641e-01, 2.16735393e-01, 1.02754058e-02, 1.27752945e-01,
                     3.24407518e-02, 4.99587685e-01, 1.18626812e-02, 8.64896700e-02,
                     1.05060078e-01, 1.02464296e-02, 3.98678243e-01, 5.93513921e-02,
                     7.72760332e-01, 2.81583704e-02, 4.69585502e-04, 7.12858289e-02,
                     2.59262621e-01, 4.24927890e-01, 9.67609212e-02, 1.64530247e-01,
                     3.55822593e-01, 2.39568744e-02, 3.57776433e-02, 2.42890522e-01,
                     1.78776324e-01, 9.08188075e-02, 2.36418732e-02, 6.76383870e-03,
                     4.54049557e-01, 8.52325782e-02, 2.50591580e-02, 3.51875305e-01],dtype=np.float32)
    conv = np.array([[-0.014191407710313797, -0.036973461508750916, -0.015319195576012135, -0.006863933522254229, -0.029552606865763664, 0.04401310905814171, 0.04806527495384216, 0.10775912553071976, 0.001493746298365295, -0.012402212247252464, 0.002029111608862877, 0.002742036944255233, -0.0046645537950098515, 0.010945827700197697, 0.006831705570220947, -0.027434252202510834, 0.0008501028642058372, 0.08175426721572876, 0.3304249346256256, -0.004849710967391729, 0.0049196877516806126, 0.0070877717807888985, -0.04503341764211655, 0.009292783215641975, 0.0012005433673039079, -0.02571120113134384, 0.011485383845865726, -0.03830144926905632, -0.021731412038207054, 0.06381881982088089, -0.007633016910403967, 0.041794296354055405, -0.02553200162947178, 0.06409578025341034, 0.004532407969236374, -0.022031264379620552, -0.0001465384557377547, 0.04227320849895477, 0.0009624777594581246, -0.02132744900882244, 0.020710542798042297, 0.0007491883588954806, -0.08225784450769424, 0.0045112683437764645, 0.021058177575469017, 0.00165481714066118, -4.923090455122292e-05, 0.04529955983161926, 0.02310376800596714, -0.051127102226018906, 0.05411333963274956, 0.018252452835440636, 0.013046816922724247, -0.0013112922897562385, -0.000997226103208959, -0.010195069015026093, 0.09466071426868439, -0.018086830154061317, -0.006584304850548506, -0.0015286762500181794, -0.01340863760560751, -0.007829035632312298, -0.000559672829695046, -0.021891526877880096], [-0.0010672933422029018, 0.08225928246974945, -0.008850830607116222, 0.053623784333467484, -0.007197312545031309, -0.0012010810896754265, -0.012972445227205753, 0.003079829504713416, 0.0011530080810189247, -0.004677408374845982, 0.004437183495610952, 0.024189377203583717, -0.0021556192077696323, -0.05904068052768707, 0.014139862731099129, 0.012843672186136246, 0.001385794603265822, 0.02520533837378025, 0.0007645302684977651, 0.01212401408702135, 0.004158062394708395, 0.0050931102596223354, -0.001415675156749785, 0.017422067001461983, 0.007134494837373495, -0.0716075599193573, 0.0051691047847270966, -0.0156190674751997, -0.01679113507270813, 0.040046513080596924, -0.00826088897883892, 0.007592677138745785, 0.021839872002601624, -0.006329275202006102, 0.002337698359042406, 0.0565684475004673, 0.015356146730482578, 0.01389353908598423, 0.006378892809152603, -0.013314097188413143, -0.001990071963518858, 0.0018812799826264381, 0.022645194083452225, 0.016935328021645546, -0.06161293387413025, 0.0034081751946359873, 9.20595175557537e-06, 0.002716902643442154, 0.11260312050580978, -0.07886161655187607, 0.028769195079803467, 0.04480674862861633, 0.08647730946540833, -0.0012067630887031555, 0.004243249073624611, 0.03177155181765556, 0.0020195296965539455, -0.007751632947474718, 0.0002545798779465258, -0.0006192399305291474, -0.05054079741239548, 0.014934931881725788, 0.011104734614491463, -0.021626537665724754], [0.021617896854877472, 0.028074242174625397, 0.0376439243555069, 0.050626181066036224, 0.06786403805017471, 0.008976769633591175, 0.012419060803949833, -0.019489169120788574, 0.00011356230243109167, -0.006080986000597477, -0.0188505407422781, 0.003043556120246649, -0.002732011256739497, -0.09720676392316818, -0.002250042976811528, -0.023357350379228592, 0.004515248350799084, 0.03125692158937454, 0.05735187605023384, 0.001956160878762603, 0.0048978435806930065, 0.0007455899612978101, -0.05936417356133461, 0.013243168592453003, 0.0037787319160997868, -0.08367769420146942, 0.008016804233193398, 0.010980209335684776, -0.008639832027256489, 0.03584562614560127, -0.01249657291918993, 0.01835247129201889, 0.048133499920368195, -0.014025489799678326, -0.0020128744654357433, 0.02942398563027382, 0.031862109899520874, 0.04260319471359253, 0.008753963746130466, -0.013772365637123585, 0.01694904826581478, 0.005899702664464712, 0.25214141607284546, 0.005331461783498526, -0.030041906982660294, 0.019650515168905258, 0.0002875585632864386, 0.024052409455180168, 0.07481331378221512, -0.051851749420166016, 0.026943380013108253, 0.006378666963428259, 0.016015976667404175, 0.0030082156881690025, -0.0012734885094687343, 0.030596459284424782, 0.03362441807985306, 0.030515363439917564, 0.017387408763170242, 0.00164253159891814, -0.09095479547977448, -0.0008745449595153332, 0.011147832497954369, -0.054756294935941696], [0.0049137636087834835, -0.011230223812162876, 0.014232083223760128, 0.012540961615741253, 0.011776325292885303, -0.004099169746041298, 0.009161446243524551, -0.03323349729180336, -0.00010075273894472048, 0.0010954027529805899, -0.029522079974412918, -0.0036082190927118063, -0.002389889908954501, -0.08343550562858582, -0.005735175684094429, -0.026300400495529175, 0.003718176856637001, 0.005615641362965107, 0.021157920360565186, -0.002590046264231205, -0.0018873974913731217, -0.0018314640037715435, -0.04343394190073013, 0.0001910625578602776, 0.000909640861209482, 0.00044824168435297906, 0.0012586952652782202, 0.005976261105388403, 0.0007900175405666232, 0.02033907175064087, -0.0027241844218224287, 0.004710058681666851, 0.02846124954521656, -0.015520960092544556, -0.002408909145742655, 0.0032836247701197863, 0.0462646409869194, 0.026319749653339386, 0.007922010496258736, 0.004832967650145292, 0.017400916665792465, 0.006005583796650171, 0.1087479367852211, -0.004926520399749279, 0.01239723339676857, 0.009709984995424747, 0.00029080145759508014, 0.007452559657394886, 0.024733267724514008, 0.059128593653440475, 0.013128252699971199, -0.02477739378809929, -0.020853953436017036, 0.003491528332233429, -0.0004962991806678474, 0.017919817939400673, 0.015245736576616764, 0.011604554951190948, -0.00037644620169885457, 0.000179930473677814, -0.012396293692290783, -0.0018680206267163157, 0.005937147885560989, -0.030059637501835823], [0.0050223516300320625, 0.0452733188867569, 0.020485153421759605, 0.014566884376108646, 0.007893156260251999, -0.012387331575155258, 0.008179829455912113, 0.01942124031484127, 0.00059700250858441, -0.003445394104346633, -0.00531879672780633, 0.00540913874283433, 0.0009997442830353975, -0.01848558895289898, 0.0016337884590029716, -0.021472861990332603, 0.0007419264875352383, -0.005526097025722265, 0.020124927163124084, 0.0009820585837587714, -0.006052721757441759, -0.006270495243370533, -0.007443533744663, 0.017672505229711533, 0.004570998717099428, 0.014114834368228912, 0.0025030409451574087, -0.008502325974404812, -0.0004407523083500564, 0.058491580188274384, -0.003441838314756751, 0.000910450704395771, 0.006124467588961124, 0.0024490696378052235, 0.0005141401197761297, 0.031176848337054253, 0.0191224105656147, 0.00656474195420742, 0.007729134056717157, -0.0019511774880811572, 0.016803471371531487, 0.0038973367772996426, 0.04781828820705414, 0.0004144645354244858, 0.12197214365005493, 0.002033144235610962, 0.00019102638179901987, 0.004701072815805674, 0.06105932593345642, 0.08205655217170715, 0.012290378101170063, 0.010609329678118229, 0.00472935289144516, 0.006338829640299082, 0.0020934876520186663, 0.025915011763572693, 0.009786544367671013, -0.01067823451012373, 0.00016243007848970592, -0.0010835540015250444, -0.012926732189953327, -0.0006195954629220068, 0.01173632126301527, -0.01838427223265171], [-0.0025973976589739323, 0.037920381873846054, 0.016625769436359406, 0.00451655313372612, -0.0001967283897101879, -0.006735800299793482, 0.006353043485432863, 0.005470772739499807, 0.0008500355761498213, -0.001678922912105918, -0.0038165475707501173, 0.006876414641737938, -0.00013507877883967012, 0.009983235970139503, -0.0014864312252029777, -0.010332457721233368, 0.0009912519017234445, -0.0036819386295974255, 0.006119404919445515, 0.001074860105291009, -0.0021744261030107737, -0.00314823049120605, 0.004131467081606388, 0.002401557983830571, 0.005866702646017075, -0.006303554400801659, 0.002455862006172538, -0.008332760073244572, 0.0012469814391806722, 0.022910868749022484, 0.00024878338444978, 0.0007250256603583694, 0.007429464254528284, 0.00437816372141242, 0.0016060189809650183, 0.02044939622282982, 0.01530710980296135, 0.012107349932193756, 0.006810991559177637, -0.0037246819119900465, 0.015159440226852894, 0.001336855930276215, 0.001730753225274384, -5.7213772379327565e-05, 0.042003776878118515, 0.001598481321707368, 3.830355854006484e-05, 0.005864759441465139, 0.031778864562511444, 0.03749363124370575, 0.0037984615191817284, -0.00770142674446106, 0.00020569475600495934, 0.0026328107342123985, 0.0012620874913409352, 0.01722322218120098, -0.0019361660815775394, -0.00455409986898303, 0.00014168776397127658, -0.0006894517573527992, 0.004164449404925108, 0.0008643926121294498, 0.008072166703641415, -0.010028106160461903], [-0.0023646876215934753, 0.01768036000430584, 0.00993821769952774, 0.005764318630099297, -0.0022276879753917456, -0.004050074145197868, 0.0024289668072015047, 0.00549050560221076, 0.0010616076178848743, -0.0014590740902349353, -0.0008655600249767303, 0.007308545988053083, -0.00021798699162900448, 0.002442510100081563, -0.0003647056582849473, -0.003496139543130994, 0.0020602887962013483, 0.0005892342305742204, 0.003968785982578993, 0.0004578713560476899, -0.0005586827755905688, 0.0007656582165509462, 0.008067023009061813, 0.0026264276821166277, 0.00506601994857192, -0.002503079827874899, 0.0025508864782750607, -0.0064495522528886795, 0.00018244178500026464, 0.017024237662553787, -0.001567151746712625, 0.0028392812237143517, 0.013425607234239578, -0.0010858834721148014, 0.002400045283138752, 0.00746294716373086, 0.017989473417401314, 0.01801884174346924, 0.004067092202603817, -0.0034874349366873503, 0.010713349096477032, 0.00038940811646170914, 0.00028071500128135085, 0.0013262488646432757, 0.0215595755726099, 0.0014608160126954317, 9.750786375661846e-06, 0.004796224646270275, 0.00909844134002924, 0.02453675866127014, 0.0005478921812027693, -0.0008065302390605211, 0.004465264733880758, 0.0011407396523281932, 0.0005212593241594732, 0.015014154836535454, -0.0023733770940452814, -0.0026715933345258236, -0.0002707208041101694, -0.0005684660281985998, 0.004955415613949299, 0.003161246422678232, 0.0036988616921007633, -0.005263757426291704], [-0.00205020047724247, 0.011587466113269329, 0.007214551791548729, 0.007477374281734228, -0.004674454219639301, -0.0016240013064816594, -0.0016872589476406574, 0.003197062062099576, 0.0007181302644312382, -0.002258619759231806, 0.0041303872130811214, 0.006661106366664171, -0.0002528303302824497, 0.002642180072143674, 0.0007713584927842021, 0.0010875387815758586, 0.0031744493171572685, 0.0017258758889511228, 0.0065711191855371, -7.859510515118018e-05, 0.001212925766594708, 0.0018236810574308038, 0.009157606400549412, 0.0028835751581937075, 0.004080669023096561, 0.0018099475419148803, 0.0024731303565204144, -0.00616542249917984, -0.0009732043254189193, 0.013760612346231937, -0.002309465315192938, 0.005051547195762396, 0.018605906516313553, -0.0051650395616889, 0.0019454286666586995, 0.0023485166020691395, 0.018929434940218925, 0.01908169500529766, 0.0021896236576139927, -0.002554020145907998, 0.008489140309393406, 5.592086381511763e-05, 0.003129372140392661, 0.0026867499109357595, 0.008494112640619278, 0.001247369684278965, -1.7441994714317843e-05, 0.004639646504074335, 0.000730262603610754, 0.02151838131248951, -0.00018538598669692874, -0.00019541803339961916, 0.006403901614248753, 0.0023352939169853926, -0.00027568908990360796, 0.013660527765750885, -0.0010624427814036608, -0.002191221108660102, -9.757623774930835e-05, -0.0004790884268004447, 0.0022832138929516077, 0.00378477293998003, 0.0024847378954291344, 0.001383969560265541], [0.0008949321345426142, 0.005229006987065077, 0.006663590669631958, 0.006799347698688507, -0.007134703919291496, -0.0017511873738840222, -0.0021382018458098173, 0.0030799421947449446, 0.000853749574162066, -0.002369379159063101, 0.0034522246569395065, 0.006387883797287941, -0.0005613677203655243, 0.0024679629132151604, 0.0014942658599466085, 0.0020293628331273794, 0.003033412853255868, 0.003095447551459074, 0.004802518989890814, -0.00047001143684610724, 0.0010808578226715326, 0.0022957217879593372, 0.007242447696626186, 0.0034540423657745123, 0.003950103186070919, 0.0029959017410874367, 0.0032100065145641565, -0.0054021067917346954, 0.00024900518474169075, 0.011620933189988136, -0.003244792576879263, 0.004227655474096537, 0.02165742591023445, -0.0037179426290094852, 0.0017697872826829553, -0.0010681753046810627, 0.016293510794639587, 0.016750456765294075, 0.0009343987912870944, -0.00112765037920326, 0.006675738841295242, 0.0001624218566576019, 0.0021808415185660124, 0.003020165953785181, 0.002598702674731612, 0.0006163189536891878, -5.852795948158018e-05, 0.003994054161012173, -0.0028491078410297632, 0.01536265853792429, 0.0004305490874685347, 0.0008134724921546876, 0.0061841909773647785, 0.0021056095138192177, 0.00025253291823901236, 0.011423520743846893, -0.0016271381173282862, 0.0019115651957690716, 3.609409986893297e-06, -0.00021437474060803652, 0.0035449534188956022, 0.0044937171041965485, 0.0014894083142280579, 0.0007938159978948534], [0.0023027898278087378, 0.002764560980722308, 0.006880316883325577, 0.007289567496627569, -0.006030529271811247, -0.0017456515925005078, -0.0013140082592144608, 0.004158297087997198, 0.0005487934686243534, -0.0030769819859415293, 0.0034221243113279343, 0.004962341394275427, -0.0007059388444758952, 0.001695251208730042, 0.0036531034857034683, 0.00043842443847097456, 0.002868116833269596, 0.004467036575078964, 0.002938198857009411, -0.00046857877168804407, 0.0031002291943877935, 0.003457156242802739, 0.0033866434823721647, 0.001787787303328514, 0.0035340748727321625, 0.00500284880399704, 0.0025903224013745785, -0.0038652888033539057, 0.00014389197167474777, 0.009782173670828342, -0.00323565024882555, 0.005586651153862476, 0.02115471474826336, -0.004291883669793606, 0.0018580231117084622, -0.003396355314180255, 0.014893736690282822, 0.012068342417478561, -7.29187813703902e-05, -0.00080128014087677, 0.0056536998599767685, 9.64988284977153e-05, 0.0019331668736413121, 0.005528878886252642, 0.00068669457687065, -4.564377377391793e-05, -7.966526027303189e-06, 0.001866438309662044, -0.0032896525226533413, 0.011308087036013603, -0.0014597284607589245, -1.8142807675758377e-05, 0.006694035604596138, 0.0014800771605223417, 0.0012545859208330512, 0.008669246919453144, -0.001975251128897071, 0.003951947670429945, 0.0003408206393942237, -7.67671808716841e-05, 0.004023435991257429, 0.0032047091517597437, 0.0007818412850610912, 0.0008243668125942349], [0.002971535548567772, 0.0037864611949771643, 0.008007075637578964, 0.00600392185151577, -0.00522661255672574, -0.001445192494429648, -0.0013344193575903773, 0.002864375477656722, 0.000443286873633042, -0.003511585760861635, 0.0027311465237289667, 0.004688642453402281, -0.0011964028235524893, 0.0015946760540828109, 0.004962467122823, -0.0011001590173691511, 0.003178269835188985, 0.0006611411808989942, 0.003980073146522045, -0.00016411558317486197, 0.0029138075187802315, 0.0040226709097623825, 0.0010164804989472032, 0.0008046628790907562, 0.0038162085693329573, 0.006768153980374336, 0.0024459350388497114, -0.0016497864853590727, 0.0006476916605606675, 0.009260185994207859, -0.0033776010386645794, 0.0016000708565115929, 0.018599359318614006, -0.004491189029067755, 0.001762032276019454, -0.003251298563554883, 0.01112289261072874, 0.006914145313203335, -0.0005223616026341915, 0.0008486814913339913, 0.006174449343234301, -0.00013009297254029661, 0.0028071680571883917, 0.0040636220946908, 0.0010909088887274265, 0.0004618608800228685, -3.291679240646772e-05, 0.002016102895140648, -0.0041226851753890514, 0.007135794032365084, -0.0007054607267491519, 0.0015045111067593098, 0.0053335498087108135, 0.001769290305674076, 0.0010558335343375802, 0.005677517037838697, -0.002094886265695095, 0.004801764618605375, 0.00042273677536286414, 9.578466415405273e-05, 0.003684025490656495, 0.003676665248349309, 6.412507354980335e-05, 0.0006462912424467504], [0.005747959017753601, 0.002075926633551717, 0.010078291408717632, 0.004075665958225727, -0.004625245928764343, 0.0005687086377292871, 0.0004232288629282266, 0.003984050825238228, 8.32284931675531e-05, -0.0034886468201875687, 0.0045469398610293865, 0.0029806545935571194, -0.0004156946379225701, -0.0014609955251216888, 0.0044043296948075294, 0.0005496907397173345, 0.001851354376412928, 0.0009191656135953963, 0.0018700570799410343, -3.7213645555311814e-05, 0.0036820725072175264, 0.004533110186457634, -0.0002542346192058176, 0.001771323150023818, 0.003268796019256115, 0.0020719359163194895, 0.0020567786414176226, 0.004151918459683657, 0.001987511059269309, 0.005179041530936956, -0.003419575747102499, 0.002035769633948803, 0.015642577782273293, -0.004204666242003441, 0.0013558504870161414, -0.00390688655897975, 0.006886928807944059, 0.005828364286571741, -0.00018205847300123423, -0.00017908142763189971, 0.006618103012442589, 0.0017185867764055729, -0.0004972091992385685, 0.003590482985600829, 0.00052403419977054, -0.0007999922963790596, 1.6782680177129805e-05, 0.0011125177843496203, -0.0044703129678964615, 0.005244309548288584, -0.00047920458018779755, 0.0009359928662888706, 0.006153404247015715, 0.001627100515179336, 0.0015225399984046817, 0.004400417674332857, -0.000663162034470588, 0.006430888082832098, -0.00018494826508685946, 0.00013924149970989674, 0.0047195847146213055, 0.004190809093415737, -8.312136196764186e-05, 0.0009781431872397661], [0.00549548864364624, 0.0008912481716834009, 0.008153390139341354, 0.002876044949516654, -0.005193850491195917, 0.0012394245713949203, 0.0009570805123075843, 0.0038211827632039785, 0.0002863262197934091, -0.0034726792946457863, 0.0023678250145167112, 0.0027260673232376575, -0.0008647952927276492, 0.0002110931818606332, 0.004691609647125006, 0.001118210144340992, 0.0016670438926666975, 0.0021178005263209343, 0.0012119077146053314, -4.879299376625568e-05, 0.00409231660887599, 0.004478817339986563, 0.0006646178662776947, 0.0007111549493856728, 0.003221470396965742, 0.0060173338279128075, 0.0014453708427026868, 0.005045673344284296, 0.0013803759356960654, 0.005827090237289667, -0.002814165549352765, 0.0005068654427304864, 0.01680942252278328, -0.0032041878439486027, 0.0014360025525093079, -0.004518551751971245, 0.006405811291188002, 0.004565994720906019, -0.0007089098799042404, 0.0016815306153148413, 0.005928492173552513, 0.00021951785311102867, -0.0009061270393431187, 0.0030794055201113224, 0.0004544545372482389, -0.0010354568948969245, 1.0868221579585224e-05, -0.0004555015475489199, -0.005296201445162296, 0.00459423940628767, -0.00025832082610577345, 0.0021670686546713114, 0.00469987140968442, 0.0012009540805593133, 0.0023078015074133873, 0.0038906324189156294, 0.0004330750089138746, 0.0072932858020067215, 0.00026615880778990686, 0.0003511002869345248, 0.002476663561537862, 0.0033134869299829006, -0.00020552361092995852, -0.00025979397469200194], [0.0034534912556409836, 0.0008737932657822967, 0.007081111427396536, 0.0012708952417597175, -0.002658290322870016, 0.0017172556836158037, 0.002438314026221633, 0.0036083550658077, -0.00014399582869373262, -0.00344551308080554, 0.003703527385368943, 0.0032606821041554213, -0.0009414106607437134, -0.0005337140755727887, 0.005466385278850794, 0.00022473404533229768, 0.0010439010802656412, 0.0005024063866585493, 0.0005033460329286754, 0.0016184239648282528, 0.0031454386189579964, 0.004130546469241381, -0.00015185773372650146, 0.0011355336755514145, 0.002090005436912179, 0.0017924868734553456, 0.00238184561021626, 0.004180844873189926, 0.001688211690634489, 0.006468991283327341, -0.003308029379695654, 0.00013539542851503938, 0.012132911011576653, -0.004090023227035999, 0.0015987252118065953, -0.004274106118828058, 0.005228639580309391, 0.0032963810954242945, -0.0008236736757680774, 0.0017526965821161866, 0.006643855478614569, 0.0012827289756387472, -0.0015028408961370587, 0.0033899913541972637, 0.0008197639253921807, -0.0008230189559981227, 2.6516217985772528e-05, -0.0019451251719146967, -0.00415444141253829, 0.0038471471052616835, -0.0002832733152899891, 0.0030128899961709976, 0.0016271575586870313, 0.002577716251835227, 0.002111211884766817, 0.0031603313982486725, 0.0015234962338581681, 0.008389146998524666, -6.651677540503442e-05, 0.00038120883982628584, 0.002072330331429839, 0.0036499297711998224, -0.00043815828394144773, 0.000636853335890919], [0.004927172791212797, -0.0012440693099051714, 0.005890808999538422, 0.0023199126590043306, -0.003575992537662387, 0.003186753485351801, 0.003124874085187912, 0.0031565360259264708, -0.00034756434615701437, -0.003816382959485054, 0.005318125244230032, 0.0029871989972889423, -0.0008624840411357582, 0.0010416432050988078, 0.0053709521889686584, -0.00027275554020889103, 0.0007510685245506465, 0.002005220390856266, -0.0012093508848920465, 0.001877254224382341, 0.0035371894482523203, 0.004255643580108881, -0.0007296202820725739, 0.00028922202182002366, 0.0026489365845918655, 0.001968120690435171, 0.003309789579361677, 0.004084128886461258, 0.0014804641250520945, 0.004733097739517689, -0.002961797872558236, 0.0011957602109760046, 0.01069321483373642, -0.0037183091044425964, 0.0007861264748498797, -0.002899233717471361, 0.0038014447782188654, 0.0002918432583101094, -0.0005393960163928568, 0.0008493122877553105, 0.006038975436240435, 0.0024056395050138235, -0.000959716911893338, 0.004103444050997496, -0.0007620599353685975, -0.0006984671927057207, 7.432411803165451e-05, -0.0011043492704629898, -0.0023595558013767004, 0.0025776647962629795, 0.0008216665009967983, 0.0025364100001752377, 0.0023232875391840935, 0.0015836602542549372, 0.0013449069811031222, 0.003068824065849185, 0.0011342598590999842, 0.007872648537158966, -0.0007412030827254057, 0.0002510432677809149, 0.002412582514807582, 0.004129078704863787, 0.00044978162623010576, -0.0002292406279593706], [0.002200763439759612, -0.001253768103197217, 0.0058267684653401375, 0.0013959784992039204, -0.001112461439333856, 0.002124077407643199, 0.001463157357648015, 0.004060937557369471, -0.0004306612827349454, -0.002544309478253126, 0.0026396354660391808, 0.0021619440522044897, -0.0007520955405198038, 0.0002992212539538741, 0.004844021983444691, 0.0006525610224343836, 0.0007374832057394087, 0.00215132930316031, -0.00025928940158337355, 0.0011834197212010622, 0.004557919222861528, 0.0035680264700204134, -0.00045163807226344943, 0.0009042970486916602, 0.002527352888137102, 0.0022214786149561405, 0.0021899230778217316, 0.004636778030544519, 0.0018702003872022033, 0.002649437403306365, -0.003180131083354354, 0.0016954938182607293, 0.0076698255725204945, -0.004158298019319773, 0.0005703841452486813, -0.002293542493134737, 0.0022786406334489584, 0.0006680734222754836, -0.00033145202905870974, 0.0018355990760028362, 0.006338789593428373, 0.00040850898949429393, -0.0012269243597984314, 0.004289968404918909, 0.000308182614389807, -0.0011489256285130978, -6.594142178073525e-05, 0.00019503255316521972, -0.002883590990677476, 0.0030307958368211985, -0.00016552071610931307, 0.0011074742069467902, 0.003042839467525482, 0.001423283712938428, 0.0013705739984288812, 0.003537846961989999, 0.0019145893165841699, 0.0057930429466068745, 0.0003402341390028596, 0.0001888953847810626, 0.0033338521607220173, 0.0031431408133357763, -1.0840385584742762e-05, 0.0003832528309430927], [0.001910148304887116, -0.0006280787056311965, 0.006176614668220282, 0.001168404589407146, -0.002762222895398736, 0.0019119601929560304, 0.0023436611518263817, 0.003478608326986432, -0.00016681858687661588, -0.0021953752730041742, 0.005332570057362318, 0.0024742470122873783, -0.000520971545483917, -0.0003643989912234247, 0.004467648919671774, 0.00013495422899723053, 0.00016872562991920859, 0.0008682612096890807, -0.0015168803511187434, 0.0013575684279203415, 0.0037611837033182383, 0.0037590370047837496, -0.0004852607671637088, 0.0010014679282903671, 0.001959615619853139, 0.0016338538844138384, 0.0024670660495758057, 0.00422227568924427, 0.002044058870524168, 0.003475972218438983, -0.0025830029044300318, -0.0004932692972943187, 0.006003637798130512, -0.0034659376833587885, 0.0006281058304011822, -0.00033185590291395783, 0.0015475301770493388, 8.767111285123974e-05, 0.00019974831957370043, 0.0021803861018270254, 0.006716919597238302, 0.0011542898137122393, 0.00024590507382526994, 0.003281622426584363, -0.0005389579455368221, -0.00082809804007411, 2.9592300052172504e-05, -0.0005384260439313948, -0.002954301191493869, 0.002450142987072468, 0.0004473854787647724, 0.0010640239343047142, 0.001389575656503439, 0.0011611711233854294, 0.001360431662760675, 0.0032755446154624224, 0.0022726678289473057, 0.005731454584747553, -0.0005340304342098534, 0.00019976760086137801, 0.0014616238186135888, 0.002593210432678461, 0.0007252786890603602, -7.962525705806911e-05], [-0.00020750674593728036, 5.006171704735607e-05, 0.0040269773453474045, 0.00015077056013979018, -0.0009839877020567656, 0.0029749860987067223, 0.002516047563403845, 0.0015053206589072943, 0.00010359626321587712, -0.001469606882892549, 0.003923994954675436, 0.0033216860610991716, -0.0008157132542692125, -0.00030989054357632995, 0.005398396402597427, -2.4138793378369883e-05, -2.0105057046748698e-05, 0.0024693869054317474, -0.000459751725429669, 0.002520815236493945, 0.0038820886984467506, 0.002844029339030385, -0.0005080268019810319, 0.0007309819338843226, 0.0009305609855800867, 0.0010230924235656857, 0.0017759465845301747, 0.00611483957618475, 0.00252056447789073, 0.003705988172441721, -0.002001744695007801, 0.0019070229027420282, 0.004695972427725792, -0.002359007950872183, 0.0010067076655104756, 0.00024657981703057885, 0.0007543040555901825, 0.0031093312427401543, -4.5168937504058704e-05, -0.0005080776754766703, 0.00394123699516058, 0.0009385105222463608, -0.00048316005268134177, 0.0025487702805548906, 0.0006412434158846736, -0.0013545608380809426, 5.991717262077145e-05, -0.0011796234175562859, -0.0019428255036473274, 0.0022036267910152674, -0.0007108165300451219, 1.681211324466858e-05, -0.00042331390432082117, 0.001244658138602972, 0.0014911058824509382, 0.003368772566318512, 0.0007800086750648916, 0.006523279007524252, 0.00023665734624955803, 0.0006230478757061064, 0.0022448610980063677, 0.0027455762028694153, -0.0004788562946487218, 7.104258111212403e-05], [0.002835929626598954, -0.0009370542247779667, 0.004500452894717455, 0.001549550099298358, -0.001125843496993184, 0.0031578538473695517, 0.0021757050417363644, 0.0037809410132467747, 0.00025009014643728733, -0.0011415918124839664, 0.003274911316111684, 0.0020643912721425295, -0.0017324549844488502, -0.00023745904036331922, 0.003993842750787735, 0.0008671864634379745, -0.0005875417846255004, 0.0021619261242449284, -0.0009100776514969766, 0.001957523636519909, 0.003941952716559172, 0.0029083306435495615, 0.0009951217798516154, -0.0009977727895602584, 0.002613134915009141, 0.0025026951916515827, 0.0010021254420280457, 0.004798517562448978, 0.0013337505515664816, 0.002078052144497633, -0.0020283041521906853, 5.3333227697294205e-05, 0.0030653264839202166, -0.0007334297988563776, 0.0004637732054106891, 0.000778289744630456, 0.0008219605078920722, 0.0018931118538603187, -0.00030993687687441707, -0.0007483959197998047, 0.005687679164111614, 0.0009324123966507614, -0.00015561353939119726, 0.0027813101187348366, -0.0017076958902180195, -0.0006303832633420825, -0.00011626954801613465, -0.0019414061680436134, -0.002193000866100192, 0.0018910205690190196, -0.0011486324947327375, 0.001841567805968225, 9.93069406831637e-05, 0.0012195399031043053, 0.0012502523604780436, 0.004414021503180265, 0.003806693246588111, 0.0023087263107299805, -0.0014576894463971257, 0.00045329489512369037, 0.0033104191534221172, 0.0016829546075314283, -0.0006410992937162519, 0.000809625256806612], [0.0010107930283993483, 0.00019029139366466552, 0.0031450435053557158, -0.0016684108413755894, 0.00043481343891471624, 0.0026738003361970186, 0.00033749453723430634, 0.0018276127520948648, -0.000883142405655235, -0.0014491381589323282, 0.0042878626845777035, 0.0003163226065225899, 2.1913017917540856e-05, 0.0009138010791502893, 0.004139982163906097, 0.0014012983301654458, -0.0010557493660598993, 0.0031647710129618645, -0.0007090235594660044, 0.002204514341428876, 0.0030385807622224092, 0.003169207600876689, -0.001537139993160963, 0.0006014092359691858, 0.0020103221759200096, 0.00035049993311986327, 0.0019635686185210943, 0.005233776289969683, 0.0012667087139561772, 0.002465367317199707, -0.0015672215959057212, 8.253476698882878e-05, 0.00132751336786896, -0.0012348862364888191, 0.0005953588406555355, -0.0007630541804246604, 0.0003675610350910574, 0.0016721468418836594, 0.0010635070502758026, 0.0010402181651443243, 0.0031971908174455166, 0.00010354258847655728, -0.0002087234315695241, 0.00201935856603086, -0.0013764927862212062, -0.001725721056573093, -4.2860385292442515e-05, -0.000751482555642724, -0.0019317325204610825, 0.0014577268157154322, -3.7529360270127654e-05, 0.0016642167465761304, 0.00037307149614207447, 0.00019419424643274397, 0.0011399294016882777, 0.004095875658094883, 0.0022979851346462965, 0.005463908426463604, -0.0009355213260278106, 0.0005409567384049296, 0.002127954736351967, 0.002303805435076356, -0.00114592129830271, 0.0006038141436874866], [-0.0006797166424803436, -0.0004321462765801698, 0.003090363461524248, -0.0008189666550606489, -0.001826524268835783, 0.0006873024394735694, 0.0026059632655233145, 0.0022061436902731657, 0.00047507372801192105, -0.0015917547279968858, 0.004418814554810524, 0.0003599978517740965, -0.0009059000876732171, -0.000850516720674932, 0.003358737099915743, 0.00041215133387595415, -0.0006072779651731253, 0.0014696301659569144, -0.001381637528538704, 0.0027350736781954765, 0.003911629319190979, 0.002013654913753271, -0.0009302643011324108, -0.0007579087396152318, 0.00033034305670298636, 0.0018493259558454156, 4.082505256519653e-05, 0.0028934015426784754, 0.0017940417164936662, 0.0031386171467602253, -0.001361915492452681, -0.0013599427184090018, -0.000948911183513701, -0.00033228713436983526, 0.0003820399288088083, 0.0004920292412862182, 0.0008828016580082476, 0.0023615078534930944, 0.000528159667737782, -0.0004542660026345402, 0.0025274495128542185, -0.00035351881524547935, 0.0009686702978797257, 0.002167223719879985, -0.0011902472469955683, -0.001447516493499279, 1.1003236068063416e-05, 0.0005539460107684135, -0.0030525140464305878, 0.00026443347451277077, -0.0007675911765545607, -0.0007822101470082998, 0.0009714200859889388, 0.000931514659896493, 0.0007847912493161857, 0.0039026138838380575, 0.0037405577022582293, 0.005465857218950987, -0.001479031052440405, 0.0007208366878330708, 0.0023068131413310766, 0.0022673318162560463, -0.00017040708917193115, 0.002011257456615567], [0.0008116044919006526, 3.760659456020221e-05, 0.0027213545981794596, 0.0004411070258356631, -0.00158395036123693, -0.0005530558410100639, -7.216417725430802e-05, 0.0006580146728083491, -0.000577749393414706, -0.0016069636913016438, 0.0023939968086779118, 0.0008578518172726035, -0.0007201666594482958, 0.000383110367693007, 0.004435707814991474, 0.0009473551763221622, -0.0003569446562323719, 0.0010557391215115786, -0.0003231187874916941, 0.0023621839936822653, 0.0028440847527235746, 0.0019090456189587712, -0.0015278193168342113, 3.652693339972757e-05, 0.00034357100958004594, -0.0005269763641990721, 0.0006679511279799044, 0.00303502194583416, 0.0013735854299739003, 0.003514382056891918, -0.00034632591996341944, -0.0009631800348870456, -0.00022193821496330202, -0.001166601083241403, 0.00030750007135793567, 0.0005263627390377223, 0.0009325282298959792, 0.0020382446236908436, -0.0004293823440093547, -0.0005638664006255567, 0.0005890281172469258, 0.00045323892845772207, -0.00015988275117706507, 0.0015828479081392288, -0.00023921120737213641, -0.0007005532388575375, 8.684951171744615e-05, -0.0006175509770400822, -0.0009875387186184525, 0.0016497799661010504, 0.001068341196514666, -0.000267628493020311, -0.0017756994348019361, 0.0002745106176007539, 0.0005159115535207093, 0.0024646930396556854, 0.000986348488368094, 0.004677886143326759, -9.421827417099848e-05, 0.00010165423736907542, 0.0034087079111486673, 0.0005373019375838339, -0.0009887107880786061, 0.0014545031590387225], [0.0003248725552111864, 0.00024033729278016835, 0.002686232328414917, 0.0004696353862527758, -0.0003573351423256099, 0.00016527045227121562, 3.191032010363415e-05, 0.0007347705541178584, -9.622483048588037e-05, -0.002045769477263093, 0.003853326663374901, 0.00013738505367655307, 0.00034093004069291055, -0.0007068737177178264, 0.0031651898752897978, 0.0008761391509324312, -0.00031973194563761353, 3.654990723589435e-05, -7.961930532474071e-05, 0.0022022188641130924, 0.003577286610379815, 0.0017960056429728866, 0.0002081061975331977, -0.00048069856711663306, 0.001119305845350027, -0.0009694057516753674, 0.00015063813771121204, -1.0322003618057352e-05, 0.0016981075750663877, 0.0030236730817705393, -0.0007029069238342345, 0.0009664195240475237, -0.00026424339739605784, 0.0004185480356682092, -0.00015289764269255102, 0.001540560508146882, -0.00038921949453651905, 0.0009504753979854286, 0.0013795121340081096, 0.0017591002397239208, 0.002290679607540369, 0.0004572374455165118, 0.00047097052447497845, 0.0025713234208524227, -0.00017381472571287304, -0.0008449372253380716, -3.034055453099427e-06, -0.0005833802279084921, -0.0030521273147314787, 0.0006643307860940695, 6.732137990184128e-05, -0.0007341336458921432, 8.682611223775893e-05, 0.00035700452281162143, 0.0005135813262313604, 0.0016317857662215829, 0.00221068668179214, 0.004503124393522739, -0.001197171164676547, 0.00036938473931513727, 0.001989027950912714, 0.0008576529216952622, -0.000302252039546147, 0.003005721140652895], [0.00013916705211158842, -0.0007247111643664539, 0.0010675153462216258, -0.0008891323232091963, 9.134371794061735e-05, 0.00035999430110678077, -0.0006452365196309984, 0.002239848719909787, -0.00044227144098840654, -0.00261635216884315, 0.0035778721794486046, 0.001536979922093451, -8.244300988735631e-05, -0.00026087972219102085, 0.002436403650790453, 0.0004909979179501534, -0.0006920695304870605, 0.00041469125426374376, -0.0009494447731412947, 0.001728581264615059, 0.0029311038088053465, 0.0011338457698002458, -0.00010115899203810841, -6.20415903540561e-06, -0.00026849392452277243, -0.0008240120951086283, 0.0017482799012213945, 0.0018796961521729827, 0.000350386108038947, 0.0007986999116837978, -0.001973796170204878, -0.0009502390748821199, -0.00044434715528041124, -0.0001617702655494213, 3.943687261198647e-05, -0.0010725747561082244, -0.00017418863717466593, 0.0014205550542101264, 5.7609569921623915e-05, 0.0004201741539873183, 0.0021414561197161674, 0.0005929389735683799, -0.00017174845561385155, 0.0013498469488695264, -0.0004534239415079355, -0.0012045922921970487, 0.00014568772166967392, -0.00025531690334901214, 0.00012314680498093367, 0.0016207937151193619, 1.1582927982090041e-05, 0.00013391704123932868, -0.0010146236745640635, 0.0005030696629546583, 0.0003017716226167977, 0.003626408986747265, 0.00229168264195323, 0.003359916852787137, 0.000464228360215202, 0.0005409981240518391, 0.00328281382098794, 7.723055023234338e-05, -0.00012062019959557801, 0.00012941677414346486], [0.0011749101104214787, -0.00021143937192391604, 0.0017835346516221762, 0.00026106214500032365, 0.0004962524981237948, 0.0010711407521739602, -0.001328926649875939, 0.00048820622032508254, 4.499829810811207e-05, -0.0005844269180670381, 0.0024926718324422836, 6.577638123417273e-05, -3.1319425033871084e-05, -0.0007347119972109795, 0.0024417093954980373, 0.002491473685950041, -0.0007084736716933548, -0.00023889959265943617, 9.79975302470848e-05, 0.002334670163691044, 0.001737999846227467, 0.001336011802777648, -0.00019383231119718403, 0.000516554107889533, -0.00026657883427105844, -0.00036007180460728705, 0.0004109657311346382, 0.0008470412576571107, 0.0021590020041912794, 0.0020955007057636976, 0.001292268862016499, -0.002144975820556283, -0.001665781601332128, 0.0009026883635669947, -0.00026956095825880766, -2.6173216610914096e-05, 0.0010921312496066093, 0.00030286191031336784, 0.0005798337515443563, 6.141324411146343e-05, 0.0007881776546128094, 0.00039351661689579487, -5.896850416320376e-05, -0.001615195069462061, 2.7441121346782893e-05, 0.00018838472897186875, -6.273326289374381e-05, -0.0004447965766303241, -0.00014366542745847255, 0.0012373027857393026, 0.00017164713062811643, 1.7036898498190567e-05, -0.0009887678315863013, -0.0003563723585102707, -9.956018766388297e-05, 0.0027985144406557083, 0.00301517266780138, 0.004586446564644575, -2.7155772841069847e-05, 0.0005122060538269579, 0.0013267024187371135, 0.0003509963571559638, -0.0005032847402617335, 0.001794428681023419], [-0.0002974257804453373, 0.00010084526729770005, 0.0019057940226048231, -0.00017233696416951716, -0.00010696853860281408, 0.0017319228500127792, -0.002484923228621483, 0.00016308411431964487, 0.00026367235113866627, -0.0009555158903822303, 0.0023265585768967867, -0.0006718939403072, 0.000708063249476254, -0.0005297765601426363, 0.0030791827011853456, 0.0012341209221631289, -0.0005353622836992145, 0.0005883065750822425, 0.00040783052099868655, 0.0008947377791628242, 0.0004398288729134947, 0.001472537056542933, -0.0002686448278836906, -0.001330100349150598, -0.0011385995894670486, 0.00048365621478296816, -0.0011682489421218634, 0.00131574971601367, 0.002117984229698777, -0.0014357395702973008, -0.001275080838240683, -0.001628590514883399, -0.00022933221771381795, 0.00021691386064048856, -0.0006720618112012744, 0.0002760190691333264, -0.000701050041243434, 0.003075175918638706, 0.0007111898739822209, -0.00011929275206057355, 0.0003300116222817451, 0.00037275743670761585, -4.4963049731450155e-05, -0.00026631387299858034, 0.00017853699682746083, -0.0007686314638704062, 2.9139677280909382e-05, -0.0015470177168026567, -6.428780761780217e-05, 0.0010143083054572344, 7.505931716877967e-05, 0.001962024485692382, -0.0013298289850354195, 0.0003281584067735821, 0.0004151244065724313, 0.002691903617233038, 0.0025767493061721325, 0.004291769117116928, -0.000799258763436228, 0.0007204454159364104, 0.0009094002889469266, -0.0003696745843626559, -0.0002141419390682131, 0.0021310094743967056], [7.193252531578764e-05, 0.002000808948650956, 0.0008931327029131353, -0.0009932521497830749, -0.000924721360206604, 0.00031228907755576074, -0.0014256630092859268, 0.0014303014613687992, 4.420544792083092e-05, -0.001471076044254005, 0.00012510252417996526, -0.0008429225999861956, 0.00012198370677651837, -0.000212606304557994, 0.0012880767462775111, 0.0024000441189855337, -0.0009828299516811967, 0.0004774047702085227, 0.000695875147357583, 0.002136772032827139, -0.0002984808525070548, 0.0009308041771873832, 0.00016361090820282698, -0.00026754909777082503, -0.0006370821502059698, 0.0013265949673950672, 2.6892677851719782e-05, -0.0007437198073603213, 0.003084983676671982, 0.0008187390048988163, 0.0004562373796943575, 7.349422230618075e-05, -0.001453627715818584, -0.0014395869802683592, 0.00046068383380770683, -0.0006482292083092034, -0.0009393134969286621, 0.0034600412473082542, 0.001427497947588563, -0.00044911704026162624, -3.3320615330012515e-05, 0.0009361837292090058, 0.0011556801619008183, -0.0010700055863708258, -0.00016336700355168432, -0.0016797214047983289, -2.95811660180334e-05, -0.0008487219456583261, 0.00037003186298534274, -6.0003319958923385e-05, -0.00031619385117664933, -0.0010211115004494786, -0.000433766923379153, 4.0441580495098606e-05, 0.000668972497805953, 0.0021163467317819595, 0.001020530122332275, 0.004003996029496193, -0.0008054219069890678, 0.0009868205524981022, 0.0027891502249985933, -0.0005570207140408456, 0.0011011610040441155, 0.0013790641678497195], [-0.0009190468699671328, 0.001152888871729374, 0.0005891546024940908, -0.002345005515962839, 0.0012301905080676079, -0.0016420719912275672, 0.0010834739077836275, 0.0013665484730154276, 0.00037007039645686746, -0.0008755362941883504, 0.0006910068914294243, 0.00139744789339602, 0.0003659435024019331, -0.0004526435805018991, 0.000917992030736059, 0.00124850042629987, -0.0008053818601183593, 0.0002039207611232996, 0.0006334168720059097, 0.002510732039809227, 0.001345583121292293, -0.0002347233530599624, -0.0010324390605092049, -0.0014454539632424712, -0.001758077065460384, -0.00017412226588930935, 0.0012937667779624462, 0.0025902404449880123, 0.00011993218504358083, 7.607533916598186e-05, -0.0010333245154470205, -0.0006347322487272322, -0.0019158468348905444, -0.0010299640707671642, 0.0004391663533169776, 0.0009755421197041869, -0.0008456239593215287, 0.0027459622360765934, -0.00034712161868810654, -0.002682104241102934, 0.0010619666427373886, 0.00024720258079469204, 0.0009288392029702663, -0.0019336412660777569, -0.0003970411198679358, -0.001957495231181383, -0.0001194048672914505, -0.0015342921251431108, -0.0009475561673752964, 0.0003311167238280177, 0.0004111566231586039, -0.001339927315711975, 0.0006435184041038156, 0.0005317936302162707, 0.000251781108090654, 0.002000430366024375, 0.0002653285046108067, 0.002657637931406498, -0.0007520537474192679, 0.0007174154743552208, 0.002641360741108656, 0.0006339548854157329, -0.0008902943227440119, 0.0029247866477817297], [0.0008221527095884085, -0.0002011916076298803, -0.0008990962523967028, -0.0013255671365186572, -0.0010068585397675633, 0.0006368500762619078, -0.0008644463960081339, 0.0007402218179777265, 0.0002218464796897024, -0.0023835350293666124, 0.0014853012980893254, 0.0005895356298424304, 0.00011362943041604012, -0.00030012920615263283, 0.0024476340040564537, 0.001988743431866169, -0.001319708302617073, 2.5506515157758258e-05, -0.0003561579214874655, 0.000981834251433611, -0.000490064499899745, 0.0006882212474010885, -0.0010386097710579634, 0.0003217097546439618, -0.0016382781323045492, -0.001551697961986065, 0.00042001347173936665, -0.00044300672016106546, -0.0008027366129681468, -0.00025066069792956114, -0.00016508856788277626, -0.001787113375030458, -0.0011386907426640391, -0.0007846600492484868, -0.00015065375191625208, 0.0008743157959543169, -4.550113226287067e-05, 0.0016235392540693283, 0.0002509852056391537, -0.000870640273205936, -0.00041993780178017914, -0.0009783610003069043, -0.0006907099741511047, -0.00037894019624218345, -0.0005706445663236082, -0.001038393471390009, -7.174840447987663e-06, -0.0009393867221660912, 0.00014582122093997896, 0.000835910439491272, 0.0012866271426901221, 0.00010039914195658639, -0.0015014612581580877, 2.699160177144222e-05, 4.3306994484737515e-05, 0.002030986361205578, 0.0016407413640990853, 0.003033481305465102, -0.00033719820203259587, 0.00033047571196220815, 0.0009187622927129269, -0.0009512965916655958, -0.0009869869099929929, 0.0008371819276362658], [0.00022995646577328444, 0.0008597806445322931, 0.0005726201925426722, -0.0004914536257274449, -0.00017108439351432025, -0.0011516402009874582, -7.643716526217759e-05, 0.0017539763357490301, 0.0007001084741204977, -0.000465704855741933, 0.0004893185687251389, -0.0007973437895998359, 0.000580055289901793, -0.0003153800207655877, 0.003112966660410166, -0.0006218696944415569, -0.0017066241707652807, -0.00026145600713789463, -0.0002639070153236389, 0.0027245376259088516, -0.000600680592469871, 7.434446888510138e-05, -0.0010329773649573326, -0.0001577321090735495, -0.002193806692957878, 0.00033060243004001677, -0.0004183533601462841, 0.0015341844409704208, 0.0012888895580545068, 0.0010787054197862744, 0.000519670604262501, -0.0017922586994245648, -0.0005685474025085568, -0.0019188961014151573, -0.0003486687783151865, 0.0010742006124928594, -0.0002868880401365459, 0.0023098867386579514, -2.7695909011526965e-05, -0.0022196287754923105, -0.00028034919523634017, 0.00028028522501699626, 1.9681809135363437e-05, 0.0009660580544732511, -0.0006382993888109922, -0.0004453966103028506, 3.5339820897206664e-05, -0.0009794533252716064, 0.0005742005305364728, 0.0006907192291691899, -0.0013482329668477178, -0.001005403115414083, -0.00152755924500525, 0.0008845074335113168, 0.00033152688411064446, -0.001322337076999247, 0.0006111137336120009, 0.0024382518604397774, -0.0008284408831968904, 0.0006547145894728601, 0.0011195902479812503, -0.0006547756493091583, -0.00043061873293481767, 0.001594800385646522], [-0.0008543158182874322, 0.000536827021278441, -0.0006574255530722439, -0.0016008639940991998, 0.0007195054204203188, -0.0005179898580536246, -0.0007732840022072196, 0.00013679212133865803, -0.0002497313544154167, -0.00134533632081002, -0.00030476352549158037, -0.001242303173057735, 4.676226672017947e-05, -0.00037950449041090906, -0.0002964219020213932, 0.0001901266659842804, -0.0004381297039799392, 0.00027754702023230493, 0.0005905759753659368, 0.001084528979845345, -0.00076786003774032, -0.0007195695652626455, -0.0010269071208313107, -0.000352171016857028, -0.002161771059036255, -0.0006298762164078653, -0.0002926689921878278, -0.000578772509470582, 0.0011587527114897966, -0.0006364476284943521, -0.00030202465131878853, -0.0005046259029768407, 0.0018524992046877742, -0.0003870893269777298, -6.115933501860127e-05, 0.0012707418063655496, -0.00025366569752804935, 0.0005220716120675206, -0.00028435466811060905, -0.00020557254902087152, -0.0013522293884307146, 0.0003221348160877824, -0.0005192485987208784, -0.0005375946057029068, -0.0008600965957157314, -0.0030493498779833317, -0.00016762365703471005, 0.0004123269463889301, -0.0010756466072052717, 0.0006264973781071603, -0.00024161803594324738, 0.0002764289965853095, -0.000633449584711343, 0.0010083809029310942, -0.0004556062340270728, 0.000958805438131094, 0.00038064291584305465, 0.00204098760150373, -0.00026415727916173637, 0.0002573131350800395, 0.00029966695001348853, 0.0007400938775390387, -4.69686574433581e-06, 0.0005929922917857766], [-0.0011483883718028665, 0.0005862537655048072, -0.002062802901491523, -0.003601779229938984, -0.0007781658787280321, -0.00026161232381127775, -0.0019842260517179966, 0.00048396544298157096, -0.00036023976281285286, -0.0016209215391427279, -0.0003672898164950311, -0.0003302951517980546, 0.0005697525921277702, 7.524374723288929e-06, 0.0004205357690807432, -9.703409159556031e-05, -0.0010961523512378335, -0.0005736987222917378, 0.00019237541710026562, 0.00044111235183663666, -0.0014377158368006349, 0.00032332848059013486, -0.0013728431658819318, -0.0006223420496098697, -0.0004346030473243445, 0.00014104635920375586, 0.00012358017556834966, 0.0015481895534321666, -0.0008828790159896016, -0.0016455447766929865, -0.001384533359669149, -0.00011311170237604529, -0.0005857969517819583, -0.0009997484739869833, 0.000422380689997226, 0.0008712983690202236, -3.620614006649703e-05, 0.002172317123040557, -0.00039582414319738746, -0.002380823018029332, -0.000697912706527859, -0.0006055221892893314, 0.00044027180410921574, -0.0009450249490328133, -0.0009661526419222355, -0.0003273762122262269, -2.918231075454969e-05, -0.0019171223975718021, 0.001166589092463255, 0.0007774872938171029, -0.0003307098522782326, 0.0008492125198245049, -0.0009376447997055948, 0.0004058915947098285, 0.00024034777015913278, -0.00036281844950281084, 0.000740811403375119, 0.0021302136592566967, -0.00026276338030584157, 0.0011001916136592627, 0.0009108055965043604, 0.0013415069552138448, -0.0007711508660577238, 0.001141105662100017]])
    return np.dot(
        X-mean,
        conv.T
    )


def load_from_tf(args, gcp):
    gcp.v_normal.data = torch.tensor(
        [-0.169884145, 0.0430190973, 0.091173254, -0.0339165181, 0.0643557236, 0.145693019, 0.112589225, -0.0952830836,
         0.0254595, -0.0693574101, -0.0650991499, 0.235404313, -0.31420821, -0.0290404037, -0.161913335, -0.09325625,
         0.298154235, -0.169444725, -0.207124308, 0.0723744854, -0.0849481523, 0.0168008488, 0.00895659439,
         -0.0171319768, -0.127776787, -0.0971129909, -0.0536339432, 0.168108433, 0.177107826, 0.320735186,
         -0.0755678415, 0.139883056, -0.388966531, -0.0078522, -0.00130009966, 0.143557593, 0.035293255, -0.12994355,
         0.1157846, -0.121418417, -0.115577929, 0.0780592263, -0.194125444, 0.113405302, 0.244302094, -0.0874284953,
         -0.0544838, 0.0926826522, 0.0209452771, 0.0718942657, 0.0228996184, 0.298201054, 0.0192331262, -0.0319460481,
         -0.17595163, -0.0833073, 0.0334902816, 0.14013885, -0.14659746, 0.181580797, -0.00996331591, -0.0195714869,
         0.160506919, 0.0497409627]).to(args.device)  # torch.Tensor(loaded_dict["V_init:0"]).to(args.device)
    gcp.c_rand.data = torch.tensor([[0.569332063, 0.19621861, 0.936044037, 0.0672274604, 0.989149, 0.916594744, 0.754,
                                     0.431524485, 0.445979536, 0.333774686, 0.732518792, 0.822434127, 0.711422324,
                                     0.753830671, 0.836414278, 0.209573701, 0.527794242, 0.3339068, 0.832167804,
                                     0.6979146, 0.807687044, 0.690893054, 0.00416331459, 0.971259296, 0.615243,
                                     0.69255811, 0.669207, 0.670641, 0.85558778, 0.00144830858, 0.76548326, 0.409540862,
                                     0.888088107, 0.717633903, 0.584715724, 0.263450205, 0.459266245, 0.986697912,
                                     0.698782682, 0.63641417, 0.400523841, 0.221628249, 0.405968219, 0.579900086,
                                     0.725307345, 0.455515683, 0.131517351, 0.763612092, 0.928811967, 0.349458158,
                                     0.832664609, 0.914531469, 0.495537758, 0.163773, 0.827578843, 0.815654,
                                     0.429762304, 0.835437894, 0.323074102, 0.756760597, 0.627905488, 0.249528378,
                                     0.8888852, 0.242653042]]).to(args.device)
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
    gcp.LSTM_v.fc.weight.data = torch.Tensor(
        loaded_dict["graph-coloring/V_cell/layer_norm_basic_lstm_cell/kernel:0"]).to(args.device).T
    gcp.LSTM_v.ln_ih.gamma.data = torch.Tensor(
        loaded_dict["graph-coloring/V_cell/layer_norm_basic_lstm_cell/input/gamma:0"]).to(args.device)
    gcp.LSTM_v.ln_ih.beta.data = torch.Tensor(
        loaded_dict["graph-coloring/V_cell/layer_norm_basic_lstm_cell/input/beta:0"]).to(args.device)
    gcp.LSTM_v.ln_ho.gamma.data = torch.Tensor(
        loaded_dict["graph-coloring/V_cell/layer_norm_basic_lstm_cell/output/gamma:0"]).to(args.device)
    gcp.LSTM_v.ln_ho.beta.data = torch.Tensor(
        loaded_dict["graph-coloring/V_cell/layer_norm_basic_lstm_cell/output/beta:0"]).to(args.device)
    gcp.LSTM_v.ln_hf.gamma.data = torch.Tensor(
        loaded_dict["graph-coloring/V_cell/layer_norm_basic_lstm_cell/transform/gamma:0"]).to(args.device)
    gcp.LSTM_v.ln_hf.beta.data = torch.Tensor(
        loaded_dict["graph-coloring/V_cell/layer_norm_basic_lstm_cell/transform/beta:0"]).to(args.device)
    gcp.LSTM_v.ln_hc.gamma.data = torch.Tensor(
        loaded_dict["graph-coloring/V_cell/layer_norm_basic_lstm_cell/forget/gamma:0"]).to(args.device)
    gcp.LSTM_v.ln_hc.beta.data = torch.Tensor(
        loaded_dict["graph-coloring/V_cell/layer_norm_basic_lstm_cell/forget/beta:0"]).to(args.device)
    gcp.LSTM_v.ln_hcy.gamma.data = torch.Tensor(
        loaded_dict["graph-coloring/V_cell/layer_norm_basic_lstm_cell/state/gamma:0"]).to(args.device)
    gcp.LSTM_v.ln_hcy.beta.data = torch.Tensor(
        loaded_dict["graph-coloring/V_cell/layer_norm_basic_lstm_cell/state/beta:0"]).to(args.device)
    gcp.LSTM_c.fc.weight.data = torch.Tensor(
        loaded_dict["graph-coloring/C_cell/layer_norm_basic_lstm_cell/kernel:0"]).to(args.device).T
    gcp.LSTM_c.ln_ih.gamma.data = torch.Tensor(
        loaded_dict["graph-coloring/C_cell/layer_norm_basic_lstm_cell/input/gamma:0"]).to(args.device)
    gcp.LSTM_c.ln_ih.beta.data = torch.Tensor(
        loaded_dict["graph-coloring/C_cell/layer_norm_basic_lstm_cell/input/beta:0"]).to(args.device)
    gcp.LSTM_c.ln_ho.gamma.data = torch.Tensor(
        loaded_dict["graph-coloring/C_cell/layer_norm_basic_lstm_cell/output/gamma:0"]).to(args.device)
    gcp.LSTM_c.ln_ho.beta.data = torch.Tensor(
        loaded_dict["graph-coloring/C_cell/layer_norm_basic_lstm_cell/output/beta:0"]).to(args.device)
    gcp.LSTM_c.ln_hf.gamma.data = torch.Tensor(
        loaded_dict["graph-coloring/C_cell/layer_norm_basic_lstm_cell/transform/gamma:0"]).to(args.device)
    gcp.LSTM_c.ln_hf.beta.data = torch.Tensor(
        loaded_dict["graph-coloring/C_cell/layer_norm_basic_lstm_cell/transform/beta:0"]).to(args.device)
    gcp.LSTM_c.ln_hc.gamma.data = torch.Tensor(
        loaded_dict["graph-coloring/C_cell/layer_norm_basic_lstm_cell/forget/gamma:0"]).to(args.device)
    gcp.LSTM_c.ln_hc.beta.data = torch.Tensor(
        loaded_dict["graph-coloring/C_cell/layer_norm_basic_lstm_cell/forget/beta:0"]).to(args.device)
    gcp.LSTM_c.ln_hcy.gamma.data = torch.Tensor(
        loaded_dict["graph-coloring/C_cell/layer_norm_basic_lstm_cell/state/gamma:0"]).to(args.device)
    gcp.LSTM_c.ln_hcy.beta.data = torch.Tensor(
        loaded_dict["graph-coloring/C_cell/layer_norm_basic_lstm_cell/state/beta:0"]).to(args.device)


def decompose_to_submatrices(M_vv, split_list):
    start_idx_i = 0
    submatrices = []
    for size_i in range(0, len(split_list), 1):
        i = split_list[size_i]
        end_idx_i = start_idx_i + i
        submatrix = M_vv[start_idx_i:end_idx_i, start_idx_i:end_idx_i]
        submatrices.append(submatrix)
        start_idx_i = end_idx_i
    return submatrices


def get_pca_mean_std(mean, conv, count, V_ret):
    pca = PCA(32)
    pca.fit(V_ret.detach().cpu().numpy())
    count += 1
    if mean is None: mean = np.zeros_like(pca.mean_)
    mean += pca.mean_
    if conv is None: conv = np.zeros_like(pca.components_)
    conv += pca.components_
    return mean, conv, count

def get_pcs(embeddings_list, color, closesed_index, assignments, adj, degrees, gcp, continues = False, global_pca=True):
    k = max([a for ass in assignments for a in ass])
    pca_results = []
    pca = PCA(n_components=3)
    pca.fit(embeddings_list[closesed_index])
    for t, embeddings in enumerate(embeddings_list):
        if not global_pca:
            pca = PCA(n_components=3)
            pca.fit(embeddings)
        reduced_embeddings = (embeddings - pca.mean_) @ pca.components_.T
        n = len(assignments[0])
        closeset = [-1] * n
        for p_i, point in enumerate(reduced_embeddings):
            pca_results.append({'time_step': t, "degree": str(degrees[p_i].item()), "degree_c": degrees[p_i].item(),
                                "ass": str(assignments[t][p_i]), 'PCA1': point[0], 'PCA2': point[1], 'PCA3': point[2],
                                "color": color[p_i],

                                })
    return pd.DataFrame(pca_results)

def create_dataframe_for_animation(embeddings_list, color, closesed_index, assignments, adj, degrees, gcp, continues = False, global_pca=True):
    # pcas = []
    # for t, embeddings in enumerate(embeddings_list):
    #     pca = PCA(n_components=3)
    #     pca.fit(embeddings)
    #     pcas.append(pca.explained_variance_ratio_[:3].tolist())
    #     pcas[-1].append(t)
    # df = pd.DataFrame(pcas, columns=["pc1","pc2","pc3", "index"])
    # sorted_df = df.sort_values(by=['pc1', 'pc2', 'pc3'], ascending=[False, False, False])
    # best_index_for_pca = int(sorted_df.iloc[0]['index'])
    #
    # print(pcas[best_index_for_pca], best_index_for_pca, closesed_index)
    mean = np.array(
        [0.08287091553211212, 0.0, 0.05255761742591858, 0.29220879077911377, 0.28744906187057495, 0.28522735834121704,
         0.2969355285167694, 0.003041481366381049, 0.20396167039871216, 0.11760088801383972, 0.003919877111911774,
         0.4614821672439575, 0.008245991542935371, 0.08896780014038086, 0.12912799417972565, 0.07441310584545135,
         0.02071746252477169, 0.7216950058937073, 0.8190121054649353, 0.005868290551006794, 0.6078723073005676,
         0.012842519208788872, 0.08054333925247192, 0.0291881300508976, 0.0038479003123939037, 0.0,
         0.018843285739421844, 0.1072654277086258, 0.10601897537708282, 0.2386658936738968, 0.03438804671168327,
         0.2529189884662628, 0.11878620833158493, 0.020043687894940376, 0.10157398879528046, 0.29636210203170776,
         0.06319785863161087, 0.8237136006355286, 0.0, 0.03621730953454971, 0.16618336737155914, 0.0009420557180419564,
         0.6496729254722595, 0.26526734232902527, 0.0, 0.12515026330947876, 0.014941827394068241, 0.07343555241823196,
         0.6150848269462585, 0.0, 0.4474858045578003, 0.0, 0.9192183613777161, 0.32544270157814026, 0.05571744218468666,
         0.5483378171920776, 0.7170514464378357, 0.3424535393714905, 0.2851882576942444, 0.0001813597627915442,
         0.24136102199554443, 0.12805874645709991, 0.1666821837425232, 0.011305602267384529])
    comp = np.array([[-0.025849997997283936, 0.0, -0.0018793940544128418, 0.04412934184074402, 0.01547686755657196,
                      -0.1112450659275055, -0.12822623550891876, 0.0007271632784977555, 0.09638133645057678,
                      0.015543663874268532, -0.0009330861503258348, 0.27184349298477173, 0.0001457457256037742,
                      -0.010690501891076565, 0.05556325614452362, 0.04286152869462967, 0.0029955976642668247,
                      -0.2823828160762787, -0.43300172686576843, 0.004267151467502117, -0.018944570794701576,
                      -0.0007401075563393533, 0.03849244862794876, -0.004486629273742437, 3.7184923712629825e-05, 0.0,
                      -0.003724454902112484, 0.013541923835873604, -5.833527393406257e-05, 0.06340427696704865,
                      -0.007040751166641712, -0.14345461130142212, 0.018625961616635323, -0.00897105410695076,
                      0.03383595868945122, 0.19800050556659698, 0.01746775582432747, -0.05037407949566841, 0.0,
                      0.0002920606348197907, -0.061737943440675735, 0.0006626739632338285, 0.013072686269879341,
                      0.07455763220787048, 0.0, -0.030413510277867317, 0.009466098621487617, -0.044120464473962784,
                      0.36160045862197876, 0.0, 0.041575346142053604, 0.0, 0.5098600387573242, -0.007551691494882107,
                      -0.00959534291177988, 0.08418335020542145, -0.35004788637161255, -0.0252783615142107,
                      0.034510981291532516, -1.1495973012642935e-05, 0.018140362575650215, 0.07335954904556274,
                      0.0036804501432925463, 0.0013911304995417595],
                     [0.016921192407608032, -2.0489096641540527e-08, 0.04417431354522705, -0.09763896465301514,
                      0.24272029101848602, -0.09658919274806976, -0.04009789228439331, -0.0004559312656056136,
                      -0.0026437067426741123, 0.06198035180568695, -0.0008732097339816391, -0.13017842173576355,
                      0.00487465551123023, 0.04071587696671486, -0.04740623012185097, -0.018722468987107277,
                      0.005053806584328413, -0.17804042994976044, -0.30460241436958313, -0.002647288143634796,
                      -0.09662222862243652, -0.0039960481226444244, -0.01574566960334778, -0.005123112816363573,
                      0.003474113065749407, 0.0, 0.007130516227334738, 0.06221820041537285, 0.03536023572087288,
                      0.03369513154029846, 0.0036811912432312965, -0.10259115695953369, 0.07344613969326019,
                      -0.0041844816878438, 0.004847946111112833, -0.1123499795794487, -0.0035931526217609644,
                      0.09556248784065247, 0.0, 0.0025215218774974346, -0.0008436244097538292, -0.00036679342156276107,
                      0.6086641550064087, -0.10209444165229797, 0.0, 0.031361907720565796, -0.004215605556964874,
                      -0.027450760826468468, -0.16229814291000366, 0.0, 0.009936630725860596, 0.0, -0.32501479983329773,
                      0.17497758567333221, -0.01640213467180729, 0.021550646051764488, -0.20708701014518738,
                      0.24795225262641907, 0.22749614715576172, 0.00016605773998890072, -0.041520942002534866,
                      -0.03731408342719078, 0.09890210628509521, -0.0016729915514588356],
                     [0.00446474552154541, 2.9802322387695312e-08, -0.0070452094078063965, -0.17529577016830444,
                      -0.03298383951187134, -0.07176638394594193, 0.085823655128479, 0.002106661908328533,
                      0.049419716000556946, 0.02773924730718136, 0.0023048410657793283, -0.12118576467037201,
                      0.002669183537364006, 0.07977332919836044, 0.010023660026490688, 0.02829509600996971,
                      -0.011312393471598625, -0.3163447976112366, -0.15978503227233887, 0.0038879886269569397,
                      0.05182650685310364, 0.006811652798205614, 0.09324956685304642, -0.013491873629391193,
                      -0.0019398819422349334, -0.0, -0.01646018773317337, 0.04914539307355881, 0.1424282193183899,
                      -0.200995072722435, 0.058152299374341965, -0.11891882866621017, 0.04324239864945412,
                      -0.0011605530744418502, 0.037919651716947556, -0.06617454439401627, 0.04569849744439125,
                      -0.2705513536930084, -0.0, 0.036653727293014526, 0.06872152537107468, -0.0005524838925339282,
                      -0.3604285717010498, -0.030390247702598572, -0.0, -0.029132336378097534, -0.017032481729984283,
                      -0.03845579922199249, -0.4294775724411011, -0.0, -0.39369478821754456, -0.0, 0.05090535804629326,
                      -0.0067183636128902435, 0.023749835789203644, -0.042482342571020126, -0.11513257026672363,
                      0.2752196490764618, -0.1931537538766861, 0.00013481623318511993, 0.13105988502502441,
                      -0.023333042860031128, -0.03187084197998047, 0.013144276104867458]])
    # data = np.load('pca_data.npz')
    # comp = data['COMP_GOOD']
    # mean = data['MEAN_GOOD']
    k = max([a for ass in assignments for a in ass])
    pca_results = []
    pca = PCA(n_components=3)
    pca.fit(embeddings_list[closesed_index])
    # pca.mean_ = mean
    # pca.components_ = comp
    print(pca.explained_variance_ratio_)
    for t, embeddings in enumerate(embeddings_list):
        if not global_pca:
            pca = PCA(n_components=3)
            pca.fit(embeddings)
        reduced_embeddings = (embeddings - pca.mean_) @ pca.components_.T
        # closeset = find_closest_kcoloring(assignments[t], adj, k)
        n = len(assignments[0])
        closeset = [-1] * n

        for p_i, point in enumerate(reduced_embeddings):
            # neighbors = assignments[closesed_index][adj[p_i].cpu() == 1]
            neighbors = assignments[t][adj[p_i].cpu() == 1]
            snd = degrees[neighbors].sum().item()
            sup = support_for_color(adj, assignments[t], 3)

            pca_results.append({'time_step': t, "degree": str(degrees[p_i].item()), "degree_c": degrees[p_i].item(),
                                "ass": str(assignments[t][p_i]), 'PCA1': point[0], 'PCA2': point[1], 'PCA3': point[2],
                                "color": color[p_i],
                                "color_s": str(color[p_i]),
                                "label": str(p_i), "neighbors_col": neighbors.tolist(),
                                "neighbors": np.arange(len(adj))[adj[p_i].cpu() == 1], "snd": snd,
                                "closeset": str(closeset[p_i] != assignments[t][p_i]),
                                })
        if not continues:
            for _ in range(1, k + 1):
                pca_results.append(
                    {'time_step': t, "degree": -1, "degree": "-1", "ass": str(_), 'PCA1': 0, 'PCA2': 0, 'PCA3': 0,
                     "color": color[_], "label": "-2", "closeset": "-1"})
            pca_results.append(
                {'time_step': t, "degree": -1, "degree": "-1", "ass": "midpoint", 'PCA1': 0, 'PCA2': 0, 'PCA3': 0,
                 "color": "black", "label": "-2", "closeset": "-1"})
                # pca_results.append({'time_step': t, "degree": -1, "degree": "-1", "ass": "2", 'PCA1': 0, 'PCA2': 0, 'PCA3': 0, "color": color[1], "label": "-2"})
                # pca_results.append({'time_step': t, "degree": -1, "degree": "-1", "ass": "3", 'PCA1': 0, 'PCA2': 0, 'PCA3': 0, "color": color[2], "label": "-2"})

            clause = gcp.clauses_hist[t].squeeze().cpu().detach().numpy()
            clause_embedding = (clause - pca.mean_) @ pca.components_.T
            if len(clause_embedding.shape)> 1:
                clause_embedding = clause_embedding[0]
            pca_results.append(
                {'time_step': t, 'PCA1': clause_embedding[0], 'PCA2': clause_embedding[1], 'PCA3': clause_embedding[2],
                 "sup": -1, "ass": "-1", "color": -1, "label": -1, "degree_c": -1, "degree": -1, "snd": -1, "closeset": "-1"})
    return pd.DataFrame(pca_results)


COMP_GOOD = torch.zeros(size=(3, 64))
COUNT_GOOD = 0
MEAN_GOOD = torch.zeros(size=(1,64))
COMP_BAD = torch.zeros(size=(3, 64))
COUNT_BAD = 0
MEAN_BAD = torch.zeros(size=(1,64))

def attributes_check(results_data, M_vv, M_vc, split,labels, cn, n, k, gcp, adj):
    # features = gcp.history[:n].clone().detach().cpu()
    # lower = my_pca(features)
    # degrees = adj.sum(axis=0).cpu()
    global COMP_GOOD, COUNT_GOOD, MEAN_GOOD, COMP_BAD, COUNT_BAD, MEAN_BAD

    # print("number of connected components", len([_ for _ in nx.connected_components(nx.from_numpy_array(adj.cpu().detach().numpy()))]))

    iters_to_run = 150
    attempts = 2
    old = gcp.tmax
    gcp.tmax = iters_to_run
    pred, means, V_ret, C_ret = gcp.forward(M_vv, M_vc, split, cn=cn, attempts=attempts)
    gcp.tmax = old
    min_disagree = []
    assignments = []
    votes = []
    missmatches_indexes = []
    centroids = []
    closesdt_history = []
    local_assignments = []

    adj_cpu = adj.cpu()

    embeddings_list = [h.clone().detach().cpu().numpy() for h in gcp.histories]

    for i in range(len(embeddings_list)):
        # assignment, _ = sklearn_k_means(adj, gcp.histories[i], k)
        assignment, centroid = sklearn_k_means(embeddings_list[i], k)
        is_k_col, disagree, missmatch_indexes = is_k_color(adj, assignment)
        # close_ass_iter = find_closest_kcoloring(assignment, adj, k)
        # miss_match_count = sum(x != y for x, y in zip(assignment, close_ass_iter))
        # closesdt_history.append(miss_match_count)
        centroids.append(centroid)
        min_disagree.append(disagree)
        assignments.append(assignment)
        local_assignments.append(assignment)
        missmatches_indexes.append(missmatch_indexes)
        votes.append(
         -1    # (gcp.vote(gcp.histories[i], split)[0] > 0.5).item()
        )
        # if disagree == 0:
        #     break
    old_closesed_index = np.argmin(min_disagree) #np.argmin(closesdt_history)# np.argmin(min_disagree)

    min_disagree = []
    assignments = []
    votes = []
    missmatches_indexes = []
    closesdt_history = []
    centroid = centroids[old_closesed_index]
    # centroid = np.array([[ 0.00000000e+00,  1.12984747e-01, -3.72529030e-09,6.78443909e-01,  2.52139807e-01,  7.45058060e-09,2.98023224e-08,  0.00000000e+00,  3.74516755e-01,0.00000000e+00,  2.24337265e-01,  1.21082827e-01,2.35551726e-02, -7.45058060e-09,  5.04705235e-02,0.00000000e+00, -7.45058060e-09, -1.49011612e-08,8.55252743e-02,  1.39751434e-02,  3.81078422e-01,0.00000000e+00,  1.70910154e-02,  8.71598441e-03,0.00000000e+00,  0.00000000e+00,  0.00000000e+00,1.86264515e-09,  5.84279969e-02,  5.12843803e-02,-1.86264515e-09,  1.75224245e-03,  1.90456212e-03,0.00000000e+00,  0.00000000e+00,  4.85255897e-01,0.00000000e+00,  5.37415385e-01,  0.00000000e+00,2.30394557e-01, -1.49011612e-08,  3.60051870e-01,1.68878466e-01,  5.74498530e-03,  0.00000000e+00,1.49011612e-08,  1.08270841e-02, -9.31322575e-10,1.13435221e+00,  0.00000000e+00,  0.00000000e+00,0.00000000e+00,  4.04889297e+00, -3.72529030e-09,3.12941462e-01,  4.74914998e-01, -2.98023224e-08,1.33267045e-02,  0.00000000e+00,  4.80023772e-03,1.90520495e-01,  0.00000000e+00,  2.77513295e-01,2.41260320e-01],[ 0.00000000e+00, -3.72529030e-09,  1.71347454e-01,1.50494829e-01,  1.16501844e+00,  3.72529030e-09,4.11791414e-01,  0.00000000e+00,  1.49011612e-08,0.00000000e+00,  0.00000000e+00,  3.72529030e-09,8.24932903e-02,  1.25259101e-01,  1.86264515e-09,0.00000000e+00,  2.63109148e-01,  5.12757152e-03,0.00000000e+00, -4.65661287e-10,  5.05748153e-01,0.00000000e+00,  4.65661287e-10, -4.65661287e-10,0.00000000e+00,  0.00000000e+00,  0.00000000e+00,5.04421964e-02,  3.94041955e-01,  6.02906942e-03,6.36159331e-02, -7.45058060e-09,  6.40475214e-01,0.00000000e+00,  0.00000000e+00, -1.49011612e-08,5.06578758e-03,  9.93744314e-01,  0.00000000e+00,2.49018371e-02,  3.31632584e-01,  1.26086846e-02,2.25194812e+00,  0.00000000e+00,  0.00000000e+00,2.22122028e-01, -4.65661287e-10,  0.00000000e+00,0.00000000e+00,  0.00000000e+00,  0.00000000e+00,0.00000000e+00,  2.85488009e-01,  1.13270037e-01,-5.96046448e-08,  8.44639122e-01,  9.21040773e-05,2.33983731e+00,  0.00000000e+00,  0.00000000e+00,2.75220051e-02,  0.00000000e+00,  2.39459097e-01,9.07675177e-03],[ 0.00000000e+00,  7.45058060e-09,  1.21620372e-02,1.30916834e-02,  1.68648511e-01,  1.29655078e-01,3.45662534e-01,  0.00000000e+00,  3.88002172e-02,0.00000000e+00,  6.34458661e-03,  3.72529030e-09,-9.31322575e-09,  1.31902784e-01,  3.07209175e-02,0.00000000e+00,  8.84811506e-02,  3.20082784e-01,2.39661694e+00,  0.00000000e+00,  3.89652431e-01,0.00000000e+00,  9.31322575e-10, -4.65661287e-10,0.00000000e+00,  0.00000000e+00,  0.00000000e+00,2.79396772e-09,  1.49011612e-08,  2.04620004e-01,6.62567746e-03,  1.10574976e-01,  2.66744494e-02,0.00000000e+00,  0.00000000e+00,  0.00000000e+00,1.16415322e-10,  1.45867813e+00,  0.00000000e+00,3.59643549e-01,  3.85800779e-01,  3.07802930e-02,1.17290020e-03,  1.54505279e-02,  0.00000000e+00,3.17776740e-01,  6.16234867e-03,  2.21770443e-02,-5.96046448e-08,  0.00000000e+00,  0.00000000e+00,0.00000000e+00,  8.94338310e-01,  4.48439121e-02,4.57218170e-01,  5.79159677e-01,  3.05478811e-01,2.72032619e-02,  0.00000000e+00,  1.62971411e-02,1.33731425e-01,  0.00000000e+00,  4.34413105e-01,0.00000000e+00]])
    closesdt_history = []
    new_centroids = []
    for i in range(len(embeddings_list)):
        # assignment, _ = sklearn_k_means(adj, gcp.histories[i], k)
        assignment, new_centroid = sklearn_k_means(embeddings_list[i], k, centroids = centroid)
        is_k_col, disagree, missmatch_indexes = is_k_color(adj, assignment)
        # close_ass_iter = find_closest_kcoloring(assignment, adj, k)
        # miss_match_count = sum(x != y for x, y in zip(assignment, close_ass_iter))
        # closesdt_history.append(miss_match_count)
        new_centroids.append(new_centroid)
        min_disagree.append(disagree)
        assignments.append(assignment)
        missmatches_indexes.append(missmatch_indexes)
        votes.append(
        -1    # (gcp.vote(gcp.histories[i], split)[0] > 0.5).item()
        )
        # if disagree == 0:
        #     break
    new_closesed_index = np.argmin(min_disagree) # np.argmin(closesdt_history) # #np.argmin(min_disagree)
    # min_value = np.min(min_disagree)
    # min_indices = np.where(min_disagree == min_value)[0]
    # closesed_index = min_indices[len(min_indices) // 2]
    attmept_range = new_closesed_index//iters_to_run
    closesed_index = new_closesed_index % iters_to_run
    gcp.histories = gcp.histories[attmept_range*iters_to_run: (attmept_range+1)*iters_to_run ]
    embeddings_list = embeddings_list[attmept_range * iters_to_run: (attmept_range + 1) * iters_to_run]
    min_disagree = min_disagree[attmept_range*iters_to_run: (attmept_range+1)*iters_to_run ]
    assignments = assignments[attmept_range*iters_to_run: (attmept_range+1)*iters_to_run ]
    local_assignments = local_assignments[attmept_range*iters_to_run: (attmept_range+1)*iters_to_run ]
    votes = votes[attmept_range*iters_to_run: (attmept_range+1)*iters_to_run ]
    gcp.clauses_hist = gcp.clauses_hist[attmept_range*iters_to_run: (attmept_range+1)*iters_to_run ]
    centroids = centroids[attmept_range*iters_to_run: (attmept_range+1)*iters_to_run ]
    closesdt_history = closesdt_history[attmept_range*iters_to_run: (attmept_range+1)*iters_to_run ]
    new_centroids = new_centroids[attmept_range*iters_to_run: (attmept_range+1)*iters_to_run ]
    # missmatches_indexes = missmatches_indexes[attmept_range*iters_to_run: (attmept_range+1)*iters_to_run ]

    results_data["closesed_index"].append(closesed_index.item())
    results_data["attmept_range"].append(attmept_range.item())

    results_data["min_disagree"].append( min_disagree[closesed_index])
    print("min disagree", results_data["min_disagree"])
    return

    # print("closesed_index diff", old_closesed_index - closesed_index)

    # close_ass = find_closest_kcoloring(assignments[closesed_index], adj, k)
    close_ass = assignments[closesed_index]
    if close_ass is None:
        hint = None
        # hint = [-1 for _ in range(len(assignments[closesed_index]))]
        # hint[0]=4
        close_ass = find_closest_kcoloring(assignments[closesed_index], adj, k+1, hint=hint)
    markers = [int(a != c) for a, c in zip(assignments[closesed_index], close_ass)]
    dist = sum(markers)
    results_data["closest_ass_dist"].append(dist)
    degrees = adj.sum(0).cpu()

    # gcp.tmax = closesed_index
    # pred, means, V_ret, C_ret = gcp.forward(M_vv, M_vc, split, cn=cn, history=False)
    # gcp.tmax = old
    # pred_int = 0 if pred < 0.5 else 1



    # pca = PCA(10)
    # # lower = pca.fit_transform(gcp.histories[closesed_index].cpu().detach().numpy())
    # # print(pca.explained_variance_ratio_)
    # x = lower[:, 0]
    # y = lower[:, 1]
    # z = lower[:, 2]

    # features = gcp.histories[closesed_index].clone().detach().cpu()
    # n_components, comp_labels = connected_components(csgraph=adj.cpu().detach().numpy(), directed=False, return_labels=True)
    all_sols = []# all_sols = find_all_k_coloring(adj.cpu(), k=3)
    # #
    # # G= nx.from_numpy_array(adj.cpu().detach().numpy())
    # # ccs = nx.connected_components(G)
    # # all_sorts_ccs = []
    # # for cc in ccs:
    # #     g = G.subgraph(list(cc))
    # #     g_np = nx.to_numpy_array(g)
    # #     all_sorts_ccs.append(find_all_k_coloring(adj.cpu(), k=3))
    # # all_sols = find_all_k_coloring(g_np, k=3)
    # ass = np.array(close_ass)
    # ass_f = assignments[closesed_index]
    # # print(ass.tolist())
    # # print(ass_f.tolist())
    # # print(adj.tolist())
    # print(len(all_sols)/6, degrees.mean().item(), degrees.std().item(), "\n",len(ass_f[ass_f == 1]), len(ass_f[ass_f == 2]), len(ass_f[ass_f == 3]),  "\n",  degrees[ass_f==1].mean().item(), degrees[ass_f==2].mean().item(), degrees[ass_f==3].mean().item(),"\n", len(ass[ass == 1]), len(ass[ass == 2]), len(ass[ass == 3]), "\n",  degrees[ass==1].mean().item(), degrees[ass==2].mean().item(), degrees[ass==3].mean().item())
    print("n",n, "k", k, "diss", min_disagree[closesed_index], "ind", closesed_index,
          # "closest", dist, "pred_opt", pred_int, "label", int(labels), "acc_opt", int(pred_int == labels), "vote", votes[closesed_index], "mean_std_count", np.mean(votes), np.std(votes), np.sum(votes)
          )

    #results_data["embeddings"].append(embeddings_list[closesed_index].tolist())

    # # centr = [centrality for node, centrality in nx.closeness_centrality(nx.from_numpy_array(adj.clone().detach().cpu().numpy())).items()]
    # support = support_for_vertex_color_assignment(adj, assignments[closesed_index],3)
    # for ass in assignments:
    #     ass[0]=4
    # df = create_dataframe_for_animation(
    #     embeddings_list,
    #     [str(d) for d in close_ass], closesed_index, assignments, adj, degrees, gcp)
    # fig = px.scatter(df, x='PCA1', y='PCA2',  # z="PCA3",
    #                  color="color",
    #                  animation_frame='time_step',
    #                  color_continuous_scale='Greys',
    #                  title=f'PCA of Embeddings Over Time dist:{min_disagree[closesed_index]} close:{dist} index: {closesed_index}',
    #                  labels={'PCA1': 'PC1', 'PCA2': 'PC2', 'PC3': "PCA3"},
    #                  hover_data={"color": True, 'label': True, 'neighbors': True, "neighbors_col": True, "degree": True,
    #                              "snd": True})
    # # fig.show()
    #
    # centr = [centrality for node, centrality in
    #          nx.closeness_centrality(nx.from_numpy_array(adj.clone().detach().cpu().numpy())).items()]
    # # support = support_for_vertex_color_assignment(adj, assignments[closesed_index],3)
    # # for ass in assignments:
    # #     ass[0]=4
    # df = create_dataframe_for_animation(
    #     embeddings_list,
    #     [str(d) for d in close_ass], # [str(d != d2) for d, d2 in zip(assignments[closesed_index], close_ass)],
    #     closesed_index, assignments, adj, degrees, gcp)
    # fig = px.scatter(df, x='PCA1', y='PCA2', #z="PCA3",
    #                     color="color",
    #                     animation_frame='time_step',
    #                     color_continuous_scale='Greys',
    #                     title=f'PCA of Embeddings Over Time dist:{min_disagree[closesed_index]} close:{dist} index: {closesed_index}',
    #                     labels={'PCA1': 'PC1', 'PCA2': 'PC2', 'PC3': "PCA3"},
    #                     hover_data={"color": True, 'label': True, 'neighbors': True, "neighbors_col": True,
    #                                 "degree": True,
    #                                 "snd": True})
    #
    from sklearn.linear_model import LinearRegression
    #
    # df = df[df["time_step"] == closesed_index]
    #
    # fig = px.scatter(df, x='PCA1', y='PCA2', #z="PCA3",
    #                     color="color",
    #                     animation_frame='time_step',
    #                     color_continuous_scale='Greys',
    #                     title=f'PCA of Embeddings Over Time dist:{min_disagree[closesed_index]} close:{dist} index: {closesed_index}',
    #                      labels={'PCA1': 'PC1', 'PCA2': 'PC2', 'PC3': "PCA3"},
    #                      hover_data={"color": True, 'label': True, 'neighbors': True, "neighbors_col": True,
    #                                  "degree": True,
    #                                  "snd": True}
    #                     )

    # df = get_pcs(
    #         embeddings_list,
    #         [str(d) for d in close_ass],
    #         closesed_index, assignments, adj, degrees, gcp)
    # d2_df =df[df["time_step"]==closesed_index]
    # fig = px.scatter(d2_df, x="PCA1", y="PCA2", color="color", title=f"{min_disagree[closesed_index]}")
    # best_centroid = new_centroids[closesed_index]
    #
    # pca = PCA(2)
    # pca.fit(embeddings_list[closesed_index])
    # best_centroid = pca.transform(best_centroid)
    #
    # points = d2_df[["PCA1", "PCA2"]].to_numpy().astype(np.float32)
    #
    # unique_points = np.unique(np.round(points,4), axis=0)
    #
    # rectified_triangle=None
    # if unique_points.shape[0] >= 3:
    #     try:
    #         ret, triangle = cv.minEnclosingTriangle(unique_points)
    #         triangle = triangle.squeeze()
    #         distances = np.linalg.norm(points[:, np.newaxis] - triangle, axis=2)
    #         triangle_assignments = np.argmin(distances, axis=1)
    #
    #         fig.add_shape( type='line', x0=triangle[0][0], y0=triangle[0][1], x1=triangle[1][0], y1=triangle[1][1], line=dict(color='LightSeaGreen', width=2))
    #         fig.add_shape(type='line', x0=triangle[1][0], y0=triangle[1][1], x1=triangle[2][0], y1=triangle[2][1], line=dict(color='LightSeaGreen', width=2))
    #         fig.add_shape(type='line', x0=triangle[2][0], y0=triangle[2][1], x1=triangle[0][0], y1=triangle[0][1], line=dict(color='LightSeaGreen', width=2))
    #
    #         best_prediction = 0
    #         best_tri_ass = []
    #         rectified_triangle = triangle
    #         permutations = list(itertools.permutations(list(range(k))))
    #         for permute in permutations:
    #             conversion = {i: permute[i] for i in range(k)}
    #             inverse_conversion = {val:key for key,val in conversion.items()}
    #             converted_centroid_assignment = [conversion[i - 1] for i in assignments[closesed_index]]
    #             traingle_classification_metric = np.mean(
    #                 [a == b for a, b in zip(triangle_assignments, converted_centroid_assignment)])
    #             if traingle_classification_metric > best_prediction:
    #                 best_prediction = traingle_classification_metric
    #                 best_tri_ass = converted_centroid_assignment
    #                 rectified_triangle = [triangle[inverse_conversion[i]] for i in range(k)]
    #
    #         results_data["triangle_metric"].append(best_prediction.item())
    #         print("triangle_metric", min_disagree[closesed_index], best_prediction)
    #
    #         results_data["centroid_from_triangle"].append(
    #             np.linalg.norm(rectified_triangle - best_centroid, axis=1).mean().item())
    #
    #         centroid_diffs = []
    #         for j, tri in enumerate(triangle):
    #             model = LinearRegression().fit([[0], [tri[0]]], [0, tri[1]])
    #             y_predict = model.predict([[best_centroid[j][0]]])
    #             diff = abs(y_predict - best_centroid[j][1])
    #             centroid_diffs.append(diff)
    #         results_data["centroid_diff_from_line"].append(
    #             np.mean(centroid_diffs).item())
    #     except Exception as e:
    #         print("ERRORRRRRRR", e)
    #         results_data["triangle_metric"].append(-2)
    # else:
    #     results_data["triangle_metric"].append(-1)
    #     results_data["centroid_diff_from_line"].append(-1)
    #     results_data["centroid_from_triangle"].append(-1)
    #
    #
    #
    # r_sqs = []
    # for color in df['color'].unique():
    #     if color not in [str(i+1) for i in range(k)]: continue
    #     cluster_df = df[(df['color'] == color) & (df["time_step"]==closesed_index)]
    #     X = cluster_df[['PCA1']].values
    #     y = cluster_df['PCA2'].values
    #
    #     # Fit linear regression model
    #     model = LinearRegression().fit(X, y)
    #     slope = model.coef_[0]
    #     intercept = model.intercept_
    #     if len(X) < 2:
    #         r_sq =1
    #     else:
    #         r_sq = model.score(X, y)
    #     r_sqs.append(r_sq)
    #
    #     # Calculate distances from the origin
    #     distances = np.sqrt(cluster_df['PCA1'] ** 2 + cluster_df['PCA2'] ** 2)
    #     max_distance_idx = distances.idxmax()
    #     min_distance_idx = distances.idxmin()
    #
    #     # Get the furthest and closest points
    #     max_point = cluster_df.loc[max_distance_idx]
    #     min_point = cluster_df.loc[min_distance_idx]
    #
    #     # Add the extended line to the plot
    #     # fig.add_shape(
    #     #     type='line',
    #     #     x0=-0, y0=0,
    #     #     x1=max_point['PCA1'], y1=max_point['PCA2'],
    #     #     line=dict(color='LightSeaGreen', width=2)
    #     # )
    # print("triangle_regression", np.mean(r_sqs))
    # results_data["triangle_regression"].append(np.mean(r_sqs).item())
    #
    # def rotate_and_normalize_data(X, y):
    #     # Fit the initial model to get the slope
    #     initial_model = LinearRegression().fit(X, y)
    #     slope = initial_model.coef_[0]
    #
    #     # Calculate the angle of rotation
    #     angle = np.arctan(slope) - np.pi / 4  # Rotate to make the slope 1
    #
    #     # Create the rotation matrix
    #     rotation_matrix = np.array([
    #         [np.cos(angle), -np.sin(angle)],
    #         [np.sin(angle), np.cos(angle)]
    #     ])
    #
    #     # Apply the rotation to the data
    #     data = np.hstack((X, y.reshape(-1, 1)))
    #     rotated_data = data.dot(rotation_matrix.T)
    #
    #     # Normalize the data
    #     max_point = np.max(np.abs(rotated_data), axis=0)
    #     normalized_data = rotated_data / max_point
    #
    #     return normalized_data[:, 0].reshape(-1, 1), normalized_data[:, 1]
    #
    # r_sqs = []
    # if rectified_triangle is not None:
    #     for color in df['color'].unique():
    #         if color not in [str(i + 1) for i in range(k)]: continue
    #         cluster_df = df[(df['color'] == color) & (df["time_step"] == closesed_index)]
    #         X = cluster_df[['PCA1']].values
    #         y = cluster_df['PCA2'].values
    #         apex = rectified_triangle[int(color) - 1]
    #
    #         # Rotate the data
    #         X_rotated, y_rotated = rotate_and_normalize_data(X, y)
    #
    #         # Fit the rectified model
    #         rectified_model = LinearRegression().fit(X_rotated, y_rotated)
    #         if len(X) < 2:
    #             r_sq =1
    #         else:
    #             r_sq = rectified_model.score(X_rotated, y_rotated)
    #         r_sqs.append(r_sq)
    #
    #         fig.add_shape(
    #             type='line',
    #             x0=0, y0=0,
    #             x1=apex[0], y1=apex[1],
    #             line=dict(color='LightSeaGreen', width=2)
    #         )
    # print("triangle_regression", np.mean(r_sqs).item() if len(r_sqs) > 0 else -2)
    # results_data["triangle_apex_regression"].append(np.mean(r_sqs).item() if len(r_sqs) > 0 else -2)
    #
    #
    # fig.add_scatter(x=best_centroid[:,0], y=best_centroid[:,1], mode='markers', marker=dict(size=10, color='red'), name='origin')
    # # fig.update_layout(title=f'PCA of Embeddings Over Time dist:{min_disagree[closesed_index]} close:{dist} index: {closesed_index} regression:{np.mean(r_sqs).item()}')
    # # fig.show()




    # # IMPORTANT
    distance_neighbors_strangers_color(adj, assignments, closesed_index, embeddings_list, min_disagree, results_data)

    # #IMPORTANT
    distances_over_time = calc_dist_over_iteartions(adj_cpu, embeddings_list)
    # spear, spear_p = plot_conf_over_time(distances_over_time, closesed_index)
    # results_data["spearman_avg_confidence_over_time"].append(spear)
    # results_data["spearman_p_avg_confidence_over_time"].append(spear_p)

    conflict_sums = []
    least_common_colors = []
    fake_closesd_index = min(len(embeddings_list)-1, (closesed_index if min_disagree[closesed_index]==0 else (150 if n>250 else 50)))
    for i in range(fake_closesd_index + 1 ):
        emb = embeddings_list[i]
        least_common_color, cs, neighbor_dist = find_least_common_neighbor_color(adj, local_assignments[i], emb, k)
        # least_common_color, cs, neighbor_dist = find_least_common_neighbor_color(adj, assignments[i], emb, k)
        conflict_sums.append(cs)
        least_common_colors.append(least_common_color)

    # IMPORTANT
    # flipped_stats2(adj, assignments, closesed_index, degrees, gcp, k, results_data,embeddings_list, least_common_colors)

    #IMPORTANT
    # confidance_same_pair_res = confidance_same_pair(adj, local_assignments, distances_over_time)
    # results_data["confidence_pair_contrudiction"].append(confidance_same_pair_res.tolist())
    above, below, interstion_70, interstion_80, interstion_90, support_percentile_of_70, support_percentile_of_80, support_percentile_of_90 = confidance_pair(adj, local_assignments, distances_over_time, fake_closesd_index, conflict_sums)
    results_data["above_below"].append([above, below])
    results_data["interstion_70"].append(interstion_70)
    results_data["interstion_80"].append(interstion_80)
    results_data["interstion_90"].append(interstion_90)

    results_data["support_percentile_of_70"].append(np.mean(support_percentile_of_70).item() if len(support_percentile_of_70)>0 else -1)
    results_data["support_percentile_of_80"].append(np.mean(support_percentile_of_80).item() if len(support_percentile_of_80)>0 else -1)
    results_data["support_percentile_of_90"].append(np.mean(support_percentile_of_90).item() if len(support_percentile_of_90)>0 else -1)



    #IMPORTANT?
    pc_corolation, degree_pc_corolation = support_axis_corolation(adj_cpu, assignments, fake_closesd_index, embeddings_list, k, conflict_sums)
    results_data["pc_corolation"].append(pc_corolation.item())
    results_data["degree_pc_corolation"].append(degree_pc_corolation.item())

    # IMPORTANT
    get_spearman_support_mean_confidence(distances_over_time, embeddings_list, k, adj_cpu, fake_closesd_index, assignments, results_data, conflict_sums)
    return


def flipped_stats2(adj, assignments, closesed_index, degrees, gcp, k, results_data, embeddings_list, least_common_colors):
    changed_nodes = []
    not_changed_nodes = []
    least_conflicting_changes = []
    weighted_least_conflicting_changes = []

    not_least_conflicting_changes = []
    not_weighted_least_conflicting_changes = []

    iters = range(1, len(assignments[:closesed_index + 1]))
    changed_to_best_color = []
    kept_to_best_color = []

    for i in iters:
        current_assignment = assignments[i]
        previous_assignment = assignments[i - 1]
        prev_feature = embeddings_list[i - 1]
        changed_nodes = []
        not_changed_nodes = []
        for node, (a,b) in enumerate(zip(current_assignment, previous_assignment)):
            if a != b:
                changed_nodes.append(node)
            else:
                not_changed_nodes.append(node)
        changed_node_abiding_least_resistance = 0
        unchanged_node_abiding_least_resistance = 0
        for node in changed_nodes:
            if current_assignment[node] == least_common_colors[i][node]: changed_node_abiding_least_resistance += 1
        for node in not_changed_nodes:
            if current_assignment[node] == least_common_colors[i][node]: unchanged_node_abiding_least_resistance += 1

        changed_to_best_color.append(changed_node_abiding_least_resistance/len(changed_nodes)
                                     if len(changed_nodes)>0 else -1)
        kept_to_best_color.append(unchanged_node_abiding_least_resistance/len(not_changed_nodes)
                                  if len(not_changed_nodes)>0 else -1)

    df = pd.DataFrame({
        'Iteration': iters,
        'Changed Nodes': changed_to_best_color,
        'UNChanged Nodes': kept_to_best_color,
    })


    changed_ratio_mean = -1
    unchanged_ratio_mean = -1

    changed_to_best_color = np.array(changed_to_best_color)
    kept_to_best_color    = np.array(kept_to_best_color)

    if len(changed_nodes) > 0:
        changed_ratio_mean = changed_to_best_color[changed_to_best_color!=-1].mean()
        unchanged_ratio_mean = kept_to_best_color[kept_to_best_color!=-1].mean()



    results_data["changed_ratio_mean"].append(changed_ratio_mean)
    results_data["unchanged_ratio_mean"].append(unchanged_ratio_mean)




def flipped_stats(adj, assignments, closesed_index, degrees, gcp, k, results_data, embeddings_list):
    changed_nodes = []
    not_changed_nodes = []
    least_conflicting_changes = []
    weighted_least_conflicting_changes = []

    not_least_conflicting_changes = []
    not_weighted_least_conflicting_changes = []

    def least_conflicting_color(node, assignment, adj, k):
        # Function to determine the least conflicting color for a node
        neighbor_colors = [assignment[neighbor] for neighbor in range(len(adj)) if adj[node][neighbor] > 0]
        color_counts = {color: neighbor_colors.count(color) for color in range(1, k + 1)}
        least_conflicting = min(list(color_counts.values()))
        return set([color for color in color_counts if color_counts[color] == least_conflicting])

    def most_distant_color(node, assignment, adj, features):
        # Function to determine the most distant color for a node
        start, end = min(assignment), max(assignment)
        color_neighbor = {c: [] for c in range(start, end + 1)}
        for neighbor in range(len(adj)):
            if adj[node][neighbor] > 0:
                color_neighbor[assignment[neighbor]].append(neighbor)
        color_distances = {}
        for color in range(start, end + 1):
            neighbor_colors = color_neighbor[color]
            # if len(neighbor_colors) == 0: continue
            color_distances[color] = np.linalg.norm(features[node] - features[neighbor_colors])
        most_distant = max(list(color_distances.values()))
        most_distant_nodes = set([color for color in color_distances if color_distances[color] == most_distant])
        least_distant = min(list(color_distances.values()))
        least_distant_nodes = set([color for color in color_distances if color_distances[color] == least_distant])
        return least_distant_nodes

    # Calculate changes and least conflicting color changes
    iters = range(1, len(assignments[:closesed_index + 1]))
    iterations_with_changes = []
    avg_degree_of_changed_nodes = []
    neihboring_distange_changed = []
    neihboring_distange_unchanged = []
    c_distanve_changed = []
    c_distanve_unchanged = []
    for i in iters:
        current_assignment = assignments[i]
        previous_assignment = assignments[i - 1]
        prev_feature = embeddings_list[i - 1]
        changed = [node for node, (a, b) in enumerate(zip(current_assignment, previous_assignment)) if a != b]
        not_changed = [node for node, (a, b) in enumerate(zip(current_assignment, previous_assignment)) if a == b]

        # get avg distance between enihbors for changed vs unchanged nodes
        neihboring_distange_changed.append(np.mean(
            [np.linalg.norm(prev_feature[node] - prev_feature[neighbor]) for node in changed for neighbor in
             range(len(adj)) if adj[node][neighbor] > 0]))
        neihboring_distange_unchanged.append(np.mean(
            [np.linalg.norm(prev_feature[node] - prev_feature[neighbor]) for node in range(len(adj)) if
             node not in changed for neighbor in range(len(adj)) if adj[node][neighbor] > 0]))
        # get avg distance between clause = gcp.clauses_hist[t].squeeze().cpu().detach().numpy() for changed nodes vs unchaved noes
        clause = gcp.clauses_hist[i].squeeze().cpu().detach()
        c_distanve_changed.append(np.mean([np.linalg.norm(clause - prev_feature[node]) for node in changed]))
        c_distanve_unchanged.append(
            np.mean([np.linalg.norm(clause - prev_feature[node]) for node in range(len(adj)) if node not in changed]))

        least_conflicting = [node for node in changed if
                             current_assignment[node] in least_conflicting_color(node, previous_assignment, adj, k)]
        weighted_least_conflicting = [node for node in changed if
                                      current_assignment[node] in most_distant_color(node, previous_assignment, adj,
                                                                                     prev_feature)]

        not_least_conflicting = [node for node in not_changed if current_assignment[node] in least_conflicting_color(node, previous_assignment, adj, k)]
        not_weighted_least_conflicting = [node for node in not_changed if current_assignment[node] in most_distant_color(node, previous_assignment, adj, prev_feature)]

        changed_nodes.append(len(changed))
        least_conflicting_changes.append(len(least_conflicting))
        weighted_least_conflicting_changes.append(len(weighted_least_conflicting))

        not_changed_nodes.append(len(not_changed))
        not_least_conflicting_changes.append(len(not_least_conflicting))
        not_weighted_least_conflicting_changes.append(len(not_weighted_least_conflicting))

        if len(changed) > 0:
            iterations_with_changes.append(i)
            avg_degree_of_changed_nodes.append(np.mean([degrees[node] for node in changed]))
    df = pd.DataFrame({
        'Iteration': iters,
        'Changed Nodes': changed_nodes,
        'Least Conflicting Changes': least_conflicting_changes,
        'Weighted Least Conflicting Changes': weighted_least_conflicting_changes,
        'NOT Changed Nodes': not_changed_nodes,
        'NOT Least Conflicting Changes': not_least_conflicting_changes,
        'NOT Weighted Least Conflicting Changes': not_weighted_least_conflicting_changes,
        'neihbors changed': neihboring_distange_changed,
        'neihbors unchanged': neihboring_distange_unchanged,
        'c changed': c_distanve_changed,
        'c unchanged': c_distanve_unchanged,
    })
    # results_data["changed_node"].append(changed_nodes)
    # results_data["least_conflict_change"].append(least_conflicting_changes)
    # results_data["weighted_least_conflict_change"].append(weighted_least_conflicting_changes)
    fig = px.line(df, x='Iteration',
                  y=['Changed Nodes', 'Least Conflicting Changes', "Weighted Least Conflicting Changes",
                     'neihbors changed', 'neihbors unchanged', 'c changed', 'c unchanged'],
                  title='Nodes Changing Assignments and Least Conflicting Changes Over Iterations')
    # fig.show()
    # IMPORTANT
    flips_ratios_least_conflict = []
    flips_ratio_weighted = []

    changed_nodes = df['Changed Nodes']
    df.loc[df['Changed Nodes'] == 0, 'Changed Nodes'] = -1
    not_changed_nodes= df['NOT Changed Nodes']
    df.loc[df['NOT Changed Nodes'] == 0, 'NOT Changed Nodes'] = -1

    changed_vs_least_ratio = -1
    changed_vs_weighted_ratio = -1
    not_changed_vs_least_ratio = -1
    not_changed_vs_weighted_ratio = -1

    if len(changed_nodes) > 0:
        changed_vs_least_ratio = np.where(changed_nodes != -1, df['Least Conflicting Changes'] / changed_nodes, 0).mean()
        changed_vs_weighted_ratio = np.where(changed_nodes != -1, df['Weighted Least Conflicting Changes'] / changed_nodes, 0).mean()

        not_changed_vs_least_ratio = np.where(changed_nodes != -1, df['NOT Least Conflicting Changes'] / not_changed_nodes, 0).mean()
        not_changed_vs_weighted_ratio = np.where(changed_nodes != -1,
                                             df['NOT Weighted Least Conflicting Changes'] / not_changed_nodes, 0).mean()

    results_data["changed_vs_least_ratio"].append(changed_vs_least_ratio)
    results_data["changed_vs_weighted_ratio"].append(changed_vs_weighted_ratio)
    results_data["not_changed_vs_least_ratio"].append(not_changed_vs_least_ratio)
    results_data["not_changed_vs_weighted_ratio"].append(not_changed_vs_weighted_ratio)


def distance_neighbors_strangers_color(adj, assignments, closesed_index, embeddings_list, min_disagree, results_data):
    def calculate_average_distances_per_iteration(embeddings_list, adj, ass):
        avg_distances_per_iteration_neighbors = []
        avg_distances_per_iteration_strangers = []
        avg_distances_per_iteration_color_group = []
        for embeddings in embeddings_list:
            avg_distances_neihobrs = []
            avg_distances_strangers = []
            avg_distances_color_group = []
            for i in range(len(embeddings)):
                neighbors = torch.where(adj[i] > 0)[0]
                neihbors_distances = [0]
                if len(neighbors) != 0:
                    neihbors_distances = [np.linalg.norm(embeddings[i] - embeddings[neighbor]) for neighbor in
                                          neighbors]
                avg_distances_neihobrs.append(np.mean(neihbors_distances))

                strangers = torch.where(adj[i] == 0)[0]
                strangers_distances = [0]
                if len(strangers) != 0:
                    strangers_distances = [np.linalg.norm(embeddings[i] - embeddings[stranger]) for stranger in
                                           strangers]
                avg_distances_strangers.append(np.mean(strangers_distances))

                color_group = [j for j, c in enumerate(ass) if c == ass[i]]
                color_group_dist = [0]
                if len(color_group) != 0:
                    color_group_dist = [np.linalg.norm(embeddings[i] - embeddings[cg]) for cg in color_group]
                avg_distances_color_group.append(np.mean(color_group_dist))

            avg_distances_per_iteration_neighbors.append(np.mean(avg_distances_neihobrs))
            avg_distances_per_iteration_strangers.append(np.mean(avg_distances_strangers))
            avg_distances_per_iteration_color_group.append(np.mean(avg_distances_color_group))
        return avg_distances_per_iteration_neighbors, avg_distances_per_iteration_strangers, avg_distances_per_iteration_color_group

    avg_distances_per_iteration_neighbors, avg_distances_per_iteration_strangers, avg_distances_per_iteration_color_group = calculate_average_distances_per_iteration(
        embeddings_list[:closesed_index], adj, assignments[closesed_index])
    df = pd.DataFrame({
        'Iteration': range(len(avg_distances_per_iteration_neighbors))[:closesed_index],
        'Average Distance': avg_distances_per_iteration_neighbors[:closesed_index],
        'Average Distance Strangers': avg_distances_per_iteration_strangers[:closesed_index],
        'Average Distance Color Group': avg_distances_per_iteration_color_group[:closesed_index],
    })
    iterations = df['Iteration']
    average_distances = df['Average Distance']
    if len(iterations) > 1:
        slope, intercept, r_value, p_value, std_err = linregress(iterations, average_distances)
    else:
        slope, intercept, r_value, p_value, std_err = 0, 0, 0, 0, 0

    statistic = -1
    pval = -1
    if len(average_distances) > 0:
        stats = sci_spearmanr(list(range(len(average_distances))), average_distances)
        statistic = stats.statistic
        pval = stats.pvalue
    results_data["node_distance_ordering"].append(statistic)
    results_data["node_distance_ordering_p"].append(pval)
    results_data["regression_slope"].append(slope)
    avg_distances_per_iteration_neighbors = np.array(avg_distances_per_iteration_neighbors[:closesed_index])
    avg_distances_per_iteration_strangers = np.array(avg_distances_per_iteration_strangers[:closesed_index])
    avg_distances_per_iteration_color_group = np.array(avg_distances_per_iteration_color_group[:closesed_index])
    signs_sitance_neighbor_strange = np.sign(
        avg_distances_per_iteration_neighbors - avg_distances_per_iteration_strangers)
    signs_sitance_neighbor_strange[signs_sitance_neighbor_strange == -1] = 0
    # signs_sitance_neighbor_strange.mean()
    results_data["signs_distance_neighbor_strange"].append(signs_sitance_neighbor_strange.mean().item() if len(signs_sitance_neighbor_strange) > 0 else -1)
    signs_sitance_stranger_color = np.sign(
        avg_distances_per_iteration_strangers - avg_distances_per_iteration_color_group)
    signs_sitance_stranger_color[signs_sitance_stranger_color == -1] = 0
    # signs_sitance_stranger_color.mean()
    results_data["signs_distance_stranger_color"].append(signs_sitance_stranger_color.mean().item() if len(signs_sitance_stranger_color)> 0 else -1)
    regression_line = slope * iterations + intercept
    fig = px.line(df, x='Iteration',
                  y=['Average Distance', 'Average Distance Strangers', 'Average Distance Color Group'],
                  title='Average Distance Between Nodes and Their Neighbors Over Time')
    fig.add_scatter(x=iterations, y=regression_line, mode='lines', name='Regression Line')
    if min_disagree[closesed_index] == 0 and slope < 0.1:
        pass
    # fig.show()


def support_axis_corolation(adj, assignments, closesed_index, embeddings_list, k, conflict_sums):
    support_over_time = []
    z_axis = []
    support_z_spearman = []
    degree_z_spearman = []
    pca_global = PCA(3)
    degrees = adj.cpu().sum(0)
    pca_global.fit(embeddings_list[closesed_index])
    for i in range(closesed_index + 1):
        emb = embeddings_list[i]

        conflict_sum = conflict_sums[i]
        support_over_time.append(conflict_sum[0])
        pca = PCA(3)
        pca.fit(emb)
        lower = (emb - pca.mean_) @ pca.components_.T
        lower_global = (emb - pca_global.mean_) @ pca_global.components_.T

        spear_x = sci_spearmanr(conflict_sum[0], lower[:, 0]).statistic
        spear_y = sci_spearmanr(conflict_sum[0], lower[:, 1]).statistic
        spear_z = sci_spearmanr(conflict_sum[0], lower[:, 2]).statistic
        global_spear_x = sci_spearmanr(conflict_sum[0], lower_global[:, 0]).statistic
        global_spear_y = sci_spearmanr(conflict_sum[0], lower_global[:, 1]).statistic
        global_spear_z = sci_spearmanr(conflict_sum[0], lower_global[:, 2]).statistic
        support_z_spearman.append(
            (spear_x, spear_y, spear_z, global_spear_x, global_spear_y, global_spear_z)
        )
        if i > 4:
            degree_spear_x = sci_spearmanr(degrees, lower[:, 0]).statistic
            degree_spear_y = sci_spearmanr(degrees, lower[:, 1]).statistic
            degree_spear_z = sci_spearmanr(degrees, lower[:, 2]).statistic
            degree_global_spear_x = sci_spearmanr(degrees, lower_global[:, 0]).statistic
            degree_global_spear_y = sci_spearmanr(degrees, lower_global[:, 1]).statistic
            degree_global_spear_z = sci_spearmanr(degrees, lower_global[:, 2]).statistic

            degree_z_spearman.append(
                (degree_spear_x, degree_spear_y, degree_spear_z, degree_global_spear_x, degree_global_spear_y, degree_global_spear_z)
            )
    if len(degree_z_spearman)==0:
        degree_z_spearman=[0]
    return np.abs(support_z_spearman).max(), np.abs(degree_z_spearman).max()


def get_spearman_support_mean_confidence(distances_over_time, embeddings_list, k, adj, closesed_index, assignments, results_data, supports):
    # distances_over_time = calc_dist_over_iteartions(adj, gcp.histories)
    rows = []
    columns = ["support", "confidence", "degree", "iteration"]
    degrees = adj.sum(0).numpy()
    for i in range(closesed_index + 1):
        support = supports[i][0]
        for j in range(len(support)):
            rows.append([support[j], distances_over_time[i][j], degrees[j], i])
    df = pd.DataFrame(rows, columns=columns)
    fig = px.scatter(df, x='support', y='confidence', animation_frame="iteration")
    # fig.show()
    iter_sup_mean_conf = df.groupby(["iteration", "support"])["confidence"].mean()
    spearman_corrs = []
    for i in range(len(df["iteration"].unique())):
        spearman_corrs.append(sci_spearmanr(iter_sup_mean_conf[i].index, iter_sup_mean_conf[i].values))
    results_data["spearman_support_mean_confidence"].append(np.mean([sc.statistic for sc in spearman_corrs]).item())

    iter_sup_mean_conf = df.groupby(["iteration", "degree"])["confidence"].mean()
    spearman_corrs = []
    for i in range(len(df["iteration"].unique())):
        spearman_corrs.append(sci_spearmanr(iter_sup_mean_conf[i].index, iter_sup_mean_conf[i].values))
    results_data["spearman_degree_mean_confidence"].append(np.mean([sc.statistic for sc in spearman_corrs]).item())

    iter_sup_mean_conf = df.groupby(["iteration"])
    for i, iter_df in iter_sup_mean_conf:
        spearman_corrs.append(sci_spearmanr(iter_df["degree"], iter_df["support"]))
    # print(np.mean([sc.statistic for sc in spearman_corrs]).item())


def run(args, gcp, critirion, results_data, dl):
    preds = []
    lables_agg = []
    plot_loss = 0
    plot_acc = 0
    t1 = time.perf_counter()

    mean = None
    conv = None
    count = 0

    for j, b in enumerate(dl):
        if j > 49: break
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
        l = critirion(means.to(DEVICE), torch.Tensor(labels).to(device=DEVICE))

        preds += pred.tolist() if len(pred.shape) != 0 else [pred.item()]
        plot_loss += l.detach()
        acc = ((pred.detach().cpu() > 0.5) == torch.Tensor(labels)).sum() / float(cn.shape[1])

        if len(labels.size()) > 0:
            k = cn[0, 0].item() + (1 - int(labels[0].item()))
        else:
            k = cn[0, 0].item() + (1 - int(labels.item()))
        k = k
        n = split[0]
        adj = M_vv[:n, :n]
        # continue

        # mean, std, count = get_pca_mean_std(mean, std, count, V_ret)
        # continue

        features = gcp.history[:n].clone().detach().cpu()
        lower = my_pca(features)
        degrees = adj.sum(axis=0).cpu()
        n = len(degrees)


        p = M_vv[:n, :n].sum() / (n * (n - 1))
        c = n * p
        results_data["n"] += [n]
        results_data["p"] += [p.item()]
        results_data["c"] += [c.item()]
        results_data["max_degree"] += [int(degrees.max().item())]
        results_data["mean_degree"] += [degrees.mean().item()]
        results_data["mean_degree_normalized"] += [degrees.mean().item()/(n-1)]
        results_data["k"] += [k]
        results_data["colorable"] += [int(labels)]

        print(n, int(degrees.max().item()), round(degrees.mean().item(), 2), round(degrees.mean().item()/(n-1), 3))

        attributes_check(results_data, M_vv, M_vc, split, labels, cn, n, k, gcp, adj)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--graph_dir', default="/home/elad/Documents/kcol/GCP_project/data_json/data")
    parser.add_argument('--embedding_size', type=int, default=64)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--tmax', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=5300)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--check_path', type=str, default=None)
    parser.add_argument('--test', type=bool, default=True)
    args = parser.parse_args()

    ds = GraphDataSet(
        args.graph_dir,
        batch_size=args.batch_size,
        filter=lambda f: f.split("/")[-1][3] == "3",
    )

    embedding_size = args.embedding_size
    gcp = GCPNet(embedding_size, tmax=args.tmax, device=args.device)
    gcp.to(args.device)
    gcp.eval()
    load_from_tf(args, gcp)
    critirion = torch.nn.BCEWithLogitsLoss(reduction="mean")

    results_data_original = {
        "n":[],
        "p":[],
        "c":[],

        "max_degree":[],
        "mean_degree":[],
        "mean_degree_normalized":[],
        "k":[],
        "colorable":[],
        "closesed_index":[],
        "attmept_range":[],
        "closest_ass_dist":[],
        "greedy_k": [],
        "sdp_k": [],
        "lovasz": [],
        "n_connected_components": [],
        "min_dist_from_neighbors": [],
        "most_common_color": [],
        "starting_color": [],
        "regression_slope": [],
        "node_distance_ordering": [],
        "node_distance_ordering_p": [],
        "diff_in_assignments": [],
        "changed_node": [],
        "avg_degree_of_changed_nodes":[],
        "least_conflict_change": [],
        "weighted_least_conflict_change": [],
        "greedy_conflicts": [],

        "min_disagree": [],
        "spearman_support_mean_confidence":[],
        "spearman_degree_mean_confidence":[],
        "spearman_degree_mean_support":[],

        "changed_ratio_mean": [],
        "unchanged_ratio_mean": [],

        "confidence_pair_contrudiction": [],
        "spearman_avg_confidence_over_time":[],
        "spearman_p_avg_confidence_over_time":[],
        "signs_distance_neighbor_strange":[],
        "signs_distance_stranger_color":[],
        "changed_vs_least_ratio": [],
        "changed_vs_weighted_ratio": [],
        "triangle_regression": [],
        "pc_corolation": [],
        "degree_pc_corolation": [],
        "triangle_metric":[],
        "triangle_apex_regression":[],

        "not_changed_vs_least_ratio": [],
        "not_changed_vs_weighted_ratio":[],

        "embeddings": [],
        "centroid_from_triangle": [],
        "centroid_diff_from_line": [],
        "above_below": [],
        "interstion_70":[],
        "interstion_80":[],
        "interstion_90":[],
        "support_percentile_of_70":[],
        "support_percentile_of_80":[],
        "support_percentile_of_90":[],
    }


    if args.check_path is not None:
        print("loading:", args.check_path)
        checkpoint = torch.load(args.check_path, map_location="cpu")
        gcp.load_all_attributes(checkpoint['model'])
    gcp.to(args.device)


    ns = [
        45, 100,
        # 250,
        500,
        1000, 2000]
    k = 3
    cs = [
        # 0.5
        1, 2,
          3, 3.5, 4, 4.5,
        # 5, 5.5
    ]
    ps = [
        # 0.3,
        0.5, 0.8]

    ds.shuffle()
    dl_files = DataLoader(ds, batch_size=1, shuffle=False)
    datasets = []

    for n in ns:
        results_data = results_data_original.copy()
        for c in cs: # Random
            if os.path.exists(f"results_data_n_{n}_c_{c}.json"):
                continue
            print("starting nc", n,c)
            if FS().file_exists(os.path.join(data_path, "experiments", "random", str(n), str(c), "data.pt")):
                random = FS().get_data(os.path.join(data_path, "experiments", "random", str(n), str(c), "data.pt"))
                random = torch.load(io.BytesIO(random))["data"]
                dl = DataLoader(GeneratedGraphs(random), batch_size=1, shuffle=False)
                datasets.append(dl)
                res = run(args, gcp, critirion, results_data, dl)
                with open(f'results_data_n_{n}_c_{c}.json', 'w') as json_file:
                    json.dump(results_data, json_file, indent=4)
            else:
                random = []
                for _ in range(1_000):
                    p = c/n
                    ass, v_mat, bad, be = create_gnp(n, k=k, c=c)
                    random.append(v_mat)
                buffer = io.BytesIO()
                torch.save({"data": random}, buffer)
                buffer.seek(0)
                FS().upload_data(buffer, os.path.join(data_path, "experiments", "random",str(n), str(c), "data.pt"))
        results_data = results_data_original.copy()
        for p in ps: # planted
            # if os.path.exists(f"results_data_n_{n}_p_{p}.json"):
            #     continue
            print("starting np", n, p)
            if FS().file_exists(os.path.join(data_path, "experiments", "planted",str(n), str(p), "data.pt")):
                # pass
                planted = FS().get_data(os.path.join(data_path, "experiments", "planted", str(n), str(p), "data.pt"))
                planted = torch.load(io.BytesIO(planted))["data"]
                dl = DataLoader(GeneratedGraphs(planted), batch_size=1, shuffle=False)
                datasets.append(dl)
                res = run(args, gcp, critirion, results_data, dl)
                # with open(f'results_data_n_{n}_p_{p}.json', 'w') as json_file:
                #     json.dump(results_data, json_file, indent=4)
            else:
                planted = []
                for _ in range(1_000):
                    c = n * p
                    ass, v_mat, bad, be = create_planted(n, k=k, p=p)
                    planted.append(v_mat)
                buffer = io.BytesIO()
                torch.save({"data": planted}, buffer)
                buffer.seek(0)
                FS().upload_data(buffer, os.path.join(data_path, "experiments", "planted",str(n), str(p), "data.pt"))

    datasets.append(dl_files)
    results_data = results_data_original.copy()
    for dl in datasets:
        run(args, gcp, critirion, results_data, dl)
    print("done")


if __name__ == '__main__':
    main()