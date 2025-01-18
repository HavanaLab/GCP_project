import itertools
import io

from scipy.cluster.vq import kmeans
from scipy.stats import spearmanr as sci_spearmanr

import plotly.io as pio

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
    confidance_same_pair
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
import json

FOLDER = "./"#"results/21 11 2024 - 3"

#list all files with the prefix 'results_data' in the directory using os module
all_results_files = [f for f in os.listdir(FOLDER) if f.startswith('results_data')]

all_results_data = {}

if len(all_results_files)==0:
    print("no files found")
    sys.exit()

keys = []
with open(os.path.join(FOLDER, all_results_files[0]), 'r') as f:
    data = json.load(f)
    for k, v in data.items():
        if len(v) > 0: keys.append(k)
    print("'" + "','".join(keys) + "'")

# keys = ['n','p','c','max_degree','mean_degree','mean_degree_normalized','k','colorable','closesed_index','attmept_range','closest_ass_dist','regression_slope','node_distance_ordering','node_distance_ordering_p','min_disagree','spearman_support_mean_confidence','spearman_degree_mean_confidence','changed_ratio_mean','unchanged_ratio_mean','confidence_pair_contrudiction','spearman_avg_confidence_over_time','spearman_p_avg_confidence_over_time','signs_distance_neighbor_strange','signs_distance_stranger_color','triangle_regression','pc_corolation','degree_pc_corolation','triangle_metric','triangle_apex_regression','embeddings','centroid_from_triangle','centroid_diff_from_line']

for file in all_results_files:
    name = ".".join(file.split('.')[:-1])
    name = name.split('_')[3:6]
    n = int(name[0])
    c_or_p = name[1]
    cp = float(name[2])
    name = (n, c_or_p, cp)
    if name in all_results_data and len(all_results_data[name]["c"]) > 0: continue
    with open(os.path.join(FOLDER, file), 'r') as f:
        data = json.load(f)
        # keys = []0
        # for k, v in data.items():
        #     if len(v) > 0: keys.append(k)
        # print("'" + "','".join(keys) + "'")
    indexes = []
    for i in range(len(data["n"])):
        if data["n"][i] == n:
            indexes.append(i)
    relavent_data = {
        key: [data[key][j] for j in indexes] for key in keys if key in data
    }
    all_results_data[name] = relavent_data


succes_triangle = []
suc = 0
fail =0
fail_triangle = []
success_triangle_metric = []
fail_triangle_metric = []

avg_slope = []
avg_spearman_slope = []
avg_spearman_slope_p = []

spearman_avg_confidence_over_time = []
spearman_p_avg_confidence_over_time = []

percentiles = []

condifence_support = []
condifence_degree = []

degree_support_pc3 = []

changed_vs_weighted_ratio= []
changed_vs_least_ratio = []

pc_corolation  = []
degree_pc_corolation = []

attmept_range = {}

changed_ratio_mean = []
unchanged_ratio_mean=[]

successful_centroid_from_triangle= []
successful_centroid_diff_from_line=[]
successful_triangle_apex_regression=[]

faile_centroid_from_triangle= []
faile_centroid_diff_from_line=[]
faile_triangle_apex_regression=[]

spearman_triangle = []

interstion_70 = []
interstion_80 = []
interstion_90 = []

support_percentile_of_70 = []
support_percentile_of_80= []
support_percentile_of_90= []

above=[]
below = []

for (n, cp_type, cp), value in all_results_data.items():
    if cp_type == "c" and float(cp) >= 5.0:
        continue

    suc = 0
    fail = 0
    single_succes_triangle = []
    single_fail_triangle = []

    single_succes_triangle_metric = []
    single_fail_triangle_metric = []

    single_success_centroid_from_triangle   = []
    single_success_centroid_diff_from_line  = []
    single_success_triangle_apex_regression = []

    single_fail_centroid_from_triangle   = []
    single_fail_centroid_diff_from_line  = []
    single_fail_triangle_apex_regression = []

    single_interstion_70 = []
    single_interstion_80 = []
    single_interstion_90 = []

    # spearman_triangle.append(sci_spearmanr(value["min_disagree"], value["centroid_from_triangle"]).statistic)

    for i in range(len(value["c"])):
        att = value["attmept_range"][i]
        if att not in attmept_range:
            attmept_range[att] = 0
        attmept_range[att] += 1

        # print("stats intance", n, cp_type, cp, value["c"][i], value["p"][i], value["n"][i])

        if value["min_disagree"][i] > max(0.0 * value["n"][i], 1): # if fail then skip -> show success
        # if value["min_disagree"][i] <= max(0.0 * value["n"][i], 1):  # if success then skip -> show fail
            # print("----", value["min_disagree"][i], max(0.1 * value["n"][i], 1))
            continue

        # if value["c"][i] > 2.5: # show only sparse
        # if value["c"][i] < 2.5 or value["c"][i] > 5:  # show only dense
        if value[cp_type][i] < 5:  # show only externe
            continue

        if value["min_disagree"][i] <= max(0.0*value["n"][i], 1):
            pass
            # pca = PCA(n_components=2)
            # embeddings = np.array(value["embeddings"][i])
            # kmeans = KMeans(n_clusters=3, max_iter=100)
            # kmeans.fit(embeddings)
            # assignments = kmeans.labels_
            # centroids = kmeans.cluster_centers_
            # embeddings = pca.fit_transform(embeddings)
            #
            # centroids = pca.transform(centroids)
            #
            # unique_points = np.unique(np.round(embeddings, 4), axis=0)
            # if len(unique_points) >2:
            #     pio.renderers.default = 'browser'
            #     ret, triangle = cv.minEnclosingTriangle(unique_points.astype(np.float32))
            #     triangle = triangle.squeeze()
            #     fig = px.scatter(x=embeddings[:,0], y=embeddings[:,1], color=[str(a) for a in assignments], title=f"{n} {c_or_p} {cp} {value['min_disagree'][i]}")
            #     xs = triangle[:,0].tolist() + [triangle[0,0]]
            #     ys = triangle[:,1].tolist() + [triangle[0,1]]
            #     fig.add_trace(px.line(x=xs, y=ys).data[0])
            #     for tri in triangle:
            #         fig.add_trace(px.scatter(x=[0, tri[0]], y=[0, tri[1]]).data[0])
            #     fig.add_scatter(x=centroids[:,0], y=centroids[:,1], mode='markers')
            #     pass
            #     # fig.show()

            suc +=1
            # succes_triangle.append(value["triangle_regression"][i])
            # single_succes_triangle.append(value["triangle_regression"][i])
            # success_triangle_metric.append(value["triangle_metric"][i])
            # single_succes_triangle_metric.append(value["triangle_metric"][i])

            # successful_centroid_from_triangle.append(value["centroid_from_triangle"][i])
            # successful_centroid_diff_from_line.append(value["centroid_diff_from_line"][i])
            # successful_triangle_apex_regression.append(value["triangle_apex_regression"][i])
            # single_success_centroid_from_triangle.append(value["centroid_from_triangle"][i])
            # single_success_centroid_diff_from_line.append(value["centroid_diff_from_line"][i])
            # single_success_triangle_apex_regression.append(value["triangle_apex_regression"][i])
        else:
            pass
            fail +=1
            # fail_triangle.append(value["triangle_regression"][i])
            # single_fail_triangle.append(value["triangle_regression"][i])
            # fail_triangle_metric.append(value["triangle_metric"][i])
            # single_fail_triangle_metric.append(value["triangle_metric"][i])

            # if value["centroid_from_triangle"][i]>0:
            #     faile_centroid_from_triangle.append(value["centroid_from_triangle"][i])
            # if value["centroid_diff_from_line"][i]>0:
            #     faile_centroid_diff_from_line.append(value["centroid_diff_from_line"][i])
            # faile_triangle_apex_regression.append(value["triangle_apex_regression"][i])
            # single_fail_centroid_from_triangle.append(value["centroid_from_triangle"][i])
            # single_fail_centroid_diff_from_line.append(value["centroid_diff_from_line"][i])
            # single_fail_triangle_apex_regression.append(value["triangle_apex_regression"][i])

        # avg_slope.append(value["regression_slope"][i])
        avg_spearman_slope.append(value["node_distance_ordering"][i])
        avg_spearman_slope_p.append(value["node_distance_ordering_p"][i])
        # spearman_avg_confidence_over_time.append(value["spearman_avg_confidence_over_time"][i])
        # spearman_p_avg_confidence_over_time.append(value["spearman_p_avg_confidence_over_time"][i])

        # if value["closesed_index"][i] > 4:
        #     percentiles.append(value["confidence_pair_contrudiction"][i])
        condifence_support.append(value["spearman_support_mean_confidence"][i])

        if "degree_spearman_support_mean_confidence" in value:
            condifence_degree.append(value["degree_spearman_support_mean_confidence"][i])

        # changed_vs_weighted_ratio.append(value["changed_vs_weighted_ratio"][i])
        # changed_vs_least_ratio.append(value["changed_vs_least_ratio"][i])
        pc_corolation.append(value["pc_corolation"][i])
        if "degree_pc_corolation" in value:
            degree_pc_corolation.append(value["degree_pc_corolation"][i])
        # changed_ratio_mean.append(value["changed_ratio_mean"][i])
        # unchanged_ratio_mean.append(value["unchanged_ratio_mean"][i])

        # if value["min_disagree"][i] <= max(0.0 * value["n"][i], 1):
        above.append(value["above_below"][i][0])
        below.append(value["above_below"][i][1])

        support_percentile_of_70.append(value["support_percentile_of_70"][i])
        support_percentile_of_80.append(value["support_percentile_of_80"][i])
        support_percentile_of_90.append(value["support_percentile_of_90"][i])

        interstion_70.append(value["interstion_70"][i])
        interstion_80.append(value["interstion_80"][i])
        interstion_90.append(value["interstion_90"][i])
        single_interstion_70.append(value["interstion_70"][i])
        single_interstion_80.append(value["interstion_80"][i])
        single_interstion_90.append(value["interstion_90"][i])

    single_succes_triangle = np.array(single_succes_triangle)
    single_succes_triangle = single_succes_triangle[single_succes_triangle != 1]
    single_fail_triangle = np.array(single_fail_triangle)
    single_fail_triangle = single_fail_triangle[single_fail_triangle != 1]
    # print(n, cp_type, cp, "single triangle regression",
    #       "[",
    #         np.mean(single_succes_triangle), np.std(single_succes_triangle), len(single_succes_triangle), "] [",
    #         np.mean(single_fail_triangle), np.std(single_fail_triangle), len(single_fail_triangle), "]",
    #       )
    # print(n, cp_type, cp, "single triangle metric",
    #       "[",
    #         np.mean(single_succes_triangle_metric), np.std(single_succes_triangle_metric), len(single_succes_triangle_metric), "] [",
    #         np.mean(single_fail_triangle_metric), np.std(single_fail_triangle_metric), len(single_fail_triangle_metric), "]",
    #       )
    # print(n, cp_type, cp, "single centroid_from_triangle",
    #       "[",
    #         np.mean(single_success_centroid_from_triangle), np.std(single_success_centroid_from_triangle), len(single_success_centroid_from_triangle), "] [",
    #         np.mean(single_fail_centroid_from_triangle), np.std(single_fail_centroid_from_triangle), len(single_fail_centroid_from_triangle), "]",
    #       )
    # print(n, cp_type, cp, "single centroid_diff_from_line",
    #         "[",
    #             np.mean(single_success_centroid_diff_from_line), np.std(single_success_centroid_diff_from_line), len(single_success_centroid_diff_from_line), "] [",
    #             np.mean(single_fail_centroid_diff_from_line), np.std(single_fail_centroid_diff_from_line), len(single_fail_centroid_diff_from_line), "]",
    #         )
    # print(n, cp_type, cp, "single triangle_apex_regression",
    #         "[",
    #             np.mean(single_success_triangle_apex_regression), np.std(single_success_triangle_apex_regression), len(single_success_triangle_apex_regression), "] [",
    #             np.mean(single_fail_triangle_apex_regression), np.std(single_fail_triangle_apex_regression), len(single_fail_triangle_apex_regression), "]",
    #         )
    single_interstion_70 = np.array(single_interstion_70)
    single_interstion_70 = single_interstion_70[single_interstion_70 >= 0]
    single_interstion_80 = np.array(single_interstion_80)
    single_interstion_80 = single_interstion_80[single_interstion_80 >= 0]
    single_interstion_90 = np.array(single_interstion_90)
    single_interstion_90 = single_interstion_90[single_interstion_90 >= 0]
    print(
        n, cp_type, cp, "interstion", single_interstion_70.mean(), single_interstion_70.std(), single_interstion_80.mean(), single_interstion_80.std(),
        single_interstion_90.mean(), single_interstion_90.std()
    )
    print("\t", suc, fail, suc + fail)

succes_triangle = np.array(succes_triangle)
succes_triangle[succes_triangle < 0] = 0

fail_triangle = np.array(fail_triangle)
fail_triangle[fail_triangle < 0] = 0

# print("triangle regression", np.mean(succes_triangle), np.std(succes_triangle), len(succes_triangle), np.mean(fail_triangle), np.std(fail_triangle), len(fail_triangle))
# print("triangle metric", np.mean(success_triangle_metric), np.std(success_triangle_metric), len(success_triangle_metric), np.mean(fail_triangle_metric), np.std(fail_triangle_metric), len(fail_triangle_metric))

successful_triangle_apex_regression = np.array(successful_triangle_apex_regression)
successful_triangle_apex_regression=successful_triangle_apex_regression[successful_triangle_apex_regression>=0]
successful_centroid_from_triangle = np.array(successful_centroid_from_triangle)
successful_centroid_from_triangle=successful_centroid_from_triangle[successful_centroid_from_triangle>=0]
successful_centroid_diff_from_line = np.array(successful_centroid_diff_from_line)
successful_centroid_diff_from_line=successful_centroid_diff_from_line[successful_centroid_diff_from_line>=0]

faile_triangle_apex_regression = np.array(faile_triangle_apex_regression)
faile_triangle_apex_regression=faile_triangle_apex_regression[faile_triangle_apex_regression>=0]
faile_centroid_from_triangle = np.array(faile_centroid_from_triangle)
faile_centroid_from_triangle=faile_centroid_from_triangle[faile_centroid_from_triangle>=0]
faile_centroid_diff_from_line = np.array(faile_centroid_diff_from_line)
faile_centroid_diff_from_line=faile_centroid_diff_from_line[faile_centroid_diff_from_line>=0]

# print("triangle apex regression", successful_triangle_apex_regression[successful_triangle_apex_regression>=0].mean(), successful_triangle_apex_regression[successful_triangle_apex_regression>=0].std(), len(successful_triangle_apex_regression), faile_triangle_apex_regression.mean(), faile_triangle_apex_regression.std(), len(faile_triangle_apex_regression))
# print("centroid_from_triangle", successful_centroid_from_triangle[successful_centroid_from_triangle>=0].mean(), successful_centroid_from_triangle[successful_centroid_from_triangle>=0].std(), len(successful_centroid_from_triangle), faile_triangle_apex_regression.mean(), faile_triangle_apex_regression.std(), len(faile_triangle_apex_regression))
# print("centroid_diff_from_line", successful_centroid_diff_from_line[successful_centroid_diff_from_line>=0].mean(), successful_centroid_diff_from_line[successful_centroid_diff_from_line>=0].std(), len(successful_centroid_diff_from_line), faile_triangle_apex_regression.mean(), faile_triangle_apex_regression.std(), len(faile_triangle_apex_regression))


above = np.array([a['(70, 70)'] for a in above])
above = above[above>=0]
below = np.array([a['(20, 20)'] for a in below])
below = below[below>=0]
# print("above and below", above.mean(), above.std(), len(above), below.mean(), below.std(), len(below))
# print("above and below", above.mean(), above.std(), len(above), below.mean(), below.std(), len(below))
print("low", below.mean(), below.std(), len(below))
print("high", above.mean(), above.std(), len(above))

# for k, v in attmept_range.items():
#     print("attempt range", k, v)


# avg_slope = np.array(avg_slope)
# avg_slope = avg_slope[avg_slope!=0]
# print("slope", avg_slope[avg_slope>0].mean(), avg_slope[avg_slope>0].std(), len(avg_slope), len(avg_slope[avg_slope<0]))
#

condifence_support = np.array(condifence_support)
print("confidence_support ", condifence_support[condifence_support>=0].mean(), condifence_support[condifence_support>=0].std(), len(condifence_support), len(condifence_support[condifence_support<0]))


avg_spearman_slope = np.array(avg_spearman_slope)
avg_spearman_slope = avg_spearman_slope[avg_spearman_slope!=-1]
print("spearman disatnce increasing", avg_spearman_slope[avg_spearman_slope>=0].mean(), avg_spearman_slope[avg_spearman_slope>=0].std(), len(avg_spearman_slope), len(avg_spearman_slope[avg_spearman_slope<0]))

avg_spearman_slope_p = np.array(avg_spearman_slope_p)
avg_spearman_slope_p = avg_spearman_slope_p[avg_spearman_slope_p!=-1]
print("spearman disatnce increasing p", avg_spearman_slope_p[avg_spearman_slope_p>=0].mean(), avg_spearman_slope_p[avg_spearman_slope_p>=0].std(), len(avg_spearman_slope_p))

# spearman_avg_confidence_over_time = np.array(spearman_avg_confidence_over_time)
# spearman_avg_confidence_over_time = spearman_avg_confidence_over_time[spearman_avg_confidence_over_time!=-2]
# print(
#     "once large always large", spearman_avg_confidence_over_time[spearman_avg_confidence_over_time>=0].mean(), spearman_avg_confidence_over_time[spearman_avg_confidence_over_time>=0].std(), len(spearman_avg_confidence_over_time), len(spearman_avg_confidence_over_time[spearman_avg_confidence_over_time<0]))
# spearman_p_avg_confidence_over_time = np.array(spearman_p_avg_confidence_over_time)
# spearman_p_avg_confidence_over_time = spearman_p_avg_confidence_over_time[spearman_p_avg_confidence_over_time!=-2]
# print(
#     "once large always large p", spearman_p_avg_confidence_over_time[spearman_p_avg_confidence_over_time>=0].mean(), spearman_p_avg_confidence_over_time[spearman_p_avg_confidence_over_time>=0].std(), len(spearman_p_avg_confidence_over_time), len(spearman_p_avg_confidence_over_time[spearman_p_avg_confidence_over_time<0]))

interstion_70 = np.array(interstion_70)
interstion_70= interstion_70[interstion_70>=0]
interstion_80 = np.array(interstion_80)
interstion_80= interstion_80[interstion_80>=0]
interstion_90 = np.array(interstion_90)
interstion_90= interstion_90[interstion_90>=0]
# print(
# "interstion", interstion_70.mean(),interstion_70.std(), interstion_80.mean(), interstion_80.std(), interstion_90.mean(), interstion_90.std()
# )

# support_percentile_of_70 = np.array(support_percentile_of_70)
# support_percentile_of_70= support_percentile_of_70[support_percentile_of_70>=0]
# support_percentile_of_80 = np.array(support_percentile_of_80)
# support_percentile_of_80= support_percentile_of_80[support_percentile_of_80>=0]
# support_percentile_of_90 = np.array(support_percentile_of_90)
# support_percentile_of_90= support_percentile_of_90[support_percentile_of_90>=0]
# print(
# "support_percentile_of_70", support_percentile_of_70.mean(),support_percentile_of_70.std(), support_percentile_of_80.mean(), support_percentile_of_80.std(), support_percentile_of_90.mean(), support_percentile_of_90.std()
# )


percentiles = np.array(percentiles)
# print(
#     "percentiles",
#     [round(percentiles[:,0][percentiles[:,0]!=-1].mean(),3), round(percentiles[:,0][percentiles[:,0]!=-1].std(),3)],
#     [round(percentiles[:,1][percentiles[:,1]!=-1].mean(),3), round(percentiles[:,1][percentiles[:,1]!=-1].std(),3)],
#     [round(percentiles[:,2][percentiles[:,2]!=-1].mean(),3), round(percentiles[:,2][percentiles[:,2]!=-1].std(),3)],
#     [round(percentiles[:,3][percentiles[:,3]!=-1].mean(),3), round(percentiles[:,3][percentiles[:,3]!=-1].std(),3)],
#     [round(percentiles[:,4][percentiles[:,4]!=-1].mean(),3), round(percentiles[:,4][percentiles[:,4]!=-1].std(),3)],
#     [round(percentiles[:,5][percentiles[:,5]!=-1].mean(),3), round(percentiles[:,5][percentiles[:,5]!=-1].std(),3)],
#     # [round(percentiles[:,6][percentiles[:,6]!=-1].mean(),3), round(percentiles[:,6][percentiles[:,6]!=-1].std(),3)],
# )


# condifence_degree = np.array(condifence_degree)
# print("confidence_degree ", condifence_degree[condifence_degree>=0].mean(), condifence_degree[condifence_degree>=0].std(), len(condifence_degree), len(condifence_degree[condifence_degree<0]))


# changed_vs_weighted_ratio = np.array(changed_vs_weighted_ratio)
# changed_vs_weighted_ratio = changed_vs_weighted_ratio[changed_vs_weighted_ratio!=-1]
# print("changed_vs_weighted_ratio", changed_vs_weighted_ratio[changed_vs_weighted_ratio>=0].mean(), changed_vs_weighted_ratio[changed_vs_weighted_ratio>=0].std(), len(changed_vs_weighted_ratio), len(changed_vs_weighted_ratio[changed_vs_weighted_ratio<0]))

# changed_vs_least_ratio = np.array(changed_vs_least_ratio)
# changed_vs_least_ratio = changed_vs_least_ratio[changed_vs_least_ratio!=-1]
# print("changed_vs_least_ratio", changed_vs_least_ratio[changed_vs_least_ratio>=0].mean(), changed_vs_least_ratio[changed_vs_least_ratio>=0].std(), len(changed_vs_least_ratio), len(changed_vs_least_ratio[changed_vs_least_ratio<0]))

pc_corolation = np.array(pc_corolation)
# print("pc_corolation", pc_corolation[pc_corolation>=0].mean(), pc_corolation[pc_corolation>=0].std(), len(pc_corolation), len(pc_corolation[pc_corolation<0]))

# degree_pc_corolation = np.array(degree_pc_corolation)
# print("degree_pc_corolation", degree_pc_corolation[degree_pc_corolation>=0].mean(), degree_pc_corolation[degree_pc_corolation>=0].std(), len(degree_pc_corolation), len(degree_pc_corolation[degree_pc_corolation<0]))

# changed_ratio_mean = np.array(changed_ratio_mean)
# changed_ratio_mean=changed_ratio_mean[changed_ratio_mean!=-1]
# print("changed_ratio_mean", changed_ratio_mean[changed_ratio_mean>=0].mean(), changed_ratio_mean[changed_ratio_mean>=0].std(), len(changed_ratio_mean), len(changed_ratio_mean[changed_ratio_mean<0]))
# unchanged_ratio_mean = np.array(unchanged_ratio_mean)
# unchanged_ratio_mean=unchanged_ratio_mean[unchanged_ratio_mean!=-1]
# print("unchanged_ratio_mean", unchanged_ratio_mean[unchanged_ratio_mean>=0].mean(), unchanged_ratio_mean[unchanged_ratio_mean>=0].std(), len(unchanged_ratio_mean), len(unchanged_ratio_mean[unchanged_ratio_mean<0]))

