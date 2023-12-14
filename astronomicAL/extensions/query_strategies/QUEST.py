import sys
import os

# sys.path.append("/jmain02/home/J2AD001/wwp02/gxs68-wwp02/.local/lib/python3.8/site-packages")

import numpy as np
import torch
from torch import is_tensor
from scipy.special import softmax
from scipy.stats import entropy

import matplotlib.pyplot as plt
# import umap
from scipy.spatial import Voronoi, voronoi_plot_2d
from shapely import geometry, ops
from shapely.geometry import MultiPoint,Point, Polygon
from geopandas import GeoSeries
from sklearn.cluster import KMeans
from .query_strategies import *
import json


# ████████╗███████╗███████╗███████╗    ███╗   ███╗███████╗████████╗██╗  ██╗ ██████╗ ██████╗ ███████╗
# ╚══██╔══╝██╔════╝██╔════╝██╔════╝    ████╗ ████║██╔════╝╚══██╔══╝██║  ██║██╔═══██╗██╔══██╗██╔════╝
#    ██║   █████╗  ███████╗███████╗    ██╔████╔██║█████╗     ██║   ███████║██║   ██║██║  ██║███████╗
#    ██║   ██╔══╝  ╚════██║╚════██║    ██║╚██╔╝██║██╔══╝     ██║   ██╔══██║██║   ██║██║  ██║╚════██║
#    ██║   ███████╗███████║███████║    ██║ ╚═╝ ██║███████╗   ██║   ██║  ██║╚██████╔╝██████╔╝███████║
#    ╚═╝   ╚══════╝╚══════╝╚══════╝    ╚═╝     ╚═╝╚══════╝   ╚═╝   ╚═╝  ╚═╝ ╚═════╝ ╚═════╝ ╚══════╝
                                                                                                  

def average(data, batch_size,tess_size,scores=[], method=[]):
    print("average func")
    idxs = []

    region_order = data[:,3][np.sort(np.unique(data[:,3], return_index=True)[1])]
    # print(len(region_order))
    region_order = np.delete(region_order, np.where(region_order == -1), axis=0)
    # print(len(region_order))
    # assert False
    avgs = []

    for i in range(len(region_order)):
        rows = np.where(data[:,3] == region_order[i])[0]
        # print(rows)
        avg = np.mean(data[rows,4])
        avgs.append(avg)

    avgs_sort = np.argsort(avgs)
    if method == "max":
        avgs_sort = avgs_sort[::-1]
    

    remainder = batch_size % tess_size
    samples = int(batch_size / tess_size)
    # print("remainder: ", remainder)
    # print("samples: ", samples)

    # count = 0
    count = 0
    overflow = 0
    other_i = 0
    # print("overflow")
    for i in range(tess_size):
        update = True
        other_i = i
        # print(f"--{i}--")
        if remainder == 0:
            # print(f"{region_order[avgs_sort[i]]} has {len(idxs_region[np.where(idxs_region[:,1] == region_order[avgs_sort[i]])[0][:samples+1],0].tolist())} points")
            max_s = samples

            selected = data[np.where(data[:,3] == region_order[avgs_sort[i]])[0]][:max_s,0].tolist()
            idxs = idxs + selected

            if len(selected) < max_s:
                other_i -= 1
            else:
                other_i = i


        elif (other_i < remainder):
            # print(f"{region_order[avgs_sort[i]]} has {len(idxs_region[np.where(idxs_region[:,1] == region_order[avgs_sort[i]])[0][:samples+1],0].tolist())} points")
            max_s = samples + 1
            selected = data[np.where(data[:,3] == region_order[avgs_sort[i]])[0]][:max_s,0].tolist()
            idxs = idxs + selected

            if len(selected) < max_s:
                other_i -= 1
            else:
                other_i = i

        else:
            # print(f"{region_order[avgs_sort[i]]} has {len(idxs_region[np.where(idxs_region[:,1] == region_order[avgs_sort[i]])[0][:samples],0].tolist())} points")
            max_s = samples + 1
            selected = data[np.where(data[:,3] == region_order[avgs_sort[i]])[0]][:max_s,0].tolist()
            idxs = idxs + selected

            if len(selected) < max_s:
                other_i -= 1
            else:
                other_i = i

        # print(other_i)
        # print("count:", count)
        # print(f"after {i} overflow: {overflow}")
    # print("")

    idxs = idxs[:batch_size]
    
    # print("total: ", len(idxs))
    # print(idxs)
    # assert False
    # print(idxs)

    # assert len(idxs) == batch_size, f"{len(idxs)} vs {batch_size}"
    return idxs    

def top_N_query(idxs_region, batch_size,tess_size,region_order=None,scores=[] ,method=[]):

    print("Top N Query")

    idxs = sequential_query(idxs_region, batch_size,tess_size,region_order=region_order, seq=1)

    assert len(idxs) == batch_size, f"{len(idxs)} vs {batch_size}"
    return idxs

    
def cycle_N_query(data, batch_size, tess_size,region_order=None, scores=[], method=[], region_scores=[]):
    idxs = []

    # region_order = data[:,3][np.sort(np.unique(data[:,3], return_index=True)[1])]
    # region_order = np.delete(region_order, np.where(region_order == -1), axis=0)
    if region_order is None:
        region_order = data[:,3][np.sort(np.unique(data[:,3], return_index=True)[1])]
        region_order = np.delete(region_order, np.where(region_order == -1), axis=0)
    
    if len(region_scores) > 0:
        remainder = batch_size % len(region_scores)
        samples = int(batch_size / len(region_scores))

    else:
        remainder = batch_size % len(region_order)
        samples = int(batch_size / len(region_order))

    last = 0
    idxs_region_temp = data.copy()
    argsort = np.argsort(idxs_region_temp[:,4])

    if method == "max":
        argsort = argsort[::-1]

    idxs_region_temp = idxs_region_temp[argsort]

    # idxs_region_temp = idxs_region.copy()

    print("cycling...")
    for j in range(samples+1):
        if j > 0:
            idxs_region_temp = idxs_region_temp[int(np.where(idxs_region_temp[:,0] == last)[0])+1:]
        # print("---")
        if len(region_scores) > 0:
            for i in range(len(region_scores)):
                if len(idxs) == batch_size:
                    break
                # print(idxs_region_temp[np.where(idxs_region_temp[:,1] == region_order[i])[0][0],0])
                matches = np.where(idxs_region_temp[:,3] == region_order[region_scores[i]])[0]
                if len(matches) > 0:
                    # print(matches)
                    # print(idxs_region_temp[matches[0],0])
                    idxs = idxs + [idxs_region_temp[matches[0],0]]
                    last = idxs[-1]
        else:
            for i in range(len(region_order)):
                if len(idxs) == batch_size:
                    break
                # print(idxs_region_temp[np.where(idxs_region_temp[:,1] == region_order[i])[0][0],0])
                matches = np.where(idxs_region_temp[:,3] == region_order[i])[0]
                if len(matches) > 0:
                    # print(matches)
                    # print(idxs_region_temp[matches[0],0])
                    idxs = idxs + [idxs_region_temp[matches[0],0]]
                    last = idxs[-1]
    if len(idxs) < batch_size:
        print("extra needed")
    loop = 1
    while len(idxs) < batch_size:
        for j in range(samples+1+loop):
            if len(idxs) == batch_size:
                break
            if j > 0:
                print(last)
                idxs_region_temp = idxs_region_temp[int(np.where(idxs_region_temp[:,0] == last)[0])+1:]
            # print("---")
            if len(region_scores) > 0:
                for i in range(len(region_scores)):
                    if len(idxs) == batch_size:
                        break
                    # print(idxs_region_temp[np.where(idxs_region_temp[:,1] == region_order[i])[0][0],0])
                    matches = np.where(idxs_region_temp[:,3] == region_order[region_scores[i]])[0]
                    if len(matches) > loop:
                        # print(matches)
                        # print(idxs_region_temp[matches[0],0])
                        print(idxs_region_temp[matches[loop],0])
                        if idxs_region_temp[matches[loop],0] not in idxs:
                            idxs = idxs + [idxs_region_temp[matches[loop],0]]
                            last = idxs[-1]
            else:
                for i in range(len(region_order)):
                    if len(idxs) == batch_size:
                        break
                    # print(idxs_region_temp[np.where(idxs_region_temp[:,1] == region_order[i])[0][0],0])
                    matches = np.where(idxs_region_temp[:,3] == region_order[i])[0]
                    if len(matches) > loop:
                        # print(matches)
                        print(idxs_region_temp[matches[loop],0])
                        if idxs_region_temp[matches[loop],0] not in idxs:
                            idxs = idxs + [idxs_region_temp[matches[loop],0]]
                            last = idxs[-1]
        loop += 1
            # print(int(np.where(idxs_region[:,0] == last)[0]))
            # print(idxs_region_temp[int(np.where(idxs_region_temp[:,0] == last)[0][0]):int(np.where(idxs_region_temp[:,0] == last)[0])+5])

    # print("total: ", len(idxs))
    # for i in range(len(idxs)):
    #     print(np.where(idxs_region == idxs[i])[0])
    # for i in range(len(idxs)):
    #     print(idxs_region[np.where(idxs_region == idxs[i])[0]])
    idxs = idxs[:batch_size]
    assert len(idxs) == batch_size, f"{len(idxs)} vs {batch_size}"
    return idxs

def sequential_query(data, batch_size,tess_size,region_order=None, seq=10,scores=[] ,method=[],region_scores=[]):
    idxs = []

    # print(data[:5])

    # region_order = data[np.sort(np.unique(data[:,3], return_index=True)[1]),3]

    # print(region_order[:10])

    # region_order = np.delete(region_order, np.where(region_order == -1), axis=0)
    if region_order is None:
        region_order = data[:,3][np.sort(np.unique(data[:,3], return_index=True)[1])]
        region_order = np.delete(region_order, np.where(region_order == -1), axis=0)

    if len(region_scores) > 0:
        remainder = batch_size % len(region_scores)
        samples = int(batch_size / len(region_scores))

    else:
        remainder = batch_size % len(region_order)
        samples = int(batch_size / len(region_order))


    # count = 0
    other_i = 0

    for i in range(len(region_order)):

        # print(f"{i}->{region_order[i]}")
        if len(region_scores) > 0:
            if i >= len(region_scores):
                continue
            matches = np.where(data[:,3] == region_order[region_scores[i]])[0]

        else:
            matches = np.where(data[:,3] == region_order[i])[0]
        if len(matches) == 0:
            continue

        if remainder == 0:
            # print(matches)
            # for i in range(len())
            # sequence = np.arange(0,len(matches),seq)
            # for i in range(sequence)
            if len(region_scores) > 0:
                results = data[np.where(data[:,3] == region_order[region_scores[i]])[0][np.arange(0,len(matches),seq)][:samples],0].tolist()
            
            else:
                results = data[np.where(data[:,3] == region_order[i])[0][np.arange(0,len(matches),seq)][:samples],0].tolist()

            if len(results) < samples:
                other_i -= 1
            else:
                other_i += 1
            idxs = idxs + results

        elif other_i < remainder:
            # print(matches)

            # print(f"{region_order[i]} has {len(idxs_region[np.where(idxs_region[:,1] == region_order[i])[0][:samples+1],0].tolist())} points")
            if len(region_scores) > 0:
                results = data[np.where(data[:,3] == region_order[region_scores[i]])[0][np.arange(0,len(matches),seq)][:samples+1],0].tolist()
            
            else:
                results = data[np.where(data[:,3] == region_order[i])[0][np.arange(0,len(matches),seq)][:samples+1],0].tolist()

            if len(results) < samples+1:
                other_i -= 1
            else:
                other_i += 1
            idxs = idxs + results
        else:
            # print(matches)
            # print(f"{region_order[i]} has {len(idxs_region[np.where(idxs_region[:,1] == region_order[i])[0][:samples],0].tolist())} points")
            if len(region_scores) > 0:
                results = data[np.where(data[:,3] == region_order[region_scores[i]])[0][np.arange(0,len(matches),seq)][:samples],0].tolist()

            else:

                results = data[np.where(data[:,3] == region_order[i])[0][np.arange(0,len(matches),seq)][:samples],0].tolist()
            if len(results) < samples:
                other_i -= 1
            else:
                other_i += 1
            idxs = idxs + results
    loop = 1
    while len(idxs) < batch_size:
        other_i = 0
        # print(len(idxs))
        for i in range(len(region_order)):

            if len(idxs) == batch_size:
                break
            if len(region_scores) > 0:
                if i >= len(region_scores):
                    continue
                matches = np.where(data[:,3] == region_order[region_scores[i]])[0]
    
            else:
                matches = np.where(data[:,3] == region_order[i])[0]
            # print(f"{i}->{region_order[i]}")

            if len(matches) == 0:
                continue

            if remainder == 0:
                # print(matches)
                # for i in range(len())
                # sequence = np.arange(0,len(matches),seq)
                # for i in range(sequence)
                if len(region_scores) > 0:
                    results = data[np.where(data[:,3] == region_order[region_scores[i]])[0][np.arange(0,len(matches),seq)][:samples+loop],0].tolist()
                
                else:
                    results = data[np.where(data[:,3] == region_order[i])[0][np.arange(0,len(matches),seq)][:samples+loop],0].tolist()

                if len(results) < samples:
                    other_i -= 1
                else:
                    other_i += 1
                
                for r in results:
                    if r not in idxs:
                        idxs.append(r)
                        break

                # idxs = idxs + results


            elif other_i < remainder:
                # print(matches)

                # print(f"{region_order[i]} has {len(idxs_region[np.where(idxs_region[:,1] == region_order[i])[0][:samples+1],0].tolist())} points")
                if len(region_scores) > 0:
                    results = data[np.where(data[:,3] == region_order[region_scores[i]])[0][np.arange(0,len(matches),seq)][:samples+1+loop],0].tolist()
                
                else:
                    results = data[np.where(data[:,3] == region_order[i])[0][np.arange(0,len(matches),seq)][:samples+1+loop],0].tolist()

                if len(results) < samples+1:
                    other_i -= 1
                else:
                    other_i += 1

                for r in results:
                    if r not in idxs:
                        idxs.append(r)
                        break
                # idxs = idxs + results
            else:
                # print(matches)
                # print(f"{region_order[i]} has {len(idxs_region[np.where(idxs_region[:,1] == region_order[i])[0][:samples],0].tolist())} points")
                if len(region_scores) > 0:
                    results = data[np.where(data[:,3] == region_order[region_scores[i]])[0][np.arange(0,len(matches),seq)][:samples+loop],0].tolist()
                
                else:
                    results = data[np.where(data[:,3] == region_order[i])[0][np.arange(0,len(matches),seq)][:samples+loop],0].tolist()

                if len(results) < samples:
                    other_i -= 1
                else:
                    other_i += 1
                
                for r in results:
                    if r not in idxs:
                        idxs.append(r)
                        break
        loop+=1
    # print("total: ", len(idxs))
    # print(idxs)
    # assert False
    idxs = idxs[:batch_size]
    assert len(idxs) == batch_size, f"{len(idxs)} vs {batch_size}"
    return idxs

# def split(a, n):
#     k, m = divmod(len(a), n)
#     ans = (a[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n))
#     print(a)
#     print(n)
#     print(list(ans))

#     assert False

#     return ans

def cut_range_query(data, batch_size,tess_size,region_order=None,scores=[] ,method=[], region_scores=[]):
    idxs = []
    if region_order is None:
        region_order = data[:,3][np.sort(np.unique(data[:,3], return_index=True)[1])]
        region_order = np.delete(region_order, np.where(region_order == -1), axis=0)

    if len(region_scores) > 0:
        remainder = batch_size % len(region_scores)
        samples = int(batch_size / len(region_scores))

    else:
        remainder = batch_size % len(region_order)
        samples = int(batch_size / len(region_order))

    # count = 0

    for i in range(len(region_order)):

        if i >= len(region_order):
            continue
        if len(region_scores) > 0:
            if i >= len(region_scores):
                continue
        r_idx = []
        if remainder == 0:
            if len(region_scores) > 0:
                matches = np.where(data[:,3] == region_order[region_scores[i]])[0]

            else:
                matches = np.where(data[:,3] == region_order[i])[0]

            s = np.linspace(0, len(matches)-1, samples)

            # print(s)

            # assert False

            # s = list(split(np.arange(0, len(matches)), samples))
            # print(s)
            # r_idx = []

            # print(idxs_region[np.where(idxs_region[:,1] == region_order[i])[0][r_idx]].tolist())
        # assert False
        elif (i < remainder):
            # print(f"{region_order[i]} has {len(idxs_region[np.where(idxs_region[:,1] == region_order[i])[0][:samples+1],0].tolist())} points")
            if len(region_scores) > 0:
                matches = np.where(data[:,3] == region_order[region_scores[i]])[0]

            else:
                matches = np.where(data[:,3] == region_order[i])[0]

            s = np.linspace(0, len(matches)-1, samples+1)
            # print(idxs_region[np.where(idxs_region[:,1] == region_order[i])[0][r_idx]].tolist())
        else:
            # print(f"{region_order[i]} has {len(idxs_region[np.where(idxs_region[:,1] == region_order[i])[0][:samples],0].tolist())} points")
            if len(region_scores) > 0:
                matches = np.where(data[:,3] == region_order[region_scores[i]])[0]

            else:
                matches = np.where(data[:,3] == region_order[i])[0]

            s = np.linspace(0, len(matches)-1, samples)
            # print(idxs_region[np.where(idxs_region[:,1] == region_order[i])[0][r_idx]].tolist())
        
        # print(matches)
        # print(s)

        for i in range(len(s)):
            if i == 0:
                r_idx.append(matches[0])
            elif i == len(s)-1:
                r_idx.append(matches[int(s[-1])])
            else:
                r_idx.append(matches[int(np.mean(int(s[i])))])
        # print(r_idx)
        idxs = idxs + data[r_idx,0].tolist()
            

    # print(idxs)
    # assert False
    assert len(idxs) == batch_size, f"{len(idxs)} vs {batch_size}"
    return idxs


tess_methods = {
    "Average":average,
    "Top-N":top_N_query,
    "Cycle-N":cycle_N_query,
    "Sequential":sequential_query,
    "Cut-Range":cut_range_query,
}



class QUEST:
    def __init__(self, X, Y, idxs_lb, qs_dict={}, embedding="UMAP", retrain_embed=False):
        self.X = X
        self.Y = Y
        self.idxs_lb = idxs_lb
        self.embed_type = embedding

        self.qs_dict = qs_dict

        if embedding == "test":
            with open('test_embed.npy', 'rb') as f:
                # trainset.data = np.load(f)
                self.X = np.load(f)
                self.Y = np.random.randint(0,10,10)

        self.create_embedding(retrain_embed)
        
        # self.initialise_metrics()
        self.combined = np.arange(0,len(self.transformed_train))[:,np.newaxis]

        print(self.combined.shape)
        print(self.combined[~self.idxs_lb].shape)

        # assert False
        # self.combined = None

        print("initialised")


    def update(self, idxs_lb):
        self.idxs_lb = idxs_lb


    def create_embedding(self, retrain_embed):
        
        ds = self.qs_dict["Dataset"]

        if self.embed_type == "UMAP":
            if (retrain_embed) or not (os.path.isfile(f'{ds}_embed.npy')):
                import umap
                # print(self.X.shape)
                # print(self.Y.shape)

                if len(self.X.shape) == 4:
                    self.data_reshape = self.X.reshape(self.X.shape[0],self.X.shape[1]*self.X.shape[2]*self.X.shape[3])
                # elif len(self.X.shape) == 1:
                #     self.data_reshape = self.X.reshape(self.X.shape[0], self.X.)
                else:
                    self.data_reshape = self.X.reshape(self.X.shape[0],self.X.shape[1]*self.X.shape[2])
                self.embedder = umap.UMAP(n_components = 2, n_neighbors = 20, verbose = True).fit(self.data_reshape)
                # self.embedder = None
                self.transformed_train = self.embedder.transform(self.data_reshape)
                self.transformed_points = GeoSeries(map(geometry.Point, self.transformed_train))
                with open(f'{self.qs_dict["Dataset"]}_embed.npy', 'wb') as f:
                    np.save(f, self.transformed_train)
            else:
                # print(len(self.X))
                with open(f'{ds}_embed.npy', 'rb') as f:
                    self.transformed_train = np.load(f)
                # self.transformed_train = self.embedder.transform(self.data_reshape)
                # print(self.transformed_train.shape)
                # assert False
                self.transformed_points = GeoSeries(map(geometry.Point, self.transformed_train))

        elif self.embed_type == "simCLR":
            
            assert os.path.isfile(f'{ds}_simclr_embedding.npy'), f'{self.qs_dict["Dataset"]} simCLR Embedding doesnt exist'
            with open(f'{ds}_simclr_embedding.npy', 'rb') as f:
                self.transformed_train = np.load(f)
            print(self.transformed_train.shape)
            
            self.transformed_points = GeoSeries(map(geometry.Point, self.transformed_train))

        elif self.embed_type == "RandomTrue":
            
            if not os.path.isfile(f'{ds}_randomtrue_{self.qs_dict["Tessellations"]}_embedding.npy'):
                import random
                import math

                tess_idx = np.random.randint(0,self.qs_dict["Tessellations"],len(self.X))

                grid = 100

                x = tess_idx % grid
                y = tess_idx // grid

                gap = 600
                offset = gap/3.0

                x_o = np.random.uniform(-offset,offset, len(self.X))
                y_o = np.random.uniform(-offset,offset, len(self.X))

                kmc = np.array([(offset + gap*x),(offset + gap*y)]).T

                coords = np.array([(offset + gap*x)+x_o,(offset + gap*y)+y_o]).T
                self.transformed_train = coords
                with open(f'{ds}_randomtrue_{self.qs_dict["Tessellations"]}_embedding.npy', 'wb') as f:
                    np.save(f, self.transformed_train)
                with open(f'{ds}_randomtrue_{self.qs_dict["Tessellations"]}_kmc.npy', 'wb') as f:
                    np.save(f, kmc)
                self.transformed_points = GeoSeries(map(geometry.Point, self.transformed_train))
            else:
                with open(f'{ds}_randomtrue_{self.qs_dict["Tessellations"]}_embedding.npy', 'rb') as f:
                    self.transformed_train = np.load(f)
                print(self.transformed_train.shape)
                
                self.transformed_points = GeoSeries(map(geometry.Point, self.transformed_train))
        elif self.embed_type == "Random":
            
            if not os.path.isfile(f'{ds}_random_embedding.npy'):
                import random
                import math

                # radius of the circle
                circle_r = 1000
                # center of the circle (x, y)
                circle_x = 0
                circle_y = 0
                self.transformed_train = np.zeros((len(self.X),2))
                for i in range(len(self.X)):


                    # random angle
                    alpha = 2 * math.pi * random.random()
                    # random radius
                    r = circle_r * math.sqrt(random.random())
                    # calculating coordinates
                    self.transformed_train[i,0] = r * math.cos(alpha) + circle_x
                    self.transformed_train[i,1] = r * math.sin(alpha) + circle_y

                with open(f'{self.qs_dict["Dataset"]}_random_embedding.npy', 'wb') as f:
                    np.save(f, self.transformed_train)
                self.transformed_points = GeoSeries(map(geometry.Point, self.transformed_train))

            else:
                print("\n\n\n Choosing Else \n\n\n")
                with open(f'{ds}_random_embedding.npy', 'rb') as f:
                    self.transformed_train = np.load(f)
                print(self.transformed_train.shape)
                
                self.transformed_points = GeoSeries(map(geometry.Point, self.transformed_train))

        elif self.embed_type == "test":
            assert os.path.isfile("test_embed.npy"), "Test Embedding doesnt exist"
            with open('test_embed.npy', 'rb') as f:
                self.transformed_train = np.load(f)
            self.transformed_points = GeoSeries(map(geometry.Point, self.transformed_train))        
        else:
            assert False, "Embedding doesn't exist"
        print("")
        print("")

        print(self.transformed_train.shape)
        print(self.transformed_points.shape)

        print("")
        print("")

    def query_by_segments(self, strategy,batch_size, tessellations,seed,num_start=1000, subset=None, uncertainty=None):
        
        self.combined = self.combined[self.combined[:,0].argsort()]

        self.combine_data(strategy, update=True, uncertainty=uncertainty)

        pts = MultiPoint([Point(i) for i in self.clusters])
        mask = pts.convex_hull
        # new_vertices = []

        # print(self.combined[:20,-1:1])
        inters = self.combined[self.combined[:,0].argsort()].copy()

        if subset is not None:
            inters = inters[subset]
            inters = strategy.sort_informativeness(inters)
            # print("not used: ", not_used[:250,0])
            self.combined = self.combined[self.combined[:,0].argsort()]
            not_used = self.combined[inters[:,0].astype(int)]
            print("\n")
            # print("final uncert ", not_used[:20,4])
            # print("unique regions:")
            print(len(np.unique(not_used[:,3])))
        
        else:
            # print("subset was None")
            # inters = self.combined[self.combined[:,0].argsort()]

            # inters = self.combine.copy()
            inters = inters[~self.idxs_lb]
            # print("not sorted comb: ", inters[:10,4])
            # print(inters.shape)
            # print("\n\n")

            not_used = strategy.sort_informativeness(inters)
            # print("sorted combined ",not_used[:10,4])
            # print(~self.idxs_l).nonzero()[:20])

            # not_used = self.combined[inters]
            # not_used = self.combined[~self.idxs_lb]

        # print("sorting informativeness...")
        
        # print(not_used[:20])

        # print(not_used[:10])



        # strategy.net.eval()

        # if self.qs_dict["QS"] == "LL":
        #     preds, outputs = self.get_pool_outputs(strategy.net, poolloader)
        # else:
        #     preds = self.get_pool_outputs(strategy.net, poolloader)
        #     outputs = preds


        outputs = None

        # print("saving data...")
        # self.sample_entropy()
        # self.save_data(tessellations, batch_size, seed, num_start, outputs, poolloader=poolloader)  


        print("get idxs")

        # print(self.combined[:5])
        # self.explore_exploit_metrics(outputs)
        idxs = self.get_idxs(not_used, self.qs_dict, batch_size, tessellations, [self.clusters, self.regions, self.vertices, self.transformed_points])
        print("after: ", idxs[:5])

        # print(type(idxs))
        idxs = np.array(idxs).astype(int)
        assert max(idxs) <= max(not_used[:,0]), f"Index {max(idxs)} with only {max(not_used[:,0])} available"

        assert set(list(idxs)).isdisjoint(set(list(self.idxs_lb.nonzero()[0]))),f"Not disjoint: {idxs[:20]} vs {self.idxs.nonzero()[0][:20]}"

        return idxs
    

    def create_tessellation(self, N, test=False):

        print(test)
        print("create tess: ", N)
        print("create tess: ", type(N))
        self.tessellations = N
        self.clusters = self._create_centre_points(int(N),test=test)

        # print(self.clusters)

        self.vor = Voronoi(self.clusters)
        self.regions, self.vertices = self._voronoi_finite_polygons_2d(self.vor)


    def _create_centre_points(self,N, test=False):

        use_kmeans = False
        if not use_kmeans:
            if not test:
                km = KMeans(N, n_init="auto",tol=1e7,max_iter=10000).fit(self.transformed_train)
            else:
                # print("running else")
                km = []
                for i in np.arange(0,1.1,(1.0/N)):
                    # print(i)
                    if len(np.where(self.transformed_train[:,0] == i)[0])>0:
                        k = np.mean(self.transformed_train[np.where(self.transformed_train[:,0] == i)[0],1])
                        print(k)
                    else:
                        k = 0
                    km.append([i,0.5])
        else:
            km = KMeans(N, n_init=10,tol=1e7,max_iter=10000).fit(self.transformed_train)

        print(km)

        if self.embed_type == "RandomTrue":
            minx = self.transformed_train[:,0].min()-200
            maxx = self.transformed_train[:,0].max()+200

            maxy = self.transformed_train[:,1].max()+200
            miny = self.transformed_train[:,1].min()-200
        else:

            minx = self.transformed_train[:,0].min()-0.1*self.transformed_train[:,0].max()
            maxx = self.transformed_train[:,0].max()+0.1*self.transformed_train[:,0].max()

            miny = self.transformed_train[:,1].min()-0.1*self.transformed_train[:,1].max()
            maxy = self.transformed_train[:,1].max()+0.1*self.transformed_train[:,1].max()
        if not use_kmeans:
            if not test:
                clusters = km.cluster_centers_
            else:
                clusters = np.array(km)
        else:
            clusters = km.cluster_centers_

        if self.embed_type == "RandomTrue":
            with open(f'{self.qs_dict["Dataset"]}_randomtrue_{self.qs_dict["Tessellations"]}_kmc.npy', 'rb') as f:
                kmc = np.load(f)
        
            clusters = kmc

            print("updated KMC")


        print(clusters.shape)
        if self.embed_type == "RandomTrue":
            # 200+(100*600)

            for i in range(100):
                clusters = np.append(clusters,[[200+ (600*i),-400]], axis=0)
                clusters = np.append(clusters,[[200+ (600*i),200+(10*600)]], axis=0)

            for i in range(10):
                clusters = np.append(clusters,[[-400,200+ (600*i)]], axis=0)
                clusters = np.append(clusters,[[200+ (600*100),200+ (600*i)]], axis=0)

        else:
            clusters = np.append(clusters,[[minx,miny]], axis=0)
            clusters = np.append(clusters,[[minx,maxy]], axis=0)
            clusters = np.append(clusters,[[maxx,miny]], axis=0)
            clusters = np.append(clusters,[[maxx,maxy]], axis=0)

        return clusters


    def _voronoi_finite_polygons_2d(self, vor, radius=None):
        """
        Reconstruct infinite voronoi regions in a 2D diagram to finite
        regions.
        Parameters
        ----------
        vor : Voronoi
            Input diagram
        radius : float, optional
            Distance to 'points at infinity'.
        Returns
        -------
        regions : list of tuples
            Indices of vertices in each revised Voronoi regions.
        vertices : list of tuples
            Coordinates for revised Voronoi vertices. Same as coordinates
            of input vertices, with 'points at infinity' appended to the
            end.
        """

        if vor.points.shape[1] != 2:
            raise ValueError("Requires 2D input")

        new_regions = []
        new_vertices = vor.vertices.tolist()

        center = vor.points.mean(axis=0)
        if radius is None:
            radius = 10000
        # Construct a map containing all ridges for a given point
        all_ridges = {}
        for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
            all_ridges.setdefault(p1, []).append((p2, v1, v2))
            all_ridges.setdefault(p2, []).append((p1, v1, v2))

        # Reconstruct infinite regions
        for p1, region in enumerate(vor.point_region):
            vertices = vor.regions[region]

            if all(v >= 0 for v in vertices):
                # finite region
                new_regions.append(vertices)
                continue

            # reconstruct a non-finite region
            try:
                ridges = all_ridges[p1]
                new_region = [v for v in vertices if v >= 0]

                for p2, v1, v2 in ridges:
                    if v2 < 0:
                        v1, v2 = v2, v1
                    if v1 >= 0:
                        # finite ridge: already in the region
                        continue

                    # Compute the missing endpoint of an infinite ridge

                    t = vor.points[p2] - vor.points[p1] # tangent
                    t /= np.linalg.norm(t)
                    n = np.array([-t[1], t[0]])  # normal

                    midpoint = vor.points[[p1, p2]].mean(axis=0)
                    direction = np.sign(np.dot(midpoint - center, n)) * n
                    far_point = vor.vertices[v2] + direction * radius

                    new_region.append(len(new_vertices))
                    new_vertices.append(far_point.tolist())
            except:
                print("failed ",p1)

            # sort region counterclockwise
            vs = np.asarray([new_vertices[v] for v in new_region])
            c = vs.mean(axis=0)
            angles = np.arctan2(vs[:,1] - c[1], vs[:,0] - c[0])
            new_region = np.array(new_region)[np.argsort(angles)]

            # finish
            new_regions.append(new_region.tolist())

        return new_regions, np.asarray(new_vertices)

    def combine_data(self,strategy, device="cuda", update=False, uncertainty=None):
        print("combine data...")

        if uncertainty is not None:
            assert len(uncertainty) == len(self.X), f"{len(uncertainty)} vs {len(self.X)}"

        # softmax_o = self.get_pool_outputs(strategy.net, poolloader)

        # print(softmax_o)
        if not update:
            print("not update")

            # print("Not update: ", strategy)
        
            data = self.transformed_train.copy()
            
            region_idxs = self.get_region_idxs()
            print("got region idxs")

            if uncertainty is not None:
                outputs = uncertainty
            else:
                outputs = strategy.get_informativeness()

            print("got informativeness")

            print(data.shape)
            print(region_idxs.shape)
            print(outputs.shape)


            print("stacking...")


            data = np.hstack((data, region_idxs[:,np.newaxis]))
            data = np.hstack((data, outputs[:,np.newaxis]))
            print(f"len transformed train: {len(self.transformed_train)}")
            # data = np.hstack((data,np.arange(len(self.transformed_train))[:,np.newaxis]))
            data = np.hstack((self.combined, data))


            print("data shape:", data.shape)

            self.combined = data

            print(self.combined[:10])
            # assert False
        else:
            print("combined else")
            self.combined = self.combined[self.combined[:,0].argsort()]

            print(f"len transformed train: {len(self.transformed_train)}")
            print(len(uncertainty))
            # print("update: ", strategy)
            if uncertainty is not None:
                print("uncertainty has val")
                outputs = uncertainty
            else:
                outputs = strategy.get_informativeness()
            
            # print("uncert: ", uncertainty[idxs_unlabeled][:20])


            # outputs = strategy.get_informativeness()
            # print(np.array(outputs).shape)
            # print(self.combined.shape)

            self.combined[:,4] = outputs
            # self.combined[:,0] = np.arange(len(self.pool.data))

            # print(self.combined[:10])
            # assert False

    def get_region_idxs(self):
        region_idxs = None
        region_idxs = np.zeros(len(self.transformed_points))-1

        pts = MultiPoint([Point(i) for i in self.clusters])
        mask = pts.convex_hull
        # print(idxs_region[:5])

        for reg_idx, region in enumerate(self.regions):


            polygon = self.vertices[region]
            shape = list(polygon.shape)
            shape[0] += 1
            
            p = Polygon(np.append(polygon, polygon[0]).reshape(*shape)).intersection(mask)

            a = np.where(self.transformed_points.within(p))[0]

            region_idxs[a] = reg_idx

            # b = np.nonzero(a[:,None] == idxs_region[:,0])[1]        
            # print(b)

        # assert len(idxs_region[np.where(idxs_region == -1)[0]]) == 0, f"{len(idxs_region[np.where(idxs_region == -1)[0]])} at {np.where(idxs_region == -1)[0]}"

        return region_idxs



    def get_idxs(self, combined, qs_dict, batch_size, tess_size, tess_data = None):

        # qs = base_qs[qs_dict["QS"]]
        data = combined.copy()

        print("get_idxs: ", data.shape)
        # print(data)
        # data = data[data[:, 4].argsort()]
        # print(data)

        print("batch size: ", batch_size)

        if qs_dict["method"] == "Average":
            region_order = data[:,3][np.sort(np.unique(data[:,3], return_index=True)[1])]


            # print(region_order)
            region_order = np.delete(region_order, np.where(region_order == -1), axis=0)
            if batch_size <= tess_size:
                new_idxs = []
                # print(len(region_order))

                # assert False
                # if qs_dict["sub"] == "Average":
                print("average smaller")

                avgs = []

                for i in range(len(region_order)):
                    rows = np.where(data[:,3] == region_order[i])[0]
                    avg = np.mean(data[rows,4])
                    avgs.append(avg)

                # print(region_order)
                # print(avgs)

                avgs_sort = np.argsort(avgs)
                # print(avgs_sort)
                
            
                if qs["opt"] == "max":
                    avgs_sort = avgs_sort[::-1]

                while len(new_idxs) < batch_size:

                    for i in range(batch_size):
                        try:
                            print(f"choosing from region {region_order[avgs_sort[i]]} which has avg {avgs[list(region_order).index(region_order[avgs_sort[i]])]}")
                            idx = data[np.where(data[:,3] == region_order[avgs_sort[i]])[0][0],0]
                            new_idxs.append(idx)
                        except:
                            continue

                new_idxs = new_idxs[:batch_size]
                
                assert len(new_idxs) == batch_size
            else:
                region_order = data[:,3][np.sort(np.unique(data[:,3], return_index=True)[1])]
                # print(len(region_order))
                region_order = np.delete(region_order, np.where(region_order == -1), axis=0)
                # print(region_order)
                # print(len(region_order))
                # assert False
                avgs = []

                for i in range(len(region_order)):
                    rows = np.where(data[:,3] == region_order[i])[0]
                    # print(rows)
                    avg = np.mean(data[rows,4])
                    avgs.append(avg)

                # print(avgs)
                

                region_order = region_order[np.argsort(avgs)]
                # print(region_order)
                # assert False
                # print(region_order)
                if qs["opt"] == "max":
                    region_order = region_order[::-1]
                new_idxs = tess_methods[qs_dict["sub"]](data, batch_size,tess_size, method=qs["opt"], region_order=region_order)
            # idxs_region = np.copy(idxs)
            # # idxs_region = idxs_region[:, np.newaxis]
            # idxs_region = np.vstack((idxs_region,np.zeros(len(idxs_region))-1)).T
            # print(idxs_region.shape)
            # print(idxs_region)

            # idxs_region = get_region_idxs(idxs_region,tess_data[0],tess_data[1],tess_data[2],tess_data[3])
        else:
            region_order = data[:,3][np.sort(np.unique(data[:,3], return_index=True)[1])]
            region_order = np.delete(region_order, np.where(region_order == -1), axis=0)
            if batch_size <= len(region_order):
                new_idxs = []
                # print(region_order)
                # print(len(region_order))

                for i in range(batch_size):
                    idx = data[np.where(data[:,3] == region_order[i])[0][0],0]
                    new_idxs.append(idx)
                # print(new_idxs)
                
                assert len(new_idxs) == batch_size
            else:
                print(batch_size)
                print(len(region_order))

                new_idxs = tess_methods[qs_dict["sub"]](data, batch_size,tess_size)

        # print(new_idxs)
        # print([int(x) for x in new_idxs])
        # print(data[[int(x) for x in new_idxs],0])
        # assert False
        curr_idxs = []
        for i in range(len(data)):
            if data[i,0] in new_idxs:
                curr_idxs.append(data[i,4])
        print(f"pool data > max: {np.max(data[:,4])}, min: {np.min(data[:,4])}")
        print(f"chosen > max: {np.max(curr_idxs)}, min: {np.min(curr_idxs)}")

        return new_idxs