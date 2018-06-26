import numpy as np
from scipy.spatial.distance import cdist
from time import time
import mrpt
from mrpt import MRPTIndex
import sys
import inspect
from threading import Thread
from collections import Counter

data = np.fromfile("../../train.bin.part001", dtype=np.float32)
data = data.reshape((128, 1116550), order="FORTRAN").T
index = mrpt.MRPTIndex(data, depth=5, n_trees=100, projection_sparsity=0.088)
index.build()
queries = np.fromfile("../../test.bin", dtype=np.float32)
queries = queries.reshape((128, 1000), order="FORTRAN").T


leaves = index.get_leaves(queries[0])

def time_counted(leaves):
	from time import time
	t = time()
	voted_dict = Counter(leaves)
	sorted_leaves = [k for k,v in voted_dict.items() if v >= 2]
	print (str(time()-t))
	print (len(sorted_leaves))
	return sorted_leaves

def time_filtered(index,leaves):
	from time import time
	t1 = time()
	filtered = index.filter_leaves_by_votes(leaves,2).tolist()
	print (str(time()-t1))
	print (len(filtered))
	return filtered

f = time_filtered(index,leaves)