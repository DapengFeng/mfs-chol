#!/usr/bin/python

from __future__ import print_function
import sys, os
import matplotlib.pyplot as plt
from scipy import sparse
from scipy.sparse import csc_matrix
import numpy as np

mat_file = sys.argv[1]
mat_name = mat_file[:-4]
mat_imag = mat_name+'.png'
print(mat_imag)

f = open(mat_file, "r")
rows = np.fromfile(f, count=1, dtype=np.int64)
cols = np.fromfile(f, count=1, dtype=np.int64)
nnz  = np.fromfile(f, count=1, dtype=np.int64)
indptr = np.fromfile(f, count=cols[0]+1, dtype=np.int32)
indices = np.fromfile(f, count=nnz[0], dtype=np.int32)
data = np.fromfile(f, count=nnz[0], dtype=np.dtype('d'))

A = csc_matrix((data, indices, indptr), shape=(rows[0], cols[0]))
print(rows, cols)

plt.spy(A, markersize=0.01)
plt.savefig(mat_imag)

# print(sys.argv)

# if len(sys.argv) == 2:
#     mat_file = sys.argv[1]

#     f = open(mat_file, "r")

#     rows = np.fromfile(f, count=1, dtype=np.int64)
#     cols = np.fromfile(f, count=1, dtype=np.int64)
#     data = np.fromfile(f, count=rows[0]*cols[0], dtype=np.float64)
#     data = np.reshape(data, (rows[0], cols[0]), 'F')

#     total_num = np.ceil(1.0*rows/10.0).astype(int)
#     print("--- matrix size and plot num: %d, %d, %d" % (rows[0], cols[0], total_num))

#     fig = plt.figure(figsize=(10, 5*total_num))

#     ptr = 0
#     cnt = 0
#     while ptr < rows:
#         print("--- plot %d segment"%cnt)
#         ax = fig.add_subplot(total_num, 1, cnt+1)        

#         end = min(ptr+10, rows)
#         for i in range(ptr, end):
#           ax.plot(data[i, :], label=i)

#         ax.legend()

#         ptr += 10
#         cnt += 1

#     plt.savefig(mat_file+".plot.pdf")

# elif len(sys.argv) == 4:
#     for i in range(1, len(sys.argv)-1):
#         print(i)
        
#         f = open(sys.argv[i], "r")

#         rows = np.fromfile(f, count=1, dtype=np.int64)
#         cols = np.fromfile(f, count=1, dtype=np.int64)
#         data = np.fromfile(f, count=rows[0]*cols[0], dtype=np.float64)

#         if i == 1:
#             plt.plot(data, label="opt D", linewidth=3)
#         if i == 2:
#             plt.plot(data, label="not opt D", linewidth=3)

#     plt.legend()
#     plt.yscale('log')
#     plt.savefig(sys.argv[-1])

# elif len(sys.argv) == 6:
#     for i in range(1, len(sys.argv)-1):
#         print(i)

#         if not os.path.isfile(sys.argv[i]):
#             continue
        
#         f = open(sys.argv[i], "r")

#         rows = np.fromfile(f, count=1, dtype=np.int64)
#         cols = np.fromfile(f, count=1, dtype=np.int64)
#         data = np.fromfile(f, count=rows[0]*cols[0], dtype=np.float64)

#         if i == 1:
#             plt.plot(data, label="D1S1", linewidth=3)
#         if i == 2:
#             plt.plot(data, label="D1S0", linewidth=3)
#         if i == 3:
#             plt.plot(data, label="D0S0", linewidth=3)
#         if i == 4:
#             plt.plot(data, label="D0S1", linewidth=3)


#     plt.legend()
#     plt.yscale('log')
#     plt.savefig(sys.argv[-1])
