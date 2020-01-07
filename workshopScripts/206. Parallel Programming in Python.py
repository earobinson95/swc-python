import numpy as np
import math
import time as t
import multiprocessing as mp
from functools import partial

def generate_grid(size):
    grid = np.random.randint(1,20,size=(size,size))
    def inf(x):
        return math.inf if x >= 5 else float(x)
    inf = np.vectorize(inf)
    return inf(grid)

# def floydWarshall(g):
#     n = g.shape[0]
#     for k in range(n): 
#         # pick all vertices as source one by one 
#         for i in range(n): 
#             # Pick all vertices as destination for the 
#             # above picked source 
#             for j in range(n): 
#                 # If vertex k is on the shortest path from  
#                 # i to j, then update the value of g[i][j] 
#                 g[i][j] = min(g[i][j],g[i][k]+ g[k][j])
#     return g

# print(generate_grid(8))

# # Start with a small grid so it's easy to view the results
# grid_size = 8
# graph = generate_grid(grid_size)
# print('Graph:\n',graph)
# shortest_paths = floydWarshall(graph)
# print('\nShortest Paths:\n',shortest_paths)

# import time as t

# grid_size = 256
# graph = generate_grid(grid_size)

# t1 = t.time()
# shortest_paths = floydWarshall(graph)
# t2 = t.time()
# print('\nShortest Pathes\n', shortest_paths)
# print('serial: ',t2 - t1, 's')

# OKAY LET's PARALELLIZE IT----------------

def floydWarshall_p2(i, g, n, k):
    # Pick all vertices as destination for the 
    # above picked source 
    for j in range(n): 
        # If vertex k is on the shortest path from  
        # i to j, then update the value of dist[i][j] 
        g[i][j] = min(g[i][j],g[i][k]+ g[k][j])
    return (i,g[i])

def floydWarshall_p1(g):
    n = g.shape[0]
    pool = mp.Pool(processes=mp.cpu_count())
    for k in range(n):
        p = partial(floydWarshall_p2, g=g,n=n,k=k)
        result_list = pool.map( p,range(n))
        for result in result_list:
            g[result[0]] = result[1]
    pool.close()
    pool.join()
    return g

if __name__ == '__main__':
    grid_size = 256
    graph = generate_grid(grid_size)
    print('Graph:\n', graph)

    t1 = t.time()
    shortest_paths = floydWarshall_p1(graph)
    t2 = t.time()
    print('\nShortest Paths:\n', shortest_paths)
    print('parallel: ',t2 - t1, 's')

    t3 = t.time()
    shortest_paths_parallel = floydWarshall_p1(graph)
    t4 = t.time()
    print('\nShortest Paths:\n', shortest_paths)
    print('parallel: ',t4 - t3, 's')

    correct = np.array_equal(shortest_paths, shortest_paths_parallel)
    print("Did we get the right answer?", correct)