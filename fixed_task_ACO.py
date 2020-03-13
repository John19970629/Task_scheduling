# -*- coding: utf-8 -*-
"""
Created on Sat Jan  4 14:19:25 2020

@author: USER
"""

import pandas as pd
import numpy as np
from numpy.random import choice as np_choice

job = ['b1','b2','b3','b4','b5','b6','b7','b8','b9','b10','b11','b12','b13','b14','b15','b16','b17','b18','b19','b20']
#       'b21','b22','b23','b24','b25','b26','b27','b28','b29','b30','b31','b32','b33','b34','b35','b36','b37','b38','b39','b40',
#       'b41','b42','b43','b44','b45','b46','b47','b48','b49','b50','b51','b52','b53','b54','b55','b56','b57','b58','b59','b60',
#       'b61','b62','b63','b64','b65','b66','b67','b68','b69','b70','b71','b72','b73','b74','b75','b76','b77','b78','b79','b80',
#       'b81','b82','b83','b84','b85','b86','b87','b88','b89','b90','b91','b92','b93','b94','b95','b96','b97','b98','b99','b100']

task = ['a1','a2','a3','a4','a5','a6','a7','a8','a9','a10','a11','a12','a13','a14','a15','a16','a17','a18','a19','a20']
#       'a21','a22','a23','a24','a25','a26','a27','a28','a29','a30','a31','a32','a33','a34','a35','a36','a37','a38','a39','a40',
#       'a41','a42','a43','a44','a45','a46','a47','a48','a49','a50','a51','a52','a53','a54','a55','a56','a57','a58','a59','a60',
#       'a61','a62','a63','a64','a65','a66','a67','a68','a69','a70','a71','a72','a73','a74','a75','a76','a77','a78','a79','a80',
#       'a81','a82','a83','a84','a85','a86','a87','a88','a89','a90','a91','a92','a93','a94','a95','a96','a97','a98','a99','a100']

f=open('Array.txt')
arr=f.read()
arr=arr.split('\n')
del arr[100]
for i in range(100):
    arr[i]=arr[i].split(',')
del arr[20:100]
for i in range(20):
    del arr[i][20:100]
    
arr1=[[row[col] for row in arr] for col in range(len(arr[0]))]  #讓job變成列
df = pd.DataFrame(arr1,index=job,columns=task)

#讀入時間
with open('data.txt','r') as dat:
     data = dat.readlines()

task_time = []
job_time = []
for i in range(1,21):
    task_time.append(float(data[i][:-1]))
task_df = pd.DataFrame(task_time,index=task)

for i in range(102,122):
    job_time.append(float(data[i][:-1]))
job_df = pd.DataFrame(job_time,index=job) 

class AntColony(object):
    def __init__(self, distances, n_ants, n_best, n_iterations, decay, alpha=1, beta=1):
        self.distances  = distances
        self.pheromone = np.ones(self.distances.shape) / len(distances)
        self.all_inds = range(len(distances))
        self.n_ants = n_ants
        self.n_best = n_best
        self.n_iterations = n_iterations
        self.decay = decay
        self.alpha = alpha
        self.beta = beta

    def run(self):
        shortest_path = None
        all_time_shortest_path = ("placeholder", np.inf)
        for i in range(self.n_iterations):
            all_paths = self.gen_all_paths()
            self.spread_pheronome(all_paths, self.n_best, shortest_path=shortest_path)
            shortest_path = min(all_paths, key=lambda x: x[1])
            print (shortest_path)
            if shortest_path[1] < all_time_shortest_path[1]:
                all_time_shortest_path = shortest_path            
            self.pheromone * self.decay            
        return all_time_shortest_path

    def spread_pheronome(self, all_paths, n_best, shortest_path):
        sorted_paths = sorted(all_paths, key=lambda x: x[1])
        for path, dist in sorted_paths[:n_best]:
            for move in path:
                self.pheromone[move] += 1.0 / self.distances[move]

    def gen_path_dist(self, path):
        total_sum_dist = 0
        for ele in path:
            total_sum_dist = 2*total_sum_dist+self.distances[ele]
        return total_sum_dist

    def gen_all_paths(self):
        all_paths = []
        for i in range(self.n_ants):
            path = self.gen_path(0)
            all_paths.append((path, self.gen_path_dist(path)))
        return all_paths

    def gen_path(self, start):
        path = []
        visited = set()
        visited.add(start)
        prev = start
        for i in range(len(self.distances) - 1):
            move = self.pick_move(self.pheromone[prev], self.distances[prev], visited)
            path.append((prev, move))
            prev = move
            visited.add(move)
        #path.append((prev, start)) # going back to where we started    
        return path

    def pick_move(self, pheromone, dist, visited):
        pheromone = np.copy(pheromone)
        pheromone[list(visited)] = 0

        row = pheromone ** self.alpha * (( 1.0 / dist) ** self.beta)
        norm_row = row / row.sum()
        move = np_choice(self.all_inds, 1, p=norm_row)[0]
        return move
    
distances = np.array([[np.inf, task_time[0], np.inf, np.inf, task_time[0],np.inf,np.inf,np.inf],
                      [np.inf, np.inf, task_time[1], task_time[1], np.inf,np.inf,np.inf,np.inf],
                      [np.inf, np.inf, np.inf, np.inf, np.inf,task_time[2],task_time[2],task_time[2]],
                      [np.inf, np.inf, job_time[0], np.inf, job_time[0] ,np.inf,np.inf,np.inf],
                      [np.inf, job_time[1], np.inf, job_time[1], np.inf,np.inf,np.inf,np.inf],
                      [np.inf,np.inf,np.inf,job_time[2],job_time[2],np.inf,job_time[2],job_time[2]],
                      [np.inf,np.inf,np.inf,job_time[3],job_time[3],job_time[3],np.inf,job_time[3]],
                      [np.inf,np.inf,np.inf,job_time[4],job_time[4],job_time[4],job_time[4],np.inf]])
    
distances[distances==np.inf]=10000

ant_colony = AntColony(distances, 1, 1, 1000, 0.95, alpha=1, beta=1)
shortest_path = ant_colony.run()
print ("shorted_path: {}".format(shortest_path))