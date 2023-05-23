import numpy as np
import copy
import networkx as nx
from heapq import heappop, heappush, heapify
from scipy.stats import chi2, ncx2
from itertools import combinations
from lib.HetoFedBandit_Data_Recluster import HetoFedBandit_Data_Recluster

class HetoFedBandit_Enhanced(HetoFedBandit_Data_Recluster):
    def __init__(self, dimension, alpha, lambda_, delta_, NoiseScale, threshold, exploration_length, neighbor_identification_alpha,T):
        super().__init__(dimension, alpha, lambda_, delta_, NoiseScale, threshold, exploration_length, neighbor_identification_alpha,T)
        self.explore_phase = False

    def compute_cluster_det_ratio(self, client_indices):
        sum = 0
        for client_id in client_indices:
            numerator = np.linalg.det(self.clients[client_id].A_local+self.lambda_ * np.identity(n=self.dimension))
            denominator = np.linalg.det(self.clients[client_id].A_local-self.clients[client_id].A_uploadbuffer+self.lambda_ * np.identity(n=self.dimension))
            sum += np.log(numerator/denominator) * self.clients[client_id].numObs_uploadbuffer
        #import pdb;pdb.set_trace()
        return sum
        

    # override the share_stats_between_cluster to use a PQ
    def share_stats_between_cluster(self):
        # check if any clusters are waiting to collaborate
        if (len(self.collab_queue) > 0):

            # find the cluster with largest benefit using the determinant ratio
            cluster_to_collab_idx = 0
            max_det_ratio = 0
            for i in range(len(self.collab_queue)):
                client_indices = self.collab_queue[i]
                cluster_det_ratio = self.compute_cluster_det_ratio(client_indices)
                if (cluster_det_ratio > max_det_ratio):
                    cluster_to_collab_idx = i
                    max_det_ratio = cluster_det_ratio

            clients_to_collab = self.collab_queue.pop(cluster_to_collab_idx)

            A_aggregated = np.zeros((self.dimension, self.dimension))
            b_aggregated = np.zeros(self.dimension)
            numObs_aggregated = 0
            # server aggregates statistics for cluster to collaborate with
            for client_id in clients_to_collab:
                self.totalCommCost+=1
                A_aggregated += self.clients[client_id].A_local_clean
                b_aggregated += self.clients[client_id].b_local_clean
                numObs_aggregated += self.clients[client_id].numObs_clean
            # then send the aggregated ss to all the clients, now all of them are synced
            for client_id in clients_to_collab:
                self.totalCommCost += 1
                
                self.clients[client_id].A_local = copy.deepcopy(A_aggregated)
                self.clients[client_id].b_local = copy.deepcopy(b_aggregated)
                self.clients[client_id].numObs_local = copy.deepcopy(numObs_aggregated)
                # clear client's upload buffer
                self.clients[client_id].A_uploadbuffer = np.zeros((self.dimension, self.dimension))
                self.clients[client_id].b_uploadbuffer = np.zeros(self.dimension)
                self.clients[client_id].numObs_uploadbuffer = 0
            return

        else:
            return
        