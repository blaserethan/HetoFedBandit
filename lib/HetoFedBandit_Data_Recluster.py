import numpy as np
import copy
import networkx as nx
from heapq import heappop, heappush, heapify
from scipy.stats import chi2, ncx2
from itertools import combinations
from lib.HetoFedBandit import LocalClient, HetoFedBandit_Simplified

class HetoFedBandit_Data_Recluster(HetoFedBandit_Simplified):
    def __init__(self, dimension, alpha, lambda_, delta_, NoiseScale, threshold, exploration_length, neighbor_identification_alpha,T):
        super().__init__(dimension, alpha, lambda_, delta_, NoiseScale, threshold, exploration_length, neighbor_identification_alpha)
        self.explore_phase = False
        self.T = T

    def updateParameters(self, articlePicked, click, currentClientID):
        # update local ss, and upload buffer
        self.clients[currentClientID].localUpdate(articlePicked.featureVector, click)
        
        if self.clients[currentClientID].syncRoundTriggered(self.threshold) and not self.explore_phase:
            
            # every time we have a communication round, re-cluster
            self.cluster_users()
            # since everyone sends their statistics to the server when we cluster, we add to the communication cost appropriately
            for client_id, client_model in self.clients.items():
                self.totalCommCost+=1

            # after re-clustering, add the clusters of currentClientID to the queue for collaboration
            if self.clients[currentClientID].cluster_indices not in self.collab_queue: #can't add the same cluster multiple times
                self.collab_queue.append(self.clusters[self.clients[currentClientID].cluster_indices])

    def share_stats_between_cluster(self):
        # check if any clusters are waiting to collaborate
        if (len(self.collab_queue) > 0):

            # pop off the index of the cluster to collaborate with
            clients_to_collab = self.collab_queue.pop(0)

            A_aggregated = np.zeros((self.dimension, self.dimension))
            b_aggregated = np.zeros(self.dimension)
            numObs_aggregated = 0

            # server aggregates statistics after recieivng them from client
            for client_id in clients_to_collab:
                self.totalCommCost+=1
                A_aggregated += self.clients[client_id].A_local_clean
                b_aggregated += self.clients[client_id].b_local_clean
                numObs_aggregated += self.clients[client_id].numObs_clean

            # then send the aggregated ss to all the clients, now all of them are synced
            for client_id in clients_to_collab:
                self.totalCommCost += 1
                # self.totalCommCost += (self.dimension**2 + self.dimension)
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

    def homogeneityTest(self, currentUser, neighborUser):
        """
        Cluster identification:
        Test whether two user models have the same ground-truth theta
        :param currentUser:
        :param neighborUser:
        :return:
        """
        n = currentUser.numObs_clean
        m = neighborUser.numObs_clean
        if n == 0 or m == 0:
            return False
        # Compute numerator
        theta_combine = np.dot(
            np.linalg.pinv(currentUser.A_local_clean + neighborUser.A_local_clean),
            currentUser.b_local_clean + neighborUser.b_local_clean)
        num = np.linalg.norm(np.dot(currentUser.X, (currentUser.UserThetaNoReg - theta_combine))) ** 2 + np.linalg.norm(
            np.dot(neighborUser.X, (neighborUser.UserThetaNoReg - theta_combine))) ** 2
        XCombinedRank = np.linalg.matrix_rank(np.concatenate((currentUser.X, neighborUser.X), axis=0))
        df1 = int(currentUser.rank + neighborUser.rank - XCombinedRank)
        chiSquareStatistic = num / (self.NoiseScale**2)
        top_matrix = np.linalg.multi_dot([neighborUser.A_local_clean, currentUser.A_local_clean + neighborUser.A_local_clean, currentUser.A_local_clean])
        w,v = np.linalg.eig(top_matrix)
        psi = w.max().real / ((self.T)*self.NoiseScale**2* len(self.clients)**2)

        df1 = int(currentUser.rank + neighborUser.rank - XCombinedRank)
        chiSquareStatistic = num / (self.NoiseScale**2)
        p_value = ncx2.sf(x=chiSquareStatistic, df=df1, nc=psi)
        #import pdb; pdb.set_trace()
        if p_value <= self.neighbor_identification_alpha:  # upper bound probability of false alarm
            return False
        else:
            return True
        
    def cluster_users(self):
        self.collab_graph = nx.Graph()

        # add all the users to the graph
        self.collab_graph.add_nodes_from(self.clients.keys())

        all_user_pairs = list(combinations(self.clients,2)) #gives you (client,clientID)

        for client_pair in all_user_pairs:
            # get the clocal client objects 
            client_1_model = self.clients[client_pair[0]]
            client_2_model = self.clients[client_pair[1]]

            # compute the pairwise homogeneity test between them both ways
            lr = self.homogeneityTest(client_1_model, client_2_model)
            rl = self.homogeneityTest(client_2_model, client_1_model)

            if (lr == True and rl == True):
                self.collab_graph.add_edge(client_pair[0],client_pair[1])

        # compute the clusters from the user graph
        self.clusters = list(nx.find_cliques(self.collab_graph))

        for client_id, client_model in self.clients.items():
            client_model.cluster_indices = [i for i in range(len(self.clusters)) if client_id in self.clusters[i]][0]