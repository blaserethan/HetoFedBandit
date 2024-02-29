import numpy as np
import copy
import networkx as nx
from heapq import heappop, heappush, heapify
from scipy.stats import chi2, ncx2
from itertools import combinations
from lib.HetoFedBandit import LocalClient, HetoFedBandit_Simplified

class ServerCopyClient:
    def __init__(self, featureDimension, lambda_, delta_, NoiseScale, client_id):
        self.d = featureDimension
        self.lambda_ = lambda_
        self.delta_ = delta_
        self.NoiseScale = NoiseScale
        self.client_id = client_id #

        # Sufficient statistics stored on the client #
        # latest local sufficient statistics
        self.A_clean = np.zeros((self.d, self.d))  #lambda_ * np.identity(n=self.d)
        self.b_clean = np.zeros(self.d)
        self.numObs_clean = 0

        self.numObs_not_yet_shared = 0 #number of new observations from client that have not been shared with its collaborators yet

        self.A_local = np.zeros((self.d, self.d))
        self.b_local = np.zeros(self.d)
        self.numObs_combined = 0

class HetoFedBandit_Data_Recluster(HetoFedBandit_Simplified):
    def __init__(self, dimension, alpha, lambda_, delta_, NoiseScale, threshold, exploration_length, neighbor_identification_alpha,T):
        super().__init__(dimension, alpha, lambda_, delta_, NoiseScale, threshold, exploration_length, neighbor_identification_alpha)
        self.explore_phase = False
        self.T = T

    def updateParameters(self, articlePicked, click, currentClientID):
        # update local ss, and upload buffer
        self.clients[currentClientID].localUpdate(articlePicked.featureVector, click)
        
        if self.clients[currentClientID].syncRoundTriggered(self.threshold):

            # every time we have a communication round, re-cluster and also clear the queue
            self.cluster_users()
            self.collab_queue = []
            if self.clients[currentClientID].cluster_indices not in self.collab_queue: #can't add the same cluster multiple times
                self.collab_queue.append(self.clients[currentClientID].cluster_indices)

    def share_stats_between_cluster(self):
        # check if any clusters are waiting to collaborate
        if (len(self.collab_queue) > 0):

            #server already has latest from all clients so don't re_cluster

            # pop off the index of the cluster to collaborate with
            cluster_to_collab_idx = self.collab_queue.pop(0)

            A_aggregated = np.zeros((self.dimension, self.dimension))
            b_aggregated = np.zeros(self.dimension)
            numObs_aggregated = 0

            for client_id in self.clusters[cluster_to_collab_idx]:
                self.totalCommCost+=1
                # aggregate statistics based on the latest versions the server has (no communication)
                A_aggregated += self.server_copies[client_id].A_clean
                b_aggregated += self.server_copies[client_id].b_clean
                numObs_aggregated += self.clients[client_id].numObs_clean
            # then send the aggregated ss to all the clients, now all of them are synced
            for client_id in self.clusters[cluster_to_collab_idx]:
                self.totalCommCost += 1
                # self.totalCommCost += (self.dimension**2 + self.dimension)
                self.server_copies[client_id].numObs_not_yet_shared = 0 # after sharing, clear this number
                self.clients[client_id].A_local = copy.deepcopy(A_aggregated)
                self.clients[client_id].b_local = copy.deepcopy(b_aggregated)
                self.clients[client_id].numObs_local = copy.deepcopy(numObs_aggregated)
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
        print('*************** Enhanced-CLUSTERING ************')
        print(self.collab_graph)

        # compute the clusters from the user graph
        self.clusters = list(nx.find_cliques(self.collab_graph))
        print(len(self.clusters))
        print('***************************************************')

        for client_id, client_model in self.clients.items():
            client_model.cluster_indices = [i for i in range(len(self.clusters)) if client_id in self.clusters[i]][0]