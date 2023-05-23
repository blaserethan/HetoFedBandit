import numpy as np
import copy
import networkx as nx
from heapq import heappop, heappush, heapify
from scipy.stats import chi2, ncx2
from itertools import combinations

class LocalClient:
    def __init__(self, featureDimension, lambda_, delta_, NoiseScale):
        self.d = featureDimension
        self.lambda_ = lambda_
        self.delta_ = delta_
        self.NoiseScale = NoiseScale
        self.cluster_indices = -1

        # Sufficient statistics stored on the client #
        # latest local sufficient statistics
        self.A_local = np.zeros((self.d, self.d))  #lambda_ * np.identity(n=self.d)
        self.b_local = np.zeros(self.d)
        self.numObs_local = 0

        # statistcs from the clients observations that are not ever contaminated by collaboration
        self.X = np.zeros((0, self.d)) 
        self.y = np.zeros((0,))
        self.A_local_clean = np.zeros((self.d, self.d))  #lambda_ * np.identity(n=self.d)
        self.b_local_clean = np.zeros(self.d)
        self.rank = 0
        self.numObs_clean = 0

        # aggregated sufficient statistics recently downloaded
        self.A_uploadbuffer = np.zeros((self.d, self.d))
        self.b_uploadbuffer = np.zeros(self.d)
        self.numObs_uploadbuffer = 0

        # for computing UCB
        self.AInv = np.linalg.inv(self.A_local+self.lambda_ * np.identity(n=self.d))
        self.UserTheta = np.zeros(self.d)
        self.UserThetaNoReg= np.zeros(self.d)

        self.alpha_t = self.NoiseScale * np.sqrt(
            self.d * np.log(1 + (self.numObs_local) / (self.d * self.lambda_)) + 2 * np.log(1 / self.delta_)) + np.sqrt(
            self.lambda_)

    def getUCB(self, alpha, article_FeatureVector):
        if alpha == -1:
            alpha = self.alpha_t

        mean = np.dot(self.UserTheta, article_FeatureVector)
        var = np.sqrt(np.dot(np.dot(article_FeatureVector, self.AInv), article_FeatureVector))
        pta = mean + alpha * var
        return pta

    def localUpdate(self, articlePicked_FeatureVector, click):
        # update local A and b
        self.A_local += np.outer(articlePicked_FeatureVector, articlePicked_FeatureVector)
        self.b_local += articlePicked_FeatureVector * click
        self.numObs_local += 1

        # update the upload buffer
        self.A_uploadbuffer += np.outer(articlePicked_FeatureVector, articlePicked_FeatureVector)
        self.b_uploadbuffer += articlePicked_FeatureVector * click
        self.numObs_uploadbuffer += 1


        self.AInv = np.linalg.inv(self.A_local+self.lambda_ * np.identity(n=self.d))
        self.UserTheta = np.dot(self.AInv, self.b_local)
        self.UserThetaNoReg = np.dot(np.linalg.pinv(self.A_local), self.b_local)
        assert self.d == articlePicked_FeatureVector.shape[0]

        self.alpha_t = self.NoiseScale * np.sqrt(
            self.d * np.log(1 + (self.numObs_local) / (self.d * self.lambda_)) + 2 * np.log(1 / self.delta_)) + np.sqrt(
            self.lambda_)
        
        # Update clean observation history
        self.A_local_clean += np.outer(articlePicked_FeatureVector, articlePicked_FeatureVector)
        self.b_local_clean += articlePicked_FeatureVector * click
        self.numObs_clean += 1
        self.X = np.concatenate((self.X, articlePicked_FeatureVector.reshape(1, self.d)), axis=0)
        self.y = np.concatenate((self.y, np.array([click])),axis=0)
        assert self.X.shape == (self.numObs_clean, self.d)
        assert self.y.shape == (self.numObs_clean, )
        self.rank = np.linalg.matrix_rank(self.X)

    def getTheta(self):
        return self.UserTheta

    def syncRoundTriggered(self, threshold):
        numerator = np.linalg.det(self.A_local+self.lambda_ * np.identity(n=self.d))
        denominator = np.linalg.det(self.A_local-self.A_uploadbuffer+self.lambda_ * np.identity(n=self.d))
        return np.log(numerator/denominator)*(self.numObs_uploadbuffer) >= threshold

# this is the server's copy of each client, which is updated every time a client shares with the central server
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


class HetoFedBandit_Simplified:
    def __init__(self, dimension, alpha, lambda_, delta_, NoiseScale, threshold, exploration_length, neighbor_identification_alpha):
        self.dimension = dimension
        self.alpha = alpha
        self.lambda_ = lambda_
        self.delta_ = delta_
        self.NoiseScale = NoiseScale
        self.threshold = threshold
        self.explore_len = exploration_length
        self.CanEstimateUserPreference = True
        self.explore_phase = True # boolean representing if we are in the exploration phase
        self.neighbor_identification_alpha = neighbor_identification_alpha
        self.collab_graph = nx.Graph()
        self.clusters = None
        self.collab_queue = []

        self.clients = {}
        self.server_copies = {}

        # records
        self.totalCommCost = 0

    def decide(self, pool_articles, clientID):
        if clientID not in self.clients:
            self.clients[clientID] = LocalClient(self.dimension, self.lambda_, self.delta_, self.NoiseScale)
            # initialize the server's copy
            self.server_copies[clientID] = ServerCopyClient(self.dimension, self.lambda_, self.delta_, self.NoiseScale, clientID)

        maxPTA = float('-inf')
        articlePicked = None

        if (self.explore_phase == True):
            articlePicked = np.random.choice(pool_articles)
        else:
            for x in pool_articles:
                x_pta = self.clients[clientID].getUCB(self.alpha, x.featureVector)
                # pick article with highest UCB score
                if maxPTA < x_pta:
                    articlePicked = x
                    maxPTA = x_pta

        return articlePicked

    def updateParameters(self, articlePicked, click, currentClientID):
        # update local ss, and upload buffer
        self.clients[currentClientID].localUpdate(articlePicked.featureVector, click)
        
        if self.clients[currentClientID].syncRoundTriggered(self.threshold) and not self.explore_phase:
            # if we trigger the communication threshold, then we need to add the clusters that the client is a member of
            # to the queue
            
            if self.clients[currentClientID].cluster_indices not in self.collab_queue: #can't add the same cluster multiple times
                self.collab_queue.append(self.clients[currentClientID].cluster_indices)

    def getTheta(self, clientID):
        return self.clients[clientID].UserTheta

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
        if n == 0 and m == 0:
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
        p_value = chi2.sf(x=chiSquareStatistic, df=df1) # DYCLU TEST
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
        print('*************** HETOFEDBANDIT-CLUSTERING ************')
        print(self.collab_graph)

        # compute the clusters from the user graph
        self.clusters = list(nx.find_cliques(self.collab_graph))
        print(len(self.clusters))
        print('***************************************************')

        for client_id, client_model in self.clients.items():
            client_model.cluster_indices = [i for i in range(len(self.clusters)) if client_id in self.clusters[i]][0]

    def share_stats_between_cluster(self):
        # check if any clusters are waiting to collaborate
        if (len(self.collab_queue) > 0):
            print(len(self.collab_queue))

            # pop off the index of the cluster to collaborate with
            cluster_to_collab_idx = self.collab_queue.pop(0)

            # server gets statistics from everyone in the popped cluster
            for client_id in self.clusters[cluster_to_collab_idx]:

                self.totalCommCost+=1
                self.server_copies[client_id].A_clean = copy.deepcopy(self.clients[client_id].A_local_clean)
                self.server_copies[client_id].b_clean = copy.deepcopy(self.clients[client_id].b_local_clean)
                self.server_copies[client_id].numObs_clean = copy.deepcopy(self.clients[client_id].numObs_clean)

                self.server_copies[client_id].A_local =  copy.deepcopy(self.clients[client_id].A_local)
                self.server_copies[client_id].b_local = copy.deepcopy(self.clients[client_id].b_local)
                self.server_copies[client_id].numObs_local = copy.deepcopy(self.clients[client_id].numObs_local)

                # clear client's upload buffer
                self.clients[client_id].A_uploadbuffer = np.zeros((self.dimension, self.dimension))
                self.clients[client_id].b_uploadbuffer = np.zeros(self.dimension)
                self.clients[client_id].numObs_uploadbuffer = 0

            A_aggregated = np.zeros((self.dimension, self.dimension))
            b_aggregated = np.zeros(self.dimension)
            numObs_aggregated = 0

            for client_id in self.clusters[cluster_to_collab_idx]:
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
        

class HetoFedBandit_PQ(HetoFedBandit_Simplified):
    def __init__(self, dimension, alpha, lambda_, delta_, NoiseScale, threshold, exploration_length, neighbor_identification_alpha):
        super().__init__(dimension, alpha, lambda_, delta_, NoiseScale, threshold, exploration_length, neighbor_identification_alpha)      

    
    def compute_cluster_det_ratio(self, cluster_idx):
        collab_A = np.zeros((self.dimension, self.dimension)) 
        for client_id in self.clusters[cluster_idx]:
            collab_A += self.server_copies[client_id].A_clean
        
        numerator = np.linalg.det(collab_A+self.lambda_ * np.identity(n=self.dimension))
        sum = 0
        for client_id in self.clusters[cluster_idx]:
            denominator = np.linalg.det(self.server_copies[client_id].A_local+self.lambda_ * np.identity(n=self.dimension))
            sum += np.log(numerator/denominator) * self.server_copies[client_id].numObs_not_yet_shared
        return sum
        

    # override the share_stats_between_cluster to use a PQ
    def share_stats_between_cluster(self):
        # check if any clusters are waiting to collaborate
        if (len(self.collab_queue) > 0):

            # find the cluster with largest benefit using the determinant ratio
            cluster_to_collab_idx = 0
            max_det_ratio = 0
            for cluster_idx in self.collab_queue:
                cluster_det_ratio = self.compute_cluster_det_ratio(cluster_idx)
                if (cluster_det_ratio >= max_det_ratio):
                    cluster_to_collab_idx = cluster_idx
                    max_det_ratio = cluster_det_ratio

            # remove it from the queue
            self.collab_queue.remove(cluster_to_collab_idx)

            # server gets statistics from everyone in the selected cluster
            for client_id in self.clusters[cluster_to_collab_idx]:

                self.totalCommCost+=1
                self.server_copies[client_id].A_clean = copy.deepcopy(self.clients[client_id].A_local_clean)
                self.server_copies[client_id].b_clean = copy.deepcopy(self.clients[client_id].b_local_clean)
                self.server_copies[client_id].numObs_clean = copy.deepcopy(self.clients[client_id].numObs_clean)

                self.server_copies[client_id].A_local =  copy.deepcopy(self.clients[client_id].A_local)
                self.server_copies[client_id].b_local = copy.deepcopy(self.clients[client_id].b_local)
                self.server_copies[client_id].numObs_local = copy.deepcopy(self.clients[client_id].numObs_local)

                # clear client's upload buffer
                self.clients[client_id].A_uploadbuffer = np.zeros((self.dimension, self.dimension))
                self.clients[client_id].b_uploadbuffer = np.zeros(self.dimension)
                self.clients[client_id].numObs_uploadbuffer = 0

            A_aggregated = np.zeros((self.dimension, self.dimension))
            b_aggregated = np.zeros(self.dimension)
            numObs_aggregated = 0

            for client_id in self.clusters[cluster_to_collab_idx]:
                self.totalCommCost += 1
                # self.totalCommCost += (self.dimension**2 + self.dimension)
                # update server's aggregated ss
                A_aggregated += self.clients[client_id].A_local_clean
                b_aggregated += self.clients[client_id].b_local_clean
                numObs_aggregated += self.clients[client_id].numObs_clean

                # clear client's upload buffer
                self.clients[client_id].A_uploadbuffer = np.zeros((self.dimension, self.dimension))
                self.clients[client_id].b_uploadbuffer = np.zeros(self.dimension)
                self.clients[client_id].numObs_uploadbuffer = 0

            # then send the aggregated ss to all the clients, now all of them are synced
            for client_id in self.clusters[cluster_to_collab_idx]:
                self.totalCommCost += 1
                # self.totalCommCost += (self.dimension**2 + self.dimension)
                self.clients[client_id].A_local = copy.deepcopy(A_aggregated)
                self.clients[client_id].b_local = copy.deepcopy(b_aggregated)
                self.clients[client_id].numObs_local = copy.deepcopy(numObs_aggregated)

                #print(self.clients[client_id].numObs_local)
            
            return

        else:
            return
        
class HetoFedBandit_Data_Recluster(HetoFedBandit_Simplified):
    def __init__(self, dimension, alpha, lambda_, delta_, NoiseScale, threshold, exploration_length, neighbor_identification_alpha,T):
        super().__init__(dimension, alpha, lambda_, delta_, NoiseScale, threshold, exploration_length, neighbor_identification_alpha)
        self.explore_phase = False
        self.T = T

    def updateParameters(self, articlePicked, click, currentClientID):
        # update local ss, and upload buffer
        self.clients[currentClientID].localUpdate(articlePicked.featureVector, click)
        
        if self.clients[currentClientID].syncRoundTriggered(self.threshold) and not self.explore_phase:

            # every time we have a communication round, re-cluster and also clear the queue
            self.cluster_users()
            self.collab_queue = []
            if self.clients[currentClientID].cluster_indices not in self.collab_queue: #can't add the same cluster multiple times
                self.collab_queue.append(self.clients[currentClientID].cluster_indices)

            # everyone send updated statistics to server
            for client_id, client_model in self.clients.items():
                self.totalCommCost+=1
                self.server_copies[client_id].A_clean = copy.deepcopy(client_model.A_local_clean)
                self.server_copies[client_id].b_clean = copy.deepcopy(client_model.b_local_clean)
                self.server_copies[client_id].numObs_clean = copy.deepcopy(client_model.numObs_clean)

                self.server_copies[client_id].A_local =  copy.deepcopy(client_model.A_local)
                self.server_copies[client_id].b_local = copy.deepcopy(client_model.b_local)
                self.server_copies[client_id].numObs_local = copy.deepcopy(client_model.numObs_local)

                # clear client's upload buffer
                self.clients[client_id].A_uploadbuffer = np.zeros((self.dimension, self.dimension))
                self.clients[client_id].b_uploadbuffer = np.zeros(self.dimension)
                self.clients[client_id].numObs_uploadbuffer = 0

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
        print('*************** HETOFEDBANDIT-CLUSTERING ************')
        print(self.collab_graph)

        # compute the clusters from the user graph
        self.clusters = list(nx.find_cliques(self.collab_graph))
        print(len(self.clusters))
        print('***************************************************')

        for client_id, client_model in self.clients.items():
            client_model.cluster_indices = [i for i in range(len(self.clusters)) if client_id in self.clusters[i]][0]