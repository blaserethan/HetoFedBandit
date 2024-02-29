# -*- coding: utf-8 -*-
# final version


import networkx as nx
import numpy as np
import copy
from bidict import bidict

S = 1

# -*- coding: utf-8 -*-
# final version
import cmath
import sys

# Some constant
# c = 0.01

# alpha = 4
# alpha2 = 3.5
alpha = 1.5
alpha2 = 2
delt = 0.1
alpha1 = 1
# alpha1 = 0.01
# epsi = 0.1
# epsi = 0.5
epsi = 1
# epsi = 2
# epsi = 4
# epsi = 6
# epsi = 8
# epsi = 10
U = 1.2
D = 1.2


# --------------------------------generate some parameters-------------------------------------- #

def isInvertible(S):
    return np.linalg.cond(S) < 1 / sys.float_info.epsilon


# generate items to recommend
def generate_items(num_items, d):
    # return a ndarray of num_items * d
    x = np.random.normal(0, 1, (num_items, d - 1))
    x = np.concatenate((np.divide(x, np.outer(np.linalg.norm(x, axis=1), np.ones(np.shape(x)[1]))) / np.sqrt(2),
                        np.ones((num_items, 1)) / np.sqrt(2)), axis=1)
    return x


# generate sigma
def sigm(delta, epsilon):
    tmp = np.power(2 * np.log(2.5 / delta), 0.5)
    # print("sigma:",6 * tmp / epsilon)
    return 6 * tmp / epsilon


# generate gamma
def gamma(t, d, alpha, sigma):
    tmp = 4 * cmath.sqrt(d) + 2 * np.log(2 * t / alpha)
    # return sigma * cmath.sqrt(rounds) * tmp
    return 1


# generate beta
def beta(sigma, alpha, gamma, S, d, t, L=1):
    # tmp1 = cmath.sqrt(2 * np.log(2 / alpha) + d * np.log(3 + rounds * np.power(L, 2) / (d * gamma)))
    # tmp2 = cmath.sqrt(3 * gamma)
    # tmp3 = cmath.sqrt((1/gamma) * d * rounds)
    # #print("beta:", sigma * tmp1 + S * tmp2 + sigma * tmp3)
    # return sigma * tmp1 + S * tmp2 + sigma * tmp3 * 0.5
    tmp1 = cmath.sqrt(2 * np.log(2 / alpha) + d * np.log(3 + t * np.power(L, 2) / d))
    return tmp1 * 0.5


# L_1 = 1, generate sigma in CDP version
def sigma_CDP(t):
    m = np.log(t + 1e-6) + 1
    tmp1 = cmath.sqrt(m * np.log(16 / (delt ** 2)))
    return 4 * (1 + 1) * tmp1 / epsi


# Intermediate variables for CDP calculation
def rou_min(t, d):
    m = np.log(t + 1e-6) + 1
    tmp1 = 4 * cmath.sqrt(d) + 2 * np.log(2 * t / alpha1)
    return cmath.sqrt(32) * m * 2 * np.log(4 / delt) * tmp1 / epsi


# Intermediate variables for CDP calculation
def rou_max(t, d):
    return rou_min(t, d)


# Intermediate variables for CDP calculation
def upsilon(t, d):
    m = np.log(t + 1e-6) + 1
    tmp1 = cmath.sqrt(d) + 2 * np.log(2 * t / alpha1)
    return cmath.sqrt(m * 2 * tmp1 / cmath.sqrt(2 * epsi))


# generate beta in CDP version, L: the number of local server
def beta_CDP(t, d, L):
    rou_min1 = rou_min(t, d)
    rou_max1 = rou_max(t, d)
    tmp1 = cmath.sqrt(2 * np.log(2 / alpha1 + 1e-6) + d * np.log(rou_max1 / rou_min1 + t / (d * rou_min1)))
    sigm = 1
    upsi = upsilon(t, d)
    return sigm * tmp1 + cmath.sqrt(L * rou_max1) + cmath.sqrt(L) * upsi


sigma = sigm(delt, epsi);

# ---------------------------------- Environment: generate user, item and feedback ----------------------------------------- #

class Environment:
    def __init__(self, d, num_users, theta, L=10, noise_scale=0.1):
        self.L = L  # the number of items to generate at each step
        self.d = d # d is the dimension
        self.user_num = num_users
        self.theta = theta
        self.noise_scale = noise_scale

    def get_items(self):
        self.items = generate_items(self.L, self.d)
        return self.items

    # get reward, best reward and then compute regret
    def feedback_Local(self, items, i, k, d):  # k: the chosen item's index , i: user_index
        x = items[k, :]  # select item from item array
        B_noise = np.random.normal(0, sigma ** 2, (d, d))
        reward = np.dot(x, self.theta[i])
        if reward < 0 or reward > 1:    # if reward is illegal
            y = 0
        else:
            y = np.random.binomial(1, reward)
        ksi_noise = np.random.normal(np.zeros(d), np.eye(d), (d, d))
        best_reward = np.max(np.dot(items, self.theta[i]))
        return reward, y, best_reward, ksi_noise, B_noise

    # get reward, best reward and then compute regret
    def feedback(self, items, i, b, M, k, d):   # k: the chosen item's index , i: user_index
        x = items[k, :]  # select item from item array
        B_noise = np.random.normal(0, sigma ** 2, (d, d))
        reward = np.dot(self.theta[i], x)
        add_noise = np.random.normal(scale=self.noise_scale)
        y = reward+add_noise
        ksi_noise = np.random.normal(np.zeros(d), np.eye(d), (d, d))
        best_reward = np.max(np.dot(items, self.theta[i]))
        return reward, y, best_reward, ksi_noise, B_noise

    def generate_users(self):  # user selection is uniform
        X = np.random.multinomial(1, [1 / self.user_num] * self.user_num)  # X: 1*d array
        I = np.nonzero(X)[0]  # I: user_index
        return I


class User:
    def __init__(self, d, user_index, T):
        self.d = d  # dimension
        self.index = user_index  # the user's index, and it's unique
        self.t = 0  # rounds that pick the user
        self.b = np.zeros(self.d)
        self.V = np.zeros((self.d, self.d))
        self.rewards = np.zeros(T)  # T: the total round
        self.best_rewards = np.zeros(T)
        self.theta = np.zeros(d)

    def store_info(self, x, y, t):
        self.t += 1
        # self.V = self.V + np.outer(x,x) + B_noise
        self.V = self.V + np.outer(x, x)
        # self.b = self.b + y*x + ksi_noise
        self.b = self.b + y * x

        # c_t=1
        self.theta = np.matmul(np.linalg.inv(np.eye(self.d) + self.V), self.b)

    def get_info(self):
        return self.V, self.b, self.t


# Base cluster
class Cluster(User):
    def __init__(self, b, V, users_begin, d, user_num, rounds, rewards, best_rewards, users={}, t=0):
        self.d = d
        if not users:  # initialization at the beginning or a split/merged new cluster
            self.users = dict()
            for i in range(users_begin, users_begin + user_num):
                self.users[i] = User(self.d, i, rounds)  # a list/array of users
        else:
            self.users = copy.deepcopy(users)
        self.users_begin = users_begin
        self.user_num = user_num
        self.b = b
        self.t = t  # the current pick round
        self.V = V
        self.rewards = rewards
        self.best_rewards = best_rewards
        self.theta = np.matmul(np.linalg.inv(np.eye(self.d) + self.V), self.b)  # now c_t = 1

    def get_user(self, user_index):
        return self.users[user_index]

    # ksi_noise and B_noise are LDP noise parameter, in our experiment we don't add it
    def store_info(self, x, y, t):
        # self.V = self.V + np.outer(x, x) + B_noise
        self.V = self.V + np.outer(x, x)
        # self.b = self.b + y * x + ksi_noise
        self.b = self.b + y * x
        self.t += 1
        self.theta = np.matmul(np.linalg.inv(np.eye(self.d) + self.V), self.b)

    def get_info(self):
        V_t = self.V
        b_t = self.b
        t = self.t
        return V_t, b_t, t


# cluster delay communication version
class DC_Cluster(User):
    # Base cluster
    def __init__(self, b, V, users_begin, d, user_num, rounds, rewards, best_rewards, l_server_index, index, users={}, t=0):
        self.d = d
        if not users:  # initialization at the beginning or a split/merged new cluster
            self.users = dict()
            for i in range(users_begin, users_begin + user_num):
                self.users[i] = User(self.d, i, rounds)  # a list/array of users
        else:
            self.users = copy.deepcopy(users)
        self.users_begin = users_begin
        self.user_num = user_num
        self.u = b  # np.eye(d)
        self.t = t  # initial 0
        self.l_server_index = l_server_index    # l_server_index: which local server the DC_cluster belongs to
        self.index = index  # index: the cluster's index in the local server
        self.S = V  # synchronized gram matrix

        # upload buffer
        self.S_up = np.zeros((d, d))
        self.u_up = np.zeros(d)
        self.T_up = 0

        # download buffer
        self.S_down = np.zeros((d, d))
        self.u_down = np.zeros(d)
        self.T_down = 0

        self.rewards = rewards
        self.best_rewards = best_rewards

        # assume c_t = 1
        self.theta = np.matmul(np.linalg.inv(np.eye(self.d) + self.S), self.u)

    def get_user(self, user_index):
        return self.users[user_index]

    # update upload buffer
    def store_info(self, x, y, t):
        # self.V = self.V + np.outer(x, x) + B_noise
        self.S_up += np.outer(x, x)
        # self.b = self.b + y * x + ksi_noise
        self.u_up += y * x
        self.T_up += 1

        self.theta = np.matmul(np.linalg.inv(np.eye(self.d) + self.S), self.u)

    def get_info(self):
        V_t = self.S_up + self.S
        b_t = self.u + self.u_up
        t = self.t + self.T_up
        return V_t, b_t, t


S = 1
phase_cardinality = 2


class Local_server:
    def __init__(self, nl, d, begin_num, T, server_index):
        self.nl = nl  # the number of users in a server
        self.d = d  # dimension
        self.rounds = T  # the number of all rounds
        user_index_list = list(range(begin_num, begin_num + nl))  # the index of users in this server
        self.G = nx.generators.classic.complete_graph(
            user_index_list)  # Generate undirected complete graph，user indexes range from begin_num to begin_num + nl
        self.clusters = {
            0: DC_Cluster(b=np.zeros(d), t=0, V=np.zeros((d, d)), users_begin=begin_num, d=d, user_num=nl,
                               rounds=self.rounds, rewards=np.zeros(self.rounds), best_rewards=np.zeros(self.rounds),
                               l_server_index=server_index,
                               index=0)}  # Initialize the cluster, there is only one at the beginning
        self.index = server_index
        self.cluster_inds = dict()  # Record the index of the cluster to which each user belongs, key:user_index, value:cluster_index
        self.begin_num = begin_num
        for i in range(begin_num, begin_num + nl):
            self.cluster_inds[i] = 0  #
        self.num_clusters = np.zeros(self.rounds, int)  # the total number of clusters in each round , which recorded for a total of `round` times
        self.num_clusters[0] = 1

    # Determine which local cluster the user belongs to
    def locate_user_index(self, user_index):
        l_cluster_index = self.cluster_inds[user_index]
        return l_cluster_index

    # decide which items should be recommended at present
    def recommend(self, l_cluster_index, items):
        cluster = self.clusters[l_cluster_index]
        V_t, b_t, t = cluster.get_info()
        gamma_t = gamma(t, self.d, alpha, sigma)
        lambda_t = gamma_t * 2
        M_t = np.eye(self.d) * np.float_(lambda_t) + V_t
        beta_t = beta(sigma, alpha, gamma_t, S, self.d, t)
        # print("beta: ", beta_t)
        Minv = np.linalg.inv(M_t)
        theta = np.dot(Minv, b_t)
        # calculate the best item
        r_item_index = np.argmax(np.dot(items, theta) + beta_t * (np.matmul(items, Minv) * items).sum(axis=1))
        return r_item_index

    # Judge whether the edge between the two users in this cluster needs to be deleted
    def if_delete(self, user_index1, user_index2, cluster):
        t1 = cluster.users[user_index1].t
        t2 = cluster.users[user_index2].t
        fact_T1 = np.sqrt((1 + np.log(1 + t1)) / (1 + t1))
        fact_T2 = np.sqrt((1 + np.log(1 + t2)) / (1 + t2))

        # calculate and update the user's theta
        gamma_1 = gamma(t1, self.d, alpha, sigma)
        theta1 = np.matmul(np.linalg.inv(gamma_1 * 2 * np.eye(self.d) + cluster.users[user_index1].V),
                           cluster.users[user_index1].b)
        cluster.users[user_index1].theta = theta1
        gamma_2 = gamma(t2, self.d, alpha, sigma)
        theta2 = np.matmul(np.linalg.inv(gamma_2 * 2 * np.eye(self.d) + cluster.users[user_index2].V),
                           cluster.users[user_index2].b)
        cluster.users[user_index2].theta = theta2
        return np.linalg.norm(theta1 - theta2) > alpha * (fact_T1 + fact_T2)

    # Delete edges in this user's cluster
    def check_update(self, user_index):
        c = self.cluster_inds[user_index]  # Find the local cluster to which the updated user belongs
        i = user_index
        A = [a for a in self.G.neighbors(i)]
        for j in A:
            user2_index = j
            c2 = self.cluster_inds[user2_index]
            user1 = self.clusters[c].users[i]
            user2 = self.clusters[c2].users[user2_index]
            if user1.t != 0 and user2.t != 0 and self.if_delete(i, user2_index, self.clusters[c]):
                self.G.remove_edge(i, j)  # delete the edge

    # update clusters in this server after several delete edge operation
    def update_cluster(self):
        user_dict = dict()

        # Delete all previous clusters and save the users' information (since many edges may have been deleted, we regenerate all clusters for convenience)
        for j in list(self.clusters.keys()):
            for i in self.clusters[j].users:
                user_dict[i] = copy.deepcopy(self.clusters[j].users[i])
            del self.clusters[j]

        c = 0  # the cluster index in this server
        # Redivide the clusters according to the current connected_components
        for cluster_set in nx.connected_components(self.G):
            all_user = list(cluster_set)
            remain_users = dict()
            for k in all_user:
                remain_users[k] = user_dict[k]

            # Generate new cluster based on the connected_components
            tmp_cluster = DC_Cluster(b=sum([remain_users[k].b for k in remain_users]),
                                          t=sum([remain_users[k].t for k in remain_users]),
                                          V=sum([remain_users[k].V for k in remain_users]),
                                          users_begin=min(remain_users), d=self.d, user_num=len(remain_users),
                                          rounds=self.rounds,
                                          users=copy.deepcopy(remain_users),
                                          rewards=sum([remain_users[k].rewards for k in remain_users]),
                                          best_rewards=sum([remain_users[k].best_rewards for k in remain_users]),
                                          l_server_index=self.index, index=c)
            self.clusters[c] = tmp_cluster
            for k in remain_users:
                self.cluster_inds[k] = c

            c += 1


class FCLUB_DC_Global_server:
    def __init__(self, L, n, userList, d, T):
        self.l_server_list = []
        self.usernum = n  # the total number of users
        self.rounds = T
        self.l_server_num = L  # the number of local server
        self.g_cluster_num = 1  # only one cluster in the beginning
        self.d = d
        self.cluster_usernum = np.zeros(L * n,
                                        int)  # Record the number of users in each global cluster in each round
        self.clusters = dict()     # the global clusters' information, key: index
        self.regret = np.zeros(self.rounds)
        self.reward = np.zeros(self.rounds)
        self.best_reward = np.zeros(self.rounds)
        self.global_time = 0
        self.stage_id = 0
        self.time_to_next_phase = 2**self.stage_id
        self.CanEstimateUserPreference = False

        self.userID2userIndex = bidict()
        self.cur_userIndex = 0

        self.totalCommCost = 0
        # Record the partition, the first dimension represents global cluster, the second dimension represents local clusters in this global cluster
        # the expression of local cluster in the second dimension: (local server index，local cluster index in local server)
        self.partition = np.zeros((self.usernum, self.usernum * 2))
        self.partition.fill(-1)  # initialization
        # the initial partition, in local server and global server, the index is from 0
        for i in range(0, L * 2, 2):
            self.partition[0][i] = i / 2
            self.partition[0][i + 1] = 0

        # only one cluster in the beginning
        self.clusters[0] = DC_Cluster(b=np.zeros(self.d), t=0, V=np.zeros((self.d, self.d)), users_begin=0,
                                           d=self.d, user_num=self.usernum, rounds=self.rounds, users={},
                                           rewards=np.zeros(self.rounds), best_rewards=np.zeros(self.rounds),
                                           l_server_index=-1, index=0)
        self.cluster_inds = np.zeros(n,
                                     int)  # index of the global cluster to which each user belongs, value: user index
        self.l_server_inds = np.zeros(n, int)  # index of the local server to which each user belongs

        # initialize local server
        user_index = 0      # the first user's index in the local server
        j = 0  # the local server index
        for i in userList:  # userList records the number of users in each local server
            self.l_server_list.append(Local_server(i, d, user_index, self.rounds, server_index=j))
            self.cluster_usernum[j] = i
            self.cluster_inds[user_index:user_index + i] = 0
            self.l_server_inds[user_index:user_index + i] = j
            user_index = user_index + i
            j = j + 1

    # Locate the local server and global cluster
    def locate_user_index(self, user_index):
        l_server_index = self.l_server_inds[user_index]
        g_cluster_index = self.cluster_inds[user_index]
        return l_server_index, g_cluster_index

    # communicate between global server and local server and update the partition
    def communicate(self):
        g_cluster_index = 0
        tmp_partition = np.zeros((self.usernum, self.usernum * 2))
        tmp_partition.fill(-1)
        for i in range(self.l_server_num):
            l_server = self.l_server_list[i]
            for cluster_index in l_server.clusters:  # for convenience, upload all local clusters and remerge
                self.clusters[g_cluster_index] = copy.deepcopy(l_server.clusters[cluster_index]);
                tmp_partition[g_cluster_index][0] = l_server.clusters[cluster_index].l_server_index
                tmp_partition[g_cluster_index][1] = l_server.clusters[cluster_index].index

                for user in l_server.cluster_inds:
                    if l_server.cluster_inds[user] == cluster_index:
                        self.cluster_inds[user] = g_cluster_index
                self.cluster_usernum[g_cluster_index] = l_server.clusters[cluster_index].user_num
                g_cluster_index += 1

        self.partition = tmp_partition

    # determine whether the two clusters need to merge or not
    def if_merge(self, cluster_id1, cluster_id2):
        cluster1 = self.clusters[cluster_id1]
        cluster2 = self.clusters[cluster_id2]
        t1 = cluster1.t
        t2 = cluster2.t
        fact_T1 = np.sqrt((1 + np.log(1 + t1)) / (1 + t1))
        fact_T2 = np.sqrt((1 + np.log(1 + t2)) / (1 + t2))
        theta1 = cluster1.theta
        theta2 = cluster2.theta
        if (np.linalg.norm(theta1 - theta2) < alpha2 * (fact_T1 + fact_T2)):
            return True
        else:
            return False

    # merge the global clusters and update the information
    def merge(self, former_partition):
        done_merge = False   # if no change, don't need to update the partition
        cluster_node = list(self.clusters.keys())
        cluster_G = nx.complete_graph(cluster_node)  # all global clusters generate a complete graph
        nodes = cluster_G.nodes()  # all global clusters
        for c1 in nodes:
            if c1 not in self.clusters:
                continue
            A = [a for a in cluster_G.neighbors(c1)]
            for c2 in A:
                if c2 not in self.clusters:
                    continue
                if not self.if_merge(c1, c2):
                    cluster_G.remove_edge(c1, c2)  # remove the edge if two clusters can't merge together
                    done_merge = True

        # if the structure of global cluster has changed, update the info of global cluster
        if done_merge and (former_partition != self.partition).any():
            for cluster_set in nx.connected_components(cluster_G):
                global_l_cluster_num = 1
                cluster_list = list(cluster_set)
                c1 = cluster_list[0]    # choose a cluster, other clusters merge to it
                # after merge, update the global clusters' information and the partition
                for i in cluster_list[1:]:
                    self.clusters[c1].S += self.clusters[i].S
                    self.clusters[c1].u += self.clusters[i].u
                    self.clusters[c1].t += self.clusters[i].t
                    self.clusters[c1].user_num += self.clusters[i].user_num
                    self.cluster_usernum[c1] += self.cluster_usernum[i]

                    # update the partition
                    self.partition[c1][global_l_cluster_num * 2] = self.clusters[i].l_server_index
                    self.partition[c1][global_l_cluster_num * 2 + 1] = self.clusters[i].index
                    self.partition[i][0] = -1
                    self.partition[i][1] = -1
                    global_l_cluster_num += 1
                    for j in range(self.usernum):
                        if self.cluster_inds[j] == i:
                            self.cluster_inds[j] = cluster_list[0]
                    for user in self.clusters[i].users:
                        self.clusters[cluster_list[0]].users.setdefault(user, self.clusters[i].users[user])
                    del self.clusters[i]
                # recompute theta
                self.clusters[c1].theta = np.matmul(np.linalg.inv(np.eye(self.d) + self.clusters[c1].S),
                                                    self.clusters[c1].u)

    # phase-based cluster detection and adjustment
    def detection(self, t):
        # Upload the local clustered information to the global server, should have some time cost
        for l_index in range(len(self.l_server_list)):
            check_server = self.l_server_list[l_index]
            for i in check_server.cluster_inds:
                user1_index = i
                check_server.check_update(user1_index)

            check_server.update_cluster()

        # Upload the local clustered information to the global server,maybe should have some time cost
        former_partition = self.partition
        self.communicate()  # update global clusters by uploading local clusters' information
        self.merge(former_partition)  # merge the global clusters
        if (former_partition != self.partition).any():
            # Renew the cluster information
            self.totalCommCost += 1  # update the cumulative communication cost
            for g_cluster_id in self.clusters:
                self.clusters[g_cluster_id].S = np.zeros((self.d, self.d))
                self.clusters[g_cluster_id].u = np.zeros(self.d)
                self.clusters[g_cluster_id].t = 0
                l_cluster_info = self.partition[g_cluster_id]
                for i in range(0, self.usernum * 2, 2):
                    l_server_id = l_cluster_info[i].astype(int)
                    l_cluster_id = l_cluster_info[i + 1].astype(int)
                    if l_cluster_id == -1 or l_server_id == -1:
                        continue
                    l_server = self.l_server_list[l_server_id]
                    l_cluster = l_server.clusters[l_cluster_id]
                    self.clusters[g_cluster_id].S += l_cluster.S
                    self.clusters[g_cluster_id].u += l_cluster.u
                    self.clusters[g_cluster_id].t += l_cluster.t

                self.clusters[g_cluster_id].theta = np.matmul(
                    np.linalg.inv(np.eye(self.d) + self.clusters[g_cluster_id].S), self.clusters[g_cluster_id].u)

                # update local cluster's information using global cluster's information
                for i in range(0, self.usernum * 2, 2):
                    l_server_id = l_cluster_info[i].astype(int)
                    l_cluster_id = l_cluster_info[i + 1].astype(int)
                    if l_cluster_id == -1 or l_server_id == -1:
                        continue
                    l_server = self.l_server_list[l_server_id]
                    l_cluster = l_server.clusters[l_cluster_id]
                    l_cluster.S = self.clusters[g_cluster_id].S
                    l_cluster.u = self.clusters[g_cluster_id].u
                    l_cluster.t = self.clusters[g_cluster_id].t
                    l_cluster.theta = np.matmul(
                        np.linalg.inv(np.eye(self.d) + l_cluster.S), l_cluster.u)

                    # create new buffers
                    l_cluster.S_up = np.zeros((self.d, self.d))
                    l_cluster.S_down = np.zeros((self.d, self.d))
                    l_cluster.u_up = np.zeros(self.d)
                    l_cluster.u_down = np.zeros(self.d)
                    l_cluster.T_up = 0
                    l_cluster.T_down = 0

    # determine global cluster according to the local cluster
    def find_global_cluster(self, l_server_id, l_cluster_id):
        for g_cluster_id in self.clusters:
            g_cluster_want = g_cluster_id
            l_cluster_info = self.partition[g_cluster_id]
            for i in range(0, self.usernum * 2, 2):
                l_server_id_tmp = l_cluster_info[i]
                l_cluster_id_tmp = l_cluster_info[i + 1]
                if l_server_id_tmp == l_server_id and l_cluster_id_tmp == l_cluster_id:
                    return g_cluster_want

        return -1   # can't find the global cluster

    # Check upload event
    def check_upload(self, l_server_id, l_cluster_id):
        l_server = self.l_server_list[l_server_id.astype(int)]
        l_cluster = l_server.clusters[l_cluster_id]
        S = l_cluster.S
        S_up = l_cluster.S_up
        if np.linalg.det(S + S_up) / np.linalg.det(S) >= U:
            self.totalCommCost += 1  # update the cumulative communication cost
            g_cluster_id = self.find_global_cluster(l_server_id, l_cluster_id)
            if g_cluster_id != -1:
                self.clusters[g_cluster_id].S += S_up
                self.clusters[g_cluster_id].u += l_cluster.u_up
                self.clusters[g_cluster_id].t += l_cluster.T_up
                self.clusters[g_cluster_id].theta = np.matmul(
                    np.linalg.inv(np.eye(self.d) + self.clusters[g_cluster_id].S), self.clusters[g_cluster_id].u)

                # global server updates other servers' download buffer
                l_cluster_info = self.partition[g_cluster_id]  # all local clusters that generate this global cluster
                for i in range(0, self.usernum * 2, 2):
                    l_server_id_other = l_cluster_info[i]
                    l_cluster_id_other = l_cluster_info[i + 1]
                    if l_server_id_other == l_server_id and l_cluster_id_other == l_cluster_id:
                        continue
                    if l_server_id_other == -1 or l_cluster_id_other == -1:
                        continue
                    l_server_other = self.l_server_list[l_server_id_other.astype(int)]
                    l_cluster_other = l_server_other.clusters[l_cluster_id_other.astype(int)]
                    l_cluster_other.S_down += S_up
                    l_cluster_other.u_down += l_cluster.u_up
                    l_cluster_other.T_down += l_cluster.T_up

                # Local server cleans the buffer
                l_cluster.S += l_cluster.S_up
                l_cluster.u += l_cluster.u_up
                l_cluster.t += l_cluster.T_up
                l_cluster.theta = np.matmul(np.linalg.inv(np.eye(self.d) + l_cluster.S), l_cluster.u)
                l_cluster.S_up = np.zeros((self.d, self.d))
                l_cluster.u_up = np.zeros(self.d)
                l_cluster.T_up = 0

    # Check download event
    def check_download(self, g_cluster_id):
        l_cluster_info = self.partition[g_cluster_id]  # all local clusters that generate this global cluster
        V_g = self.clusters[g_cluster_id].S
        for i in range(0, self.usernum * 2, 2):
            l_server_id = l_cluster_info[i]
            l_cluster_id = l_cluster_info[i + 1]
            if l_server_id == -1 or l_cluster_id == -1:
                continue
            l_server = self.l_server_list[l_server_id.astype(int)]
            l_cluster = l_server.clusters[l_cluster_id.astype(int)]
            if np.linalg.det(V_g) / np.linalg.det(l_cluster.S) >= D:
                self.totalCommCost += 1  # update the cumulative communication cost

                # update local cluster's information
                l_cluster.S += l_cluster.S_down
                l_cluster.u += l_cluster.u_down
                l_cluster.t += l_cluster.T_down
                l_cluster.theta = np.matmul(np.linalg.inv(np.eye(self.d) + l_cluster.S), l_cluster.u)

                # clean the download buffer
                l_cluster.S_down = np.zeros((self.d, self.d))
                l_cluster.u_down = np.zeros(self.d)
                l_cluster.T_down = 0
    
    def decide(self, pool_articles, userID):
        if userID not in self.userID2userIndex:
             self.userID2userIndex[userID] = self.cur_userIndex
             self.cur_userIndex += 1

        user_index = self.userID2userIndex[userID]

        l_server_index, g_cluster_index = self.locate_user_index(user_index)
        l_server = self.l_server_list[l_server_index]
        l_cluster_index = l_server.locate_user_index(user_index)
        l_cluster = l_server.clusters[l_cluster_index]
        

        # gather x of all articles into array num_items * d
        items = np.zeros((0, self.d))
        for article in pool_articles:
            #import pdb; pdb.set_trace()
            items = np.concatenate((items, article.featureVector[:self.d].reshape(1, self.d)), axis=0)
        
        article_id = l_server.recommend(l_cluster_index=l_cluster_index, items=items)

        return pool_articles[article_id]

    def updateParameters(self, articlePicked, click, userID):

        # get the relevant info
        user_index = self.userID2userIndex[userID]
        x = articlePicked.featureVector[:self.d]
        y = click
        i = self.global_time

        # get the user index
        l_server_index, g_cluster_index = self.locate_user_index(user_index)
        l_server = self.l_server_list[l_server_index]
        l_cluster_index = l_server.locate_user_index(user_index)
        l_cluster = l_server.clusters[l_cluster_index]


        l_cluster.users[user_index].store_info(x, y, i - 1)
        l_cluster.store_info(x, y, i - 1)

        # check upload
        self.check_upload(l_server_index, l_cluster_index)
        # check download
        self.check_download(g_cluster_index)

        self.global_time += 1

        self.time_to_next_phase -= 1
        if self.time_to_next_phase == 0:
            self.stage_id += 1
            self.time_to_next_phase = 2**self.stage_id

    # # Phase-based FCLUB with communication delay
    # def run(self, envir, phase, number, all_round):
    #     result_final = list()  # to save the users' final theta information
    #     communication_cost = list()  # to save the cumulative communication cost
    #     for s in range(1, phase + 1):
    #         # detect and adjust clusters
    #         self.detection(phase_cardinality ** s - 1)
    #         for i in range(1, phase_cardinality ** s + 1):
    #             # compute the total time step
    #             t = (phase_cardinality ** s - 1) // (phase_cardinality - 1) + i - 1
    #             if t >= all_round:
    #                 break
    #             user_all = envir.generate_users()  # random user arrives
    #             user_index = user_all[0]
    #             l_server_index, g_cluster_index = self.locate_user_index(user_index)
    #             l_server = self.l_server_list[l_server_index]
    #             l_cluster_index = l_server.locate_user_index(user_index)
    #             l_cluster = l_server.clusters[l_cluster_index]
    #             # the context set
    #             items = envir.get_items()
    #             r_item_index = l_server.recommend(l_cluster_index=l_cluster_index, items=items)
    #             x = items[r_item_index]
    #             # receive the feedback and update the user's information
    #             # ksi_noise, B_noise are used to calculating LDP, but in this version we don't add
    #             self.reward[t - 1], y, self.best_reward[t - 1], ksi_noise, B_noise = envir.feedback_Local(items=items,
    #                                                                                                       i=user_index,
    #                                                                                                       k=r_item_index,
    #                                                                                                       d=self.d)
    #             l_cluster.users[user_index].store_info(x, y, t - 1, self.reward[t - 1], self.best_reward[t - 1],
    #                                                    ksi_noise[0], B_noise)
    #             l_cluster.store_info(x, y, t - 1, self.reward[t - 1], self.best_reward[t - 1], ksi_noise[0], B_noise)
    #             # check upload
    #             self.check_upload(l_server_index, l_cluster_index)
    #             # check download
    #             self.check_download(g_cluster_index)
    #             # calculate regret
    #             self.regret[t - 1] = self.best_reward[t - 1] - self.reward[t - 1]
    #             communication_cost.append(self.totalCommCost)

    #     return self.regret, result_final, self.reward, communication_cost
