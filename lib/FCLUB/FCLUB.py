# -*- coding: utf-8 -*-
# final version


import networkx as nx
import numpy as np
import copy

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
U = 1.01
D = 1.01


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
    def store_info(self, x, y, t, r, br, ksi_noise, B_noise):
        # self.V = self.V + np.outer(x, x) + B_noise
        self.S_up += np.outer(x, x)
        # self.b = self.b + y * x + ksi_noise
        self.u_up += y * x
        self.T_up += 1
        self.best_rewards[t] += br
        self.rewards[t] += r
        self.theta = np.matmul(np.linalg.inv(np.eye(self.d) + self.S), self.u)

    def get_info(self):
        V_t = self.S_up + self.S
        b_t = self.u + self.u_up
        t = self.t + self.T_up
        return V_t, b_t, t

# cluster delay communication version with CDP
class CDP_Cluster(DC_Cluster):
    def __init__(self, b, V, users_begin, d, user_num, rounds, rewards, best_rewards, l_server_index, index, users={},
                 t=0):
        super(CDP_Cluster, self).__init__(b, V, users_begin, d, user_num, rounds, rewards, best_rewards, l_server_index,
                                          index, users, t)
        self.H_queue = dict()
        self.h_queue = dict()
        self.serial_num = 1
        self.H_now = np.zeros((self.d, self.d))
        self.h_now = np.zeros(self.d)
        self.H_former = np.zeros((self.d, self.d))
        self.h_former = np.zeros(self.d)

    # update upload buffer
    def store_info(self, x, y, t, r, br):
        self.S_up += np.outer(x, x)
        self.u_up += y * x
        self.T_up += 1
        self.best_rewards[t] += br
        self.rewards[t] += r
        self.theta = np.matmul(np.linalg.inv(np.eye(self.d) + self.S), self.u)

    def countBits(self, n):
        List = list()
        bit_num = 0
        t = n
        while n != 0:
            if n % 2 != 0:
                left = t - 2 ** bit_num + 1
                right = t
                List.append(tuple([left, right]))
                t = left - 1

            n = n // 2
            bit_num += 1

        return List

    def phase_update(self):
        self.H_now = np.zeros((self.d, self.d))
        self.h_now = np.zeros(self.d)
        self.H_former = np.zeros((self.d, self.d))
        self.h_former = np.zeros(self.d)
        self.H_queue.clear()
        self.h_queue.clear()
        self.serial_num = 1

    def privatizer(self, t, delt, epsi):
        m = np.log(t + 1).astype(np.int) + 1
        num_list = self.countBits(self.serial_num)
        H_tmp = np.zeros((self.d, self.d))
        h_tmp = np.zeros(self.d)
        for i in num_list:
            if i in self.H_queue.keys():
                H_tmp += self.H_queue[i]
                h_tmp += self.h_queue[i]
            else:
                N_tmp = np.random.normal(0, 64 * m * (np.log(2 / delt)) ** 2 / epsi ** 2, (self.d + 1, self.d + 1))
                N = (N_tmp + N_tmp.T) / np.sqrt(2)
                H = N[0:self.d, 0:self.d]
                h = N[0:self.d, self.d - 1:self.d]
                self.H_queue[i] = H
                self.h_queue[i] = np.squeeze(h)

                H_tmp += self.H_queue[i]
                h_tmp += self.h_queue[i]

        self.h_now = h_tmp
        self.H_now = H_tmp

        self.serial_num += 1

# cluster in SCLUB
class sclub_Cluster(Cluster):
    def __init__(self, b, V, users_begin, d, user_num, rounds, rewards, best_rewards, theta_phase, users={}, t=0, T_phase=0):
        super(sclub_Cluster, self).__init__(b, V, users_begin, d, user_num, rounds, rewards, best_rewards, users, t)
        self.T_phase = T_phase
        self.theta_phase = theta_phase
        self.checks = {i: False for i in self.users}
        self.checked = len(self.users) == sum(self.checks.values())

    def phase_update(self):
        self.T_phase = self.t
        self.theta_phase = self.theta

    def update_check(self, i):
        self.checks[i] = True
        self.checked = len(self.users) == sum(self.checks.values())





class Local_server:
    def __init__(self, nl, d, begin_num, T, edge_probability=0.8):
        self.nl = nl    # the number of users in a server
        self.d = d     # dimension
        self.rounds = T  # the number of all rounds
        user_index_list = list(range(begin_num, begin_num + nl))    # the index of users in this server
        self.G = nx.generators.classic.complete_graph(user_index_list)   # Generate undirected complete graph，user indexes range from begin_num to begin_num + nl
        self.clusters = {
            0: Cluster(b=np.zeros(d), t=0, V=np.zeros((d, d)), users_begin=begin_num, d=d, user_num=nl,
                            rounds=self.rounds, rewards=np.zeros(self.rounds),
                            best_rewards=np.zeros(self.rounds))}
        self.cluster_inds = dict()  # Record the index of the cluster to which each user belongs, key:user_index, value:cluster_index
        self.begin_num = begin_num  # the beginning of the users' index in this server
        for i in range(begin_num, begin_num + nl):
            self.cluster_inds[i] = 0   # index of the cluster to which each user belongs, key:user_index ,value:cluster_index
        self.num_clusters = np.zeros(self.rounds, np.int64)  # the total number of clusters in each round , which recorded for a total of `round` times
        self.num_clusters[0] = 1    # only one cluster in the beginning

    # calculate the item to recommend
    def recommend(self, M, b, beta, items):
        Minv = np.linalg.inv(M)
        theta = np.dot(Minv, b)
        r_item_index = np.argmax(np.dot(items, theta) + beta * (np.matmul(items, Minv) * items).sum(axis=1))
        return r_item_index

    # Judge whether the edge between the two users in this cluster needs to be deleted
    def if_delete(self, user_index1, user_index2, cluster):
        t1 = cluster.users[user_index1].t
        t2 = cluster.users[user_index2].t
        fact_T1 = np.sqrt((1 + np.log(1 + t1)) / (1 + t1))
        fact_T2 = np.sqrt((1 + np.log(1 + t2)) / (1 + t2))
        theta1 = cluster.users[user_index1].theta
        theta2 = cluster.users[user_index2].theta
        return np.linalg.norm(theta1 - theta2) > alpha * (fact_T1 + fact_T2)

    # update cluster due to edge deleting
    def update(self, user_index, t):
        update_cluster = False
        c = self.cluster_inds[user_index]  # Find the local cluster to which the updated user belongs
        i = user_index
        # store the origin cluster
        origin_cluster = self.clusters[c]
        A = [a for a in self.G.neighbors(i)]    # find the connected-component of this user
        for j in A:
            user2_index = j
            c2 = self.cluster_inds[user2_index]
            user1 = self.clusters[c].users[i]
            user2 = self.clusters[c2].users[user2_index]
            if user1.t != 0 and user2.t != 0 and self.if_delete(i, user2_index, self.clusters[c]):
                self.G.remove_edge(i, j)    # delete the edge if the user should split
                update_cluster = True

        # may split the cluster
        if update_cluster:
            C = nx.node_connected_component(self.G, i)     # current user in the updated cluster
            remain_users = dict()     # user waiting to be assigned to a new cluster
            for m in C:
                remain_users[m] = self.clusters[c].get_user(m)

            if len(C) < len(self.clusters[c].users):    # if the cluster  has been split
                all_users_index = set(self.clusters[c].users)
                all_users = dict()  # all users in the origin cluster
                for user_index_all in all_users_index:
                    all_users[user_index_all] = self.clusters[c].get_user(user_index_all)
                # generate new cluster
                tmp_cluster = Cluster(b=sum([remain_users[k].b for k in remain_users]),
                                           t=sum([remain_users[k].t for k in remain_users]),
                                           V=sum([remain_users[k].V for k in remain_users]),
                                           users_begin=min(remain_users), d=self.d, user_num=len(remain_users),
                                           rounds=self.rounds,
                                           users=copy.deepcopy(remain_users),
                                           rewards=sum([remain_users[k].rewards for k in remain_users]),
                                           best_rewards=sum([remain_users[k].best_rewards for k in remain_users]))
                self.clusters[c] = tmp_cluster

                # Remove the users constituting the new cluster from the origin cluster's userlist
                for user_index3 in all_users_index:
                    if remain_users.__contains__(user_index3):
                        all_users.pop(user_index3)

                c = max(self.clusters) + 1
                while len(all_users) > 0:   # some users haven't been put into cluster
                    j = np.random.choice(list(all_users))
                    C = nx.node_connected_component(self.G, j)
                    new_cluster_users = dict()
                    for k in C:
                        new_cluster_users[k] = origin_cluster.get_user(k)
                    self.clusters[c] = Cluster(b=sum([new_cluster_users[n].b for n in new_cluster_users]),
                                                    t=sum([new_cluster_users[n].t for n in new_cluster_users]),
                                                    V=sum([new_cluster_users[n].V for n in new_cluster_users]),
                                                    users_begin=min(new_cluster_users), d=self.d,
                                                    user_num=len(new_cluster_users),
                                                    rounds=self.rounds, users=copy.deepcopy(new_cluster_users),
                                                    rewards=sum(
                                                        [new_cluster_users[k].rewards for k in new_cluster_users]),
                                                    best_rewards=sum(
                                                        [new_cluster_users[k].best_rewards for k in new_cluster_users]))
                    for k in C:
                        self.cluster_inds[k] = c

                    c += 1
                    for user_index in all_users_index:
                        if new_cluster_users.__contains__(user_index):
                            all_users.pop(user_index)
        #print(t)
        self.num_clusters[t] = len(self.clusters)


class FCLUB_Global_server:
    def __init__(self, L, n, userList, d, T):
        self.l_server_list = []
        self.usernum = n    # the total number of users
        self.rounds = T
        self.l_server_num = L   # the number of local server
        self.d = d
        self.cluster_usernum = np.zeros(L * n, np.int64)  # Record the number of users in each global cluster in each round
        self.clusters = dict()  # global cluster
        self.regret = np.zeros(self.rounds)
        self.reward = np.zeros(self.rounds)
        self.best_reward = np.zeros(self.rounds)
        self.totalCommCost=0
        user_begin = 0     # the first user's index in a cluster
        self.global_time = 1 # for some reason their time starts with 1'
        self.CanEstimateUserPreference = False
        # initialize global cluster, there are L cluster at first, corresponding to L local server
        for i in range(L):
            self.clusters[i] = Cluster(b=np.zeros(self.d), t=0, V=np.zeros((d, d)), users_begin=user_begin,
                                            d=self.d, user_num=userList[i], rounds=self.rounds, users={},
                                            rewards=np.zeros(self.rounds), best_rewards=np.zeros(
                    self.rounds))
            user_begin += userList[i]
        self.cluster_inds = np.zeros(n, np.int64)  # index of the global cluster to which each user belongs, value: user index
        self.l_server_inds = np.zeros(n,
                                      np.int64)  # index of the local server to which each user belongs
        # initialize local server
        user_index = 0     # the first user's index in the local server
        j = 0   # the local server index
        for i in userList:  # userList records the number of users in each local server
            self.l_server_list.append(Local_server(i, d, user_index, self.rounds))
            self.cluster_usernum[j] = i
            self.cluster_inds[user_index:user_index + i] = j
            self.l_server_inds[user_index:user_index + i] = j
            user_index = user_index + i
            j = j + 1

    # Locate the local server and global cluster
    def locate_user_index(self, user_index):
        # which local server
        l_server_index = self.l_server_inds[user_index]
        # which global cluster
        g_cluster_index = self.cluster_inds[user_index]
        return l_server_index, g_cluster_index

    # get the global cluster's information for recommendation
    def global_info(self, g_cluster_index):
        g_cluster = self.clusters[g_cluster_index]
        V = g_cluster.V
        b = g_cluster.b
        T = g_cluster.t
        gamma_t = gamma(T + 1, self.d, alpha, sigma)
        lambda_t = gamma_t * 2
        # S=L=1
        beta_t = beta(sigma, alpha, gamma_t, S, self.d, T + 1, self.l_server_num)
        M_t = np.eye(self.d) * np.float_(lambda_t) + V
        return M_t, b, beta_t

    # communicate between local server and global server
    def communicate(self):
        g_cluster_index = 0
        for i in range(self.l_server_num):
            l_server = self.l_server_list[i]
            for cluster_index in l_server.clusters:
                self.clusters[g_cluster_index] = copy.deepcopy(l_server.clusters[cluster_index])
                for user in l_server.cluster_inds:
                    if l_server.cluster_inds[user] == cluster_index:
                        self.cluster_inds[user] = g_cluster_index
                self.cluster_usernum[g_cluster_index] = l_server.clusters[cluster_index].user_num
                g_cluster_index += 1

    # merge two global clusters if they are close enough
    def merge(self):
        cmax = max(self.clusters)
        for c1 in range(cmax - 1):
            if c1 not in self.clusters:
                continue
            for c2 in range(c1 + 1, cmax):
                if c2 not in self.clusters:
                    continue
                T1 = self.clusters[c1].t
                T2 = self.clusters[c2].t
                fact_T1 = np.sqrt((1 + np.log(1 + T1)) / (1 + T1))
                fact_T2 = np.sqrt((1 + np.log(1 + T2)) / (1 + T2))
                theta1 = self.clusters[c1].theta
                theta2 = self.clusters[c2].theta
                # judge if two cluster should be merged
                if (np.linalg.norm(theta1 - theta2) >= alpha2 * (fact_T1 + fact_T2)):
                    continue
                else:
                    # merge two clusters and update the cluster's information
                    for i in range(self.usernum):
                        if self.cluster_inds[i] == c2:
                            self.cluster_inds[i] = c1

                    self.cluster_usernum[c1] = self.cluster_usernum[c1] + self.cluster_usernum[c2]
                    self.cluster_usernum[c2] = 0

                    self.clusters[c1].V = self.clusters[c1].V + self.clusters[c2].V
                    self.clusters[c1].b = self.clusters[c1].b + self.clusters[c2].b
                    self.clusters[c1].t = self.clusters[c1].t + self.clusters[c2].t
                    self.clusters[c1].user_num = self.clusters[c1].user_num + self.clusters[c2].user_num
                    # compute theta
                    self.clusters[c1].theta = np.matmul(np.linalg.inv(np.eye(self.d) + self.clusters[c1].V),
                                                        self.clusters[c1].b)
                    for user in self.clusters[c2].users:
                        self.clusters[c1].users.setdefault(user, self.clusters[c2].users[user])
                    del self.clusters[c2]
    
    def decide(self, pool_articles, userID):
        # if userID not in self.userID2userIndex:
        #     self.userID2userIndex[userID] = self.cur_userIndex
        #     self.cur_userIndex += 1

        user_index = userID
        
        l_server_index, g_cluster_index = self.locate_user_index(user_index)
        M_t, b, beta_t = self.global_info(g_cluster_index)
        l_server = self.l_server_list[l_server_index]
        g_cluster = self.clusters[g_cluster_index]
        l_cluster = l_server.clusters[l_server.cluster_inds[user_index]]
        # gather x of all articles into array num_items * d
        items = np.zeros((0, self.d))
        for article in pool_articles:
            #import pdb; pdb.set_trace()
            items = np.concatenate((items, article.featureVector[:self.d].reshape(1, self.d)), axis=0)
        
        article_id = l_server.recommend(M_t, b, beta_t, items)

        # self.cluster = []
        # cid = self.cluster_inds[self.userID2userIndex[userID]]
        # uind_cluster = np.where(self.cluster_inds==cid)[0].tolist()
        # for uind in uind_cluster:
        #     if uind in self.userID2userIndex.inverse:
        #         self.cluster.append(self.userID2userIndex.inverse[uind])

        return pool_articles[article_id]

    def updateParameters(self, articlePicked, click, userID):

        # self.store_info(self.userID2userIndex[userID], articlePicked.featureVector[:self.d], click, self.global_time)
        # self.split(self.userID2userIndex[userID], self.global_time)
        # self.merge(self.global_time)

        # get the relevant info
        user_index = userID
        x = articlePicked.featureVector[:self.d]
        y = click
        i = self.global_time

        # get the user index
        l_server_index, g_cluster_index = self.locate_user_index(user_index)
        l_server = self.l_server_list[l_server_index]
        g_cluster = self.clusters[g_cluster_index]
        l_cluster = l_server.clusters[l_server.cluster_inds[user_index]]

        l_cluster.users[user_index].store_info(x, y, i - 1)
        l_cluster.store_info(x, y, i - 1)
        # delete edge and aggregated information
        l_server.update(user_index, i - 1)
        # update the global information
        g_cluster.store_info(x, y, i - 1)
        self.communicate()
        self.totalCommCost += 2*self.l_server_num # EDITED because we need to count 2 for every server
        self.merge()

        self.global_time += 1
