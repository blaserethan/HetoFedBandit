import copy
import numpy as np
from scipy.stats import ortho_group, special_ortho_group
import random
from random import sample, shuffle, choice
from scipy.sparse import csgraph
import datetime
import os.path
import json
import matplotlib.pyplot as plt
import argparse
from sklearn.decomposition import TruncatedSVD
from sklearn import cluster
from sklearn.decomposition import PCA
# local address to different datasets
from conf import *
from operator import truediv

# real dataset
from dataset_utils.LastFM_util_functions_2 import readFeatureVectorFile, parseLine
from lib.SyncLinUCB import SyncLinUCB
from lib.FCLUB.LDP_FCLUB_DC import FCLUB_DC_Global_server
from lib.DyClu import DyClu
from lib.HetoFedBandit import HetoFedBandit_Simplified
from lib.HetoFedBandit_Enhanced import HetoFedBandit_Enhanced

class Article():
    def __init__(self, aid, FV=None):
        self.article_id = aid
        self.contextFeatureVector = FV
        self.featureVector = FV


class experimentOneRealData(object):
    def __init__(self, namelabel, dataset, context_dimension, batchSize=25, plot=True, Write_to_File=False):

        self.namelabel = namelabel
        assert dataset in ["LastFM", "Delicious", "MovieLens"]
        self.dataset = dataset
        self.context_dimension = context_dimension
        self.Plot = plot
        self.Write_to_File = Write_to_File
        self.batchSize = batchSize
        if self.dataset == 'LastFM':
            self.relationFileName = LastFM_relationFileName
            self.address = LastFM_address
            self.save_address = LastFM_save_address
            FeatureVectorsFileName = LastFM_FeatureVectorsFileName
            self.event_fileName = self.address + '/randUserShuffledTime_N75_ObsMoreThan300.dat'           #"/randUserOrderedTime_N75_ObsMoreThan300.dat" 
            # Read Feature Vectors from File
        self.FeatureVectors = readFeatureVectorFile(FeatureVectorsFileName)
        self.articlePool = []

    def batchRecord(self, iter_):
        print("Iteration %d" % iter_, "Pool", len(self.articlePool), " Elapsed time",
              datetime.datetime.now() - self.startTime)

    def runAlgorithms(self, algorithms, startTime):
        self.startTime = startTime
        timeRun = self.startTime.strftime('_%m_%d_%H_%M_%S')

        filenameWriteReward = os.path.join(self.save_address, 'AccReward' + str(self.namelabel) + timeRun + '.csv')

        end_num = 0
        while os.path.exists(filenameWriteReward):
            filenameWriteReward = os.path.join(self.save_address,'AccReward' + str(self.namelabel) + timeRun + str(end_num) + '.csv')
            end_num += 1

        filenameWriteCommCost = os.path.join(self.save_address, 'AccCommCost' + str(self.namelabel) + timeRun + '.csv')
        end_num = 0
        while os.path.exists(filenameWriteCommCost):
            filenameWriteCommCost = os.path.join(self.save_address,'AccCommCost' + str(self.namelabel) + timeRun + str(end_num) + '.csv')
            end_num += 1
        tim_ = []
        AlgReward = {}
        BatchCumlateReward = {}
        BatchCommCost = {}
        CommCostList = {}
        AlgReward["random"] = []
        BatchCumlateReward["random"] = []
        for alg_name, alg in algorithms.items():
            AlgReward[alg_name] = []
            BatchCumlateReward[alg_name] = []
            CommCostList[alg_name] = []
            BatchCommCost[alg_name]= []

        if self.Write_to_File:
            with open(filenameWriteReward, 'w') as f:
                f.write('Time(Iteration)')
                f.write(',' + ','.join([str(alg_name) for alg_name in algorithms.keys()]))
                f.write('\n')

            with open(filenameWriteCommCost, 'w') as f:
                f.write('Time(Iteration)')
                f.write(',' + ','.join([str(alg_name) for alg_name in algorithms.keys()]))
                f.write('\n')

        userIDSet = set()
        with open(self.event_fileName, 'r') as f:
            f.readline()
            iter_ = 0
            for _, line in enumerate(f, 1):
                userID, _, pool_articles = parseLine(line)
                if userID not in userIDSet:
                    userIDSet.add(userID)
                # ground truth chosen article
                article_id_chosen = int(pool_articles[0])
                
                # Construct arm pool
                self.article_pool = []
                for article in pool_articles:
                    article_id = int(article.strip(']'))
                    article_featureVector = self.FeatureVectors[article_id]
                    article_featureVector = np.array(article_featureVector, dtype=float)
                    assert type(article_featureVector) == np.ndarray
                    assert article_featureVector.shape == (self.context_dimension,)
                    self.article_pool.append(Article(article_id, article_featureVector))

                # Random strategy
                RandomPicked = choice(self.article_pool)
                if RandomPicked.article_id == article_id_chosen:
                    reward = 1
                else:
                    reward = 0  # avoid division by zero
                AlgReward["random"].append(reward)

                if (iter_ == algorithms['HetoFedBandit'].explore_len):
                    # set the explore_phase flag to False
                    #HetoFedBanditSimplified and HetoFedBandit_PQ both cluster only once
                    algorithms['HetoFedBandit'].explore_phase = False
                    # algorithms['HetoFedBandit_PQ'].explore_phase = False
                    # algorithms['HetoFedBandit_Data_Dependent_Recluster_PQ'].explore_phase = False
                    algorithms['HetoFedBandit'].cluster_users()    

                for alg_name, alg in algorithms.items():
                    # Observe the candiate arm pool and algoirhtm makes a decision

                    pickedArticle = alg.decide(self.article_pool, userID)

                    # Get the feedback by looking at whether the selected arm by alg is the same as that of ground truth
                    if pickedArticle.article_id == article_id_chosen:
                        reward = 1
                    else:
                        reward = 0
                    # The feedback/observation will be fed to the algorithm to further update the algorithm's model estimation
                    alg.updateParameters(pickedArticle, reward, userID)

                    # Record the reward
                    AlgReward[alg_name].append(reward)
                    CommCostList[alg_name].append(alg.totalCommCost)

                    # # check if the phase is right and then call decide
                    if alg_name=='FCLUB_DC':
                        if alg.time_to_next_phase == 1:
                            alg.detection(alg.global_time)

                if (not algorithms['HetoFedBandit'].explore_phase and (iter_ % config["n_users"] == 0)):
                    algorithms['HetoFedBandit'].share_stats_between_cluster()
                
                algorithms['HetoFedBandit_Enhanced'].share_stats_between_cluster()

                if iter_ % self.batchSize == 0:
                    self.batchRecord(iter_)
                    tim_.append(iter_)
                    BatchCumlateReward["random"].append(sum(AlgReward["random"]))
                    for alg_name in algorithms.keys():
                        BatchCumlateReward[alg_name].append(sum(AlgReward[alg_name]))
                        BatchCommCost[alg_name].append(CommCostList[alg_name][-1])

                    if self.Write_to_File:
                        with open(filenameWriteReward, 'a+') as f:
                            f.write(str(iter_))
                            f.write(',' + ','.join([str(BatchCumlateReward[alg_name][-1]) for alg_name in
                                                    list(algorithms.keys()) + ["random"]]))
                            f.write('\n')
                        with open(filenameWriteCommCost, 'a+') as f:
                            f.write(str(iter_))
                            f.write(',' + ','.join([str(BatchCommCost[alg_name][-1]) for alg_name in algorithms.keys()]))
                            f.write('\n')
                iter_ += 1

        if self.Plot:  # only plot
            linestyles = ['o-', 's-', '*-', '>-', '<-', 'g-', '.-', 'o-', 's-', '*-']
            markerlist = ['.', ',', 'o', 's', '*', 'v', '>', '<']

            # # plot the results
            fig, axa = plt.subplots(2, 1, sharex='all')
            # Remove horizontal space between axes
            fig.subplots_adjust(hspace=0)
            # fig.suptitle('Accumulated Regret and Communication Cost')
            # f, axa = plt.subplots(1)
            print("=====reward=====")
            count = 0
            for alg_name, alg in algorithms.items():
                labelName = alg_name
                axa[0].plot(tim_, [x / (y + 1) for x, y in zip(BatchCumlateReward[alg_name], BatchCumlateReward["random"])],
                        linewidth=1, marker=markerlist[count], markevery=2000, label=labelName)
                count += 1
            axa[0].legend(loc='upper left', prop={'size': 9})
            axa[0].set_xlabel("Iteration")
            axa[0].set_ylabel("Normalized reward")

            print("=====Comm Cost=====")
            count = 0
            for alg_name, alg in algorithms.items():
                labelName = alg_name
                axa[1].plot(tim_, BatchCommCost[alg_name], linewidth=1, marker=markerlist[count], markevery=2000, label=labelName)
                count += 1
            # axa[1].legend(loc='upper left',prop={'size':9})
            axa[1].set_xlabel("Iteration")
            axa[1].set_ylabel("Communication Cost")
            plt_path = os.path.join(self.save_address, str(self.namelabel) + str(timeRun) + '.png')
            plt.savefig(plt_path, dpi=300,bbox_inches='tight', pad_inches=0.0)
            plt.show()

        for alg_name in algorithms.keys():
            print('%s: %.2f' % (alg_name, BatchCumlateReward[alg_name][-1]))

        return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--alg', dest='alg',
                        help='Select a specific algorithm, could be CoLin, hLinUCB, factorUCB, LinUCB, etc.')
    parser.add_argument('--namelabel', dest='namelabel', help='Name')
    parser.add_argument('--dataset', default='MovieLens', dest='dataset', help='dataset')
    parser.add_argument('--dCLUB_alpha', dest='dCLUB_alpha', help='dCLUB_alpha')

    args = parser.parse_args()
    algName = str(args.alg)
    namelabel = str(args.namelabel)
    dataset = str(args.dataset)
    # Configuration about the environment
    Write_to_File = True
    plot = True
    config = {}

    # environment parameters
    if dataset == "LastFM":
        config["n_users"] = 75 #57
        config["testing_iterations"] = 41284 # 35438 
    elif dataset == "Delicious":
        config["n_users"] = 1867
        config["testing_iterations"] = 104799
    elif dataset == "MovieLens":
        config["n_users"] = 54
        config["testing_iterations"] = 214729
    else:
        assert False
    config["context_dimension"] = 25  # Feature dimension

    # Algorithm parameters
    config["lambda_"] = 0.2  # regularization in ridge regression
    config["alpha"] = 0.3
    config["gamma"] = 5
    config["D2"] = (config["testing_iterations"]) / (config["n_users"] * config["context_dimension"] * np.log(config["testing_iterations"]))
    config["D3"] = (config["testing_iterations"]) / (config["context_dimension"] * np.log(config["testing_iterations"]))

    # DyClu Parameters
    config["delta_1"] = 1e-1
    config["delta_2"] = 1e-1
    config["tau"] = 20  # size of sliding window



    realExperiment = experimentOneRealData(namelabel=namelabel,
                                           dataset=dataset,
                                           context_dimension=config["context_dimension"],
                                           plot=plot,
                                           Write_to_File=Write_to_File)

    print("Starting for {}, context dimension {}".format(realExperiment.dataset, realExperiment.context_dimension))
    algorithms = {}
    if not args.alg:

        algorithms['DisLinUCB'] = SyncLinUCB(dimension=config["context_dimension"], alpha=config["alpha"],
                                                         lambda_=config["lambda_"],
                                                         delta_=1e-1,
                                                         NoiseScale=0.1, threshold=config["D2"])

        algorithms['NIndepLinUCB'] = SyncLinUCB(dimension=config["context_dimension"], alpha=0.9,
                                            lambda_=config["lambda_"],
                                            delta_=1e-1,
                                            NoiseScale=0.1, threshold=np.Inf)

        
        algorithms['FCLUB_DC'] = FCLUB_DC_Global_server(L=config["n_users"], n=config["n_users"], userList= [1]*config["n_users"], d=config["context_dimension"], T=config["testing_iterations"])

        algorithms['HetoFedBandit'] = HetoFedBandit_Simplified(dimension=config["context_dimension"], alpha=0.2, lambda_=config["lambda_"],
                                                delta_=1e-1,
                                                NoiseScale=0.1, threshold=config['D2'], exploration_length= 5000, neighbor_identification_alpha =0.01)
        
        algorithms['HetoFedBandit_Enhanced'] = HetoFedBandit_Enhanced(dimension=config["context_dimension"], alpha=0.2, lambda_=config["lambda_"],
                                                delta_=1e-1,
                                                NoiseScale=0.1, threshold=config['D3']/5, exploration_length= 1000, neighbor_identification_alpha =0.01,T=config['testing_iterations'])
    
        algorithms['DyClu'] = DyClu(dimension=config["context_dimension"], alpha=config["alpha"],
                                lambda_=config["lambda_"],
                                NoiseScale=0.1, tau_e=config["tau"],
                                delta_1=config["delta_1"], delta_2=config["delta_2"],
                                change_detection_alpha=0, neighbor_identification_alpha=0.01,
                                dataSharing=False,
                                aggregationMethod="combine", useOutdated=False,
                                maxNumOutdatedModels=None)


    startTime = datetime.datetime.now()
    if dataset == "LastFM":
        address = LastFM_save_address
    else:
        address = "Invalid Dataset"
    print(address)

    cfg_path = os.path.join(address, 'Config' + str(namelabel) + startTime.strftime('_%m_%d_%H_%M_%S') + '.json')

    end_num = 0
    while os.path.exists(cfg_path):
        cfg_path = os.path.join(address, 'Config' + str(namelabel) + startTime.strftime('_%m_%d_%H_%M_%S') + str(
            end_num) + '.json')
        end_num += 1

    with open(cfg_path, 'w') as fp:
        json.dump(config, fp)
    realExperiment.runAlgorithms(algorithms, startTime)