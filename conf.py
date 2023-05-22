'''
Created on May 12, 2015
'''
import os

sim_files_folder = "./Simulation_MAB_files"
save_address = "./Results/SimulationResults"
LastFM_save_address = "./Results/LastFMResults"

save_addressResult = "./Results/Sparse"

datasets_address = '.'  # should be modified accoring to the local address

LastFM_address = datasets_address + '/Dataset/hetrec2011-lastfm-2k/processed_data'

LastFM_FeatureVectorsFileName = os.path.join(LastFM_address, 'Arm_FeatureVectors_2.dat')
LastFM_relationFileName = os.path.join(LastFM_address, 'user_friends.dat.mapped')

