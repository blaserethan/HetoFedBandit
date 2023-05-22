# -*- coding: utf-8 -*-

from bidict import bidict
import random
import numpy as np

dataset = "lastfm"
data_path = "./hetrec2011-{}-2k/raw_data/".format(dataset)
save_path = "./hetrec2011-{}-2k/processed_data/".format(dataset)
userObservations_filename = data_path+"processed_events_shuffled.dat"

user2ItemSeqs = {}

with open(userObservations_filename) as f:
    n=0
    for line in f:
        if n > 0:
            split_list=line.split('\t')
            userID = split_list[0]
            timeStamp = split_list[1]
            itemList = split_list[2]
            #print(itemList)

            if int(userID) not in user2ItemSeqs:
                user2ItemSeqs[int(userID)] = []
            user2ItemSeqs[int(userID)].append(timeStamp+"\t"+itemList)
        else:
            first_line = line
        n += 1

threshold_len = 300
user2ItemSeqs_100 = {}
print(len(user2ItemSeqs.keys()))
for userID in user2ItemSeqs.keys():
    if len(user2ItemSeqs[userID]) >= threshold_len:
        user2ItemSeqs_100[userID] = user2ItemSeqs[userID]
print('Number of users: '+str(len(user2ItemSeqs_100)))

# only keep users with over 300 observations


global_time = 0
file = open(save_path+"randUserShuffledTime_N{}_ObsMoreThan{}.dat".format(len(user2ItemSeqs_100), threshold_len),"w")
import pdb;pdb.set_trace()
final_sequence = []
while user2ItemSeqs_100:
    userID = random.choice(list(user2ItemSeqs_100.keys()))
    l = user2ItemSeqs_100[userID].pop(0)
    global_time += 1

    final_sequence.append(str(userID)+"\t"+l)
    if not user2ItemSeqs_100[userID]:
        del user2ItemSeqs_100[userID]

print("global_time {}".format(global_time))

random.shuffle(final_sequence)

global_time=0

file.write(first_line)
for l in final_sequence:
    file.write(l)
    global_time+=1
file.close()

print("global_time {}".format(global_time))