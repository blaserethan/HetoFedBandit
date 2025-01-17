# Code for "Federated Linear Contextual Bandits with Heterogeneous Clients"

This repository contains implementation of the proposed algorithms HetoFedBandit and HetoFedBandit-Enhanced, and baseline algorithms for comparison:
- FCLUB_DC
- DisLinUCB
- N-IndependentLinUCB
- DyClu

For experiments on the synthetic dataset, directly run:
```console
python Simulation.py --T 2500 --n 30 --m 5
```
To experiment with different environment settings, specify parameters:
- T: number of iterations to run
- n: number of users
- m: number of clusters
- sigma: standard deviation of Gaussian noise in observed reward

Detailed description of how the simulation environment works can be found in Section 4 of the paper.

Experiment results for the simulated environment can be found in the "./Results/SimulationResults/" folder, which contains:
- "[namelabel]\_[startTime].png": plot of accumulated regret over iteration for each algorithm
- "[namelabel]\_AccRegret\_[startTime].csv": regret at each iteration for each algorithm
- "[namelabel]\_ParameterEstimation\_[startTime].csv": l2 norm between estimated and ground-truth parameter at each iteration for each algorithm
- "Config\_[startTime].json": stores hyper parameters of all algorithms for this experiment

For experiments on the already pre-processed LastFM dataset,  directly run:
```console
python Simulation_Realworld.py --dataset LastFM
```
If you want to preprocess the dataset from LastFM yourself, with different random shuffling of the events, you can utilize [Dataset/getProcessedEvents.py](Dataset/getProcessedEvents.py) to re-process the dataset using the procedure described in Section 4.3 of the paper.
The original dasetet can be downloaded at [LastFM-2011 Dataset](https://grouplens.org/datasets/hetrec-2011/).
