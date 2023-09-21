from environment import GridWorld
import numpy as np
from learning import Learning
from qlearning import QLearning
import matplotlib.pyplot as plt
from drawing import Plotter

colors = ['b', 'g', 'r', 'c', 'm', 'k', 'y']

def movingAverage(data, n=50):
    ret = np.cumsum(data, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n-1:]/n

def computeConfInterval(avg, std_dev, n):
	return avg - 1.96 * (std_dev/np.sqrt(n)), avg + 1.96 * (std_dev/np.sqrt(n))

def exponentiate(M, exp):
    
    ret = np.zeros_like(M)
    for i in range(ret.shape[0]):
        for j in range(ret.shape[1]):
            if M[i, j] != 0:
                ret[i, j] = M[i, j]**exp
    return ret

def discoverOptions(env:GridWorld, epsilon, verbose=False, discoverNegation=False, plotGraphs=False):
    
    options = []
    actionSetPerOption = []
    
    numStates = env.getNumStates()
    
    W = env.getAdjacencyMatrix()
    D = np.zeros((numStates, numStates))

    for i in range(numStates):
        for j in range(numStates):
            D[i, i] = np.sum(W[i])

    for i in range(numStates):
        if D[i, i] == 0.0:
            D[i, i] = 1.0

    L = D - W
    
    expD = exponentiate(D, -0.5)
    normalizedL = expD.dot(L).dot(expD)
        
    eigenvalues, eigenvectors = np.linalg.eig(normalizedL)
    eigenvectors = np.real_if_close(eigenvectors, tol=1)
    
    # SORT
    
    idx = eigenvalues.argsort()[::-1]
    
    
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    
    if plotGraphs:
        plot = Plotter('outputs/', env)
        # plot.plotBasisFunctions(eigenvalues, eigenvectors)
    
    guard = len(eigenvectors[0])
    
    for i in range(guard):
        idx = guard-i-1
        
        polIter = Learning(0.9, env, augmentActionSet=True)
        env.defineRewardFunction(eigenvectors[:, idx])
        
        
        V, pi = polIter.solvePolicyIteration()
        
        
        for j in range(len(V)):
            if V[j] < epsilon:
                pi[j] = len(env.getActionSet())
                
        if plotGraphs:
            plot.plotValueFunction(V[0:numStates], str(idx) + '_')
            plot.plotPolicy(pi[0:numStates], str(idx)+'_')
        
        options.append(pi[0:numStates])
        optionsActionSet = env.getActionSet()
        optionsActionSet.append('terminate')
        actionSetPerOption.append(optionsActionSet)
    
    env.defineRewardFunction(None)
    env.reset()
    
    return options, actionSetPerOption
    
    
def qLearningWithOptions(env, alpha, gamma, options_eps, epsilon,
	nSeeds, maxLengthEp, nEpisodes, verbose, useNegation,
	genericNumOptionsToEvaluate, loadedOptions=None):

	numSeeds = nSeeds
	numEpisodes = nEpisodes
	# We first discover all options
	options = None
	actionSetPerOption = None

	if loadedOptions == None:
		if verbose:
			options, actionSetPerOption = discoverOptions(env, options_eps, verbose,
				useNegation, plotGraphs=True)
		else:
			options, actionSetPerOption = discoverOptions(env, options_eps, verbose,
				useNegation, plotGraphs=False)
	else:
		options = loadedOptions
		actionSetPerOption = []

		for i in range(len(loadedOptions)):
			tempActionSet = env.getActionSet()
			tempActionSet.append('terminate')
			actionSetPerOption.append(tempActionSet)

	returns_eval = []
	returns_learn = []
	# Now I add all options to my action set. Later we decide which ones to use.
	i = 0
	#genericNumOptionsToEvaluate = [1, 2, 4, 32, 64, 128, 256]
	totalOptionsToUse = []
	maxNumOptions = 0
	if useNegation and loadedOptions == None:
		maxNumOptions = int(len(options)/2)
	else:
		maxNumOptions = len(options)
	while i < len(genericNumOptionsToEvaluate) and genericNumOptionsToEvaluate[i] <= maxNumOptions:
		totalOptionsToUse.append(genericNumOptionsToEvaluate[i])
		i += 1

	for idx, numOptionsToUse in enumerate(totalOptionsToUse):
		returns_eval.append([])
		returns_learn.append([])

		if verbose:
			print('Using', numOptionsToUse, 'options') 

		for s in range(numSeeds):
			if verbose:
				print('Seed: ', s + 1) 

			returns_eval[idx].append([])
			returns_learn[idx].append([])
			actionSet = env.getActionSet()

			for i in range(numOptionsToUse):
				actionSet.append(options[i])

			if useNegation and loadedOptions == None:
				numOptions = 2*numOptionsToUse
			else:
				numOptions = numOptionsToUse

			learner = QLearning(alpha=alpha, gamma=gamma, epsilon=epsilon,
				environment=env, seed=s, useOnlyPrimActions=True,
				actionSet=actionSet, actionSetPerOption=actionSetPerOption)

			for i in range(numEpisodes):
				returns_learn[idx][s].append(learner.learnOneEpisode(timestepLimit=maxLengthEp))
				returns_eval[idx][s].append(learner.evaluateOneEpisode(eps=0.01, timestepLimit=maxLengthEp))

	returns_learn_primitive = []
	returns_eval_primitive  = []
	for s in range(numSeeds):
		returns_learn_primitive.append([])
		returns_eval_primitive.append([])
		learner = QLearning(alpha=alpha, gamma=gamma, epsilon=epsilon, environment=env, seed=s)
		for i in range(numEpisodes):
			returns_learn_primitive[s].append(learner.learnOneEpisode(timestepLimit=maxLengthEp))
			returns_eval_primitive[s].append(learner.evaluateOneEpisode(eps=0.01, timestepLimit=maxLengthEp))


	return returns_eval_primitive, returns_eval, totalOptionsToUse


if __name__ == '__main__':
    env = GridWorld('mdps/4rooms.mdp', useNegativeRewards=False)
    n = env.getNumStates()
    num_seeds = 1
    # i, j = env.getStateXY(0)
    # print(i, j)
    a = discoverOptions(env, 0.0, plotGraphs=True)
    # print(a[0])
    # returns_eval_primitive, returns_eval, totalOptionsToUse = qLearningWithOptions(
	# 		env=env, alpha=0.1, gamma=0.9, options_eps=0.0, epsilon=1.0, nSeeds=num_seeds,
	# 		maxLengthEp=100, nEpisodes=500,
	# 		verbose=False, useNegation=False,
	# 		genericNumOptionsToEvaluate = [1, 2, 4, 32, 64, 128],
	# 		loadedOptions=None)
    
    # color_idx = 0
    # average = np.mean(returns_eval_primitive, axis=0)
    # std_dev = np.std(returns_eval_primitive, axis=0)
    
    # minConfInt, maxConfInt = computeConfInterval(average, std_dev, num_seeds)
    
    # plt.plot(movingAverage(average), label='prim. act.', color=colors[color_idx])
    # plt.fill_between(range(len(movingAverage(average))), 
    #                  movingAverage(minConfInt), movingAverage(maxConfInt), 
    #                  alpha=0.5, color=colors[color_idx])
    
    # for idx, numOptionsToUse in enumerate(totalOptionsToUse):
    #     color_idx += 1
    #     average = np.mean(returns_eval[idx], axis=0)
    #     std_dev = np.std(returns_eval[idx], axis=0)
    #     minConfInt, maxConfInt = computeConfInterval(average, std_dev, num_seeds)

        
    #     plt.plot(movingAverage(average),
    #         label=str(numOptionsToUse) + ' opt.', color=colors[color_idx])

    #     plt.fill_between(range(len(movingAverage(average))),
    #         movingAverage(minConfInt), movingAverage(maxConfInt),
    #         alpha=0.5, color=colors[color_idx])

    # plt.legend(loc='upper left', prop={'size':10}, bbox_to_anchor=(1,1))
    # plt.tight_layout(pad=7)
    # plt.show()
    
    