''' 
Created by: Kevin De Angeli
12/28/18
'''


import random
import string
import matplotlib.pyplot as plt
from matplotlib.collections import EventCollection
from matplotlib.pyplot import figure

import numpy as np

population= 100
generations=100
numberOfRandomGames = 20
nunberOfGamesAgainstAgents = 50
mutationParaneter = 30
numOfSelectedIndForReproduction =40
selection1_selectivity_percent = numOfSelectedIndForReproduction/100




class Agent:
    def __init__(self):
        self.strategy = [] #A list of strategies for any given stage
        self.createRandomPopulation()
        self.fitness = 0
        self.generation = "" #Not currently being used, but it could be usefull to keep track of
                             #of the generation in which the agent was "born"

    #Initializes a random set of strategies for each possible stage encountered.
    def createRandomPopulation(self):
        for i,x in enumerate(total_game_scenarios):
            available = generatePossibleMoves(x)
            self.strategy.append(list(random.choice(list(available))))
    #Note: In order to associate an strategy with a stage in the game, I use the indexes.
    #There is a 1-1 relationship between the index of a stage and the strategy to play at that stage in the strategy list.



#############################################################################
#Fitness1 (Only one fitness function is used per run)
#"fitness1" just uses one of the agents against a random agent.
#The problem with this, is that it coverges to a local minima.
#That's because it does not play against more skilled players.
#############################################################################
def fitness1(agents):
    for allAgents,agent_i in enumerate(agents):
        agent_i.fitness = play(agent_i)
    #agents = adjusted_fitness(agents)
    return agents
#Play() is defined to play a game with one agent from the population
#and one opponents that executed random moves. It then calculates
#the win/total_games ratio. It's only used in fitness1()
def play(agent_i):
    numberOfGames = numberOfRandomGames
    agent_i_total_wins=0
    for i in range(numberOfGames):
        state = getNewGameChips()
        #i = random.randint(0, 9)
        i = 2 # the player moves first. This is because the first have a winning move in a 3,5,7 game
        while sum(state) != 0:
            if i % 2 == 0:  # Player 1 plays here
                move_index = total_game_scenarios.index(state)
                move = agent_i.strategy[move_index]
                state = moveExecute(state=state, move=move)
                winner = gameover(state)
                agent_i_total_wins += winner
            else:  # Player 2 plays when here
                move = randomMove(state)
                state = moveExecute(state=state, move=move)
                winner = gameover(state)
            i = i + 1
    return agent_i_total_wins/numberOfGames
#----------------------------------------------------------------------------

def fitness1_2Combined(agents):
    fitness_1 =0
    fitness_2 =0
    for allAgents,agent_i in enumerate(agents):
        fitness_1 = play(agent_i)
        fitness_2 = fitness2_play_against_other_agents(agent_i=agent_i, agents= agents)
        agent_i.fitness=(fitness_1+fitness_2)/2

    agents.sort(key=lambda x: x.fitness, reverse=True)
    return agents

#############################################################################
#Fitness function 2. (only one of the Fitness function
#  is implemented during the run)
#############################################################################
def fitness2(agents):
    random_agent_rival = Agent()
    for allAgents,agent_i in enumerate(agents):
        random_agent_rival= random.choice(agents)
        while random_agent_rival == agent_i:
            random_agent_rival = random.choice(agents)
        agent_i.fitness = fitness2_play_against_other_agents(agent_i,random_agent_rival)
    #agents = adjusted_fitness(agents)
    return agents
#Implemented for "fitness2" a new way to compute fitness.
#here, the agents play against other agents.
#This can have a negative effect if the population is not mantained
#Diverse, because they play against the same strategy over and over.
def fitness2_play_against_other_agents(agent_i, agents):
    numberOfGames = nunberOfGamesAgainstAgents
    agent_i_total_wins=0
    random_agent_rival = Agent()
    for i in range(numberOfGames):
        random_agent_rival = random.choice(agents)
        while random_agent_rival == agent_i:
            random_agent_rival = random.choice(agents)
        state = getNewGameChips()
        i = 2 # the player moves first. This is because the first have a winning move in a 3,5,7 game
        while sum(state) != 0:
            if i % 2 == 0:  # Player 1 plays here
                move_index = total_game_scenarios.index(state)
                move = agent_i.strategy[move_index]
                state = moveExecute(state=state, move=move)
                winner = gameover(state)
                agent_i_total_wins += winner
            else:  # Player 2 plays when here
                move_index = total_game_scenarios.index(state)
                move = random_agent_rival.strategy[move_index]
                state = moveExecute(state=state, move=move)
            i = i + 1
    return (agent_i_total_wins/numberOfGames)
#----------------------------------------------------------------------------


#############################################################################
#This function promotes diversity by scaling the fitness
#based on how many agents have exactly the same fitness
#It was stolen from the PDF in TicTT. It's used in both Fitness1() and Fitness()
#############################################################################
def adjusted_fitness(agents):
    newListAgents = []
    newListAgents = agents.copy()
    for i,x in enumerate(agents):
        count = 1
        for p in range(len(agents)):
            if x.fitness == newListAgents[p].fitness:
                count += 1
        x.fitness = x.fitness/count
    return agents
#----------------------------------------------------------------------------


#############################################################################
#sellection1() just picks the top "selecivity_percent" out of the population
#############################################################################
#This variables controls how many individuals will reproduce
#and how many will be taken out of the population it does not grow too much. (Selection() + cutPopulation())
def selection1(agents):
    #selectivity_percent = .10 #% of individuals that will reproduce
    newList = agents.copy()
    newList.sort(key=lambda x: x.fitness, reverse=True)
    newList = newList[:int(selection1_selectivity_percent * len(agents))] #Top 20%. Note you will get an error if the pop and generatios is not big enough
    return newList


#############################################################################
#sellection2() will implement a Tournament sellection.
#It choosed 5 random agents, and then selects the one with
#the highest fitnesss
#############################################################################
def selection2_TournamentSelection(agents):
    numberOfParticipants = 8
    listOfSelectedIndividuals =[]

    for q in range(numOfSelectedIndForReproduction):
        listOfParticiapants = []

        for i in range(numberOfParticipants):
            listOfParticiapants.append(random.choice(agents))

        #Chose the one with highest fitness
        listOfParticiapants.sort(key=lambda x: x.fitness, reverse=True)
        listOfSelectedIndividuals.append(listOfParticiapants[0])
    return listOfSelectedIndividuals


#Keep 80% of the population and add the new children
#Without this, the population keeps growing.
def cutPopulation(agents):
    agents.sort(key=lambda x: x.fitness, reverse=True)
    agents = agents[:int((1-selection1_selectivity_percent) * len(agents))]
    return agents


def crossover(agents, selectedIndividuals):
    newList = selectedIndividuals.copy()
    newChildList = []
    agents = cutPopulation(agents)
    isRepeated = True
    ParentIndex1 = -1
    ParentIndex2 = -2
    parent1 = Agent()
    parent2 = Agent()
    for i in range(int(len(selectedIndividuals))):
        child = Agent()
        while ParentIndex1 == ParentIndex2: #prevents reproduction of the same parent
            parent1 = random.choice(newList)
            ParentIndex1 = newList.index(parent1)
            parent2 = random.choice(newList)
            ParentIndex2 = newList.index(parent2)
        for p,q in enumerate(parent1.strategy):
                if q == parent2.strategy[p]:
                    child.strategy[p]=q
                else:
                    strategy_random = [q,parent2.strategy[p]]
                    strategy_random_select = random.choice(strategy_random)
                    child.strategy[p]=strategy_random_select
        child.fitness = ((parent1.fitness + parent2.fitness)/2) -0 #initial child fitness =0?
        newChildList.append(child)                                 #This affect how Mutation will treat it.
    agents= agents + newChildList
    return agents

#Implements half of the strategies from parent1, and half from parent2
def crossover2(agents, selectedIndividuals):
    newList = selectedIndividuals.copy()
    newChildList = []
    agents = cutPopulation(agents)
    isRepeated = True
    ParentIndex1 = -1
    ParentIndex2 = -2
    parent1 = Agent()
    parent2 = Agent()
    for i in range(int(len(selectedIndividuals) / 2)):
        child = Agent()
        while ParentIndex1 == ParentIndex2:  # prevents reproduction of the same parent
            parent1 = random.choice(newList)
            ParentIndex1 = newList.index(parent1)
            parent2 = random.choice(newList)
            ParentIndex1 = newList.index(parent2)
        for p, q in enumerate(parent1.strategy): #Half of the strategy from Paret1
            if p <= int(len(parent1.strategy)/2):
                child.strategy[p] = q
            else:                               #Half from Parent2
                strategy_random = [q, parent2.strategy[p]]
                strategy_random_select = random.choice(strategy_random)
                child.strategy[p] = strategy_random_select
        # child.fitness = ((parent1.fitness + parent2.fitness)/2) -.2 #initial child fitness =0?
        newChildList.append(child)  # This affect how Mutation will treat it.
    agents = agents + newChildList
    return agents


#This mutation2 function does not have a "elite preservation"
def mutation2(agents):
    for i,x in enumerate(agents):
        num_strategies_to_change = int((1-(x.fitness))*mutationParaneter) + 0 #This line is Key:
        for ii in range(num_strategies_to_change):              #It changes the number of strategies of an angent
            a= random.choice(total_game_scenarios)              #Based on the fitness
            b= randomMove(a)
            c= total_game_scenarios.index(a)
            x.strategy[c]=b
    return agents

#The higher the fitness, the less strategies that are randomly changed.
def mutation(agents):
    for i,x in enumerate(agents):
        random_num  = random.random()
        if random_num > x.fitness:
            num_strategies_to_change = int((1-(x.fitness))*mutationParaneter) + 0 #This line is Key:
            for ii in range(num_strategies_to_change):              #It changes the number of strategies of an angent
                a= random.choice(total_game_scenarios)              #Based on the fitness
                b= randomMove(a)
                c= total_game_scenarios.index(a)
                x.strategy[c]=b
    return agents


def randomMove(state):
    available = generatePossibleMoves(state)
    #print("available Moves: ", available)
    return list(random.choice(list(available)))
def gameover(state):
    for i, x in enumerate(state):
        if x != 0:
            return 0
    return 1
def moveExecute(state, move):
    newState = state.copy()
    pile = move[0]
    previousChips = newState[pile - 1]
    newChips = previousChips - move[1]
    newState[move[0] - 1] = newChips
    return newState
def getNewGameChips():
    return [3,5,7]
def generateAllScenarios():
    scenarios = []
    for i in range(4):
        for k in range(6):
            for q in range(8):
                scenarios.append([i, k, q])
    return scenarios
def generatePossibleMoves(state):  # Will this function go into the Agent Class?
    q = []
    if state == [0,0,0]:
        q.append((0,0))
        return q
    for i, x in enumerate(state):
        j = 1
        while j <= x:
            b = (i + 1, j)
            q.append(b)
            j = j + 1
    return q
def nim_sum():
    state = [3, 5, 7]
    nim = 0
    SumZeroNimStages = []

    for i, x in enumerate(total_game_scenarios):
        nim = 0
        rockList = x
        # Calculate nim sum for all elements in the rockList
        for i in rockList:
            nim = nim ^ i

            # Determine how many rocks to remove from which pile
        stones_to_remove = max(rockList) - nim
        stones_to_remove = abs(stones_to_remove)

        if nim == 0 and rockList not in SumZeroNimStages:
            SumZeroNimStages.append(rockList)

    # Note that there are 4*6*8=192 possible scenarios.
    # print("Sum Zero Stages: ", SumZeroNimStages)
    # print("Number of Zero sum Stages = ", len(SumZeroNimStages))
    return SumZeroNimStages
def analyzeResults(agent):
    nimSumZero = []
    agentStatesAfterMoves = []
    totalNumCoincidencesWithNimSumZero = 0
    for i, x in enumerate(agent.strategy):
        agentStatesAfterMoves.append(moveExecute(state=total_game_scenarios[i], move=x))
    nim_states = nim_sum()
    check = False
    for i, x in enumerate(nim_states):
        check = False
        for a, b in enumerate(agentStatesAfterMoves):
            if x == b:
                check = True
        if check == True:
            totalNumCoincidencesWithNimSumZero += 1
            nimSumZero.append(x)
            #print(x)
    if totalNumCoincidencesWithNimSumZero >= 20:
        print(nimSumZero)
    #print("total num Coincidences: ", totalNumCoincidencesWithNimSumZero)
    return totalNumCoincidencesWithNimSumZero
def init_population():
    return [Agent() for _ in range(population)]
def plotData(y1, y2,generation=0):

    if generation == 0:
        generation=generations

    xdata1 = list(range(0,generation))
    figure(figsize=(10, 7))

    plt.subplot(2, 1, 1)
    plt.plot(xdata1, y1, 'o-')
    plt.title('{0} Random Games \n{1} Games against other agents \n {2} Mutation Parameter\n {3} Selected for reproduccion '.format(numberOfRandomGames,nunberOfGamesAgainstAgents,mutationParaneter,numOfSelectedIndForReproduction))
    plt.ylabel('Fitness of best agent')


    plt.subplot(2, 1, 2)
    plt.plot(xdata1, y2, '.-')
    plt.xlabel('Generations')
    plt.ylabel('Nim Zero Coincidences')

    plt.savefig('Graph#.png')
    plt.show()

def ga():

    agents = init_population() #Contains a list of agents.
    newAgentsForReproduction = []
    dataForPlottingFitness =[]
    dataForPlottingNimSumZero =[]
    num_ofCoincidences = 0

    for generation in range(generations):
        print("Gen:",generation)


        agents = fitness1_2Combined(agents)

        num_ofCoincidences=(analyzeResults(agents[0]))
        print("Coincidence",num_ofCoincidences)
        dataForPlottingFitness.append(agents[0].fitness)
        dataForPlottingNimSumZero.append(analyzeResults(agents[0]))


        newAgentsForReproduction = selection2_TournamentSelection(agents) #Only a few are selected for rep.
        agents = crossover(agents=agents, selectedIndividuals=newAgentsForReproduction)
        agents = mutation(agents)
    print("strategy:", agents[0].strategy)
    print("scenarios:", total_game_scenarios)
    plotData(y1=dataForPlottingFitness, y2=dataForPlottingNimSumZero)



def gaUntilFitnessEqual1():

    agents = init_population() #Contains a list of agents.
    newAgentsForReproduction = []
    dataForPlottingFitness =[]
    dataForPlottingNimSumZero =[]
    newGeneration = 0
    num_ofCoincidences = 0

    while num_ofCoincidences <= 22:
        print("Gen",newGeneration)
        print(num_ofCoincidences)
        newGeneration += 1

        agents = fitness1_2Combined(agents)

        dataForPlottingFitness.append(agents[0].fitness)
        dataForPlottingNimSumZero.append(analyzeResults(agents[0]))
        num_ofCoincidences=(analyzeResults(agents[0]))

        newAgentsForReproduction = selection2_TournamentSelection(agents) #Only a few are selected for rep.
        agents = crossover(agents=agents, selectedIndividuals=newAgentsForReproduction)
        agents = mutation(agents)


    generations = newGeneration
    plotData(y1=dataForPlottingFitness, y2=dataForPlottingNimSumZero, generation=newGeneration)


#Exectution of the program:
#Population is the initial number of agents.
#Generations is how many times the loop is going to run.


total_game_scenarios = generateAllScenarios()

#gaUntilFitnessEqual1()
ga()


