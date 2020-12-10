import numpy as np
import unicodedata
import sys
from data_sets import challenge_set,training_set
import time, datetime

def y_counter(f):
    ycount = {"START" : 0}
    previous_state = ""
    counter = 0
    for line in f:
        
        if previous_state == "":
            if len(line.split(" ")) > 1:
                y = line.split(" ")[1]
                if y in ycount:
                    ycount[y] +=1
                    ycount["START"] +=1
                else:
                    ycount[y] = 1
                    ycount["START"] +=1
                previous_state = y

        else:
            if len(line.split(" ")) > 1:
                y = line.split(" ")[1] 
                if y in ycount:
                    ycount[y] +=1
                else:
                    ycount[y] = 1
                previous_state = y
    return ycount

ycount = y_counter(training_set("EN"))

############################# initializing parameters ####################################
list_of_states = list(ycount.keys())
list_of_states.insert(0,'START1')
list_of_states[1] = 'START2'
list_of_states.append('STOP1')
list_of_states.append('STOP2')

possible_states = {}
counter = 0
for state in list_of_states:
    possible_states[state] = counter
    counter +=1

emission_parameter = tuple([{} for i in range(len(list_of_states)-2)])
observation_space = set()

transmission_parameter = np.zeros((len(list_of_states), len(list_of_states), len(list_of_states)))

iterations = 50

def forward(preScore, x):

    layer = []
    for i in range(2, 23): 
        temp_score = []

        if ((x in observation_space) & (x in emission_parameter[i])):
            b = emission_parameter[i][x]
        elif (x in observation_space):
            b = 0.1   
        else:
            b = 1
        for j in range(2, 23):  #
            # score = preScore*a*b
            for k in range(2, 23):
                kj_score = preScore[j - 2][0] + transmission_parameter[k][j][i] + b 
                temp_score.append(kj_score)
        max_value = max(temp_score)
        max_index = temp_score.index(max_value) / 21
        layer.append((max_value, max_index + 2))
    return layer


def viterbiAlgo(X):

    n = len(X)
    Y = []
    prev_layer = []


    x = X[0]
    for j in range(2, 23):
        if ((x in observation_space) & (x in emission_parameter[j])):
            b = emission_parameter[j][x]
        elif (x in observation_space):
            b = 0.1                                                       
        else:
            b = 1
        probability = transmission_parameter[0][1][j] + b
        prev_layer.append((probability, 1))
    layers = [[(1, -1)], prev_layer]



    if len(X) > 1:
        x = X[1]
        layer = []
        for j in range(2, 23):
            temp_score = []
            if ((x in observation_space) & (x in emission_parameter[j])):
                b = emission_parameter[j][x]
            elif (x in observation_space):
                b = 0.1                                                          
            else:
                b = 1
            for k in range(2,23):
                temp_score.append(transmission_parameter[1][k][j] + b)  
            max_value = max(temp_score)
            max_index = temp_score.index(max_value)
            layer.append((max_value, max_index + 2))
        layers.append(layer)  #



    for i in range(2, n):
        score = forward(layers[i], X[i])
        layers.append(score)

    temp_score = []
    for j in range(2, 23):
        for k in range(2,23):

            kj_score = layers[n][j-2][0] + (transmission_parameter[k][j][possible_states['STOP1']])
            temp_score.append(kj_score)
    max_value = max(temp_score)
    max_index = temp_score.index(max_value) / 21
    layers.append([(max_value, max_index + 2)])

    parent = 2
    for i in range(n + 1, 1, -1):
        parent = int(layers[i][parent-2][1])
        
        Y.insert(0, list_of_states[parent])
    return Y


def update_parameters(XGOLD, YGOLD, Ytrain): 
    # concept: make transitions and emissions observed in trainingset more likely
    # while making transitions and emissions observed in viterbi output less likely

    # this first for loop is for emission parameters
    for i in range(2, len(YGOLD) - 2):
        if (XGOLD[i - 2] in emission_parameter[possible_states[YGOLD[i]]]): 
            emission_parameter[possible_states[YGOLD[i]]][XGOLD[i - 2]] += 1
        elif (XGOLD[i - 2] in observation_space): 
            emission_parameter[possible_states[YGOLD[i]]][XGOLD[i - 2]] = 1
        else:
            emission_parameter[possible_states[YGOLD[i]]][XGOLD[i - 2]] = 1
            observation_space.add(XGOLD[i - 2])

    # this second for loop is for transition parameters
    for i in range(2, len(YGOLD) - 2):
        if (XGOLD[i - 2] in emission_parameter[possible_states[Ytrain[i]]]):
            emission_parameter[possible_states[Ytrain[i]]][XGOLD[i - 2]] -= 1
        elif (XGOLD[i - 2] in observation_space):
            emission_parameter[possible_states[Ytrain[i]]][XGOLD[i - 2]] = -1
        else:
            emission_parameter[possible_states[Ytrain[i]]][XGOLD[i - 2]] = -1
            observation_space.add(XGOLD[i - 2])
    for i in range(2, len(YGOLD)):
        transmission_parameter[possible_states[YGOLD[i - 2]]][possible_states[YGOLD[i - 1]]][possible_states[YGOLD[i]]] += 1                    
        transmission_parameter[possible_states[Ytrain[i - 2]]][possible_states[Ytrain[i - 1]]][possible_states[Ytrain[i]]] -= 1


def train(language):
    start_time = time.time()
    for iteration in range(iterations):  
        print ('   Iteration:  ', iteration)
        print("Time Elapsed:  ", time.strftime("%H:%M:%S", time.gmtime(time.time()-start_time)))

        training_set = open(language+'/train', 'r', encoding='utf-8')
        Ygold = ['START1', 'START2']
        Sentence = []

        for observation in training_set:
            try:
                observation, state_label = observation.split()
                observation = observation.strip()

                state_label = state_label.strip()
                Sentence.append(observation)
                Ygold.append(state_label)

            except:  
                Ygold.extend(['STOP1', 'STOP2'])
                Ytrain = ['START1', 'START2']
                Ytrain.extend(viterbiAlgo(Sentence))
                Ytrain.extend(['STOP1', 'STOP2'])

                update_parameters(Sentence, Ygold, Ytrain) 


                Ygold = ['START1', 'START2']
                Sentence = []


def perception_algo(language):
    input_file = challenge_set("EN")
    out_file = open(language+'/dev.test.out', 'w', encoding='utf-8')
    input_file = training_set("EN")
    out_file = open(language+'/dev.p5.out', 'w', encoding='utf-8')
    Sentence = []
    for line in input_file:
        line = line.strip()
        if (line == ''):
            Y = viterbiAlgo(Sentence)
            for i in range(0, len(Sentence)):
                out_file.write('' + Sentence[i] + " " + Y[i] + '\n')
            out_file.write('\n')
            Sentence = []
        else:

            Sentence.append(line)


for language in ['EN']:
    train(language)
    perception_algo(language)
