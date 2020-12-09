import numpy as np
import unicodedata
import sys
from data_sets import challenge_set
############################# initialize parameter ####################################
possible_states = {'START1':0, 'START2':1, 'B-NP' : 2 , 'I-NP' :3, 'B-VP':4, 'B-ADVP':5, 'B-ADJP':6, 'I-ADJP':7, 'B-PP':8, 'O':9, 'B-SBAR':10, 'I-VP':11, 'I-ADVP':12, 'B-PRT':13, 'I-PP':14, 'B-CONJP':15, 'I-CONJP':16, 'B-INTJ':17, 'I-INTJ':18, 'I-SBAR':19, 'B-UCP':20, 'I-UCP':21, 'B-LST':22, 'STOP1':23, 'STOP2':24}
list_of_states = ['START1','START2', 'B-NP', 'I-NP', 'B-VP', 'B-ADVP', 'B-ADJP', 'I-ADJP', 'B-PP', 'O', 'B-SBAR', 'I-VP', 'I-ADVP', 'B-PRT', 'I-PP', 'B-CONJP', 'I-CONJP', 'B-INTJ', 'I-INTJ', 'I-SBAR', 'B-UCP', 'I-UCP', 'B-LST', 'STOP1', 'STOP2']

emission_parameter = ({}, {}, {}, {}, {}, {}, {}, {},{}, {}, {}, {}, {}, {}, {}, {},{}, {}, {}, {}, {}, {}, {})  ## 1st,2nd possible_statest empty
observation_space = set()

states = 25
transmission_parameter = np.zeros((states, states, states))

iterations = 15


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


def updateParam(XGOLD, YGOLD, Ytrain): 
    for i in range(2, len(YGOLD)):
        transmission_parameter[possible_states[YGOLD[i - 2]]][possible_states[YGOLD[i - 1]]][possible_states[YGOLD[i]]] += 1                     #UPDATES 
        transmission_parameter[possible_states[Ytrain[i - 2]]][possible_states[Ytrain[i - 1]]][possible_states[Ytrain[i]]] -= 1

    for i in range(2, len(YGOLD) - 2):
        if (XGOLD[i - 2] in emission_parameter[possible_states[YGOLD[i]]]): #IF observationervation in emission_parameter[possible_states[ygold[i]]]
            emission_parameter[possible_states[YGOLD[i]]][XGOLD[i - 2]] += 1
        elif (XGOLD[i - 2] in observation_space): #IF observationervation in emission_parameter[possible_states[ygold[i]]] but in observationervation space
            emission_parameter[possible_states[YGOLD[i]]][XGOLD[i - 2]] = 1
        else:
            emission_parameter[possible_states[YGOLD[i]]][XGOLD[i - 2]] = 1
            observation_space.add(XGOLD[i - 2])

    for i in range(2, len(YGOLD) - 2):
        if (XGOLD[i - 2] in emission_parameter[possible_states[Ytrain[i]]]):
            emission_parameter[possible_states[Ytrain[i]]][XGOLD[i - 2]] -= 1
        elif (XGOLD[i - 2] in observation_space):
            emission_parameter[possible_states[Ytrain[i]]][XGOLD[i - 2]] = -1
        else:
            emission_parameter[possible_states[Ytrain[i]]][XGOLD[i - 2]] = -1
            observation_space.add(XGOLD[i - 2])



def train(language):

    for iteration in range(iterations):  
        print ('Iteration: ', iteration)

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

                updateParam(Sentence, Ygold, Ytrain) 


                Ygold = ['START1', 'START2']
                Sentence = []


def perception_algo(language):
    input_file = challenge_set("EN")
    out_file = open(language+'/dev.test.out', 'w', encoding='utf-8')
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

            X.append(line)


for language in ['EN']:
    train(language)
    perception_algo(language)
