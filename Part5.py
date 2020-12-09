import numpy as np
import unicodedata
import sys
############################# initialize parameter ####################################
dic = {'PRESTART':0, 'START':1, 'B-NP' : 2 , 'I-NP' :3, 'B-VP':4, 'B-ADVP':5, 'B-ADJP':6, 'I-ADJP':7, 'B-PP':8, 'O':9, 'B-SBAR':10, 'I-VP':11, 'I-ADVP':12, 'B-PRT':13, 'I-PP':14, 'B-CONJP':15, 'I-CONJP':16, 'B-INTJ':17, 'I-INTJ':18, 'I-SBAR':19, 'B-UCP':20, 'I-UCP':21, 'B-LST':22, 'STOP':23, 'POSTSTOP':24}
l = ['PRESTART','START', 'B-NP', 'I-NP', 'B-VP', 'B-ADVP', 'B-ADJP', 'I-ADJP', 'B-PP', 'O', 'B-SBAR', 'I-VP', 'I-ADVP', 'B-PRT', 'I-PP', 'B-CONJP', 'I-CONJP', 'B-INTJ', 'I-INTJ', 'I-SBAR', 'B-UCP', 'I-UCP', 'B-LST', 'STOP', 'POSTSTOP']

# store emission parameters
# data structure: tuple + dictionary
emission_parameter = ({}, {}, {}, {}, {}, {}, {}, {},{}, {}, {}, {}, {}, {}, {}, {},{}, {}, {}, {}, {}, {}, {})  ## 1st,2nd dict empty
observation_space = set()

# store transition parameters
# initialize as a 11*11*11 matrix of zeros
a = 25
transmission_parameter = np.zeros((25, 25, 25))

# b_inSpace = float(sys.argv[1])
# b_notInSpace = float(sys.argv[2])  # 1/count+1
b_inSpace = 0.1
b_notInSpace = 1
# num of iteration over the training set
T = 1


def forward(preScore, x):
    """
    This function takes in the list of scores and parents, 
    """
    """inputs: preScore: list of (pre_score real_num, pre_parent int)
               x: current word
       output: list of max(score real_num, parent int) for all states, len=7
    """

    # k j i
    layer = []
    for i in range(2, 23):  # i: 2~8
        temp_score = []
        # calculate emission first
        if ((x in observation_space) & (x in emission_parameter[i])):
            b = emission_parameter[i][x]
        elif (x in observation_space):
            b = b_inSpace                                       # !!!!!!!!!!!!!!!!!!!!!!!!!!
        else:
            b = b_notInSpace
        for j in range(2, 23):  #
            # score = preScore*a*b
            for k in range(2, 23):
                kj_score = preScore[j - 2][0] + transmission_parameter[k][j][i] + b  # trans 2~8 -> 2-8
                temp_score.append(kj_score)
        max_value = max(temp_score)
        max_index = temp_score.index(max_value) / 21
        layer.append((max_value, max_index + 2)) # index: 0-6 + 2
    return layer


def viterbiAlgo(X):
    # initialization
    n = len(X)
    Y = []
    prev_layer = []

    # prestart, start -> 1
    x = X[0]
    for j in range(2, 23): # 2-8
        if ((x in observation_space) & (x in emission_parameter[j])):
            b = emission_parameter[j][x]
        elif (x in observation_space):
            b = b_inSpace                                                        # to be tuned
        else:
            b = b_notInSpace
        probability = transmission_parameter[0][1][j] + b   # prestart * start * y1
        prev_layer.append((probability, 1))  # (probability, PRESTART, START)            ????
    layers = [[(1, -1)], prev_layer]


    # start, 1 -> 2
    if len(X) > 1:
        x = X[1]
        layer = []
        for j in range(2, 23): # 2-8
            temp_score = []
            if ((x in observation_space) & (x in emission_parameter[j])):
                b = emission_parameter[j][x]
            elif (x in observation_space):
                b = b_inSpace                                                           # to be tuned
            else:
                b = b_notInSpace
            for k in range(2,23):
                temp_score.append(transmission_parameter[1][k][j] + b)   #  start + y1 + y2
            max_value = max(temp_score)
            max_index = temp_score.index(max_value)
            layer.append((max_value, max_index + 2))
        layers.append(layer)  #


    # calculate path i=(2,...,n)

    for i in range(2, n):  # 2 -> n-1
        score = forward(layers[i], X[i])  # a list of max(score: real, parent: int) for all 7 states
        layers.append(score)


    # calculate score(n+1, STOP), and get max
    temp_score = []
    for j in range(2, 23):
        for k in range(2,23):
            # score = preScore*a
            kj_score = layers[n][j-2][0] + (transmission_parameter[k][j][dic['STOP']])
            temp_score.append(kj_score)
    max_value = max(temp_score)
    max_index = temp_score.index(max_value) / 21
    layers.append([(max_value, max_index + 2)])
    # print(scores)

    # backtracking
    parent = 2  # only 1 entry in STOP
    for i in range(n + 1, 1, -1):  # index range from N to 2
        parent = int(layers[i][parent-2][1]) # 0-6
        
        Y.insert(0, l[parent])  # 2-8
    # print(Y)
    return Y


def updateParam(XGOLD, YGOLD, Ytrain): 
    for i in range(2, len(YGOLD)):
        transmission_parameter[dic[YGOLD[i - 2]]][dic[YGOLD[i - 1]]][dic[YGOLD[i]]] += 1                     #UPDATES 
        transmission_parameter[dic[Ytrain[i - 2]]][dic[Ytrain[i - 1]]][dic[Ytrain[i]]] -= 1

    for i in range(2, len(YGOLD) - 2):
        if (XGOLD[i - 2] in emission_parameter[dic[YGOLD[i]]]): #IF observationervation in emission_parameter[dic[ygold[i]]]
            emission_parameter[dic[YGOLD[i]]][XGOLD[i - 2]] += 1
        elif (XGOLD[i - 2] in observation_space): #IF observationervation in emission_parameter[dic[ygold[i]]] but in observationervation space
            emission_parameter[dic[YGOLD[i]]][XGOLD[i - 2]] = 1
        else:
            emission_parameter[dic[YGOLD[i]]][XGOLD[i - 2]] = 1
            observation_space.add(XGOLD[i - 2])

    for i in range(2, len(YGOLD) - 2):
        if (XGOLD[i - 2] in emission_parameter[dic[Ytrain[i]]]):
            emission_parameter[dic[Ytrain[i]]][XGOLD[i - 2]] -= 1
        elif (XGOLD[i - 2] in observation_space):
            emission_parameter[dic[Ytrain[i]]][XGOLD[i - 2]] = -1
        else:
            emission_parameter[dic[Ytrain[i]]][XGOLD[i - 2]] = -1
            observation_space.add(XGOLD[i - 2])



def train(language):
    ############################# initialize parameter ####################################
    """T number of iterations"""
    for trainStep in range(T):  
        print ('Iteration: ', trainStep)
        ## read and parse file
        train_file = open(language+'/train', 'r', encoding='utf-8')
        Ygold = ['PRESTART', 'START'] #best sequence of states so far
        X = []

        for observation in train_file:
            try:
                observation, v = observation.split()
                observation = observation.strip()
                
                # observation = preprocess(observation)
                v = v.strip()
                X.append(observation)
                Ygold.append(v)

            except:  #will only have an exception when sentence has been extracted.
                # meaning the end of a sentence: x->STOP
                Ygold.extend(['STOP', 'POSTSTOP'])
                Ytrain = ['PRESTART', 'START']
                Ytrain.extend(viterbiAlgo(X))
                Ytrain.extend(['STOP', 'POSTSTOP'])

                updateParam(X, Ygold, Ytrain) 

                # reset
                Ygold = ['PRESTART', 'START']
                X = []


def runPerceptron_MAI_GEY_KIANG_LAH(language):
    dev_file = open(language+'/dev.in', 'r', encoding='utf-8')
    out_file = open(language+'/dev.p5.out', 'w', encoding='utf-8')
    # dev_file = open(language+'/test.in', 'r')
    # out_file = open(language+'/test.out', 'w')
    X = []
    for r in dev_file:
        r = r.strip()
        if (r == ''):
            # end of a sequence
            Y = viterbiAlgo(X)
            for i in range(0, len(X)):
                out_file.write('' + X[i] + " " + Y[i] + '\n')
            out_file.write('\n')
            X = []
        else:
            # r = preprocess(r)
            X.append(r)


for language in [ 'EN']:
    train(language)
    runPerceptron_MAI_GEY_KIANG_LAH(language)
