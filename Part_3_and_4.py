# for quick access, to get the location of each sentiment in the dicentraintionary
from data_sets import training_set, dev_in, dev_out

def new_maximum_likelihood_estimator(f,k=0.5):
    ycount={}
    y_to_x_count={}
    emission_x_given_y={}
    for line in f:
        if line.strip() != "":
            line = line.split(" ")
            y=line[1]
            x=line[0]
            if y in ycount:
                ycount[y]+=1
            else:
                ycount[y]=1
            
            ytox=tuple((y,x))
            if ytox in y_to_x_count:
                y_to_x_count[ytox]+=1
            else:
                y_to_x_count[ytox]=1

    for key in y_to_x_count.keys():
        emission_x_given_y[key]=y_to_x_count[key]/(ycount[key[0]]+k)
    for key in ycount.keys():
        transition=tuple((key,"#UNK#"))
        emission_x_given_y[transition]=k/(ycount[key]+k)

    return emission_x_given_y, ycount

from Eval.evalResult import get_observed, get_predicted,compare_observed_to_predicted

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

states = 0
def train(language):
    global states
    FILE = training_set(language)
    ycount = y_counter(FILE)
    states = list(ycount.keys())
    
    emission_counter = tuple([{} for i in range(len(states))])
    emission_parameter = tuple([{} for i in range(len(states))])
    transmission_parameter = [[0]*(len(states)+1) for i in range(len(states))]
    states.append("STOP")
    counter = 0
    dic = {}
    for state in states:
        dic[state] = counter
        counter +=1
    dic["STOP"] = counter 
    u = 'START'
    observation_space = set()
    count = [0] * (len(states)-1)
    for observation in FILE:
        try:
            observation, v = observation.split()
            observation = observation.strip()
            v = v.strip()

            position = dic[v]
            if (observation in emission_counter[position]):
                emission_counter[position][observation] += 1
            else:
                emission_counter[position][observation] = 1


            pre_position = dic[u]
            transmission_parameter[pre_position][position] += 1
            u = v
            if observation not in observation_space:
                observation_space.add(observation)
        except:
            pre_position = dic[u]
            transmission_parameter[pre_position][len(states)-1] += 1
            u = 'START'

    for i in range(0, len(states)-1):
            temp_sum = 0
            for j in range(0, len(states)):
                temp_sum = temp_sum + transmission_parameter[i][j]
            count[i] = temp_sum + 1

    for i in range(0, len(states)-1):
        for j in range(0, len(states)-1):
            transmission_parameter[i][j] = 1.0 * transmission_parameter[i][j] / count[i]
    for i in range(len(states)-1):
        for observation in observation_space:
            if observation not in emission_counter[i]:
                emission_parameter[i][observation] = 0 / max(count)
            else:
                emission_parameter[i][observation] = 1.0 * emission_counter[i][observation] / count[i]
    return observation_space,emission_parameter, transmission_parameter, count

#Part 3 Begins Here

def forward(preScore, x, language, num_of_states):
    layer = []
    for i in range(1, num_of_states):
        temp_score = []

        if (x in observation_space):
            b = emission_parameter[i][x]
        else:
            b = 1.0 / count[i]
        for j in range(1, num_of_states):
            j_score = preScore[j-1][0] * (transmission_parameter[j][i]) * b  # trans 1~7 -> 1-7
            temp_score.append(j_score)
        max_value = max(temp_score)
        max_index = temp_score.index(max_value)
        layer.append((max_value, max_index))
    return layer

#Part 4 has DP in the function
def forwardDP(prev_layer, x, language, k,num_of_states):
    """inputs: prev_layer: list of list of top k best [score, partent_index (0, 6), parent_sub (0, k-1)] for all states
               x: current word
               k: top k best
       output: list of top k best [score, partent_index (0, 6), parent_sub (0, k-1)] for all states, len=7
    """

    layer = []
    for i in range(1, num_of_states):
        temp_score = []
        states = []
        n = len(prev_layer[0])
        # calculate emission first
        if (x in observation_space):
            b = emission_parameter[i][x]
        else:
            b = 1.0 / count[i]
        for j in range(1, num_of_states):  # j:1-7
            for sub in range(0, n):
                j_score = prev_layer[j-1][sub][0] * (transmission_parameter[j][i]) * b
                temp_score.append([j_score, j-1, sub])
        temp_score.sort(key=lambda tup:tup[0],reverse=True)
        for sub in range(0, k):
            states.append(temp_score[sub])
        layer.append(states)
    return layer


#for Part 3
def viterbiAlgo(X, language, num_of_states):

    n = len(X)
    Final_sequence = []
    prev_layer = []
    x = X[0]
    for j in range(1, num_of_states):
        if (x in observation_space):
            b = emission_parameter[j][x]
        else:
            b = 1.0 / count[j]
        probability = transmission_parameter[0][j] * b
        prev_layer.append((probability, 0))
    layers = [[(1,-1)],prev_layer]


    for i in range(1, n):
        score = forward(layers[i], X[i], language,num_of_states) 
        layers.append(score)


    temp_score = []
    for j in range(1, num_of_states):
        t_score = layers[n][j-1][0] * (transmission_parameter[j][num_of_states])
        temp_score.append(t_score)
    max_value = max(temp_score)
    max_index = temp_score.index(max_value)
    layers.append([(max_value, max_index)])

    parent = 0 
    for i in range(n+1, 1, -1):
        parent = layers[i][parent][1]
        Final_sequence.insert(0, states[parent + 1])
    return Final_sequence


#for Part 4
def viterbiAlgoDP(X, language, k, num_of_states):
    prev_layer = []
    n = len(X)
    x = X[0]
    Y = []

    for j in range(1, num_of_states):
        state = []
        if (x in observation_space):
            b = emission_parameter[j][x]
        else:
            b = 1.0 / count[j]
        probability = transmission_parameter[0][j] * b
        state.append([probability, 0, 0])
        prev_layer.append(state)
    layers = [[(1, -1, 0)], prev_layer]


    for i in range(1, n): 
        layer = forwardDP(layers[i], X[i], language, k,num_of_states)
        layers.append(layer)



    layer = []
    temp_score = []
    viterbi_states = []
    failed = False
    for j in range(1, num_of_states):
        for sub in range(0, len(layers[n][0])):
            t_score = layers[n][j - 1][sub][0] * (transmission_parameter[j][num_of_states])
            temp_score.append([t_score, j - 1, sub])

    temp_score.sort(key=lambda tup: tup[0], reverse=True)
    for sub in range(0, k):  # get top k best
        viterbi_states.append(temp_score[sub])
    layer.append(viterbi_states)
    layers.append(layer)



    parent_index = 0
    parent_sub = k-1
    for i in range(n+1, 1, -1): 
        a = layers[i][parent_index][parent_sub][1]
        b = layers[i][parent_index][parent_sub][2]

        Y.insert(0, states[a + 1])
        parent_index = a
        parent_sub = b

    return Y


def runPart3(language,observation_space, emission_parameter, transmission_parameter, count):
    dev_file = open(language+'/dev.in', 'r', encoding= 'utf-8')
    out_file = open(language+'/dev.p3.out', 'w',  encoding= 'utf-8')
    X = []
    for r in dev_file:
        r = r.strip()
        if (r == ''):

            Y = viterbiAlgo(X,language,len(transmission_parameter))
            for i in range(0, len(X)):
                out_file.write('' + X[i] + " " + Y[i] + '\n')
            out_file.write('\n')
            X = []
        else:
            X.append(r)

def runPart4(language,observation_space, emission_parameter, transmission_parameter, count, k):
    dev_file = open(language+'/dev.in', 'r', encoding= 'utf-8')
    out_file = open(language+'/dev.p4.out', 'w', encoding= 'utf-8')
    X = []
    for r in dev_file:
        r = r.strip()
        if (r == ''):
            Y = viterbiAlgoDP(X, language, k, len(transmission_parameter))
            for i in range(0, len(X)):
                out_file.write('' + X[i] + " " + Y[i] + '\n')
            out_file.write('\n')
            X = []
        else:
            X.append(r)

from Eval.evalResult import get_observed, get_predicted,compare_observed_to_predicted

#if its part 3, we set part=3 to get the results for part 3, else we set part=4 to get the 3rd best result as required.
part = 3  #change to part 4 for part 4


#simply click run
for language in ["EN","CN","SG"]:

    print ("\n...Running part {} on ".format(part) + language+ "...")
    observation_space, emission_parameter, transmission_parameter, count = train(language)

    if part == 3:
        runPart3(language,observation_space, emission_parameter, transmission_parameter, count)
        output = open(language+ "/dev.p3.out", encoding="utf-8")
    else:
        runPart4(language,observation_space, emission_parameter, transmission_parameter, count, 3)
        output = open(language+ "/dev.p4.out", encoding="utf-8")


    #results
    print("#results for " + language)
    compare_observed_to_predicted(get_observed(dev_out(language)),get_predicted(output))