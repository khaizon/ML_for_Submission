import pprint
pp = pprint.PrettyPrinter(indent=5)

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

def transition_estimator(f):
    ycount = {"START" : 0}
    transition_y_count = {}
    transition_prob = {}
    prev_state = ""
    counter = 0
    for line in f:
        
        if prev_state == "":
            if len(line.split(" ")) > 1:
                y = line.split(" ")[1]
                if tuple(("START", y)) in transition_y_count:
                    transition_y_count[tuple(("START", y))] += 1
                else:
                    transition_y_count[tuple(("START", y))] = 1
                if y in ycount:
                    ycount[y] +=1
                    ycount["START"] +=1
                else:
                    ycount[y] = 1
                    ycount["START"] +=1
                prev_state = y

        else:
            if len(line.split(" ")) > 1:
                y = line.split(" ")[1] 
                if tuple((prev_state, y)) in transition_y_count:
                    transition_y_count[tuple((prev_state, y))] += 1
                else:
                    transition_y_count[tuple((prev_state, y))] = 1
                if y in ycount:
                    ycount[y] +=1
                else:
                    ycount[y] = 1
                prev_state = y
            else:
                if tuple((prev_state, "STOP")) in transition_y_count:
                    transition_y_count[tuple((prev_state, "STOP"))] += 1
                else:
                    transition_y_count[tuple((prev_state, "STOP"))] = 1
                

                prev_state = ""
        
    for key in transition_y_count.keys():
        transition_prob[key] = transition_y_count[key]/ycount[key[0]]
    return ycount
states = 0
def train(language):
    global states
    FILE = training_set(language)
    ycount = transition_estimator(FILE)

    states = list(ycount.keys())
    
    e_count = tuple([{} for i in range(len(states))])
    e_param = tuple([{} for i in range(len(states))])
    t_param = [[0]*(len(states)+1) for i in range(len(states))]
    states.append("STOP")
    counter = 0
    dic = {}
    for state in states:
        dic[state] = counter
        counter +=1
    dic["STOP"] = counter 
    u = 'START'
    obs_space = set()
    count = [0] * (len(states)-1)
    for obs in FILE:
        try:
            obs, v = obs.split() #obs , state
            obs = obs.strip()
            v = v.strip()
            if language == "EN":
                position = dic[v]  ## position: 1~7
            else:
                position = dic[v]
            # update e_count
            if (obs in e_count[position]):
                e_count[position][obs] += 1
            else:
                e_count[position][obs] = 1

            # update t_param
            pre_position = dic[u]
            t_param[pre_position][position] += 1
            u = v
            # add into train_obs_set
            if obs not in obs_space:
                obs_space.add(obs)
        except:
            pre_position = dic[u]
            t_param[pre_position][len(states)-1] += 1
            u = 'START'

    for i in range(0, len(states)-1):
            temp_sum = 0
            for j in range(0, len(states)):
                temp_sum = temp_sum + t_param[i][j]
            count[i] = temp_sum + 1
    ## convert transision param to probablity
    for i in range(0, len(states)-1):
        for j in range(0, len(states)-1):
            t_param[i][j] = 1.0 * t_param[i][j] / count[i]

    # building emission params table: a list of 8 dicentraints, each dicentraint has all obs as keys,
    # value is 0 if obs never appears for this state

    for i in range(len(states)-1): # state 1-7
        for obs in obs_space:
            if obs not in e_count[i]:
                e_param[i][obs] = 0 / max(count) ## ???????????????????????????????????????????????????????????????????????????????????????? whether should be 0?? or lowest prob of all
            else:
                e_param[i][obs] = 1.0 * e_count[i][obs] / count[i]
    return obs_space,e_param, t_param, count

#main function
# for language in ["EN"]:
#     test_input = dev_in(language)
#     best_states = getmaxy(new_maximum_likelihood_estimator(training_set(language))[0])
#     print(new_maximum_likelihood_estimator(training_set(language))[0])
#     finaloutput(test_input,best_states, language)
#     output = open(language+ "/dev.p2.out", encoding="utf-8")

#     #results
#     print("\n#results for " + language)
#     compare_observed_to_predicted(get_observed(dev_out(language)),get_predicted(output))


############################### Part 3 ######################################

def forward(preScore, x, language, num_of_states):
    """inputs: preScore: list of (pre_score real_num, pre_parent int)
               x: current word
       output: list of max(score real_num, parent int) for all states, len=7
    """

    layer = []
    for i in range(1, num_of_states):  # i: 1~7
        temp_score = []
        # calculate emission first
        if (x in obs_space):
            b = e_param[i][x]
        else:
            b = 1.0 / count[i]
        for j in range(1, num_of_states):  # j:1-7
            # score = preScore*a*b
            j_score = preScore[j-1][0] * (t_param[j][i]) * b  # trans 1~7 -> 1-7
            temp_score.append(j_score)
        max_value = max(temp_score)
        max_index = temp_score.index(max_value)  # index: 0-6
        layer.append((max_value, max_index))
    return layer


def forwardDP(prev_layer, x, language, k,num_of_states):
    """inputs: prev_layer: list of list of top k best [score, partent_index (0, 6), parent_sub (0, k-1)] for all states
               x: current word
               k: top k best
       output: list of top k best [score, partent_index (0, 6), parent_sub (0, k-1)] for all states, len=7
    """

    layer = []
    for i in range(1, num_of_states):  # i: 1~7
        temp_score = []
        states = []
        n = len(prev_layer[0])
        # calculate emission first
        if (x in obs_space):
            b = e_param[i][x]
        else:
            b = 1.0 / count[i]
        for j in range(1, num_of_states):  # j:1-7
            for sub in range(0, n): # n scores for each prev_node
                # score = prev_layer*a*b
                j_score = prev_layer[j-1][sub][0] * (t_param[j][i]) * b
                temp_score.append([j_score, j-1, sub])  # 7*n scores with their parents
        temp_score.sort(key=lambda tup:tup[0],reverse=True) # sort by j_score
        for sub in range(0, k):   # get top k best
            states.append(temp_score[sub])
        layer.append(states)
    return layer



def viterbiAlgo(X, language, num_of_states):
    """input: X, words list
       output: Y, sentiment list
    """

    # initialization
    n = len(X)
    Y = []
    prev_layer = []
    # start -> 1
    x = X[0]
    for j in range(1, num_of_states):
        if (x in obs_space):
            b = e_param[j][x]
        else:
            b = 1.0 / count[j]
        prob = t_param[0][j] * b
        prev_layer.append((prob, 0))  # (prob, START)
    layers = [[(1,-1)],prev_layer]

    # calculate path i=(1,...,n)
    for i in range(1, n):  # 1 -> n-1
        score = forward(layers[i], X[i], language,num_of_states)  # a list of max(score: real, parent: int) for all 7 states
        layers.append(score)

    # calculate score(n+1, STOP), and get max
    temp_score = []
    for j in range(1, num_of_states):
        # score = preScore*a
        t_score = layers[n][j-1][0] * (t_param[j][num_of_states])
        temp_score.append(t_score)
    max_value = max(temp_score)
    max_index = temp_score.index(max_value)
    layers.append([(max_value, max_index)])
    # pp.pprint(scores)

    # backtracking
    parent = 0  # only 1 entry in STOP
    for i in range(n+1, 1, -1):  # index range from N to 2
        parent = layers[i][parent][1]
        if language == "EN":
            Y.insert(0, states[parent + 1])  # 1-7
        else:
            Y.insert(0, states[parent + 1])  # 1-7
    # print(Y)
    return Y



def viterbiAlgoDP(X, language, k, num_of_states):
    """input:  X, words list
               k, top k best
       output: Y, sentiment list
    """
    # initialization
    n = len(X)
    Y = []
    prev_layer = []
    # calculate layer (start ->) 1
    x = X[0]
    for j in range(1, num_of_states):
        state = []
        if (x in obs_space):
            b = e_param[j][x]
        else:
            b = 1.0 / count[j]
        prob = t_param[0][j] * b
        state.append([prob, 0, 0])  # [prob, START, 1st best]
        prev_layer.append(state)
    layers = [[(1, -1, 0)], prev_layer]
    # pp.pprint(prev_layer)


    # calculate layer (2,...,n)
    for i in range(1, n):  # prev_layer: 1 -> n-1
        layer = forwardDP(layers[i], X[i], language, k,num_of_states)  # a list of top k best scores for all 7 states
        layers.append(layer)
        # pp.pprint("--------layer "+ str(i)+"----------")
        # pp.pprint(layer)


    # calculate layer n+1 (STOP), and get top k best
    layer = []
    temp_score = []
    viterbi_states = []
    failed = False
    for j in range(1, num_of_states):  # j:1-7
        for sub in range(0, len(layers[n][0])):  # kth score for each prev_node
            # score = prev_layer*a
            t_score = layers[n][j - 1][sub][0] * (t_param[j][num_of_states]) # TODO: for ES data set, index out of range caused by sub, for sub = 1,2,3,4
            temp_score.append([t_score, j - 1, sub])  # 7*k scores with thier parents

    temp_score.sort(key=lambda tup: tup[0], reverse=True)  # sort by j_score
    for sub in range(0, k):  # get top k best
        viterbi_states.append(temp_score[sub])
    layer.append(viterbi_states)
    layers.append(layer)
    # pp.pprint(layer)
    # pp.pprint(layers)

    # backtracking
    parent_index = 0    # only 1 state in STOP
    parent_sub = k-1   # kth best score in STOP layer
    for i in range(n+1, 1, -1):  # index range from N to 2
        a = layers[i][parent_index][parent_sub][1]
        b = layers[i][parent_index][parent_sub][2]

        Y.insert(0, states[a + 1])  # 1-7
        parent_index = a
        parent_sub = b
    # print(Y)
    return Y


def runPart3(language,obs_space, e_param, t_param, count):
    dev_file = open(language+'/dev.in', 'r', encoding= 'utf-8')
    out_file = open(language+'/dev.p3.out', 'w',  encoding= 'utf-8')
    X = []
    for r in dev_file:
        r = r.strip()
        if (r == ''):
            # end of a sequence
            Y = viterbiAlgo(X,language,len(t_param))
            for i in range(0, len(X)):
                out_file.write('' + X[i] + " " + Y[i] + '\n')
            out_file.write('\n')
            X = []
        else:
            X.append(r)

def runPart4(language,obs_space, e_param, t_param, count, k):
    dev_file = open(language+'/dev.in', 'r', encoding= 'utf-8')
    out_file = open(language+'/dev.p4.out', 'w', encoding= 'utf-8')
    X = []
    for r in dev_file:
        r = r.strip()
        if (r == ''):
            # end of a sequence
            Y = viterbiAlgoDP(X, language, k, len(t_param))
            for i in range(0, len(X)):
                out_file.write('' + X[i] + " " + Y[i] + '\n')
            out_file.write('\n')
            X = []
        else:
            X.append(r)

from Eval.evalResult import get_observed, get_predicted,compare_observed_to_predicted

for language in ["EN","CN","SG"]:
# for language in ["EN"]:
    print ("Doing " + language)
    obs_space, e_param, t_param, count = train(language)
    # runPart2(language,obs_space, e_param, count)
    runPart3(language,obs_space, e_param, t_param, count)



    output = open(language+ "/dev.p3.out", encoding="utf-8")

    #results
    print("\n#results for " + language)
    compare_observed_to_predicted(get_observed(dev_out(language)),get_predicted(output))