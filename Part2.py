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


def getmaxy(get_e_y_x):
    maxy=0
    max_y_from_x = {}
    max_x_probability = {}
    for key, item in get_e_y_x.items():
        if key[1] not in max_x_probability:
            max_x_probability[key[1]] =item
            max_y_from_x[key[1]] = key[0]
        else:
            if max_x_probability[key[1]] < item:
                max_x_probability[key[1]] = item
                max_y_from_x[key[1]] = key[0]

    return max_y_from_x


def finaloutput(testinput, max_y_from_x, language):
    f= open(language+ "/dev.p2.out", "w", encoding="utf-8")
    for line in testinput:
        if len(line) >= 1:
            try:
                y = max_y_from_x[line]
            except:
                y = max_y_from_x["#UNK#"]
            f.write("{} {}\n".format(line,y))
        else:
            f.write("\n")
    f.close()

from Eval.evalResult import get_observed, get_predicted,compare_observed_to_predicted

def transition_estimator(f):
    ycount = {"START" : 0}
    transition_y_count = {}
    transition_probability = {}
    previous_state = ""
    counter = 0
    for line in f:
        
        if previous_state == "":
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
                previous_state = y

        else:
            if len(line.split(" ")) > 1:
                y = line.split(" ")[1] 
                if tuple((previous_state, y)) in transition_y_count:
                    transition_y_count[tuple((previous_state, y))] += 1
                else:
                    transition_y_count[tuple((previous_state, y))] = 1
                if y in ycount:
                    ycount[y] +=1
                else:
                    ycount[y] = 1
                previous_state = y
            else:
                if tuple((previous_state, "STOP")) in transition_y_count:
                    transition_y_count[tuple((previous_state, "STOP"))] += 1
                else:
                    transition_y_count[tuple((previous_state, "STOP"))] = 1
                

                previous_state = ""
        
    for key in transition_y_count.keys():
        transition_probability[key] = transition_y_count[key]/ycount[key[0]]
    return ycount

def train(language):
    FILE = training_set(language)
    ycount = transition_estimator(FILE)

    states =  list(ycount.keys())
    states.append("STOP")
    emission_counter = tuple([{} for i in range(len(states))])
    transmission_parameter = [[0]*(len(states)+1) for i in range(len(states))]
    counter = 0
    dic = {}
    for state in states:
        dic[state] = counter
        counter +=1
    dic["STOP"] = counter 
    print(dic)
    u = 'START'
    observation_space = set()

    for observation in FILE:
        try:
            observation, v = observation.split() #observation , state
            observation = observation.strip()
            v = v.strip()
            if type == "EN":
                position = dic[v]  ## position: 1~7
            else:
                position = dic[v]
            # update emission_counter
            if (observation in emission_counter[position]):
                emission_counter[position][observation] += 1
            else:
                emission_counter[position][observation] = 1

            # update transmission_parameter
            pre_position = dic[u]
            transmission_parameter[pre_position][position] += 1
            u = v
            # add into train_observation_set
            if observation not in observation_space:
                observation_space.add(observation)
        except:
            pre_position = dic[u]
            transmission_parameter[pre_position][8] += 1
            u = 'START'

    return observation_space

print(len(train("EN")))

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
