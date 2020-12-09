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

import time

def getmaxy(get_e_y_x):
    maxy=0
    max_y_from_x = {}
    max_x_prob = {}
    for key, item in get_e_y_x.items():
        if key[1] not in max_x_prob:
            max_x_prob[key[1]] =item
            max_y_from_x[key[1]] = key[0]
        else:
            if max_x_prob[key[1]] < item:
                max_x_prob[key[1]] = item
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

#main function
for language in ["EN","CN","SG"]:
    test_input = dev_in(language)
    best_states = getmaxy(new_maximum_likelihood_estimator(training_set(language))[0])
    finaloutput(test_input,best_states, language)