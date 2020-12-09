def training_set(typ):
    file = open(typ+"/train",encoding="utf-8")
    return file.read().splitlines()

def dev_in(typ):
    file = open(typ+"/dev.in",encoding="utf-8")
    return file.read().splitlines()

def dev_out(typ):
    file = open(typ+"/dev.out",encoding="utf-8")
    return file.read().splitlines()

def new_maximum_likelihood_estimator(f,k=0.5):
    ycount={}
    y_to_x_count={}
    emission_x_given_y={}
    for line in f:
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