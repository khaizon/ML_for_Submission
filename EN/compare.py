kf = open("EN/cui.dev.p4.out",encoding="utf-8")
bryan = open("EN/dev.p4.out",encoding="utf-8")
gold = open("EN/dev.out",encoding= "utf-8")
kf = kf.read().splitlines()
bryan = bryan.read().splitlines()
gold = gold.read().splitlines()

for i in range(27224):
    if kf[i] != bryan[i]:
        print(kf[i])
        print(bryan[i])
        print(gold[i])

        print(i)