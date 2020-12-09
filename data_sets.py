def training_set(language):
    file = open(language+"/train",encoding="utf-8")
    return file.read().splitlines()

def dev_in(language):
    file = open(language+"/dev.in",encoding="utf-8")
    return file.read().splitlines()

def dev_out(language):
    file = open(language+"/dev.out",encoding="utf-8")
    return file.read().splitlines()


def challenge_set(language):
    file = open(language+"/test.in",encoding="utf-8")
    return file.read().splitlines()