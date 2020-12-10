from data_sets import training_set, dev_in, dev_out

arr= [[]*5 for i in range(5)]
print(arr)
params = open("EN"+'/dev.p5.params', 'w', encoding='utf-8')
params.write(str(arr))
