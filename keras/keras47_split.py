import numpy as np

a = np.array(range(1, 11))
size = 5

def split_x(dataset, size): #a, timesteps만큼 자름
    aaa = []                #aaa = 
    for i in range(len(dataset) - size +1): # i = data길이 - timesteps +1
        subset = dataset[i : (i + size)]
        aaa.append(subset)                  # 반복
        
    return np.array(aaa)


bbb = split_x(a,size)
print(bbb)
# print(bbb.shape)


x = bbb[:, :-1]
y = bbb[:, -1]
print(x.shape)
print(y.shape)


print(x)
print(y)


