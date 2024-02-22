import torch
import torchvision
import torchvision.transforms as tr
from torch.utils.data import DataLoader,Dataset
import numpy as np
import matplotlib.pyplot as plt

transf = tr.Compose([tr.Resize(16),tr.ToTensor()]) #원본 이미지 너비, 높이가 다를경우 각각 지정 #미리 전처리를 선언하는것.
trainset = torchvision.datasets.CIFAR10(root='C:\\_data\\image\\torch\\', 
                                        train = True, download = True, 
                                        transform = transf)

testset = torchvision.datasets.CIFAR10(root='C:\\_data\\image\\torch\\',
                                       train = False, download = True, #test 데이터는 훈련을 시키지않을거기때문에 train에 False로 넣을것.
                                       transform=transf) 
print(trainset[0][0].size()) #trainset[0][0]은 이미지 #토치에서는 채널수가 앞에 옴 #torch.Size([3, 16, 16])
#trainset[0][1]은 라벨 

trainloader = DataLoader(trainset, batch_size= 50, shuffle = True)
testloader = DataLoader(testset, batch_size= 50, shuffle = False) # 테스트는 셔플하면 안됨!

print(len(trainloader)) #1000

images,labels = iter(trainloader).__next__()
print(images.size()) #torch.Size([50, 3, 16, 16]) 배치크기*채널수*너비*높이
oneshot = images[1].permute(1,2,0).numpy() #이미지를 그려주기 위해서는 채널수가 가장 뒤로 가야하기 때문에 
plt.figure(figsize=(2,2))
plt.imshow(oneshot)
plt.axis("off")
plt.show()

'''
#############전처리 커스텀##########

class ToTensor:
    def __call__(self,sample):
        inputs, labels = sample
        inputs = torch.FloatTensor(inputs)
        inputs = inputs.permute(2,0,1)
        return inputs, torch.LongTensor(labels)
'''    
    
        

