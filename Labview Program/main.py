import numpy as np
from torchvision import transforms
def change_channel(a):
    a = np.array(a)
    #print(a.shape)
    a = np.uint8(a)
    #print(a.shape)
    transform = transforms.Compose([transforms.ToTensor()
                                   ])
    a = transform(a) 
    a = a.transpose(0,1)
    a = a.transpose(1,2) 
    a = a.unsqueeze(0)    
    a=a.tolist()
    return a
