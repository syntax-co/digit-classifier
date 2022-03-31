import idx2numpy as idx
import matplotlib.pyplot as plt
from model import conv_network
import numpy as np

import time



eyes=conv_network()
eyes.load_model()

targ_key=[[0 for i in range(10)] for k in range(10)]
for i in range(10):
    targ_key[i][i]=1




training_files=['train-images-idx3-ubyte\\train-images.idx3-ubyte',
               'train-labels-idx1-ubyte\\train-labels.idx1-ubyte'
               ]

testing_files=['t10k-images-idx3-ubyte\\t10k-images.idx3-ubyte',
                't10k-labels-idx1-ubyte\\t10k-labels.idx1-ubyte'
                ]
                



run_file=testing_files
images=idx.convert_from_file(run_file[0])
labels=idx.convert_from_file(run_file[1])
training_size=images.shape[0]

epochs=1

correct=0

for epoch in range(epochs):
    print('^'*50)
    print('epoch: {}'.format(epoch))
    

    for index in range(training_size):
        # time.sleep(1)
        
        image=images[index]
        target=targ_key[labels[index]]
        image=np.expand_dims(image,0)
        
        # eyes.train_on_image(image,target)
        prediction,real = eyes.test_on_image(image,target)
        
        if prediction.item()==real.item():
            correct+=1
        
        if index%(training_size//6)==0:
            print(round((index/training_size)*100,4))
    
    print(round(correct/training_size,4))
    print('^'*50)

eyes.save()























