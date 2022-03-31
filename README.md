# digit-classifier
a neural network that classifies hand written digits between 0 and 9


This is a hand written digit classifier network and the structure can be seen withing the 'model.py' file. 
I used 3 convolutional layers and 2 linear layers within the model. Honestly while adding the convolutional layers
trying to find out the ammount of features for each layer was kind of tricky for myself. the model ended up being
really slow and for a while I could find out why. Then I had realized that it is the last convolutional layer 
before being feed into the 2 linear layers that gave me a problem. The exact problem was in the structuring I had done
where the conv layers (x,y,z) had output feature sizes similar to (12,24,32) this was a big problem especially with larger
and high quality images. the final output of the last conv layer would have had 32 3-dimensional matrices (3 channel or 1 channel)
with L x W x Channels. if the image was 4000 x 6000 with say a single channel image, then when flattened the entire single input of
the first linear layer would have had to be around 1,536 nodes for the input of the layer and running this is entirly time consuming.
the solution I saw was to create a structure where more features would be created somewehere in the middle and the later conv layers
would bring the feature size to a more managable quantity and work alot faster and takes up less memory on the gpu.
