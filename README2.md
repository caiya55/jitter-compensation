This is an assignment for Ascent Robotics

Desing description: First I realized that the "Neural network and Deep learning" ebook provides a very basic but useful python based code, 
so I used the netowrk class from that book. And since the dataset if different in the ebook, so I write a new file for dataset download 
and preprocessing. Then it comes to fine-tuing. I realized that the basic model have some flaws, so I added decay learning rate and 
weight decay regulation. The loss curve becomes smooth then. After trying several parameters, the evaulation accuracy becomes higher
and more stable. Since I choose the training dataset randomly everytime running the code, so the performance is slightly different in each 
running. 

In my understanding, training data is used for traing, evaluation dataset is used to monitor the training process and avoid overfitting, as
for the test dataset, normal in the competation or in practice, its labels are unknown and the pretrained model need predict the labels of
the test dataset. So I split the dataset randomly into 100 samples for training, 40 samples for evaluation, and 10 samples for test.

How to run:

environment: Python3.0
Extral libraries required: numpy, matplotlib(for plot), sklearn(for dataset download)

run the train.py   
-python train.py
