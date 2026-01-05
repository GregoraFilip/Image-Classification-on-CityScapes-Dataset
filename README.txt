

# STUDENT's UCO: 525265

############################################################################################################
# Write short answers to the questions. Please do not exceed a total of 250 words for all of your answers. #
############################################################################################################

1. Project code:
Code is written in python and can be found in files: dataset.py, inference.py, network.py, training.py. 

For training I used classic fit method with early stopping and picking the best model.



2. What model architecture did you use? Why did you choose it?
I used Resnet architecture, which is good at training very deep neural networks effectively.
Resnet contains residual connections and skip connections,
which help to mitigate the vanishing gradient problem by allowing gradients to flow directly through the network.
Resnet also achieved very good results in image classification tasks. 
I tried to use another architectures as AlexNet and VGG16, but Resnet has achieved best results so I use this architecture.
