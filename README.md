# Visual Memorability with Caffe Model

> @inproceedings{ICCV15_Khosla, author = "Aditya Khosla and Akhil S. Raju and Antonio Torralba and Aude Oliva", title = "Understanding and Predicting Image Memorability at a Large Scale", booktitle = "International Conference on Computer Vision (ICCV)", year = "2015" }

-------------------------

Score the memorability of pictures by Running LaMem (image process model) through Caffe (deep learning framework)

### Interface:
* IPython Notebook

### Platform:
* Ubuntu

## Knowledge Applied:
* Convolutional Neural Network and Gradient Descent
* Loss Functions and Optimization
* Activation Functions and Weight Regularization

> Mini-batch SGD: 
* Sample a batch of data
* Forward prop it through the graph, get loss
* Backprop to calculate the gradients
* Update the parameters using the gradient

> Convoluntion Layer: COnvolvve the filter with the image and convolve(slide) over all spatial locations

> Pooling Layer: make the representations smaller and more manageable and operate over each activation map independently

> Fully Connected Layer(FC layer): contain neurons that connect to the entire input volume, as in ordinary Neural Networks
### Summary
- ConNets stack CONV,POOL, FC layers
- Trend towards smaller filters and deeper architectures
- Trend towards getting rid of POOL/FC layers(just CONV)
- Trend towards smaller 

