# Melanoma analysis with Fractal neural networks
The purpose of this research is to compare the performance of 
a [convolutional neural network](https://github.com/amyshenin-tech/masters-thesis---melanoma-analysis-with-fnn/blob/fnn/convnet/notebook-ENG.ipynb), 
[MobileNet V2](https://github.com/amyshenin-tech/masters-thesis---melanoma-analysis-with-fnn/blob/fnn/mobilenet_v2/notebook-ENG.ipynb), 
[Inception ResNet V2](https://github.com/amyshenin-tech/masters-thesis---melanoma-analysis-with-fnn/blob/fnn/inception_resnet_v2/notebook-ENG.ipynb), 
and our own [fractal neural network](https://github.com/amyshenin-tech/masters-thesis---melanoma-analysis-with-fnn/blob/fnn/fractalnet/notebook-ENG.ipynb).

# Melanoma
<p>
  <b>Melanoma</b>, also redundantly known as <b>malignant melanoma</b>, is a type of skin cancer that develops from the pigment-producing cells known as melanocytes. 
  Melanomas typically occur in the skin, but may rarely occur in the mouth, intestines, or eye (uveal melanoma). 
  In women, they most commonly occur on the legs, while in men, they most commonly occur on the back. About 25% of melanomas develop from moles. 
  Changes in a mole that can indicate melanoma include an increase in size, irregular edges, change in color, itchiness, or skin breakdown.
</p>

![](./assets/melanoma.jpg)<br/>

<p> Pic.1. A melanoma of approximately 2.5 cm (1 in) by 1.5 cm (0.6 in)</p>

# Getting started
## Prerequisites
- docker
- docker-compose >= 1.28
- [NVIDIA Container Toolkit](https://github.com/NVIDIA/nvidia-docker)
## Steps
1. Clone or download the repo.
2. In the terminal, go to `/docker`.
3. Run `docker-compose up -d`.
4. Run `docker logs jp`.
5. Open a browser and go to the URL, you got after step 4.
