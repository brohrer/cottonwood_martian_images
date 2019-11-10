# Image Compression using Autoencoders

[work in progress]

This is an End-to-End case study showing how to use an autoencoder
(a type of deep neural network) to compress a set of images.

We make use of the [Cottonwood machine learning framework](
https://github.com/brohrer/cottonwood), a flexible tool for
building and experimenting with neural networks and other
machine learning methods.

Our goal is to compress images taken from cameras on the
Mars Rover Curiosity so that they can be efficiently transmitted
back to Earth.

![Trained autoencoder](doc/mars_autoencoder.png)

The methods we user here are explained fully in a sequence of courses
in the [End-to-End Machine Learning School](http://e2eml.school),
especially
* [Course 311, Autoencoder Visualization](https://end-to-end-machine-learning.teachable.com/p/neural-network-visualization)
* [Course 312, Neural Network Framework](https://end-to-end-machine-learning.teachable.com/p/write-a-neural-network-framework)
* [Course 313, Advanced Neural Network Methods](https://end-to-end-machine-learning.teachable.com/p/advanced-neural-network-methods)
* [Course 314, Hyperparameter Optimization (January 2020)](http://e2eml.school)

## Installation

To use this code, clone this repository and run it locally.
It requires Cottonwood as well.
You can do all this at the command line. Just a heads-up that this
repository includes a dataset of 270 images and weighs in at 177 MB.

```bash
git clone https://github.com/brohrer/cottonwood_martian_images.git
python3 -m pip install cottonwood --user --no-cache
cd cottonwood_martian_images
python3 build_patch_dictionary.py
```
