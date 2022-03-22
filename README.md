# Predictive Uncertainty Estimation for Camouflaged Object Detection

------

# Introduction

Uncertainty is inherent in machine learning methods, especially those for camouflaged object detection aiming to finely segment the objects concealed in background. The strong “center bias” of the training dataset leads to models of poor generalization ability as the models learn to find camouflaged objects around image center, which we define as “model bias”. Further, due to the similar appearance of camouflaged object and its surroundings, it is difficult to label the accurate scope of the camouflaged object, especially along object boundaries, which we term as “data bias”. To effectively model the two types of biases, we resort to uncertainty estimation and introduce predictive uncertainty estimation technique, which is the sum of model uncertainty and data uncertainty, to estimate the two types of biases simultaneously. Specifically, we present a predictive uncertainty estimation network (PUENet) that consists of a Bayesian conditional variational auto-encoder (BCVAE) to achieve predictive uncertainty estimation, and a predictive uncertainty approximation (PUA) module to avoid the expensive sampling process at test-time. Experimental results show that our PUENet achieves both highly accurate prediction, and reliable uncertainty estimation representing the biases within both model parameters and the datasets.

:running: :running: :running: ***KEEP UPDATING***.

------

# Performance

<p align="center">
    <img src="./figures/fig_performance_quan.jpg"/> <br />
    <em> 
    Figure 1: Performance comparison with state-of-the-art COD models.
    </em>
</p>

<p align="center">
    <img src="./figures/fig_performance_qual.jpg"/> <br />
    <em> 
    Figure 2: Visual results of our PUENet and the competing models.
    </em>
</p>

------

# Implementation

The codes are available at: 

The training and testing datasets can be downloaded at [COD10K-train] and [COD10K-test + CAMO-test + CHAMELEON], respectively.

The results of our PUENet are available at [PUENet-results].

