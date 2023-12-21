---
title:  "Using physics to train neural networks: a conversation with Ben Scellier"
mathjax: true
layout: post
categories: media
---

# Using physics to train neural networks: a conversation with Ben Scellier

By Tatsuhiro Onodera

Recently, I had the chance to attend the [Frontiers of Neuromorphic Computing conference](https://indico.mpl.mpg.de/event/15/) in Erlangen, Germany. The conference was well-organized, featuring excellent speakers and an ideal size that made it easy for everyone to interact. Thus, Logan and I realized that this would be a great opportunity to write a blog post centered around at a scientific result that interests us, and to interview a scientist who has done pioneering work in this space at the conference.

Since this is our first time writing a blog post of this kind, I want to start by providing an idea of its structure. First, I'll offer a quick summary of the emerging field that explores "self-learning through physics-based neural networks." After that, the post will transition into a conversation I had with Ben, focusing on how he got into the field of computing with physical systems. Inspired by the podcast [“what’s your problem”](https://www.pushkin.fm/podcasts/whats-your-problem), we'll conclude with a lightning round of questions where I ask Ben two quick questions and share his answers.

# A quick introduction to self-learning with equilibrium propagation

Self-learning in physical systems refers to a novel paradigm where physical systems learns to perform a task without reliance on an external computer, as data is fed to the system. Unlike classical deep learning on a digital computer that relies on a Von-Neumann architecture where memory, computation, and algorithm is separate — a physical computer that self-learns intertwine the physical medium and the computational algorithm. The crudest (somewhat wrong) example of self-learning is our own brains, where we simply learn by being exposed to new experiences (aka there is no external computer governing our learning)!

Beyond being a scientifically interesting and deep subject, self-learning is also of technologically importance for two reason: 1) it may enable energy efficient training as they leverage the natural dynamics of the system for computation/learning without needing large amounts of energy to control the learning process. 2) In the long term, It could enable extremely large and complex neural networks with more parameters than we can obtain with conventional means, as the system learns on it’s own in a decentralized, non-global way (again a crude example of this is human intelligence).

[Introduced by Ben Scellier and Yoshua Bengio in 2017](http://journal.frontiersin.org/article/10.3389/fncom.2017.00024/full), equilibrium propagation (EP) is a prominent framework in the domain of self-learning. In the following, I will be providing a quick introduction to it. As a firm believer of the principle that *good explanations provide the truth, but not the whole truth*, the following will provide a simplified (perhaps over-simplified) explanation -- please see the paper for the full detail.

<center><img src="../images/2023-10-28/fig1.png" width="30%"></center>



The model of computation used in EP is fundamentally different from that found in regular multilayer NN, as it relies of bidirectional connections/weights and is described in continuous time as a dynamical system. In particular, it uses an energy-based model, which is to say that the *dynamics of the systems naturally decreases the energy*. For simplicity, let’s assume the energy takes the simplest quadratic form:

$$E(u) = - \sum_{ij} W_{ij} u_i u_j$$

As a gradient dynamical system, it obeys the following linear ODE:

$$\dot u_i = - \frac{\partial E}{\partial u_i} = \sum_{j} W_{ij} u_j$$

To most easily relate to these equations as a physicist, imagine that $u_i$ represents the displacements of objects in 1D that are connected by springs with coupling constant $W_{ij}$. Now why isn't there a momentum variable, as we have learned in physics 101? The reason is that this system essentially resides in a highly viscous liquid, such that the momentum variable is essentially frozen out (for the mathematically inclined: one can derive this formally with adiabatic elimination). In other words, the dissipation is large enough that the system is always over-damped and does not show any oscillatory characteristics. For other physicists that are closer to condensed matter, it may be easier to reason about this as a spin-glass. In any case, a positive $W_{ij}$ seeks to align two nodes (make $u_i$  and $u_j$ adopt the same sign), while a negative $W_{ij}$ does the opposite.

To perform machine learning with this system, we need some way to encode data into the system. In particular, as we will be dealing with supervised learning, there will be inputs $x^{(\text{ML})}_i$ and outputs $y^{(\text{ML})}_i$. Similar to a Boltzmann machine, we can encode the desired information in some set of nodes (see figure above). Thus, some nodes in the dynamical systems will be *clamped* (aka fixed to some value for all time) to represent input data $u_i(t) = x_i^{\text{(ML)}},  \forall t$ On the other hand, there will be some nodes in the system that we want to directly correspond to the predicted outputs after **equilibration** (ideally $u_i(t\rightarrow \infty) = y_i^{\text{(ML)}}$) The rest of the nodes in the system are known as hidden nodes (represented as $h_i$ in the figure) which are essentially there to increase the information processing capacity of the network.


To train the network via self-learning, the approach works as follows:

-----------
More explanation on this (aka fill in)

----------

It should noted that approaches similar to this involving a 2 step approach of introducing both the input data in the first step, and the target data in the latter step has been known in a purely computer-science context, in field of contrastive Hebbian learning. Ben's key contribution is introducing this subtle approach of nudging the network, and mathematically proving that this approach results in accurate gradients (of the cost function) in the limit of small nudges. Furthermore, he illucidated that these types of algorithms previously only known in the "math" world, has important implications for machine learning with novel physical hardware.

## How Ben got into the field of physical computing

Ben completed his Master's degree in applied mathematics from École Polytechnique in France and the National University of Singapore in 2015. Shortly thereafter, he took up a role as a data scientist in Singapore. It was during this period that a friend introduced him to the backpropagation algorithm and the field of deep learning. Struck by the algorithm's simplicity and its fascinating links to neuroscience, Ben found himself captivated. He spent his evenings diving into the subject, completing Geoffrey Hinton's deep learning course and poring over academic papers. His dedication even led him to develop a graphical interface for visualizing deep belief networks.

This self-directed learning journey had a rewarding outcome: he reached out to Yoshua Bengio with screenshots of his GUI and eventually secured a spot to pursue his PhD in Bengio's research group. Once enrolled, Ben continued to explore his interest in understanding the brain as a computational machine. He began to investigate how the brain's computational principles could inform the design of more efficient physical computing systems.

I found Ben's journey particularly engaging, not least because he comes from a background in mathematics rather than physics, which is more typical for the field of computing with novel physical systems. His story underscores the value of having a diverse range of perspectives in research, enriching the environment for groundbreaking work. It also serves as a tangible reminder for aspiring researchers: if you're passionate about a field, immerse yourself in independent study, and follow it up with an email!

## Lightning-round questions

1. What is the coolest paper that you read this year?

	> [Dual Propagation: Accelerating Contrastive Hebbian Learning with Dyadic Neurons](https://arxiv.org/abs/2302.01228) - They managed to do extremely fast inference by minimizing some functional with one forward pass!

2. What is your best estimate for the probability for a non-digital electronics hardware to conquer a decent market share (~10%) of digital electronics hardware for computing in the next 10 years?

	> Probably negligible (<1%) in the next 10 years. In my view, the most significant technological leap on the horizon is digital in-memory computing, a direction that companies like Rain Computing are actively exploring. This will be followed by adding some select analog components, which will start performing some computationally expensive part of the computation (such as using memristers for matrix-vector products). Overall, I anticipate a hybrid landscape where computation is executed through a blend of analog and digital components for the foreseeable future.
