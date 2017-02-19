#**CarND MiniFlow - Deep Learning assigments** 
In this lab, you’ll build a library called MiniFlow which will be your own version of TensorFlow! (link for China)

TensorFlow is one of the most popular open source neural network libraries, built by the team at Google Brain over just the last few years.

Following this lab, you'll spend the remainder of this module actually working with open-source deep learning libraries like TensorFlow and Keras. So why bother building MiniFlow? Good question, glad you asked!

The goal of this lab is to demystify two concepts at the heart of neural networks - backpropagation and differentiable graphs.

Backpropagation is the process by which neural networks update the weights of the network over time. (You may have seen it in this video earlier.)

Differentiable graphs are graphs where the nodes are differentiable functions. They are also useful as visual aids for understanding and calculating complicated derivatives. This is the fundamental abstraction of TensorFlow - it's a framework for creating differentiable graphs.

With graphs and backpropagation, you will be able to create your own nodes and properly compute the derivatives. Even more importantly, you will be able to think and reason in terms of these graphs.

Now, let's take the first peek under the hood...