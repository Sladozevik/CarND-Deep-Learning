# -*- coding: utf-8 -*-
"""
Created on Sun Jan 29 11:30:23 2017

@author: aslado
"""
from miniflow import *

x, y, z = Input(), Input(), Input()

f = Mul(x, y, z)

feed_dict = {x: 4, y: 5, z: 10}

graph = topological_sort(feed_dict)
output = forward_pass(f, graph)

# should output 19
print("{} * {} * {} = {} (according to miniflow)".format(feed_dict[x], feed_dict[y], feed_dict[z], output))
