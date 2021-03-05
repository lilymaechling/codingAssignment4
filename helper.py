# -*- coding: utf-8 -*-
"""
Helper File Created on Mon Mar 23 22:25:37 2020
@author: deepc
"""
# In this "alive" code below, we will provide many subroutines (functions) in Python 3 which may be useful.
# We will often provide tips as well

import matplotlib.pyplot as plt

def read_pi(n):
    #opens the file name "pi" and reads the first n digits
    #puts it in the list pi, and returns that list
    pi = list()
    f = open('pi','r')
    for i in range(n):
        d = f.read(1)
        pi.append(int(d))
    return pi


def graph_plot(n):
    # needs matplotlib.pyplot
    # X-axis: numbers 1 to n
    # Y-axis: i^2 for i in range 1 to n
    X = list()
    Y = list()
    for i in range(1,n):
        X.append(i)
        Y.append(i*i)
    plt.plot(X,Y)
    plt.show()
    #Please feel free to google matplotlib and find how to modify its various features.


    


def convert_int_to_list(N):
    #takes a number N and makes a list with its single digits
    #Example: it takes 1729 and forms [1,7,2,9]
    L = list()
    s = str(N) #make N into a string
    for i in s:
        L.append(int(i))
    return L
    #The above can be succinctly written as "return [int(i) for i in str(N)]" 
    #That would be much faster
    
def convert_list_to_int(L):
    #Takes a list of single digit integers and makes it into one single integer read left to write
    #Again, it makes a string concatenating the whole list
    #and then uses the int() function
    s = ""
    for i in L:
        s = s + str(i)
    return int(s)


def read_countries():
    #opens the file named "country.txt" 
    #returns a list of all countries which are 7 letters or more (counting spaces)
    #all lower case
    countries = list()
    with open("countries.txt") as file:
        for line in file: 
            line = line.strip() #or some other preprocessing
            if(len(line) > 6):
                c = line.lower()    
                countries.append(c)
    return countries
    
