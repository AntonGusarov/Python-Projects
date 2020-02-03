import os
import numpy as np

#%% Part 1: Parsing the txt file. 
# get data out of file, parse it, turn into useful format so called ETL (extract-transform-load) process. 

# 1. EXTRACT: 
# use open() built-in canonical function. It returns a file object 
# -> this object should have its own methods: read(), readline(), readlines(), write()

# dir_fd = os.open('somedir', os.O_RDONLY)  # TODO opener callable

#def opener(path, flags):  # TODO opener callable
    # return os.open(path, flags, dir_fd=dir_fd)
    # return pass
    
    # open('ca1_step1_input_data.txt', 'r', errors='strict', opener=opener)

#%%
# with statement automatically takes care of closing the file once it leaves the with block, even in cases of error.
with open('ca1_step1_input_data.txt', 'r', errors='strict') as file: # opener is a custom callable   
    # use  "text-file iterator functionality to only read the file line by line":
    # file is iterable?
    for i in range(2):
        line = file.readline()  # defalult argument is size = -1 i.e.reads the whole lite
        print(line, '\n')  # read and print the two first lines
    


# %%
