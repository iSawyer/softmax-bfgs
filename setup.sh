#########################################################################
# File Name: setup.sh
# Created Time: 日  5/31 12:20:15 2015
#########################################################################
#!/bin/bash

rm -f bfgs.o test.o softmax.o bfgs.a softmax.a a.out
make
./a.out

