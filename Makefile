#
test: softmax.a bfgs.a test.o
	g++  -I/usr/local/Cellar/boost/1.57.0/include  test.o  ./softmax.a ./bfgs.a
test.o: test.cpp
	g++ -c test.cpp -o test.o
softmax.a: softmax.hpp
	g++ -c softmax.hpp -o softmax.o
	ar -rc softmax.a softmax.o
bfgs.a: bfgs.hpp
	g++ -c bfgs.hpp -o bfgs.o
	ar -rc bfgs.a bfgs.o
clean:
	rm -f bfgs.o test.o softmax.o bfgs.a softmax.a a.out
