
#include "bfgs.hpp"
#include "softmax.hpp"
#include <iostream>
#include <string>
using namespace std;

int main(){
    
    
    string train_data_path = "data/train_data";
    string train_label_path = "data/train_label";
    string test_data_path = "test_data_path";
    string test_label_path = "test_label_path";
    /*  fixed:
     线搜最大迭代数 = 100
     gradToler = 1e-10; grad norm minimal bound
     toler = 1e-10;     linesearch toler: cost_cur - cost_prev
     */
    double c = 0.01;            // backtracking linesearch ps
    double beta = 0.8;          // backtracking linesearch ps
    double lambda = 0.0001;     // softmax 正则参数
    size_t max_iter = 100;      // 最大总迭代数
    
    Solver test(c,max_iter,beta,lambda);
    test.init_mnist_train(train_data_path,train_label_path);
    test.train();

    return 0;
}