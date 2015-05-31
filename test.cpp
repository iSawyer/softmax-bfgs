//
//  main.cpp
//  softmax
//
//  Created by baicai on 15/5/31.
//  Copyright (c) 2015年 baicai. All rights reserved.
//

#include "bfgs.hpp"
#include "softmax.hpp"
#include <iostream>
#include <string>
using namespace std;

int main(){
    
    
    string train_data_path = "/Users/baicai/Documents/git/softmax/data/train_data";
    string train_label_path = "/Users/baicai/Documents/git/softmax/data/train_label";
    string test_data_path = "/Users/baicai/Documents/git/softmax/data/test_data_path";
    string test_label_path = "/Users/baicai/Documents/git/softmax/data/test_label_path";
    /*  fixed:
     线搜最大迭代数 = 30
     gradToler = 1e-10; grad norm minimal bound
     toler = 1e-10;     linesearch toler: cost_cur - cost_prev
     */
    double c = 0.01; 			// backtracking linesearch ps
    double beta = 0.8; 			// backtracking linesearch ps
    double lambda = 0.0001;  	// softmax 正则参数
    size_t max_iter = 100;		// 最大总迭代数
    
    Solver test(c,max_iter,beta,lambda);
    test.init_mnist_train(train_data_path,train_label_path);
    test.train();
    test.init_mnist_test(test_data_path,test_label_path);
    double precision = test.cal_precision();
    cout<<"precision is "<< precision<<endl;
    return 0;
}