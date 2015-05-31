#ifndef SOFTMAX_HPP
#define SOFTMAX_HPP

#include <stdio.h>
#include <vector>
#include <math.h>
#include <stdio.h>



class Softmax{
public:
	
	Softmax(size_t num_fea, size_t num_l, double lamb):  num_feature(num_fea), num_label(num_l),lambda(lamb) {
		fprintf(stdout,"num_feature:%d, num_label:%d\n",num_feature,num_label);
		vec_prob = new double[num_label];
		for(int i = 0; i < num_label; i++){
			vec_prob[i] = 0.0;
		}
		fprintf(stdout, "softmax init Done\n");
		// decay
		// initilize done
	}

	// 计算目标函数值 for line search
	double cal_cost(const std::vector<std ::vector<double> >& x, const std::vector<int>& y, double* weight){
		double cost = 0;
		for(size_t i = 0; i < x.size(); i++){
            // P(y = num_label | x[i],weith)
			prob(x[i], weight);
			cost += log(vec_prob[y[i]]);
		}
		cost = -cost/x.size();
		// 不将bias加入正则项中
		for (int i = 0; i < num_feature * num_label; ++i)
		{
			cost += lambda * weight[i] * weight[i] / 2;
		}
		return cost;
	}


	inline double cal_grad_norm(double* grad){
		double sum = 0;
		for (int i = 0; i < num_feature * num_label + num_label; ++i)
		{
			sum += grad[i] * grad[i];
		}
		return sum;
	}
    
    
	// 计算梯度 
	void cal_grad(const std::vector<std::vector<double> >& x, const std::vector<int>& y,const double* weight,double* grad){
       // int j = num_feature;
		for (size_t j = 0; j < (num_feature*num_label + num_label); j++)
		{
			grad[j] = 0;
		}
	
		for(size_t i = 0; i < num_label;i++){
			std::vector<double> sum_x(num_feature,0);
			double sum_bias = 0;
			for(size_t j = 0; j < x.size();j++){
				
				prob(x[j],weight);
                double indicator = (y[j] == i ? 1 : 0);
				for(size_t k = 0; k < num_feature;k++){
					sum_x[k] += x[j][k] * (indicator - vec_prob[i]);
					sum_bias += indicator - vec_prob[i];
				}
			}

			for(size_t k = 0; k < num_feature;k++){
				grad[i * num_feature + k] = -sum_x[k]/x.size() + lambda * weight[i * num_feature + k];
			}
			// bias
			grad[num_label * num_feature + i] = -sum_bias/x.size();
		}
    }

	inline int predict(const std::vector<double> &row_sample, const double* weight){
        int index = 0;
		prob(row_sample,weight);
		double max = -6666;
		for(int i = 0; i < num_label; i++){
			if(vec_prob[i] > max){
				max = vec_prob[i];
				index = i;
			}
		}
		return index;
	}


	~Softmax(){
		fprintf(stdout, "Goodbye cruel world2\n");
		delete []vec_prob;
	}

private:
	double* vec_prob;
	double lambda; 
	size_t num_feature;
	size_t num_label;

	// 计算 exp(w_k * x)
 	inline	double linear_sum(const std:: vector<double>& row_sample, int k, const double* weight){
		// check row_sample's size
		if(row_sample.size() != num_feature){
			fprintf(stderr,"dimension unequal in linear_sum\n");
			return -1;
		}
		double sum = 0;
		for (int i = 0; i < num_feature;i++)
		{	
			sum += weight[k * num_feature + i] * row_sample[i];
		}
        // add bias
		sum += weight[num_label*num_feature + k];
		return sum;
	}	

	// 计算所有  exp(w_i * x) 将概率存储在向量 vec_prob中
	void prob(const std:: vector<double>& row_sample, const double* weight){
		if(row_sample.size() != num_feature){
			fprintf(stderr,"dimension unequal in linear_sum\n");
			return;
		}
		double sum = 0;
		for (int i = 0; i < num_label; i++)
		{
			vec_prob[i] = linear_sum(row_sample, i, weight);
		}
        double max = vec_prob[0];
		for(int i = 0; i < num_label;i++){
			// some trick here learn from stackoverflow
			if(max < vec_prob[i]){
				max = vec_prob[i];
			}
		}
		for (int i = 0; i < num_label; ++i)
		{
			vec_prob[i] = exp(vec_prob[i] - max);
			sum += vec_prob[i];
		}

		for (int i = 0; i < num_label; ++i)
		{
			vec_prob[i] /= sum;
		}

	}


};  // Softmax done
 

#endif



