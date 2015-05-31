#ifndef BFGS_HPP
#define BFGS_HPP
#include "softmax.hpp"
#include <iostream>
#include <vector>
#include <string>
#include <stdlib.h>
#include <fstream>
#include <math.h>
#include <stdio.h>

class Solver{
public:
	Solver(double dd = 0.5, size_t iter = 100, double be = 0.8,double lamb = 0.0001): alpha(dd), MaxIter(iter),iter_num(1), beta(be),lambda(lamb)
	{	
		gradToler = 1e-10; // grad norm bound
		toler = 1e-10; // linesearch toler: cost_cur - cost_prev
		fprintf(stdout,"alpha:%lf, MaxIter:%d, beta:%lf, lambda:%lf\n",alpha,MaxIter,beta,lambda);
	}
    ~Solver(){
        delete []bm_cur;
        delete []bm_prev;
        delete []weight_solver;
        delete []grad_solver;
        delete model;
        fprintf(stdout, "Goodbye cruel world\n");
    }

	void init_mnist_train(const std::string data_path, const std::string label_path){
        
		size_t num_sample = 60000;
		size_t num_feature = 784;
		size_t num_label = 10;
		std::vector<double> tmp(num_feature,0);
		x_.resize(num_sample,tmp);
		y_.resize(num_sample,0);
		std:: ifstream data_in;
        data_in.open(data_path);
		std:: ifstream label_in;
        label_in.open(label_path);
        if(data_in.is_open() && label_in.is_open()){
            fprintf(stdout, "open ok\n");
        }

        double tmp_1;
        int tmp_2;
		for(size_t i = 0; i < num_sample; i++){
			for(size_t j = 0; j < num_feature; j++){
                data_in>>tmp_1;
                x_[i][j] = tmp_1;
			}
			label_in>>tmp_2;
            y_[i] = tmp_2;
		}
        ndim = num_feature * num_label + num_label;
        

		data_in.close();
		label_in.close();
        
        // 归一化
		//ndim = num_feature * num_label + num_label;
        for (size_t i = 0; i < num_sample; i++) {
            for(size_t j = 0; j < num_feature; j++){
                x_[i][j] = x_[i][j] / 255.0;
            }
        }
        
        /*
		boost:: mt19937 gen;
		boost:: normal_distribution<double> random_distribution(0,1);
		boost:: variate_generator<boost::mt19937&, boost:: normal_distribution<double> > rnumber(gen,random_distribution);
         */
		weight_solver = new double[ndim];
		grad_solver = new double[ndim];
		weight_model = new double[ndim];
		grad_model = new double[ndim];
		for(int i = 0; i < ndim; i++){
			weight_solver[i] = 0;
			weight_model[i] = weight_solver[i];
			grad_solver[i] = 0;
			grad_model[i] = 0;
		}
		bm_cur = new double[ndim*ndim];
		bm_prev = new double[ndim*ndim];
		memset(bm_cur,0,sizeof(double)*ndim*ndim);
		memset(bm_prev,0,sizeof(double)*ndim*ndim);
		//初始化为单位矩阵
		for (int i = 0; i < ndim; ++i)
		{
			bm_cur[i*ndim + i] = 1.0;
		}

		Sm.resize(ndim,0);
		Ym.resize(ndim,0);
		p.resize(ndim,0);
        this->model = new Softmax(num_feature,num_label,lambda);
		
	}

	void init_mnist_test(const std::string data_path, const std:: string label_path){
		size_t num_sample = 10000;
		size_t num_feature = 784;
		//size_t num_label = 10;
        size_t num_label = 10;
		std::vector<double> tmp(num_feature,0);
		x_.resize(num_sample,tmp);
		y_.resize(num_sample,0);
        std:: ifstream data_in;
        data_in.open(data_path);
        std:: ifstream label_in;
        label_in.open(label_path);
        if(data_in.is_open() && label_in.is_open()){
            fprintf(stdout, "open ok\n");
        }
        
        double tmp_1;
        int tmp_2;
        for(size_t i = 0; i < num_sample; i++){
            for(size_t j = 0; j < num_feature; j++){
                data_in>>tmp_1;
                x_[i][j] = tmp_1;
            }
            label_in>>tmp_2;
            y_[i] = tmp_2;
        }
        ndim = num_feature * num_label + num_label;
        data_in.close();
		label_in.close();
        // 归一化
        for (size_t i = 0; i < num_sample; i++) {
            for(size_t j = 0; j < num_feature; j++){
                x_[i][j] = x_[i][j] / 255.0;
            }
        }

	}


	inline int predict(const std:: vector<double> row_sample){
		return model->predict(row_sample,weight_model);
	}

	double cal_precision(){
		int y_predict;
		double cor = 0;
		for(size_t i = 0; i < x_.size(); i++){
			y_predict = predict(x_[i]);
          //  cout<<y_predict<<" ";
			if(y_predict == y_[i]){
				cor++;

			}
		}
        //cout<<endl;
		return cor/y_.size();
	}

	void train(){
		
		size_t iter = MaxIter;
		double* V = new double[ndim*ndim];
        std:: vector<double> yb(ndim,0);
        std:: vector<double> by(ndim,0);

		model->cal_grad(x_,y_,weight_model,grad_model);
        
		while(iter--){

			if(!backtraking()){
				return;
			}
            
			// 计算Sm
			for (size_t i = 0; i < ndim; ++i)
			{
				Sm[i] = weight_solver[i] - weight_model[i];
			}
			
			memcpy(weight_model,weight_solver,sizeof(double)*ndim);

			
			model->cal_grad(x_,y_,weight_model,grad_solver);
			double err = model->cal_grad_norm(grad_solver) / ndim;
			if(err <= gradToler){
				fprintf(stdout, "Quit with grad's norm %d in %d iter \n",err,iter_num);
				return;
			}
            
            //计算Ym
			for(size_t i = 0; i < y_.size(); ++i){
				Ym[i] = grad_solver[i] - grad_model[i];
			}

			memcpy(grad_model,grad_solver,sizeof(double)*ndim);
			memcpy(bm_prev,bm_cur,sizeof(double)*ndim*ndim);
            
			// 计算拟二阶逆矩阵 bm
            // 速度慢问题， 两个矩阵乘法 o(n^3)
            // 解决 化为矩阵向量乘 矩阵矩阵加 o(n^2)
			double p_sum = 0;
			for(size_t i = 0; i < ndim; i++){
				p_sum += Ym[i] * Sm[i];
			}
			p_sum = 1 / p_sum;
            // yi' * Bi * yi / p_sum

            double r_sum = 0;
            for(size_t i = 0; i < ndim; i++){
                for(size_t j = 0; j < ndim;j ++){
                    yb[i] += Ym[j] * bm_prev[j*ndim + i];
                }
            }
            
            for(size_t i = 0; i < ndim; i++){
                r_sum += yb[i]*Ym[i];
            }
            r_sum = (r_sum * p_sum + 1) * p_sum;
            for (size_t i = 0; i < ndim; i++) {
                for (size_t j = 0; j < ndim; j++) {
                    V[i*ndim + j] = Sm[i]* Sm[j] * r_sum;
                }
            }
            // B * y
            for (size_t i = 0; i < ndim; i++) {
                for (size_t j = 0; j < ndim; j++) {
                    by[i] += bm_prev[i*ndim + j]*Ym[j];
                }
            }
            
            for (size_t i = 0; i < ndim; i++) {
                for (size_t j = 0; j < ndim; j++) {
                    size_t index = i * ndim + j;
                    V[index] -= p_sum *(Sm[i] * yb[j] + by[i] * Sm[j]);
                    bm_cur[index] = bm_prev[index] + V[index];
                }
            }
            
            for (size_t i = 0; i < ndim; i++) {
                yb[i] = 0;
                by[i] = 0;
            }
            
            /* 太慢
			for(size_t i = 0; i < ndim; i++){
				for(size_t j = 0; j < ndim; j++){
					V[i*ndim + j] = -p_sum * Ym[i] * Sm[j];
				}
				V[i*ndim + i] += 1;
			}
            
        
			for(size_t i = 0; i < ndim; i++){
				for(size_t j = 0; j < ndim; j++){
					double tmp = 0;
					for(size_t k = 0; k < ndim; k++){
						//tmp += V[i*ndim + k] * bm_prev[k*ndim + j];
						tmp += bm_prev[i * ndim + k] * V[k*ndim + j];
					}
					bm_cur[i*ndim + j] = tmp;
				}

			}
			// transpose V
			for(size_t i = 0; i < ndim; i++){
				for(size_t j = i+1; j < ndim; j++){
					double tmp = V[i*ndim + j];
					V[i*ndim + j] = V[j*ndim + i];
					V[j*ndim + i] = tmp;
				}
			}

			for(size_t i = 0; i < ndim; i++){
				for(size_t j = 0; j < ndim; j++){
					double tmp = 0;
					for(size_t k = 0; k < ndim; k++){
						tmp += V[i*ndim + k] * bm_cur[k*ndim + j];
					}
					bm_cur[i*ndim + j] = tmp;
				}
			}

			for(size_t i = 0; i < ndim; i++){
				for(size_t j = 0; j < ndim; j++){
					bm_cur[i*ndim + j] += p_sum * Sm[i] * Sm[j];	
				}
			}
            */
            
			iter_num++;
		}
		delete []V;
		//model->set_weight(weight_model);
		fprintf(stdout, "Ttrain Done in %d iter\n",iter_num);
	}

	std::vector<std::vector<double> > get_x(){
		return x_;
	}
	std:: vector<int> get_y(){
		return y_;
	}


private:
	Softmax* model;
	//data
	size_t MaxIter;	
	double lambda;
	size_t ndim;
	size_t iter_num;
	std::vector<std::vector<double> > x_;
	std::vector<int> y_;
	//vector<double> y_predict;
	double* weight_solver;
	double* weight_model;
	double* grad_solver;
	double* grad_model;
	double gradToler; // grad norm bound
	double toler; // linesearch toler: cost_cur - cost_prev

	std::vector<double> Sm;
	std::vector<double> Ym;
	std::vector<double> p;
	double* bm_cur;  // size: (num_feature*(num_label+1) * num_feature*(num_label + 1))
	double* bm_prev;
	
	double alpha;
	double beta;

	bool backtraking(){
		
		double cost_prev = model->cal_cost(x_,y_,weight_model);
		double cost_cur = 0.0;
        //fprintf(stdout, "cost_prev:%lf\n",cost_prev);
		double tol = 9;
		double t = 0.5;
		size_t max_iter = 100;
		double m = 0;
		//计算方向
        cal_p();
        
		for (int i = 0; i < ndim; ++i)
		{
            // g' * p
			m += grad_model[i] * p[i];
		}
        //m = m > 0 :m ? -m;
		for (int i = 0; i < max_iter && tol > toler; ++i)
		{
			update_weight(t);
            /*
            for (size_t i = 0; i < ndim; i++) {
                fprintf(stdout, "weight_solver[%d]:%lf\n",i,weight_solver[i]);
            }*/
           // getchar();
			cost_cur = model->cal_cost(x_,y_,weight_solver);
            //fprintf(stdout, "cost_cur:%lf\n",cost_cur);
            //cout<<"getchar"<<endl;
            //getchar();


			// cost_cur < cost_prev - c * t
			if(cost_cur < (cost_prev + alpha * t)) {
				fprintf(stdout, "Linesearch Succused: %d iter, %lf obj_val\n", iter_num, cost_cur);
				return true;
			}
            tol = (cost_cur - cost_prev)>0 ? ((cost_cur - cost_prev)): (cost_prev-cost_cur);
			if(tol > toler){
				t *= beta;
			}
		}
		
		fprintf(stdout, "Linesearch Failed:  %d iter, %lf obj_val\n",iter_num, cost_cur);
        for (size_t i = 0; i < ndim; i++) {
            fprintf(stdout, "weight_solver[%d]:%lf\n",i,weight_solver[i]);
        }
        getchar();
		return false;
	}

	// p = -bm_cur*grad
	void cal_p(){
        
        for (size_t i = 0; i < ndim; i++) {
            p[i] = 0.0;
        }
        
		for (size_t i = 0; i < ndim; ++i)
		{
			for (size_t j = 0; j < ndim; ++j)
			{
				p[i] -= bm_cur[i*ndim + j] * grad_model[j];
			}
		}
		return;
	}

	// w = w + t*p
	void update_weight(double t){
	
		for (int i = 0; i < ndim; ++i)
		{
			weight_solver[i] = weight_model[i] + t * p[i];	
		}
	
    
    }
	

}; //bfgs done






#endif