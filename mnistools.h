#include<fstream>
#include<vector>
#include<Eigen/Dense>
#include<iostream>
#include<opencv2/opencv.hpp>

void MNISTLoader(std::ifstream& data_file, std::vector<Eigen::MatrixXd*>& data_vect,
	std::vector<Eigen::MatrixXd*>& answers_vect, const char* filename = "data");

Eigen::MatrixXd pngToMatrix(const char* file_name);