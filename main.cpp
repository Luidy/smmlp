//
// Created by sm19 on 9/30/17.
//

#include <iostream>
#include <fstream>
#include "sample.h"
#include "network.h"

using namespace std;

#define TRAINING_FILE  "training.dat"
#define DELIMITER "    "
#define TRAINING_DATA_LEN 75

#define TESTING_FILE  "testing.dat"
#define TESTING_DATA_LEN 75


vector<TrainingSample> ReadTrainingDataSet() {
	vector<TrainingSample> train_set;

	ifstream open_file(TRAINING_FILE);
	size_t pos = 0;
	string delimiter = DELIMITER;

	if (open_file.is_open()) {
		string line;
		string token;
		for (int i = 0; i < TRAINING_DATA_LEN; i++) {
			getline(open_file, line);
			vector<double> input_vector;
			for (int x = 0; x < 4; x++) {
				pos = line.find(delimiter);
				token = line.substr(0, pos);
				input_vector.push_back(stod(token));
				line.erase(0, pos + delimiter.length());
			}
			vector<double> output_vector; 
			// custom ouput value 
			if (i < 25) output_vector = { 1, 0, 0 };
			else if (i > 24 && i < 50) output_vector = { 0, 1, 0 };
			else output_vector = { 0, 0, 1 };

			TrainingSample ts(input_vector, output_vector);
			train_set.push_back(ts);
		}
	}
	open_file.close();

	return train_set;
}

vector<vector<double>> ReadTestingDataSet() {
	vector<vector<double>> test_set;

	ifstream open_file(TESTING_FILE);
	size_t pos = 0;
	string delimiter = DELIMITER;

	if (open_file.is_open()) {
		string line;
		string token;
		for (int i = 0; i < TESTING_DATA_LEN; i++) { 
			getline(open_file, line);
			vector<double> temp;
			for (int x = 0; x < 4; x++) {
				pos = line.find(delimiter);
				token = line.substr(0, pos);
				temp.push_back(stod(token));
				line.erase(0, pos + delimiter.length());
			}
			test_set.push_back(temp);
		}
	}
	open_file.close();

	return test_set;
}

int main() {
    std::cout << "\n--Start MLP--" << std::endl;

	vector<TrainingSample> train_set = ReadTrainingDataSet();

    std::vector<TrainingSample> train_set_bias(train_set);
    //set up bias
    for (auto &training_sample_with_bias : train_set_bias) {
        training_sample_with_bias.AddBiasValue(1);
    }

    // network parameters
    int num_features = train_set_bias[0].GetInputVectorSize();
    int num_outputs = train_set_bias[0].GetOutputVectorSize();
    double learning_rate = 0.01;
    int max_iterations = 500;
    double min_error_cost = 0.025;

    // create net
    Network network({num_features, 2, num_outputs}, {"sigmoid", "linear"});

    //Train and_gat_net
    network.Train(train_set_bias, learning_rate, max_iterations, min_error_cost);

    // test
	std::cout << "********** Test Running... **********" << std::endl;
	vector<vector<double>> test_set = ReadTestingDataSet(); //테스트 데이터를 읽어 트레이닝 데이터 셋으로 변환함
	for (const auto& testing_sample : test_set) {
		std::vector<double> output;
		network.GetOutput(testing_sample, &output);
		bool predicted_output_0 = output[0] > 0.5;
		bool predicted_output_1 = output[1] > 0.5;
		bool predicted_output_2 = output[2] > 0.5;

		std::cout << "Input: ";
		std::vector<double> input_vector = testing_sample;
		for (const auto& v : input_vector) {
			std::cout << v << ", ";
		}
		std::cout << " predicted value: " << predicted_output_0 << " " << predicted_output_1 << " " << predicted_output_2 << std::endl;

	}

	cout << "\n********** MLP is done... **********" << endl;
	return 0;

    // Save Model
    std::string model_path = "add_gat_net.bin";
    network.SaveNetwork(model_path);

    std::cout << "--Trained With Success--" << std::endl;
    return 0;
}