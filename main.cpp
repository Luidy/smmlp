//
// Created by sm19 on 9/30/17.
//

#include <iostream>
#include "sample.h"
#include "network.h"

int main() {
    std::cout << "\n--AND Function Traning--" << std::endl;
    std::vector<TrainingSample> train_set = {
            {{0, 0}, {0.0}},
            {{0, 1}, {0.0}},
            {{0, 1}, {0.0}},
            {{1, 0}, {0.0}},
            {{1, 1}, {1.0}},
            {{1, 1}, {1.0}},
            {{1, 1}, {1.0}}};


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
    Network and_gat_net({num_features, 2, num_outputs}, {"sigmoid", "linear"});

    //Train and_gat_net
    and_gat_net.Train(train_set_bias, learning_rate, max_iterations, min_error_cost);

    // test
    std::cout << "--Simple AND Function test--" << std::endl;
    for (const auto &training_sample : train_set_bias) {
        std::vector<double> output;
        and_gat_net.GetOutput(training_sample.input_vector(), &output);
        for (int i = 0; i < num_outputs; i++) {
            bool predicted_output = output[i] > 0.5;

            std::cout << "Input: ";
            std::vector<double> input_vector = training_sample.input_vector();
            for (const auto &v: input_vector) {
                std::cout << v << ", ";
            }
            std::cout << " predicted-value: " << predicted_output << std::endl;
        }
    }

    // Save Model
    std::string model_path = "add_gat_net.bin";
    and_gat_net.SaveNetwork(model_path);

    std::cout << "--Trained With Success--" << std::endl;
    return 0;
}