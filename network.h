//
// Created by sm19 on 9/30/17.
//

#ifndef SMMLP_NETWORK_H
#define SMMLP_NETWORK_H


#include <vector>
#include <string>
#include "layer.h"
#include "sample.h"

class Network {
public:
    Network(const std::vector<int> &layers_nodes,
            const std::vector<std::string> &activations,
            bool use_constant_weight_init = false,
            double constant_weight_init = 0.5);

    explicit Network(std::string &filename);

    ~Network();

    void SaveNetwork(std::string &filename) const;

    void LoadNetwork(std::string &filename);

    void GetOutputClass(const std::vector<double> &output, size_t *class_id) const;

    void GetOutput(const std::vector<double> &input,
                   std::vector<double> *output,
                   std::vector<std::vector<double>> *all_layers_activations = nullptr) const;


    void Train(const std::vector<TrainingSample> &training_sample_set_with_bias,
               double learning_rate,
               int max_iterations = 5000,
               double min_error_cost = 0.001,
               bool output_log = true);

protected:
    void UpdateWeights(const std::vector<std::vector<double>> &all_layers_activations,
                       const std::vector<double> &error,
                       double learning_rate);

private:
    void CreateMLP(const std::vector<int> &layers_nodes,
                   const std::vector<std::string> &activations,
                   bool use_constant_weight_init,
                   double constant_weight_init = 0.5);

    int m_num_inputs{0};
    int m_num_outputs{0};
    int m_num_hidden_layers{0};
    std::vector<int> m_layers_nodes;
    std::vector<Layer> m_layers;
};

#endif //SMMLP_NETWORK_H
