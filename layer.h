//
// Created by sm19 on 9/30/17.
//

#ifndef SMMLP_LAYER_H
#define SMMLP_LAYER_H

#include "utils.h"
#include "node.h"

#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <cassert>

class Layer {
public:
    Layer();

    Layer(int num_inputs_per_node,
          int num_nodes,
          const std::string &activation_function,
          bool use_constant_weight_init = true,
          double constant_weight_init = 0.5);

    ~Layer();

    int GetInputSize() const;

    int GetOutputSize() const;

    const std::vector<Node> &GetNodes() const;

    void GetOutputAfterActivationFunction(const std::vector<double> &input,
                                          std::vector<double> *output) const;

    void UpdateWeights(const std::vector<double> &input_layer_activation,
                       const std::vector<double> &deriv_error,
                       double m_learning_rate,
                       std::vector<double> *deltas);

    void SaveLayer(FILE *file) const;

    void LoadLayer(FILE *file);

protected:
    int m_num_inputs_per_node{0};
    int m_num_nodes{0};
    std::vector<Node> m_nodes;

    std::string m_activation_function_str;
    std::function<double(double)> m_activation_function;
    std::function<double(double)> m_deriv_activation_function;
};

#endif //SMMLP_LAYER_H
