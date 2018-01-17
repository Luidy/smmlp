//
// Created by sm19 on 9/30/17.
//

#ifndef SMMLP_NODE_H
#define SMMLP_NODE_H


#include "utils.h"
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <cassert>
#include <functional>

class Node {
public:
    Node();

    Node(int num_inputs,
         bool use_constant_weight_init/* = true*/,
         double constant_weight_init /*= 0.5*/);

    ~Node();

    void WeightInitialization(int num_inputs,
                              bool use_constant_weight_init = true,
                              double constant_weight_init = 0.5);

    int GetInputSize() const;

    void SetInputSize(int num_inputs);

    double GetBias() const;

    void SetBias(double bias);

    std::vector<double> &GetWeights();

    const std::vector<double> &GetWeights() const;

    int GetWeightsVectorSize() const;


    void GetInputInnerProdWithWeights(const std::vector<double> &input,
                                      double *output) const;

    void GetOutputAfterActivationFunction(const std::vector<double> &input,
                                          std::function<double(double)> activation_function,
                                          double *output) const;

    void GetBooleanOutput(const std::vector<double> &input,
                          std::function<double(double)> activation_function,
                          bool *bool_output,
                          double threshold = 0.5) const;

    void UpdateWeights(const std::vector<double> &x,
                       double error,
                       double learning_rate);

    void UpdateWeight(int weight_id,
                      double increment,
                      double learning_rate);

    void SaveNode(FILE *file) const;

    void LoadNode(FILE *file);

protected:
    int m_num_inputs{0};
    double m_bias{0.0};
    std::vector<double> m_weights;
};

#endif //SMMLP_NODE_H
