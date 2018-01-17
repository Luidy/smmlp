//
// Created by sm19 on 9/30/17.
//

#include "node.h"

#include <utility>

Node::Node() {
    m_num_inputs = 0;
    m_bias = 0.0;
    m_weights.clear();
}

Node::~Node() {
    m_num_inputs = 0;
    m_bias = 0.0;
    m_weights.clear();
};

Node::Node(int num_inputs,
           bool use_constant_weight_init = true,
           double constant_weight_init = 0.5) {
    m_num_inputs = num_inputs;
    m_bias = 0.0;
    m_weights.clear();

    //initialize weight vector
    WeightInitialization(m_num_inputs,
                         use_constant_weight_init,
                         constant_weight_init);
};

void Node::WeightInitialization(int num_inputs,
                                bool use_constant_weight_init /*= true*/,
                                double constant_weight_init /*= 0.5*/) {
    m_num_inputs = num_inputs;
    //initialize weight vector
    if (use_constant_weight_init) {
        m_weights.resize(m_num_inputs, constant_weight_init);
    } else {
        m_weights.resize(m_num_inputs);
        std::generate_n(m_weights.begin(),
                        m_num_inputs,
                        utils::gen_rand());
    }
}

void Node::GetInputInnerProdWithWeights(const std::vector<double> &input,
                                        double *output) const {
    assert(input.size() == m_weights.size());
    double inner_prod = std::inner_product(begin(input),
                                           end(input),
                                           begin(m_weights),
                                           0.0);
    *output = inner_prod;
}

void Node::GetOutputAfterActivationFunction(const std::vector<double> &input,
                                            std::function<double(double)> activation_function,
                                            double *output) const {
    double inner_prod = 0.0;
    GetInputInnerProdWithWeights(input, &inner_prod);
    *output = activation_function(inner_prod);
}

int Node::GetInputSize() const {
    return m_num_inputs;
}

void Node::SetInputSize(int num_inputs) {
    m_num_inputs = num_inputs;
}

double Node::GetBias() const {
    return m_bias;
}

void Node::SetBias(double bias) {
    m_bias = bias;
}

std::vector<double> &Node::GetWeights() {
    return m_weights;
}

const std::vector<double> &Node::GetWeights() const {
    return m_weights;
}

int Node::GetWeightsVectorSize() const {
    return (int) m_weights.size();
}

void Node::GetBooleanOutput(const std::vector<double> &input,
                            std::function<double(double)> activation_function,
                            bool *bool_output,
                            double threshold) const {
    double value;
    GetOutputAfterActivationFunction(input, std::move(activation_function), &value);
    *bool_output = value > threshold;
};

void Node::UpdateWeights(const std::vector<double> &x,
                         double error,
                         double learning_rate) {
    assert(x.size() == m_weights.size());
    for (size_t i = 0; i < m_weights.size(); i++)
        m_weights[i] += x[i] * learning_rate * error;
};

void Node::UpdateWeight(int weight_id,
                        double increment,
                        double learning_rate) {
    m_weights[weight_id] += learning_rate * increment;
}

void Node::SaveNode(FILE *file) const {
    fwrite(&m_num_inputs, sizeof(m_num_inputs), 1, file);
    fwrite(&m_bias, sizeof(m_bias), 1, file);
    fwrite(&m_weights[0], sizeof(m_weights[0]), m_weights.size(), file);
};

void Node::LoadNode(FILE *file) {
    m_weights.clear();

    fread(&m_num_inputs, sizeof(m_num_inputs), 1, file);
    fread(&m_bias, sizeof(m_bias), 1, file);
    m_weights.resize(m_num_inputs);
    fread(&m_weights[0], sizeof(m_weights[0]), m_weights.size(), file);
};