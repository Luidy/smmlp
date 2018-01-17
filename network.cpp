//
// Created by sm19 on 9/30/17.
//

#include "network.h"

Network::Network(const std::vector<int> &layers_node,
                 const std::vector<std::string> &activations,
                 bool use_constant_weight_init,
                 double constant_weight_init) {
    assert(layers_node.size() >= 2);
    assert(activations.size() + 1 == layers_node.size());

    CreateMLP(layers_node, activations, use_constant_weight_init, constant_weight_init);
};

Network::Network(std::string &filename) {
    LoadNetwork(filename);
}

Network::~Network() {
    m_num_inputs = 0;
    m_num_outputs = 0;
    m_num_hidden_layers = 0;
    m_layers_nodes.clear();
    m_layers.clear();
};

void Network::CreateMLP(const std::vector<int> &layers_node,
                        const std::vector<std::string> &activations,
                        bool use_constant_weight_init,
                        double constant_weight_init) {
    m_layers_nodes = layers_node;
    m_num_inputs = m_layers_nodes[0];
    m_num_outputs = m_layers_nodes[m_layers_nodes.size() - 1];
    m_num_hidden_layers = m_layers_nodes.size() - 2;

    for (int i = 0; i < m_layers_nodes.size() - 1; i++) {
        m_layers.emplace_back(Layer(m_layers_nodes[i],
                                    m_layers_nodes[i + 1],
                                    activations[i],
                                    use_constant_weight_init,
                                    constant_weight_init));
    }
};

void Network::SaveNetwork(std::string &filename) const {

    FILE *file;
    file = fopen(filename.c_str(), "wb");
    fwrite(&m_num_inputs, sizeof(m_num_inputs), 1, file);
    fwrite(&m_num_outputs, sizeof(m_num_outputs), 1, file);
    fwrite(&m_num_hidden_layers, sizeof(m_num_hidden_layers), 1, file);
    if (!m_layers_nodes.empty())
        fwrite(&m_layers_nodes[0], sizeof(m_layers_nodes[0]), m_layers_nodes.size(), file);

    for (const auto &m_layer : m_layers) {
        m_layer.SaveLayer(file);
    }
    fclose(file);
};

void Network::LoadNetwork(std::string &filename) {
    m_layers_nodes.clear();
    m_layers.clear();

    FILE *file;
    file = fopen(filename.c_str(), "rb");
    fread(&m_num_inputs, sizeof(m_num_inputs), 1, file);
    fread(&m_num_outputs, sizeof(m_num_outputs), 1, file);
    fread(&m_num_hidden_layers, sizeof(m_num_hidden_layers), 1, file);
    m_layers_nodes.resize(m_num_hidden_layers + 2);
    if (!m_layers_nodes.empty())
        fread(&m_layers_nodes[0], sizeof(m_layers_nodes[0]), m_layers_nodes.size(), file);
    m_layers.resize(m_layers_nodes.size() - 1);
    for (int i = 0; i < m_layers.size(); i++) {
        m_layers[i].LoadLayer(file);
    }
    fclose(file);
};

void Network::GetOutput(const std::vector<double> &input,
                        std::vector<double> *output,
                        std::vector<std::vector<double>> *all_layers_activations) const {
    assert(input.size() == m_num_inputs);
    int temp_size;
    if (m_num_hidden_layers == 0)
        temp_size = m_num_outputs;
    else
        temp_size = m_layers_nodes[1];

    std::vector<double> temp_in(m_num_inputs, 0.0);
    std::vector<double> temp_out(temp_size, 0.0);
    temp_in = input;

    for (int i = 0; i < m_layers.size(); ++i) {
        if (i > 0) {
            //Store this layer activation
            if (all_layers_activations != nullptr)
                all_layers_activations->emplace_back(std::move(temp_in));

            temp_in.clear();
            temp_in = temp_out;
            temp_out.clear();
            temp_out.resize(m_layers[i].GetOutputSize());
        }
        m_layers[i].GetOutputAfterActivationFunction(temp_in, &temp_out);
    }

    if (temp_out.size() > 1)
        utils::Softmax(&temp_out);
    *output = temp_out;

    //Add last layer activation
    if (all_layers_activations != nullptr)
        all_layers_activations->emplace_back(std::move(temp_in));
}

void Network::GetOutputClass(const std::vector<double> &output, size_t *class_id) const {
    utils::GetIdMaxElement(output, class_id);
}

void Network::UpdateWeights(const std::vector<std::vector<double>> &all_layers_activations,
                            const std::vector<double> &deriv_error,
                            double learning_rate) {

    std::vector<double> temp_deriv_error = deriv_error;
    std::vector<double> deltas{};
    //m_layers.size() equals (m_num_hidden_layers + 1)
    for (int i = m_num_hidden_layers; i >= 0; --i) {
        m_layers[i].UpdateWeights(all_layers_activations[i], temp_deriv_error, learning_rate, &deltas);
        if (i > 0) {
            temp_deriv_error.clear();
            temp_deriv_error = std::move(deltas);
            deltas.clear();
        }
    }
};

void Network::Train(const std::vector<TrainingSample> &training_sample_set_with_bias,
                    double learning_rate,
                    int max_iterations,
                    double min_error_cost,
                    bool output_log) {

    double current_iteration_cost_function = 0.0;
    int epoch = 0;
    for (epoch = 0; epoch < max_iterations; epoch++) {
        current_iteration_cost_function = 0.0;
        for (auto &training_sample_with_bias : training_sample_set_with_bias) {

            std::vector<double> predicted_output;
            std::vector<std::vector<double> > all_layers_activations;

            GetOutput(training_sample_with_bias.input_vector(),
                      &predicted_output,
                      &all_layers_activations);

            const std::vector<double> &correct_output =
                    training_sample_with_bias.output_vector();

            assert(correct_output.size() == predicted_output.size());
            std::vector<double> deriv_error_output(predicted_output.size());

            if (output_log && ((epoch % 5) == 0)) {
                std::stringstream temp_training;

                temp_training << training_sample_with_bias << "\t";
                temp_training << "Predicted output: [";
                for (int j = 0; j < predicted_output.size(); j++) {
                    if (j != 0)
                        temp_training << ", ";
                    temp_training << predicted_output[j];
                }
                temp_training << "]";
                std::cout << temp_training.str() << std::endl;
            }

            for (int j = 0; j < predicted_output.size(); j++) {
                current_iteration_cost_function += (std::pow)((correct_output[j] - predicted_output[j]), 2);
                deriv_error_output[j] = -2 * (correct_output[j] - predicted_output[j]);
            }
            UpdateWeights(all_layers_activations, deriv_error_output, learning_rate);
        }

        if (output_log && (epoch % 5 == 0))
            std::cout << "Iteration " << epoch << " cost function f(error): "
                      << current_iteration_cost_function << std::endl << std::endl;
        if (current_iteration_cost_function < min_error_cost) break;
    }
};


