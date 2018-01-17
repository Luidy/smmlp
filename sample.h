//
// Created by sm19 on 9/30/17.
//

#ifndef SMMLP_SAMPLE_H
#define SMMLP_SAMPLE_H


#include <cstdlib>
#include <vector>
#include <ostream>

class sample {
public:
    explicit sample(const std::vector<double> &input_vector) {
        input = input_vector;
    }

    const std::vector<double> &input_vector() const {
        return input;
    }

    int GetInputVectorSize() const {
        return (int)input.size();
    }

    void AddBiasValue(double bias_value) {
        input.insert(input.begin(), bias_value);
    }

    friend std::ostream &operator<<(std::ostream &stream, sample const &obj) {
        obj.PrintSample(stream);
        return stream;
    };
protected:
    virtual void PrintSample(std::ostream &stream) const {
        stream << "Input vector: [";
        for (int i = 0; i < input.size(); i++) {
            if (i != 0)
                stream << ", ";
            stream << input[i];
        }
        stream << "]";
    }

    std::vector<double> input;
};


class TrainingSample : public sample {
public:
    TrainingSample(const std::vector<double> &input_vector,
                   const std::vector<double> &output_vector) :
            sample(input_vector) {
        output = output_vector;
    }

    const std::vector<double> &output_vector() const {
        return output;
    }

    int GetOutputVectorSize() const {
        return (int)output.size();
    }

protected:
    void PrintSample(std::ostream &stream) const override {
        stream << "Input vector: [";
        for (int i = 0; i < input.size(); i++) {
            if (i != 0)
                stream << ", ";
            stream << input[i];
        }
        stream << "]";
        stream << "; ";
        stream << "Output vector: [";
        for (int i = 0; i < output.size(); i++) {
            if (i != 0)
                stream << ", ";
            stream << output[i];
        }
        stream << "]";
    }

    std::vector<double> output;
};

#endif //SMMLP_SAMPLE_H
