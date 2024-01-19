/********************************************************************************
 * @brief Implementation of simple neural network in C++
 ********************************************************************************/
#include <vector>
#include <neural_network.hpp>
#include "button.hpp"
#include "led.hpp"

namespace rpi = yrgo::rpi;

using namespace yrgo::machine_learning;

int main(void) {
    
    const std::vector<std::vector<double>> train_input{
        {0, 0, 0, 0}, {0, 0, 0, 1}, {0, 0, 1, 0}, {0, 0, 1, 1},
        {0, 1, 0, 0}, {0, 1, 0, 1}, {0, 1, 1, 0}, {0, 1, 1, 1},
        {1, 0, 0, 0}, {1, 0, 0, 1}, {1, 0, 1, 0}, {1, 0, 1, 1},
        {1, 1, 0, 0}, {1, 1, 0, 1}, {1, 1, 1, 0}, {1, 1, 1, 1}};

    const std::vector<std::vector<double>> train_output{
        {0}, {1}, {1}, {0},
        {1}, {0}, {0}, {1},
        {1}, {0}, {0}, {1},
        {0}, {1}, {1}, {0}};

    NeuralNetwork network{4, 8, 1, ActFunc::kTanh};
    network.AddTrainingData(train_input, train_output);
    if (network.Train(100000, 0.01)) {
        network.PrintPredictions(train_input, 2);

    }
    return 0;
}