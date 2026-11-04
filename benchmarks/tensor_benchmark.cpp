#include "../Tensor.hpp"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

using Clock = std::chrono::steady_clock;

double benchmarkSink = 0.0;

template <typename Function>
void measure(
    std::ofstream& csv,
    const std::string& operation,
    std::size_t elements,
    int repetitions,
    int operationsPerIteration,
    const Function& function)
{
    // Warm-up
    for (int i = 0; i < 3; ++i)
        function();

    std::vector<double> times;
    times.reserve(repetitions);

    for (int r = 0; r < repetitions; ++r)
    {
        auto start = Clock::now();

        for (int i = 0; i < operationsPerIteration; ++i)
            function();

        auto end = Clock::now();

        double elapsed =
            std::chrono::duration<double, std::milli>(end - start).count();

        times.push_back(elapsed / operationsPerIteration);
    }

    std::sort(times.begin(), times.end());

    double median = times[times.size() / 2];
    double average = 0.0;

    for (double t : times)
        average += t;

    average /= times.size();

    csv << operation << ','
        << elements << ','
        << repetitions << ','
        << median << ','
        << average << '\n';

    std::cout << operation
              << " | elements: " << elements
              << " | median: " << median << " ms\n";
}

int main()
{
    std::ofstream csv("benchmarks/results/tensor_cpp.csv");

    if (!csv)
    {
        std::cerr << "Could not create CSV file.\n";
        return 1;
    }

    csv << "operation,elements,repetitions,median_ms,average_ms\n";

    const std::vector<std::size_t> dimensions = {
        100, 500, 1000, 2000
    };

    constexpr int repetitions = 15;
    constexpr int accessBatch = 10000;

    for (std::size_t side : dimensions)
    {
        const std::size_t elements = side * side;

        const std::vector<std::size_t> shape = {side, side};
        const std::vector<std::size_t> flatShape = {elements};
        const std::vector<std::size_t> index = {
            side / 2, side / 2
        };

        std::vector<float> values(elements, 2.5f);

        Tensor<float> a(shape, values);
        Tensor<float> b(shape, values);

        std::cout << "\nTesting " << side << " x " << side
                  << " (" << elements << " elements)\n";

        // 1. Construction: allocation and zero initialization
        measure(csv, "construction", elements, repetitions, 1, [&]() {
            Tensor<float> t(shape);
            benchmarkSink += static_cast<double>(t.size());
        });

        // 2. Fill an existing tensor
        measure(csv, "fill", elements, repetitions, 1, [&]() {
            a.fill(3.14f);
            benchmarkSink += a.data()[0];
        });

        // 3. Copy values into an existing tensor
        measure(csv, "fillTensor", elements, repetitions, 1, [&]() {
            a.fillTensor(values);
            benchmarkSink += a.data()[0];
        });

        // 4. Read one element repeatedly
        measure(csv, "getElement", elements, repetitions,
                accessBatch, [&]() {
            benchmarkSink += a.getElement(index);
        });

        // 5. Modify one element repeatedly
        float nextValue = 1.0f;

        measure(csv, "setElement", elements, repetitions,
                accessBatch, [&]() {
            a.setElement(nextValue, index);
            nextValue += 0.001f;
        });

        // 6. Reshape back and forth, preserving element count
        measure(csv, "reshape", elements, repetitions, 1, [&]() {
            a.reshape(flatShape);
            a.reshape(shape);
            benchmarkSink += static_cast<double>(a.rank());
        });

        // 7. Addition
        measure(csv, "addition", elements, repetitions, 1, [&]() {
            Tensor<float> result = a + b;
            benchmarkSink += result.data()[0];
        });

        // 8. Subtraction
        measure(csv, "subtraction", elements, repetitions, 1, [&]() {
            Tensor<float> result = a - b;
            benchmarkSink += result.data()[0];
        });

        // 9. Element-wise multiplication
        measure(csv, "multiplication", elements, repetitions, 1, [&]() {
            Tensor<float> result = a * b;
            benchmarkSink += result.data()[0];
        });
    }

    std::cout << "\nC++ benchmark complete.\n";
    std::cout << "Checksum: " << benchmarkSink << '\n';
    std::cout << "Results: benchmarks/results/tensor_cpp.csv\n";

    return 0;
}
