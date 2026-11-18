#include "Tensor.hpp"
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

using namespace std;
using namespace std::chrono;

const int WARMUP_RUNS = 5;
const int MEASURED_RUNS = 30;
const string CSV_FILE = "new_tensor_benchmark.csv";

struct BenchmarkResult {
    string pb_id, shape, operation;
    double average, median;
    long long checksum_value;
};

volatile long long benchmarkCheck = 0;

long long checksum(const Tensor<float>& tensor) {
    long long sum = 0;
    for (int value : tensor.data()) sum += value;
    return sum;
}

double medianOf(vector<double> values) {
    if (values.empty()) throw runtime_error("No timing results available.");
    sort(values.begin(), values.end());
    size_t n = values.size();
    return n % 2 == 0 ? (values[n/2-1] + values[n/2]) / 2.0 : values[n/2];
}

string shapeString(const vector<size_t>& shape) {
    string s = "{";
    for (size_t i = 0; i < shape.size(); ++i) {
        if (i) s += ",";
        s += to_string(shape[i]);
    }
    return s + "}";
}

void saveToCSV(const vector<BenchmarkResult>& results) {
    ofstream file(CSV_FILE);
    if (!file) throw runtime_error("Could not open CSV file.");
    file << fixed << setprecision(8);
    file << "PB_ID,Shape,Operation,Average (ms),Median (ms),Checksum\n";
    for (const auto& r : results)
        file << r.pb_id << ",\"" << r.shape << "\",\"" << r.operation
             << "\"," << r.average << "," << r.median << "," << r.checksum_value << "\n";
}

void recordResult(const string& pbId, const vector<size_t>& shape,
                  const string& operation, const vector<double>& times,
                  long long check, vector<BenchmarkResult>& results) {
    double average = accumulate(times.begin(), times.end(), 0.0) / times.size();
    double median = medianOf(times);
    cout << "\n" << pbId << " | " << operation << " | Shape " << shapeString(shape)
         << "\nAverage: " << average << " ms\nMedian: " << median
         << " ms\nChecksum: " << check << '\n';
    results.push_back({pbId, shapeString(shape), operation, average, median, check});
    saveToCSV(results);
}

// PB01-PB03: same operation label, with shape distinguishing each case.
void benchmarkCreation(const vector<size_t>& shape, const string& pbId,
                       vector<BenchmarkResult>& results) {
    vector<double> times;
    long long check = 0;
    for (int i = 0; i < WARMUP_RUNS; ++i) {
        Tensor<float> temp(shape);
        check += checksum(temp);
    }
    for (int i = 0; i < MEASURED_RUNS; ++i) {
        auto start = steady_clock::now();
        Tensor<float> temp(shape);
        auto end = steady_clock::now();
        times.push_back(duration<double, milli>(end-start).count());
        check += checksum(temp);
    }
    benchmarkCheck += check;
    recordResult(pbId, shape, "Tensor Creation / Allocation", times, check, results);
}

void benchmarkDestruction(const vector<size_t>& shape, const string& pbId,
                          vector<BenchmarkResult>& results) {
    vector<double> times;
    long long check = 0;
    for (int i = 0; i < WARMUP_RUNS; ++i) {
        auto temp = make_unique<Tensor<float>>(shape);
        check += checksum(*temp);
        temp.reset();
    }
    for (int i = 0; i < MEASURED_RUNS; ++i) {
        auto temp = make_unique<Tensor<float>>(shape);
        check += checksum(*temp);
        auto start = steady_clock::now();
        temp.reset();
        auto end = steady_clock::now();
        times.push_back(duration<double, milli>(end-start).count());
    }
    benchmarkCheck += check;
    recordResult(pbId, shape, "Tensor Destruction / Deallocation", times, check, results);
}

// PB04: time a complete sequential traversal and checksum.
void benchmarkSequentialRead(Tensor<float>& tensor, const vector<size_t>& shape,
                             vector<BenchmarkResult>& results) {
    vector<double> times;
    long long totalCheck = 0;
    for (int i = 0; i < WARMUP_RUNS; ++i) totalCheck += checksum(tensor);
    for (int i = 0; i < MEASURED_RUNS; ++i) {
        long long sum = 0;
        auto start = steady_clock::now();
        for (int value : tensor.data()) sum += value;
        auto end = steady_clock::now();
        times.push_back(duration<double, milli>(end-start).count());
        totalCheck += sum;
    }
    benchmarkCheck += totalCheck;
    recordResult("PB04", shape, "Sequential Read (Checksum)", times, totalCheck, results);
}

// PB05: 100,000 random reads; indices are generated before timing.
void benchmarkRandomRead(Tensor<float>& tensor, const vector<size_t>& shape,
                         vector<BenchmarkResult>& results) {
    constexpr size_t READ_COUNT = 100000;
    mt19937 generator(20261117);
    uniform_int_distribution<size_t> distribution(0, tensor.size()-1);
    vector<size_t> indices(READ_COUNT);
    for (auto& index : indices) index = distribution(generator);
    const float* values = tensor.data().data();

    vector<double> times;
    long long totalCheck = 0;
    for (int i = 0; i < WARMUP_RUNS; ++i) {
        long long sum = 0;
        for (size_t index : indices) sum += values[index];
        totalCheck += sum;
    }
    for (int i = 0; i < MEASURED_RUNS; ++i) {
        long long sum = 0;
        auto start = steady_clock::now();
        for (size_t index : indices) sum += values[index];
        auto end = steady_clock::now();
        times.push_back(duration<double, milli>(end-start).count());
        totalCheck += sum;
    }
    benchmarkCheck += totalCheck;
    recordResult("PB05", shape, "Seeded Random Read (100000 reads)", times, totalCheck, results);
}

void benchmarkAddition(const vector<size_t>& shape, vector<BenchmarkResult>& results) {
    Tensor<float> a(shape), b(shape), c(shape);
    a.fill(1); b.fill(2);
    vector<double> times;
    for (int i = 0; i < WARMUP_RUNS; ++i) c.addInto(a,b);
    for (int i = 0; i < MEASURED_RUNS; ++i) {
        auto start = steady_clock::now(); c.addInto(a,b); auto end = steady_clock::now();
        times.push_back(duration<double,milli>(end-start).count());
        benchmarkCheck += checksum(c);
    }
    long long check = checksum(c);
    if (!c.data().empty() && c.data()[0] != 3) throw runtime_error("Addition correctness check failed.");
    recordResult("PB06", shape, "Element-wise Addition", times, check, results);
}

void benchmarkSubtraction(const vector<size_t>& shape, vector<BenchmarkResult>& results) {
    Tensor<float> a(shape), b(shape), c(shape);
    a.fill(5); b.fill(1);
    vector<double> times;
    for (int i = 0; i < WARMUP_RUNS; ++i) c.subtractInto(a,b);
    for (int i = 0; i < MEASURED_RUNS; ++i) {
        auto start = steady_clock::now(); c.subtractInto(a,b); auto end = steady_clock::now();
        times.push_back(duration<double,milli>(end-start).count());
        benchmarkCheck += checksum(c);
    }
    long long check = checksum(c);
    if (!c.data().empty() && c.data()[0] != 4) throw runtime_error("Subtraction correctness check failed.");
    recordResult("PB07", shape, "Element-wise Subtraction", times, check, results);
}

void benchmarkMultiplication(const vector<size_t>& shape, vector<BenchmarkResult>& results) {
    Tensor<float> a(shape), b(shape), c(shape);
    a.fill(2); b.fill(3);
    vector<double> times;
    for (int i = 0; i < WARMUP_RUNS; ++i) c.multiplyInto(a,b);
    for (int i = 0; i < MEASURED_RUNS; ++i) {
        auto start = steady_clock::now(); c.multiplyInto(a,b); auto end = steady_clock::now();
        times.push_back(duration<double,milli>(end-start).count());
        benchmarkCheck += checksum(c);
    }
    long long check = checksum(c);
    if (!c.data().empty() && c.data()[0] != 6) throw runtime_error("Multiplication correctness check failed.");
    recordResult("PB08", shape, "Element-wise Multiplication", times, check, results);
}

// PB12: reshape a {1000,1000} tensor to {100,10000}.
void benchmarkReshape(vector<BenchmarkResult>& results) {
    const vector<size_t> oldShape = {1000,1000}, newShape = {100,10000};
    Tensor<float> tensor(oldShape);
    for (size_t i = 0; i < tensor.size(); ++i) tensor.data()[i] = static_cast<int>(i);

    vector<double> times;
    for (int i = 0; i < WARMUP_RUNS; ++i) {
        tensor.reshape(newShape);
        tensor.reshape(oldShape);
    }
    for (int i = 0; i < MEASURED_RUNS; ++i) {
        tensor.reshape(oldShape); // outside timed interval
        auto start = steady_clock::now();
        tensor.reshape(newShape);
        auto end = steady_clock::now();
        times.push_back(duration<double,milli>(end-start).count());
        if (tensor.shape() != newShape) throw runtime_error("Reshape correctness check failed.");
        benchmarkCheck += static_cast<long long>(tensor.size());
    }
    recordResult("PB12", oldShape, "Reshape {1000,1000} to {100,10000}",
                 times, static_cast<long long>(tensor.size()), results);
}

int main() {
    try {
        vector<BenchmarkResult> results;

        // PB01-PB03: creation and destruction are reported with identical labels
        // for each specified shape; the Shape column distinguishes the cases.
        const vector<vector<size_t>> allocationShapes = {
            {100,100}, {1000,1000}, {4000,4000}
        };
        const vector<string> allocationIds = {"PB01","PB02","PB03"};
        for (size_t i = 0; i < allocationShapes.size(); ++i) {
            benchmarkCreation(allocationShapes[i], allocationIds[i], results);
            benchmarkDestruction(allocationShapes[i], allocationIds[i], results);
        }

        const vector<size_t> readShape = {1000,1000};
        Tensor<float> readTensor(readShape);
        readTensor.fill(1);
        benchmarkSequentialRead(readTensor, readShape, results);
        benchmarkRandomRead(readTensor, readShape, results);

        // PB06-PB08 core size and required scaling sizes.
        const vector<vector<size_t>> arithmeticShapes = {
            {1000,1000}, {100,100}, {500,500}, {4000,4000}
        };
        for (const auto& shape : arithmeticShapes) {
            benchmarkAddition(shape, results);
            benchmarkSubtraction(shape, results);
            benchmarkMultiplication(shape, results);
        }

        // PB09-PB11 matrix multiplication intentionally omitted.
        benchmarkReshape(results);

        cout << "\nBenchmark runs completed.\nResults saved to " << CSV_FILE
             << "\nBenchmark checksum: " << benchmarkCheck << '\n';
    } catch (const exception& error) {
        cerr << "Error: " << error.what() << '\n';
        return 1;
    }
    return 0;
}
