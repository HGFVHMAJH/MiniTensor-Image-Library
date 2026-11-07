#include "Tensor.hpp"

#include <iostream>
#include <chrono>
#include <vector>
#include <algorithm>
#include <fstream>
#include <numeric>
#include <string>
#include <stdexcept>
#include <memory>
#include <iomanip>
#include <cstdint>

using namespace std;
using namespace std::chrono;

const int WARMUP_RUNS = 3;
const int MEASURED_RUNS = 10;
const string CSV_FILE = "new_tensor_benchmark.csv";

// Stores timing statistics
struct BenchmarkResult
{
    string operation;
    double average;
    double median;
};

// Prevents benchmark results from being discarded
volatile long long benchmarkCheck = 0;

// Consumes every tensor element outside the timed section
long long checksum(const Tensor<int>& tensor)
{
    long long sum = 0;

    for (int value : tensor.data())
        sum += value;

    return sum;
}

// Reusable function to calculate average and median
BenchmarkResult calculateResults(const string& operation, vector<double> times)
{
    if (times.empty())
        throw runtime_error("No timing results available.");

    double total = accumulate(times.begin(), times.end(), 0.0);
    double average = total / times.size();

    sort(times.begin(), times.end());

    double median;

    if (times.size() % 2 == 0)
    {
        median = (times[times.size() / 2 - 1] +
                  times[times.size() / 2]) / 2.0;
    }
    else
    {
        median = times[times.size() / 2];
    }

    return {operation, average, median};
}

// Saves all completed results to CSV
void saveToCSV(const vector<BenchmarkResult>& results)
{
    ofstream file(CSV_FILE);

    if (!file.is_open())
        throw runtime_error("Could not open CSV file.");

    file << fixed << setprecision(8);
    file << "Operation,Average (ms),Median (ms)\n";

    for (const auto& result : results)
    {
        file << result.operation << ","
             << result.average << ","
             << result.median << "\n";
    }
}

// Prints and stores each benchmark result
void recordResult(
    const string& operation,
    const vector<double>& times,
    vector<BenchmarkResult>& results)
{
    BenchmarkResult result = calculateResults(operation, times);

    cout << "\n" << operation << "\n";
    cout << "Average: " << result.average << " ms\n";
    cout << "Median: " << result.median << " ms\n";

    results.push_back(result);
    saveToCSV(results);
}

// Measures tensor allocation / creation
void benchmarkCreation(vector<BenchmarkResult>& results)
{
    vector<double> times;

    for (int i = 0; i < WARMUP_RUNS; i++)
    {
        Tensor<int> temp({2, 2});
        benchmarkCheck += checksum(temp);
    }

    for (int i = 0; i < MEASURED_RUNS; i++)
    {
        auto start = steady_clock::now();

        Tensor<int> temp({2, 2});

        auto end = steady_clock::now();

        times.push_back(duration<double, milli>(end - start).count());

        benchmarkCheck += checksum(temp);
    }

    recordResult("Tensor Creation / Allocation", times, results);
}

// Measures tensor destruction separately
void benchmarkDestruction(vector<BenchmarkResult>& results)
{
    vector<double> times;

    for (int i = 0; i < WARMUP_RUNS; i++)
    {
        auto temp = make_unique<Tensor<int>>(vector<size_t>{2, 2});
        temp.reset();
    }

    for (int i = 0; i < MEASURED_RUNS; i++)
    {
        auto temp = make_unique<Tensor<int>>(vector<size_t>{2, 2});

        auto start = steady_clock::now();

        temp.reset();

        auto end = steady_clock::now();

        times.push_back(
            duration<double, milli>(end - start).count()
        );
    }

    recordResult("Tensor Destruction / Deallocation", times, results);
}

// Measures element access
void benchmarkElementAccess(
    Tensor<int>& a,
    vector<BenchmarkResult>& results)
{
    vector<double> times;
    // access first element
    vector<size_t> indices = {0, 0};

    volatile int check = 0;

    for (int i = 0; i < WARMUP_RUNS; i++)
        check = a.getElement(indices);

    for (int i = 0; i < MEASURED_RUNS; i++)
    {
        auto start = steady_clock::now();

        int value = a.getElement(indices);

        auto end = steady_clock::now();

        check = value;

        times.push_back(
            duration<double, milli>(end - start).count()
        );
    }

    benchmarkCheck += check;

    recordResult("Element Access", times, results);
}

// Measures addition with preallocated output
void benchmarkAddition(Tensor<int>& a, Tensor<int>& b, Tensor<int>& c, vector<BenchmarkResult>& results)
{
    vector<double> times;

    for (int i = 0; i < WARMUP_RUNS; i++)
        c.addInto(a, b);

    for (int i = 0; i < MEASURED_RUNS; i++)
    {
        auto start = steady_clock::now();

        c.addInto(a, b);

        auto end = steady_clock::now();

        times.push_back(
            duration<double, milli>(end - start).count()
        );

        benchmarkCheck += checksum(c);
    }

    if (c.data()[0] != a.data()[0] + b.data()[0])
        throw runtime_error("Addition correctness check failed.");

    recordResult("Addition (Preallocated Output)", times, results);
}

// Measures subtraction with preallocated output
void benchmarkSubtraction(Tensor<int>& a, Tensor<int>& b, Tensor<int>& c, vector<BenchmarkResult>& results)
{
    vector<double> times;

    for (int i = 0; i < WARMUP_RUNS; i++)
        c.subtractInto(a, b);

    for (int i = 0; i < MEASURED_RUNS; i++)
    {
        auto start = steady_clock::now();

        c.subtractInto(a, b);

        auto end = steady_clock::now();

        times.push_back(
            duration<double, milli>(end - start).count()
        );

        benchmarkCheck += checksum(c);
    }

    if (c.data()[0] != a.data()[0] - b.data()[0])
        throw runtime_error("Subtraction correctness check failed.");

    recordResult("Subtraction (Preallocated Output)", times, results);
}

// Measures elementwise multiplication
void benchmarkMultiplication(
    Tensor<int>& a,
    Tensor<int>& b,
    Tensor<int>& c,
    vector<BenchmarkResult>& results)
{
    vector<double> times;

    for (int i = 0; i < WARMUP_RUNS; i++)
        c.multiplyInto(a, b);

    for (int i = 0; i < MEASURED_RUNS; i++)
    {
        auto start = steady_clock::now();

        c.multiplyInto(a, b);

        auto end = steady_clock::now();

        times.push_back(
            duration<double, milli>(end - start).count()
        );

        benchmarkCheck += checksum(c);
    }

    if (c.data()[0] != a.data()[0] * b.data()[0])
        throw runtime_error("Multiplication correctness check failed.");

    recordResult(
        "Elementwise Multiplication (Preallocated Output)",
        times,
        results
    );
}

// Measures reshape
void benchmarkReshape(vector<BenchmarkResult>& results)
{
    vector<double> times;

    for (int i = 0; i < WARMUP_RUNS; i++)
    {
        Tensor<int> temp({2, 2});
        temp.reshape({1, 4});
    }

    for (int i = 0; i < MEASURED_RUNS; i++)
    {
        Tensor<int> temp({2, 2});

        auto start = steady_clock::now();

        temp.reshape({1, 4});

        auto end = steady_clock::now();

        times.push_back(
            duration<double, milli>(end - start).count()
        );

        if (temp.shape() != vector<size_t>{1, 4})
            throw runtime_error("Reshape correctness check failed.");

        benchmarkCheck += static_cast<long long>(temp.size());
    }

    recordResult("Reshape", times, results);
}

int main()
{
    try
    {
        vector<BenchmarkResult> results;

        // sample data for benchmark
        // Prepare input tensors outside timed sections
        Tensor<int> a({2, 2});
        Tensor<int> b({2, 2});
        Tensor<int> c({2, 2});

        a.fill(10);
        b.fill(20);

        benchmarkCreation(results);
        benchmarkDestruction(results);
        benchmarkElementAccess(a, results);
        benchmarkAddition(a, b, c, results);
        benchmarkSubtraction(a, b, c, results);
        benchmarkMultiplication(a, b, c, results);
        benchmarkReshape(results);

        cout << "\nAll benchmarks completed successfully.\n";
        cout << "Results saved to " << CSV_FILE << "\n";
        cout << "Benchmark checksum: " << benchmarkCheck << "\n";
    }
    catch (const exception& error)
    {
        cerr << "Error: " << error.what() << '\n';
        return 1;
    }

    return 0;
}
