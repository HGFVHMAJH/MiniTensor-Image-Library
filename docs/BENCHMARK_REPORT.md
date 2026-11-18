# Mini Tensor and BMP Image Library — Benchmark Report

## 1. Objective

This report compares the execution times of the custom C++ Mini Tensor and BMP Image Library with reference implementations using PyTorch and OpenCV. The measurements are intended to provide an initial performance profile and identify areas for optimization.

## 2. System Specifications

The benchmark environment used for development and testing:

| Specification | Details |
|---|---|
| Operating System | Ubuntu 24.04 |
| Compiler | GCC 13.3.0 |
| Build System | CMake 3.28.3 |
| C++ Standard | C++17 |
| CPU Cores Available | 2 |
| RAM | Approximately 7.8 GB |
| Compiler Optimization | `-O3 -DNDEBUG` |
| OpenMP | Not used |

## 3. Image benchmark results

| Operation | Custom C++ (ms) | OpenCV Python API (ms) | C++ / OpenCV | Lower measured time |
|---|---:|---:|---:|---|
| loading | 1.473880 | 0.311531 | 4.73× | OpenCV |
| grayscale | 1.475100 | 0.112060 | 13.16× | OpenCV |
| horizontal reflection | 0.847852 | 0.060603 | 13.99× | OpenCV |
| vertical flip | 0.732867 | 0.041177 | 17.80× | OpenCV |
| blur | 7.389460 | 0.219369 | 33.69× | OpenCV |
| edges | 14.190300 | 2.595852 | 5.47× | OpenCV |
| sepia | 3.648590 | 0.641126 | 5.69× | OpenCV |
| brightness | 3.234300 | 0.169075 | 19.13× | OpenCV |
| saving | 3.483740 | 3.403937 | 1.02× | OpenCV |

## 4. Tensor benchmark results

[View Benchmark CSV](tensor_benchmark.csv)

Results of operations in Pytorch
[View Refrence CSV](tensor_pytorch.csv)


## 5. Test Results

The project includes functional test results for the Mini Tensor library.

The test suite covers:
- Tensor creation and initialization
- Element indexing and modification
- Addition and subtraction
- Element-wise multiplication
- Matrix multiplication support status
- Tensor reshaping
- Memory allocation and copy behaviour

Test results are exported in CSV format using the required submission columns.

**Results file:** [Test Results](testcase_results.csv)
**Testing Code:** [Test Code](../testcases.cpp)

## Summary Table

| Category | Total | Passed | Failed | Not supported | Remarks |
|---|---:|---:|---:|---:|---|
| Functional tests | 50 | 41 | 0 | 9 | Matrix Multiplication not supported |
| Performance benchmarks | 12 core benchmark IDs | 8 | 1 | 3 |  Wrongly initialized Tensor |

## 6. Optimization Techniques

### Contiguous Memory Storage Using std::vector

The Tensor class uses `std::vector<T>` to store tensor elements in contiguous memory.

This provides:
- Efficient memory allocation and management.
- Direct element access.
- Better cache locality during sequential operations.
- Simple storage management without manual memory allocation.

The tensor uses a row-major layout, meaning elements are stored consecutively in memory.

### Preallocated Output Operations

The library provides three operations:

- `addInto()`
- `subtractInto()`
- `multiplyInto()`

These functions perform arithmetic operations and store their results in an existing output tensor.

For example:

```cpp
c.addInto(a, b);
```

This computes the elementwise sum of tensors `a` and `b` and stores the result in `c`.

Similarly:

```cpp
c.subtractInto(a, b);
c.multiplyInto(a, b);
```

The output tensor is allocated before the operation begins. This avoids creating a new tensor and allocating new output storage for every arithmetic operation.

As a result, these functions reduce unnecessary memory allocations and allow the arithmetic operations to reuse existing storage.

### Separation of Computation and Benchmark Preparation

Input tensors are prepared before the timed sections.

Only the target operation is measured, while correctness checks and result consumption are performed outside the timed sections.

This helps make the benchmark more representative of the execution time of the individual operation.
