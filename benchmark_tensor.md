# Benchmarking

## 1. Overview

The operations considered include:

- Tensor creation
- Element access
- Addition
- Subtraction
- Elementwise multiplication
- Reshape
- Memory allocation and deallocation

## 2. Source Code

The following files contain the implementation of the project:

| File | Description |
|---|---|
| [Tensor.hpp](Tensor.hpp) | Tensor class implementation |
| [Image.hpp](Image.hpp) | Image class declaration |
| [Image.cpp](Image.cpp) | Image processing implementation |
| [main.cpp](main.cpp) | Main application |
| [CMakeLists.txt](CMakeLists.txt) | Build configuration |

## 3. Benchmarking Program

The benchmarking program used for the updated tensor benchmarks is:

[benchmark.cpp](benchmark.cpp)

It measures tensor creation, destruction, element access, arithmetic operations and reshape.

The benchmarking procedure uses:

- 3 warm-up runs
- 10 measured runs
- Average execution time
- Median execution time

The benchmark excludes input tensor preparation and result-checking work from the timed arithmetic sections.

## 4. Benchmark Results

The tensor benchmark results are available here:

[View Benchmark CSV](sample_tensor_benchmark.csv)

The CSV contains average and median execution times for each measured operation.


## 5. System Specifications

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
