import csv
import os
import statistics
import time

import torch

# CPU-only, one-thread configuration
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

os.makedirs("benchmarks/results", exist_ok=True)

DIMENSIONS = [100, 500, 1000, 2000]
REPETITIONS = 15
ACCESS_BATCH = 10000

results = []


def measure(operation, elements, function, operations_per_iteration=1):
    # Warm-up
    for _ in range(3):
        function()

    times = []

    for _ in range(REPETITIONS):
        start = time.perf_counter_ns()

        for _ in range(operations_per_iteration):
            function()

        end = time.perf_counter_ns()

        elapsed_ms = (end - start) / 1_000_000
        times.append(elapsed_ms / operations_per_iteration)

    median = statistics.median(times)
    average = statistics.mean(times)

    results.append({
        "operation": operation,
        "elements": elements,
        "repetitions": REPETITIONS,
        "median_ms": median,
        "average_ms": average,
    })

    print(
        f"{operation:16} | elements: {elements:>8} "
        f"| median: {median:.6f} ms"
    )


checksum = 0.0

for side in DIMENSIONS:
    elements = side * side
    shape = (side, side)
    flat_shape = (elements,)

    print(f"\nTesting {side} x {side} ({elements} elements)")

    # Prepare inputs outside the timed sections
    a = torch.full(shape, 2.5, dtype=torch.float32, device="cpu")
    b = torch.full(shape, 2.5, dtype=torch.float32, device="cpu")
    values = torch.full(shape, 2.5, dtype=torch.float32, device="cpu")

    index = (side // 2, side // 2)

    # 1. Construction: allocation and zero initialization
    def construction():
        global checksum
        t = torch.zeros(shape, dtype=torch.float32)
        checksum += t.numel()

    measure("construction", elements, construction)

    # 2. Fill
    def fill():
        global checksum
        a.fill_(3.14)
        checksum += a[0, 0].item()

    measure("fill", elements, fill)

    # 3. Copy values into an existing tensor
    def fill_tensor():
        global checksum
        a.copy_(values)
        checksum += a[0, 0].item()

    measure("fillTensor", elements, fill_tensor)

    # 4. Element access
    def get_element():
        global checksum
        checksum += a[index].item()

    measure(
        "getElement",
        elements,
        get_element,
        operations_per_iteration=ACCESS_BATCH,
    )

    # 5. Element modification
    next_value = [1.0]

    def set_element():
        next_value[0] += 0.001
        a[index] = next_value[0]

    measure(
        "setElement",
        elements,
        set_element,
        operations_per_iteration=ACCESS_BATCH,
    )

    # 6. Reshape: produce views of the same contiguous tensor
    def reshape():
        global checksum
        flat = a.reshape(flat_shape)
        restored = flat.reshape(shape)
        checksum += restored.numel()

    measure("reshape", elements, reshape)

    # 7. Addition
    def addition():
        global checksum
        result = a + b
        checksum += result[0, 0].item()

    measure("addition", elements, addition)

    # 8. Subtraction
    def subtraction():
        global checksum
        result = a - b
        checksum += result[0, 0].item()

    measure("subtraction", elements, subtraction)

    # 9. Element-wise multiplication
    def multiplication():
        global checksum
        result = a * b
        checksum += result[0, 0].item()

    measure("multiplication", elements, multiplication)


output_path = "benchmarks/results/tensor_pytorch.csv"

with open(output_path, "w", newline="") as file:
    writer = csv.DictWriter(
        file,
        fieldnames=[
            "operation",
            "elements",
            "repetitions",
            "median_ms",
            "average_ms",
        ],
    )
    writer.writeheader()
    writer.writerows(results)

print("\nPyTorch benchmark complete.")
print("Checksum:", checksum)
print("Results:", output_path)
