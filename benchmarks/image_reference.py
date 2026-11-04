
import cv2
import numpy as np
import csv
import os
import time
import statistics

# Use CPU and one OpenCV thread for the comparison.
cv2.setNumThreads(1)

INPUT = "yard.bmp"
OUTPUT_DIR = "benchmarks/results"

REPETITIONS = 15

os.makedirs(OUTPUT_DIR, exist_ok=True)

results = []


def measure(operation, width, height, function):
    # Warm-up
    for _ in range(3):
        output = function()

    # Keep output alive and consume it outside the timed section.
    checksum = float(np.sum(output, dtype=np.float64))

    times = []

    for _ in range(REPETITIONS):
        start = time.perf_counter_ns()
        output = function()
        end = time.perf_counter_ns()

        times.append((end - start) / 1_000_000)

        checksum += float(np.sum(output, dtype=np.float64))

    median = statistics.median(times)
    average = statistics.mean(times)

    results.append({
        "operation": operation,
        "width": width,
        "height": height,
        "repetitions": REPETITIONS,
        "median_ms": median,
        "average_ms": average,
    })

    print(
        f"{operation:24} | "
        f"median: {median:.6f} ms"
    )

    return checksum


# -------------------------------------------------
# LOAD IMAGE
# -------------------------------------------------

loading_times = []

# Warm-up
warmup = cv2.imread(INPUT, cv2.IMREAD_COLOR)

if warmup is None:
    raise RuntimeError(f"Could not load {INPUT}")

for _ in range(REPETITIONS):
    start = time.perf_counter_ns()
    image = cv2.imread(INPUT, cv2.IMREAD_COLOR)
    end = time.perf_counter_ns()

    if image is None:
        raise RuntimeError("OpenCV failed to load image")

    loading_times.append((end - start) / 1_000_000)

height, width = image.shape[:2]

print(f"\nImage dimensions: {width} x {height}")

results.append({
    "operation": "loading",
    "width": width,
    "height": height,
    "repetitions": REPETITIONS,
    "median_ms": statistics.median(loading_times),
    "average_ms": statistics.mean(loading_times),
})

print(
    f"{'loading':24} | "
    f"median: {statistics.median(loading_times):.6f} ms"
)


# -------------------------------------------------
# IMAGE OPERATIONS
# -------------------------------------------------

# 1. Grayscale
def grayscale():
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


# 2. Horizontal reflection
def horizontal_reflection():
    return cv2.flip(image, 1)


# 3. Vertical flip
def vertical_flip():
    return cv2.flip(image, 0)


# 4. Blur
def blur():
    return cv2.blur(image, (3, 3))


# 5. Edge detection
def edges():
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    return cv2.Canny(gray, 100, 200)


# 6. Sepia
def sepia():
    kernel = np.array([
        [0.272, 0.534, 0.131],
        [0.349, 0.686, 0.168],
        [0.393, 0.769, 0.189]
    ], dtype=np.float32)

    # OpenCV image is BGR; transform in that channel order.
    result = cv2.transform(
        image.astype(np.float32),
        kernel
    )

    return np.clip(result, 0, 255).astype(np.uint8)


# 7. Brightness adjustment
def brightness():
    return cv2.convertScaleAbs(image, alpha=1.2, beta=0)


operations = [
    ("grayscale", grayscale),
    ("horizontal_reflection", horizontal_reflection),
    ("vertical_flip", vertical_flip),
    ("blur", blur),
    ("edges", edges),
    ("sepia", sepia),
    ("brightness", brightness),
]

checksum = 0.0

for name, operation in operations:
    checksum += measure(
        name,
        width,
        height,
        operation
    )


# -------------------------------------------------
# SAVE IMAGE
# -------------------------------------------------

save_image = image.copy()
save_path = os.path.join(OUTPUT_DIR, "opencv_temp.bmp")

saving_times = []

# Warm-up
cv2.imwrite(save_path, save_image)

for _ in range(REPETITIONS):
    start = time.perf_counter_ns()
    success = cv2.imwrite(save_path, save_image)
    end = time.perf_counter_ns()

    if not success:
        raise RuntimeError("OpenCV failed to save image")

    saving_times.append((end - start) / 1_000_000)

results.append({
    "operation": "saving",
    "width": width,
    "height": height,
    "repetitions": REPETITIONS,
    "median_ms": statistics.median(saving_times),
    "average_ms": statistics.mean(saving_times),
})

print(
    f"{'saving':24} | "
    f"median: {statistics.median(saving_times):.6f} ms"
)


# -------------------------------------------------
# SAVE RESULTS
# -------------------------------------------------

output_csv = os.path.join(
    OUTPUT_DIR,
    "image_opencv.csv"
)

with open(output_csv, "w", newline="") as file:
    writer = csv.DictWriter(
        file,
        fieldnames=[
            "operation",
            "width",
            "height",
            "repetitions",
            "median_ms",
            "average_ms",
        ],
    )

    writer.writeheader()
    writer.writerows(results)

print("\nOpenCV image benchmark completed.")
print("Checksum:", checksum)
print("Results saved to:", output_csv)
