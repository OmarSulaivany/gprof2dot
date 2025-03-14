import subprocess
import re
import numpy as np
import os

# Fix paths by expanding ~ to absolute paths
PROJECT_DIR = os.path.expanduser("~/gprof2dot/Project/knn-ver1.0a-orig-knn-stored-data-points")
OUTPUT_DIR = os.getcwd()

# Available optimization levels
available_opts = {"1": "-O1", "2": "-O2", "3": "-O3"}
num_runs = 10  # Number of runs per optimization level
time_pattern = re.compile(r"Total execution time:\s+([0-9.]+)\s+s")
results = {}

# User selects optimization level
print("Select the optimization level:\n1. O1\n2. O2\n3. O3\n4. All")
choice = input("Enter your choice (1-4): ").strip()

if choice in available_opts:
    optimizations = [available_opts[choice]]
elif choice == "4":
    optimizations = list(available_opts.values())
else:
    print("❌ Invalid choice! Exiting.")
    exit()

# User provides a filename for the output
output_filename = input("Enter the name for the results file (without extension): ").strip()
output_filepath = os.path.join(OUTPUT_DIR, f"{output_filename}.txt")

for opt in optimizations:
    print(f"Running tests for {opt}...")
    execution_times = []

    # Compile the program without profiling
    compile_cmd = f"gcc -fopenmp {opt} -D K=3 -D SPECIALIZED=1 -D MATH_TYPE=1 -D DIST_METHOD=1 -D USE_SQRT=1 -D SCENARIO_FEATURES=2 -D READ=4 -D DIMEM=0 -D VERIFY=0 -D STREAMING=1 -Wall -Wextra -std=gnu99 {PROJECT_DIR}/knn_adjusted.c {PROJECT_DIR}/utils.c {PROJECT_DIR}/timer.c {PROJECT_DIR}/io.c {PROJECT_DIR}/features.c {PROJECT_DIR}/main.c -lm -o {PROJECT_DIR}/knn"
    subprocess.run(compile_cmd, shell=True, check=True)

    # Run the program 10 times and record execution time
    for i in range(num_runs):
        print(f"  Run {i+1}/{num_runs}...")
        output = subprocess.run(f"{PROJECT_DIR}/knn", shell=True, capture_output=True, text=True, cwd=PROJECT_DIR).stdout

        # Extract execution time
        match = time_pattern.search(output)
        if match:
            exec_time = float(match.group(1))
            execution_times.append(exec_time)

    # Compute average execution time
    avg_time = np.mean(execution_times)
    results[opt] = avg_time
    print(f"  Average execution time for {opt}: {avg_time:.4f} s\n")

# Save benchmark results and configuration
try:
    with open(output_filepath, "w") as f:
        f.write("Benchmark Configuration:\n")
        f.write(f"Selected Optimization: {', '.join(optimizations)}\n")
        f.write(f"Number of Runs per Optimization: {num_runs}\n")
        f.write(f"Compilation Command: {compile_cmd}\n\n")
        f.write("Benchmark Results:\n")
        
        for opt, avg_time in results.items():
            f.write(f"{opt}: {avg_time:.4f} s\n")
        
        f.flush()  # Ensure the data is written immediately
    
    print(f"✅ Benchmarking completed! Results saved in {output_filepath}")
except Exception as e:
    print(f"❌ Error saving results: {e}")
