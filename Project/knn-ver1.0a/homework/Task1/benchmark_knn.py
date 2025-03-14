import subprocess
import re
import numpy as np
import os
import glob

# Fix paths by expanding ~ to absolute paths
PROJECT_DIR = os.path.expanduser("~/gprof2dot/Project/knn-ver1.0a")

# Get the current directory where the script is executed
OUTPUT_DIR = os.getcwd()

# Optimization levels to test
optimizations = ["-O1", "-O2", "-O3"]
num_runs = 10  # Number of runs per optimization level
time_pattern = re.compile(r"Total execution time:\s+([0-9.]+)\s+s")
results = {}

for opt in optimizations:
    print(f"Running tests for {opt}...")
    execution_times = []

    # with no vectorization -fno-tree-vectorize.
    # Compile the program with profiling (-pg).
    # Take avantage of the whole CPU personalized to the machine -march=native -mtune=native 
    compile_cmd = f"gcc -fopenmp {opt} -pg -D K=3 -D SPECIALIZED=1 -D MATH_TYPE=1 -D DIST_METHOD=1 -D USE_SQRT=1 -D SCENARIO_FEATURES=2 -D READ=4 -D DIMEM=0 -D VERIFY=0 -D STREAMING=1 -Wall -Wextra -std=gnu99 {PROJECT_DIR}/knn_openmp.c {PROJECT_DIR}/utils.c {PROJECT_DIR}/timer.c {PROJECT_DIR}/io.c {PROJECT_DIR}/features.c {PROJECT_DIR}/main.c -lm -o {PROJECT_DIR}/knn"
    subprocess.run(compile_cmd, shell=True, check=True)

    # Run the program 10 times and save gmon.out files
    for i in range(num_runs):
        print(f"  Run {i+1}/{num_runs}...")
        output = subprocess.run(f"{PROJECT_DIR}/knn", shell=True, capture_output=True, text=True, cwd=PROJECT_DIR).stdout

        # Extract execution time
        match = time_pattern.search(output)
        if match:
            exec_time = float(match.group(1))
            execution_times.append(exec_time)

        # Rename gmon.out to keep all profiling data
        gmon_filename = f"{OUTPUT_DIR}/gmon_{opt}_{i+1}.out"
        if os.path.exists(f"{PROJECT_DIR}/gmon.out"):
            os.rename(f"{PROJECT_DIR}/gmon.out", gmon_filename)

    # Compute average execution time
    avg_time = np.mean(execution_times)
    results[opt] = avg_time
    print(f"  Average execution time for {opt}: {avg_time:.4f} s\n")

    # Merge all gmon.out files into one
    print(f"  Merging profiling data for {opt}...")
    merge_cmd = f"gprof {PROJECT_DIR}/knn {' '.join([f'{OUTPUT_DIR}/gmon_{opt}_{i+1}.out' for i in range(num_runs)])} > {OUTPUT_DIR}/gprof_{opt}.txt"
    subprocess.run(merge_cmd, shell=True, check=True)

    # Generate call graph
    print(f"  Generating call graph for {opt}...")
    subprocess.run(f"gprof {PROJECT_DIR}/knn {OUTPUT_DIR}/gmon_{opt}_1.out | gprof2dot | dot -Tpng -o {OUTPUT_DIR}/callgraph_{opt}.png", shell=True, check=True)

# Save benchmark results
with open(f"{OUTPUT_DIR}/benchmark_results.txt", "w") as f:
    for opt, avg_time in results.items():
        f.write(f"{opt}: {avg_time:.4f} s\n")

# Cleanup: Delete all gmon_*.out files after execution
print("🧹 Cleaning up gmon.out files...")
for file in glob.glob(f"{OUTPUT_DIR}/gmon_*.out"):
    os.remove(file)

print(f"✅ Benchmarking and profiling completed! All results are saved in {OUTPUT_DIR}")
