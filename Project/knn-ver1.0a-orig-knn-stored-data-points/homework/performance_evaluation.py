import subprocess
import re
import numpy as np
import os
import glob

# Fix paths by expanding ~ to absolute paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))  # Get the script's directory
PROJECT_DIR = os.path.expanduser("~/gprof2dot/Project/knn-ver1.0a-orig-knn-stored-data-points")
OUTPUT_DIR = os.getcwd()

# Find the correct gprof2dot path
GPROF2DOT_PATH = subprocess.run("which gprof2dot", shell=True, capture_output=True, text=True).stdout.strip()
if not GPROF2DOT_PATH:
    print("❌ Error: gprof2dot not found. Make sure it's installed and in the system path.")
    exit()

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

# User chooses whether to enable profiling
profile_choice = input("Enable profiling? (y/n): ").strip().lower()
enable_profiling = profile_choice == "y"

# User provides a name for the results folder and file
result_name = input("Enter the name for the results: ").strip()
result_folder = os.path.join(OUTPUT_DIR, f"Results of {result_name}")

# Ensure the output folder is created immediately
os.makedirs(result_folder, exist_ok=True)
os.sync()  # Force sync to disk

# Define the path for the results file inside the folder
output_filepath = os.path.join(result_folder, f"{result_name}.txt")

for opt in optimizations:
    print(f"Running tests for {opt} (Profiling: {'Enabled' if enable_profiling else 'Disabled'})...")
    execution_times = []
    gmon_files = []

    # Set compilation flags
    profiling_flag = "-pg" if enable_profiling else ""
    compile_cmd = f"gcc -fopenmp {opt} {profiling_flag} -D K=3 -D SPECIALIZED=1 -D MATH_TYPE=1 -D DIST_METHOD=1 -D USE_SQRT=1 -D SCENARIO_FEATURES=2 -D READ=4 -D DIMEM=0 -D VERIFY=0 -D STREAMING=1 -Wall -Wextra -std=gnu99 {PROJECT_DIR}/knn.c {PROJECT_DIR}/utils.c {PROJECT_DIR}/timer.c {PROJECT_DIR}/io.c {PROJECT_DIR}/features.c {PROJECT_DIR}/main.c -lm -o {PROJECT_DIR}/knn"
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

        # Check if gmon.out was generated and is non-empty
        gmon_path = os.path.join(result_folder, f"gmon_{opt}_{i+1}.out")
        if enable_profiling and os.path.exists(f"{PROJECT_DIR}/gmon.out") and os.path.getsize(f"{PROJECT_DIR}/gmon.out") > 0:
            os.rename(f"{PROJECT_DIR}/gmon.out", gmon_path)
            gmon_files.append(gmon_path)
        else:
            print(f"⚠️ Warning: gmon.out not found or empty for run {i+1}.")

    # Compute average execution time
    avg_time = np.mean(execution_times)
    results[opt] = avg_time
    print(f"  Average execution time for {opt}: {avg_time:.4f} s\n")

    # If profiling is enabled and we have valid gmon files, generate profiling reports
    if enable_profiling and gmon_files:
        print(f"  Merging profiling data for {opt}...")
        gprof_output_file = os.path.join(result_folder, f"gprof_{opt}.txt")

        try:
            merge_cmd = f"gprof {PROJECT_DIR}/knn {' '.join(gmon_files)} > {gprof_output_file}"
            subprocess.run(merge_cmd, shell=True, check=True)
        except subprocess.CalledProcessError:
            print(f"❌ Error: gprof failed for {opt}. Check if gmon files are valid.")

        print(f"  Generating call graph for {opt}...")
        callgraph_file = os.path.join(result_folder, f"callgraph_{opt}.png")

        try:
            callgraph_cmd = f"gprof {PROJECT_DIR}/knn {gmon_files[0]} | {GPROF2DOT_PATH} | dot -Tpng -o {callgraph_file}"
            subprocess.run(callgraph_cmd, shell=True, check=True)
        except subprocess.CalledProcessError:
            print(f"❌ Error: gprof2dot or dot failed for {opt}. Check dependencies.")

# Save benchmark results and configuration
try:
    with open(output_filepath, "w") as f:
        f.write("Benchmark Configuration:\n")
        f.write(f"Selected Optimization: {', '.join(optimizations)}\n")
        f.write(f"Number of Runs per Optimization: {num_runs}\n")
        f.write(f"Profiling Enabled: {'Yes' if enable_profiling else 'No'}\n")
        f.write(f"Compilation Command: {compile_cmd}\n\n")
        f.write("Benchmark Results:\n")

        for opt, avg_time in results.items():
            f.write(f"{opt}: {avg_time:.4f} s\n")

        f.flush()  # Ensure data is written immediately
        os.fsync(f.fileno())  # Force sync to disk

    print(f"✅ Benchmarking completed! Results saved in {output_filepath}")

    # Cleanup: Delete profiling files if enabled
    if enable_profiling:
        print("🧹 Cleaning up gmon.out files...")
        for file in glob.glob(os.path.join(result_folder, "gmon_*.out")):
            os.remove(file)

except Exception as e:
    print(f"❌ Error saving results: {e}")

# Ensure everything is written to disk immediately
os.sync()
