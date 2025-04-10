import os
import psutil

logical_cpus = psutil.cpu_count(logical=False)

print(logical_cpus)

os.environ['OMP_NUM_THREADS'] = str(logical_cpus)