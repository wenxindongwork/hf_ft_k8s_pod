import psutil
used_memory_gb = psutil.virtual_memory().used / (1024 ** 3)
total_ram_gb = psutil.virtual_memory().total / (1024 ** 3)
print(f"Used {used_memory_gb:.2f} GB out of a total of {total_ram_gb:.2f} GB RAM")