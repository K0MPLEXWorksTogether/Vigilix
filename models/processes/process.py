import psutil

def list_system_processes():
    processes = psutil.process_iter(['pid', 'name', 'username'])

    print(f"{'PID':<10}{'Process Name':<25}{'User'}")
    print("-" * 50)

    for process in processes:
        try:
            pid = process.info['pid']
            name = process.info['name']
            username = process.info['username']
            print(f"{pid:<10}{name:<25}{username}")
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            continue

if __name__ == "__main__":
    list_system_processes()
