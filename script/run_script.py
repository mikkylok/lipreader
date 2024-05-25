import subprocess

def run_script():
    while True:
        try:
            subprocess.run(['python', 'script_full.py'], check=True)
        except subprocess.CalledProcessError:
            print("Error occurred, restarting script.")
            continue

run_script()