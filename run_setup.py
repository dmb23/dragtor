import subprocess

def run_setup():
    try:
        subprocess.run(["bash", "setup.sh"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error during setup: {e}")

if __name__ == "__main__":
    run_setup()
