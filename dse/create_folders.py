import subprocess

directories = {
    "6.8.37": [
        "cdc_raw",
        "commitlog",
        "data",
        "hints",
        "insights_data",
        "metadata",
        "saved_caches"
    ],
    "7.0.0-alpha.4": [
        "commitlog",
        "data",
        "hints",
        "insights_data",
        "metadata",
        "saved_caches"
    ],
}

try:
    subprocess.check_output(["useradd", "dse"])
except subprocess.CalledProcessError:
    print("User 'dse' already exists, or insufficient permissions to create user")

try:
    for directory in directories:
        subprocess.check_output(["mkdir", directory])
        subprocess.check_output(["chown", "-R", "root:root", directory])
        subprocess.check_output(["chmod", "755", directory])
        for subdirectory in directories[directory]:
            subprocess.check_output(["mkdir", f"{directory}/{subdirectory}"])
            subprocess.check_output(["chown", "-R", "dse:dse", f"{directory}/{subdirectory}"])
            subprocess.check_output(["chmod", "777", f"{directory}/{subdirectory}"])
except subprocess.CalledProcessError:
    print("Error creating directories, please ensure you are running as root")