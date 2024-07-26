import sys

# Using a simple workaround to to output logs to a file
class Logger:
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "w")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        
    def flush(self):
        self.terminal.flush()
        self.log.flush()
        
    def isatty(self):
        return False

sys.stdout = Logger("output.log")

# function to read the logs
def read_logs():
    sys.stdout.flush()
    with open("output.log", "r") as f:
        # return the last 25 lines of the log
        return ''.join(f.readlines()[-25:])