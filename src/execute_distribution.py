import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

from distribution.load_balancer import execute
if __name__ == "__main__":
    execute()