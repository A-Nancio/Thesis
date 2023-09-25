import os
from distribution.db_utils import reset_database, delete_states
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

from distribution.load_balancer import execute
if __name__ == "__main__":
    reset_database()
    execute()
    delete_states()