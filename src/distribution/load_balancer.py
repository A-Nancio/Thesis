import numpy as np
import tensorflow as tf
import sys
import argparse
from machine_learning.models import CARD_ID_COLUMN
import multiprocessing
from worker import worker_function

HOST = "127.0.0.1"  # Standard loopback interface address (localhost)
PORT = 65432  # Port to listen on (non-privileged ports are > 1023)



def main(argv):
    argParser = argparse.ArgumentParser()
    argParser.add_argument("-n", "--num_workers", type=int, help="Number of workers")

    args = argParser.parse_args()

    num_workers = args.num_workers
    worker_list = []
    worker_queues = []

    # initalize workers
    for id in range(len(num_workers)):
        new_queue = multiprocessing.Queue()
        worker_queues.append(new_queue)
        
        new_worker = multiprocessing.Process(target=worker_function, args=(new_queue, id))
        new_worker.start()
        worker_list.append(new_worker)

    # send data to each worker
    transactions_list = np.load(f'src/data/test/transactions.npy')
    labels = np.load(f'src/data/test/all_transaction_labels.npy')
    for transaction, label in zip(transactions_list, labels):
        worker_id = transaction[CARD_ID_COLUMN] % num_workers

        worker_queues[worker_id].put([transaction, label])


    # extract metrics

    return

if __name__ == "__main__":
    main(sys.argv[1:])