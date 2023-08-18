import argparse
from machine_learning.models import CARD_ID_COLUMN
import multiprocessing
from .worker import worker_function
import tensorflow as tf
import numpy as np

def execute():
    datasets = []
    worker_list = []

    worker_0 = multiprocessing.Process(target=worker_function, 
                                       args=(0,))
    worker_1 = multiprocessing.Process(target=worker_function, 
                                       args=(1,))
    worker_0.start()
    worker_1.start()

    worker_0.join()
    worker_1.join()

    #for worker in worker_list:
    #    worker.start()
    #
    #for worker in worker_list:
    #    worker.join()
