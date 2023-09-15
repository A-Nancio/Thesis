import argparse
from machine_learning.models import CARD_ID_COLUMN
import multiprocessing
from .worker import worker_function
import tensorflow as tf
import numpy as np

def execute():
    worker_0 = multiprocessing.Process(target=worker_function, 
                                       args=(0,))
    worker_1 = multiprocessing.Process(target=worker_function, 
                                       args=(1,))
    worker_2 = multiprocessing.Process(target=worker_function, 
                                        args=(2,))
    worker_3 = multiprocessing.Process(target=worker_function, 
                                        args=(3,))
    worker_4 = multiprocessing.Process(target=worker_function, 
                                        args=(4,))
    worker_5 = multiprocessing.Process(target=worker_function, 
                                        args=(5,))
    worker_6 = multiprocessing.Process(target=worker_function, 
                                        args=(6,))
    worker_7 = multiprocessing.Process(target=worker_function, 
                                        args=(7,))
    worker_8 = multiprocessing.Process(target=worker_function, 
                                        args=(8,))
    worker_9 = multiprocessing.Process(target=worker_function, 
                                        args=(9,))
    worker_0.start()
    worker_1.start()
    worker_2.start()
    worker_3.start()
    worker_4.start()
    worker_5.start()
    worker_6.start()
    worker_7.start()
    worker_8.start()
    worker_9.start()
 
    worker_0.join()
    worker_1.join()
    worker_2.join()
    worker_3.join()
    worker_4.join()
    worker_5.join()
    worker_6.join()
    worker_7.join()
    worker_8.join()
    worker_9.join()