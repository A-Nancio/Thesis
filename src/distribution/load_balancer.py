import multiprocessing

from distribution.db_utils import delete_states
from .worker import worker_function
from multiprocessing import Pool

def execute():
    # PARAMETERS FOR EXECUTION
    threshold_limit = 300000
    max_workers = 10

    # NOTE uncommend for specific configuration
    # print(f"\n-------\nCONFIGURATION -> num_workers: {1}, threshold: {threshold}\n------\n")
    # args = [(id, threshold, 1) for id in range(1)]
    # 
    # with Pool(1) as p:
    #     p.starmap(worker_function, args)
    # 
    # delete_states()
    # threshold *= 2
    for pool_size in range(2, max_workers+1):
        threshold = 1
        while threshold < threshold_limit:
            print(f"\n-------\nCONFIGURATION -> num_workers: {pool_size}, threshold: {threshold}\n------\n")
            args = [(id, threshold, pool_size) for id in range(pool_size)]
            
            with Pool(pool_size) as p:
                p.starmap(worker_function, args)

            threshold *= 2
        
