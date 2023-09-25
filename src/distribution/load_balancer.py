import multiprocessing

from distribution.db_utils import delete_states
from .worker import worker_function
from multiprocessing import Pool

def execute():
    # PARAMETERS FOR EXECUTION
    threshold_limit = 20
    threshold = 8
    max_workers = 2


    for pool_size in range(1, max_workers+1):
        threshold = 8
        while threshold < threshold_limit:
            print(f"\n-------\nCONFIGURATION -> num_workers: {pool_size}, threshold: {threshold}\n------\n")
            args = [(id, threshold, pool_size) for id in range(pool_size)]
            
            with Pool(pool_size) as p:
                p.starmap(worker_function, args)

            delete_states()
            threshold *= 2
    # worker_0 = multiprocessing.Process(target=worker_function, 
    #                                 args=(0, threshold))
    # worker_1 = multiprocessing.Process(target=worker_function, 
    #                                 args=(1, threshold))
    # worker_2 = multiprocessing.Process(target=worker_function, 
    #                                      args=(2,))
    # worker_3 = multiprocessing.Process(target=worker_function, 
    #                                      args=(3,))
    # worker_4 = multiprocessing.Process(target=worker_function, 
    #                                      args=(4,))
    # worker_5 = multiprocessing.Process(target=worker_function, 
    #                                      args=(5,))
    # worker_6 = multiprocessing.Process(target=worker_function, 
    #                                      args=(6,))
    # worker_7 = multiprocessing.Process(target=worker_function, 
    #                                      args=(7,))
    # worker_8 = multiprocessing.Process(target=worker_function, 
    #                                      args=(8,))
    # worker_9 = multiprocessing.Process(target=worker_function, 
    #                                      args=(9,))
    # worker_0.start()
    # worker_1.start()
    # worker_2.start()
    # worker_3.start()
    # worker_4.start()
    # worker_5.start()
    # worker_6.start()
    # worker_7.start()
    # worker_8.start()
    # worker_9.start()
# 
    # worker_0.join()
    # worker_1.join()
    # worker_2.join()
    # worker_3.join()
    # worker_4.join()
    # worker_5.join()
    # worker_6.join()
    # worker_7.join()
    # worker_8.join()
#    worker_9.join()
    #while threshold < threshold_limit:

   #     threshold *= 2