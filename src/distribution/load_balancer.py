import multiprocessing

from distribution.db_utils import delete_states
from .worker import worker_function
from multiprocessing import Pool
from machine_learning.models import DoubleProduction, DoubleConcatProduction, DoubleExtraProduction, DoubleExtraConcatProduction
from distribution.watcher import AsynchronousAverage, AsynchronousBoundSum, SynchronousBoundSum,SynchronousAverage, NoSynchronization


def execute():
    # PARAMETERS FOR EXECUTION
    merge_methods = [SynchronousBoundSum, SynchronousAverage]
    model_list = [DoubleExtraConcatProduction]
    threshold_limit = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
    num_workers = [2, 4, 8]

    checkpoint = (SynchronousBoundSum, DoubleExtraConcatProduction, 2, 1)
    reached_checkpoint = False
    for model_type in model_list:
        for merge in merge_methods:
            for pool_size in num_workers:
                for threshold in threshold_limit:
                    if (merge, model_type, pool_size, threshold) == checkpoint:
                        reached_checkpoint = True
                    
                    if reached_checkpoint:
                        delete_states()
                        print(f"\n-------\nCONFIGURATION -> merge method: {merge.name}, model: {model_type.__name__}, num_workers: {pool_size}, threshold: {threshold}\n------\n")
                        args = [(id, model_type, merge, threshold, pool_size) for id in range(pool_size)]
                        
                        with Pool(pool_size) as p:
                            p.starmap(worker_function, args)
                    
                    else:
                        print(f"Skipping: {(merge, model_type, pool_size, threshold)}")

        
