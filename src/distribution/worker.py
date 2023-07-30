import multiprocessing

from machine_learning.models import FeedzaiProduction
import redis

def worker_function(queue: multiprocessing.Queue, id):

    database = redis.Redis(host='localhost', port=6379, decode_responses=True)

    # compile model
    model = FeedzaiProduction()
    weight_path = "PATH TO WEIGHTS"
    model.load_weights(weight_path, by_name=True)
    model.reset_gru()
    model.reset_metrics()

    while True:
        if not queue.empty():
            input = queue.get()
            
            if type(input) is bool:
                break   # end execution
            
            metrics = model.compute_metrics(input[0], input[1], model(input[0]))

            # Register state
            # state = model.fetch_state()

            


    print(f'[WORKER {id}]: {metrics}')

