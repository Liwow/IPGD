import os
from multiprocessing import Process, Queue

import ecole


def generate_single_instance(queue, istrain, size, generator):
    while True:
        i = queue.get()
        if i is None:
            break  # No more tasks

        instance = next(generator)
        instance_dir = f"instance/{istrain}/{size}"
        os.makedirs(instance_dir, exist_ok=True)
        instance_path = os.path.join(instance_dir, f"{size}_{i}.lp")
        instance.write_problem(instance_path)
        print(f"第{i}个问题实例已生成：{instance_path}")


def generate_instances(num_instances, istrain, size):
    if size == "CF":
        generator = ecole.instance.CapacitatedFacilityLocationGenerator(50, 100)
    elif size == "IS":
        generator = ecole.instance.IndependentSetGenerator(1500)
    elif size == "CA":
        generator = ecole.instance.CombinatorialAuctionGenerator(700, 1500)
    elif size == "SC":
        generator = ecole.instance.SetCoverGenerator(1000, 2000)
    else:
        raise ValueError("Invalid type")

    generator.seed(4202)
    observation_function = ecole.observation.MilpBipartite()

    # Create a queue to hold tasks
    task_queue = Queue()

    # Add tasks to queue
    for i in range(num_instances):
        task_queue.put(i)

    # Number of worker processes
    num_workers = 20

    # Create worker processes
    workers = []
    for _ in range(num_workers):
        worker = Process(target=generate_single_instance,
                         args=(task_queue, istrain, size, generator))
        workers.append(worker)
        worker.start()

    # Add None to the queue to signal workers to exit
    for _ in range(num_workers):
        task_queue.put(None)

    # Wait for all worker processes to finish
    for worker in workers:
        worker.join()


if __name__ == '__main__':
    generate_instances(3, "test", "CF")
