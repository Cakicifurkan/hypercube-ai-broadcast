import time
import threading
from queue import Queue
from model_vgg16 import prepare_data

node_queues = {
    "00": Queue(),
    "01": Queue(),
    "10": Queue(),
    "11": Queue()
}

node_datasets = {}
data_lock = threading.Lock()
step1_event = threading.Event()
broadcast_barrier = threading.Barrier(4)


def node_00_worker(dataset):
    with data_lock:
        node_datasets["00"] = dataset

    # Step 1: 00 -> 01
    print("[STEP 1] Node 00 -> Node 01")
    node_queues["01"].put(dataset)
    step1_event.wait()

    # Step 2: 00 -> 10 (Parallel with 01 -> 11)
    print("[STEP 2] Node 00 -> Node 10")
    node_queues["10"].put(dataset)
    broadcast_barrier.wait()


def node_01_worker():
    # Step 1: Receive from 00
    dataset = node_queues["01"].get()
    with data_lock:
        node_datasets["01"] = dataset
    print("[STEP 1] Node 01 received dataset from Node 00")
    step1_event.set()

    # Step 2: 01 -> 11 (Parallel with 00 -> 10)
    print("[STEP 2] Node 01 -> Node 11")
    node_queues["11"].put(dataset)
    broadcast_barrier.wait()


def node_10_worker():
    # Step 2: Receive from 00
    dataset = node_queues["10"].get()
    with data_lock:
        node_datasets["10"] = dataset
    print("[STEP 2] Node 10 received dataset from Node 00")
    broadcast_barrier.wait()


def node_11_worker():
    # Step 2: Receive from 01
    dataset = node_queues["11"].get()
    with data_lock:
        node_datasets["11"] = dataset
    print("[STEP 2] Node 11 received dataset from Node 01")
    broadcast_barrier.wait()


def run_hypercube_broadcast():
    dataset = prepare_data()
    start_time = time.time()

    threads = [
        threading.Thread(target=node_00_worker, args=(dataset,), name="Thread-00"),
        threading.Thread(target=node_01_worker, name="Thread-01"),
        threading.Thread(target=node_10_worker, name="Thread-10"),
        threading.Thread(target=node_11_worker, name="Thread-11")
    ]

    for t in threads:
        t.start()

    for t in threads:
        t.join()

    broadcast_time = round(time.time() - start_time, 6)

    print(f"\n[BROADCAST COMPLETE] Elapsed Time: {broadcast_time} sec")
    for node_id in ["00", "01", "10", "11"]:
        X_train, X_test, _, _ = node_datasets[node_id]
        print(f"Node {node_id} -> X_train: {X_train.shape} | X_test: {X_test.shape}")

    return node_datasets, broadcast_time


if __name__ == "__main__":
    run_hypercube_broadcast()