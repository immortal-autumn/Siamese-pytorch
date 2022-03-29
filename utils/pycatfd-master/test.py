import threading
from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image
from concurrent.futures import ThreadPoolExecutor, as_completed, wait, ALL_COMPLETED

from catfd import detect_and_return

pool = ThreadPoolExecutor(max_workers=20)

current_path = Path(__file__).parent.parent.parent
cat_face = current_path / 'datasets' / 'images_background'

# cat_face_directory = Path(__file__).parent / 'samples' / 'zelda.jpg'

total = 0
success = 0
glock = threading.Lock()


def add_val():
    global success
    glock.acquire()
    success += 1
    glock.release()


def print_result(result):
    output, _, _ = result.result()
    print(result.result())

    if output['face_count'] == 1:
        add_val()


thread_pool = []

for entry in cat_face.glob('*'):
    for img in entry.glob('*'):
        total += 1
        command = pool.submit(detect_and_return, (img.parent.name, img.name, img))
        command.add_done_callback(print_result)
        thread_pool.append(command)

wait(thread_pool, return_when=ALL_COMPLETED)
print("Completed!")
print(f"The success / total is: {success} / {total}!")
