import os
import shutil
import random

classes = ["A", "B", "C", "D"]
SPLIT = 2

ROOT = 'dataset1\\gray\\'
INPUT = 'training'
OUTPUT = 'validation'

for the_class in classes:
    files = os.listdir(ROOT + INPUT + "\\" + the_class)
    files_move = random.sample(files, len(files)//SPLIT)

    for file_move in files_move:
        shutil.move(os.path.join(ROOT + INPUT + "\\" + the_class, file_move),
                    ROOT + OUTPUT + "\\" + the_class)
