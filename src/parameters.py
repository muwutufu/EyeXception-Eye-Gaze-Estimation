#!/usr/bin/env python3

import os


LEARNING_RATE = 0.001   # Initial model learning rate
EPOCHS = 9              # How many epochs to go through at max
PATIENCE = 2            # how many epochs to wait without progress before early stopping
BATCH_SIZE = 16         # Batch - GPU RAM limited, 16 is the maximum 10GB will hold
BASEFILTER_SIZE = 16    # Nr of filters in the initial layer of Xception lite model, gets proportionally larger deeper in the model

SAVE_PATH = "./tf/"     # Path to save models in
LOGS_PATH = "./tb/"     # Path to save logging in
LOAD_MODEL = True
if not os.path.exists(SAVE_PATH):
    os.makedirs(SAVE_PATH)
if not os.path.exists(LOGS_PATH):
    os.makedirs(LOGS_PATH)
MODEL_NAME = 'model'    # Model name to LOAD FROM (LOOKS IN SAVE_PATH DIRECTORY)

PATHT = "/media/peter/E410A0F210A0CCBC/MPPROJDATA/project3/DATA0"       # Training dataset path
PATHV = "/media/peter/E410A0F210A0CCBC/MPPROJDATA/project3/DATA1"       # Validation dataset path
PATHTE = "/media/peter/E410A0F210A0CCBC/MPPROJDATA/project3/DATA2"      # Test dataset path
