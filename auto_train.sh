#!/bin/bash

python3.7 auto_training.py
git add .
git commit -a -m "UPDATE model at `date`"
git push -u orgin master

