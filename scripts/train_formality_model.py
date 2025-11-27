#!/usr/bin/env python
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.train_regressor import train

if __name__ == "__main__":
    train()
