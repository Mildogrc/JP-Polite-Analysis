#!/usr/bin/env python
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.make_dataset import main

if __name__ == "__main__":
    main()
