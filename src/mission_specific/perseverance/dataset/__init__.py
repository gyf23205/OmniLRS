__author__ = "Aleksa Stanivuk"
__copyright__ = "Copyright 2025-26, JAOPS"
__license__ = "BSD-3-Clause"
__version__ = "2.0.0"
__maintainer__ = "Louis Burtz"
__email__ = "ljburtz@jaops.com"
__status__ = "development"

"""
Dataset generation for fault detection.

Drives the rover through randomized episodes, injects faults on a randomized schedule, and records
everything the run produces - split into what a real mission could see and what only the simulator
knows. See fault_scheduler.py for the sampling rules and recorder.py for the observable/oracle
contract.
"""
