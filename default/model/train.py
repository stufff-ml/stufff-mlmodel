#!/usr/bin/env python

import argparse
import model
from datetime import datetime

PARAMS = model.initialise( argparse.ArgumentParser())

def main():

    # Run the train and evaluate experiment
    time_start = datetime.utcnow()
    print('')
    print(" --> Training started at {}".format(time_start.strftime("%H:%M:%S")))
    print('')
    print(PARAMS)
    print('')

    # do some real stuff ...
    model.train(PARAMS)

    # Done.
    time_end = datetime.utcnow()
    time_elapsed = time_end - time_start

    print('')
    print(" --> Training finished at {}".format(time_end.strftime("%H:%M:%S")))
    print(" --> Training elapsed time: {} seconds".format(time_elapsed.total_seconds()))
    print('')


if __name__ == '__main__':
    main()
