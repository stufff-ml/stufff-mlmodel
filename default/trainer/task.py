#!/usr/bin/env python

import os
import argparse
import pandas as pd

from datetime import datetime
from google.cloud import storage
from io import BytesIO

def main():

    run_config = {}

    # Run the train and evaluate experiment
    time_start = datetime.utcnow()
    print('')
    print(" --> Training started at {}".format(time_start.strftime("%H:%M:%S")))
    print(' Parameters:')
    print(PARAMS)
    print('')

    run_experiment(run_config)

    time_end = datetime.utcnow()
    print('')
    print(" --> Training finished at {}".format(time_end.strftime("%H:%M:%S")))
    print('')
    time_elapsed = time_end - time_start
    print(" --> Training elapsed time: {} seconds".format(time_elapsed.total_seconds()))
    print('')


def initialise_params(args_parser):
    """
    Define the arguments with the default values,
    parses the arguments passed to the task,
    and set the PARAMS global variable

    Args:
        args_parser
    """

    args_parser.add_argument(
        '--job-dir',
        help='Location used for all tmp data',
        required=True
    )
    args_parser.add_argument(
        '--model-id',
        help='ID of the model to be trained',
        required=True
    )
    args_parser.add_argument(
        '--model-rev',
        help='Revision of the model to be trained',
        required=True
    )
    args_parser.add_argument(
        '--predict-batch-size',
        help='Batch size for each prediction',
        type=int,
        default=10
    )
    args_parser.add_argument(
        '--source-bucket',
        help='Bucket where training data is expected',
        default='models.stufff.review'
    )
    args_parser.add_argument(
        '--model-bucket',
        help='Bucket where models will be uploaded to',
        default='models.stufff.review'
    )

    args_parser.add_argument(
        '--split-percentage',
        help='Percentage of the data used for training',
        type=float,
        default=0.8
    )
    args_parser.add_argument(
        '--split-seed',
        help='Randome value seed for data splitting',
        type=int,
        default=5
    )

    args_parser.add_argument(
        '--output-dir',
        help='Temporary file location for data, models, exports etc.',
        default='output'
    )

    # Argument to turn on all logging
    args_parser.add_argument(
        '--verbosity',
        choices=[
            'DEBUG',
            'ERROR',
            'FATAL',
            'INFO',
            'WARN'
        ],
        default='INFO',
    )

    return args_parser.parse_args()


def run_experiment(run_config):
    """Train, evaluate, and export the model using Turicreate"""

    source_name = PARAMS.model_id + '/' + PARAMS.model_id + '.' + PARAMS.model_rev + '.csv'
    target_name = PARAMS.model_id + '/model/' + PARAMS.model_id + '.' + PARAMS.model_rev + '.csv'
    temp_name = PARAMS.model_id + '.' + PARAMS.model_rev + '.csv'

    client = storage.Client()
    source_bucket = client.get_bucket(PARAMS.source_bucket)

    # Load the training data as a panda dataframe
    print(' --> Load data and prepare training set')
    print("")

    source_blob = source_bucket.get_blob(source_name)
    content = source_blob.download_as_string()
    _data = pd.read_csv(BytesIO(content))
    _data.columns = ['event','entity_type','entity_id','target_entity_type','target_entity_id','timestamp','properties']


args_parser = argparse.ArgumentParser()
PARAMS = initialise_params(args_parser)

if __name__ == '__main__':
    main()
