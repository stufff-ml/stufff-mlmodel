
import argparse

import os
import argparse
from io import BytesIO

from google.cloud import storage

import pandas as pd
import numpy as np


def train(params):

    print(" +++ Training +++")


def initialise(args_parser):
    
    args_parser.add_argument(
        '--job-id',
        help='ID of this instance',
        required=True
    )
    args_parser.add_argument(
        '--client-id',
        help='ID of the client resource',
        required=True
    )
    args_parser.add_argument(
        '--model-name',
        help='Name of the model',
        required=True
    )

    # other setup
    args_parser.add_argument(
        '--job-dir',
        help='Location used for exports and temp data',
        required=True
    )
    args_parser.add_argument(
        '--source-bucket',
        help='Bucket where training data is expected',
        default='models.stufff.review'
    )
    args_parser.add_argument(
        '--model-bucket',
        help='Bucket where model data will be exported to',
        default='models.stufff.review'
    )

    return args_parser.parse_args()

