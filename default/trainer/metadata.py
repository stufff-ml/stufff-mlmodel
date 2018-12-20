#!/usr/bin/env python

import argparse
import wals

def initialise_params(args_parser):
    """
    Define the arguments with the default values,
    parses the arguments passed to the task,
    and set the PARAMS global variable

    Args:
        args_parser
    """

    # model adnd data selection
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
        '--test-percentage',
        help='Percentage of the data used to valiate the model',
        type=float,
        default=0.2
    )
    
    # hyper params for model
    args_parser.add_argument(
        '--latent_factors',
        type=int,
        help='Number of latent factors',
    )
    args_parser.add_argument(
        '--num_iters',
        type=int,
        help='Number of iterations for alternating least squares factorization',
    )
    args_parser.add_argument(
        '--regularization',
        type=float,
        help='L2 regularization factor',
    )
    args_parser.add_argument(
        '--unobs_weight',
        type=float,
        help='Weight for unobserved values',
    )
    args_parser.add_argument(
        '--wt_type',
        type=int,
        help='Rating weight type (0=linear, 1=log)',
        default=wals.LINEAR_RATINGS
    )
    args_parser.add_argument(
        '--feature_wt_factor',
        type=float,
        help='Feature weight factor (linear ratings)',
    )
    args_parser.add_argument(
        '--feature_wt_exp',
        type=float,
        help='Feature weight exponent (log ratings)',
    )
    
    # other setup
    args_parser.add_argument(
        '--job-dir',
        help='Location used for all tmp data',
        required=True
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

