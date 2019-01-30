#!/usr/bin/env python

import argparse
import urllib2
from datetime import datetime
import model

# default hyperparameters
DEFAULT_HYPERPARAMS = {
    'weights': True,
    'latent_factors': 5,
    'num_iters': 20,
    'regularization': 0.07,
    'unobs_weight': 0.01,
    'wt_type': 0,
    'feature_wt_factor': 130.0,
    'feature_wt_exp': 0.08
}

def setup(parser):
    
    # job specific setup parameters
    
    parser.add_argument(
        '--job-id',
        help='ID of this instance',
        required=True
    )
    parser.add_argument(
        '--client-id',
        help='ID of the client resource',
        required=True
    )
    parser.add_argument(
        '--model-name',
        help='Name of the model',
        required=True
    )
    parser.add_argument(
        '--job-dir',
        help='Location used for exports and temp data',
        required=True
    )
    parser.add_argument(
        '--callback',
        help='Callback URL',
        default='http://localhost:8080'
    )

    # generic setup parameters
    parser.add_argument(
        '--batch-size',
        help='Batch size for each prediction',
        type=int,
        default=25
    )
    parser.add_argument(
        '--source-bucket',
        help='Bucket where training data is expected',
        default='exports.stufff.review'
    )
    parser.add_argument(
        '--target-bucket',
        help='Bucket where model data will be exported to',
        default='models.stufff.review'
    )

    return parser.parse_args()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args = setup( parser)

    print(" --> PARAMS")
    print(args)
    print('')

    status = 'ok'
    try:
        model.train(args)
        urllib2.urlopen(args.callback + '&status=ok').read()
    except Exception as e:
        urllib2.urlopen(args.callback + '&status=error').read()
        print(e)
        raise e
    