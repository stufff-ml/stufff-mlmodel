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

    # model hyperparams
    parser.add_argument(
        '--weights',
        default=DEFAULT_HYPERPARAMS['weights']
    )
    parser.add_argument(
        '--latent-factors',
        default=DEFAULT_HYPERPARAMS['latent_factors']
    )
    parser.add_argument(
        '--num-iters',
        default=DEFAULT_HYPERPARAMS['num_iters']
    )
    parser.add_argument(
        '--regularization',
        default=DEFAULT_HYPERPARAMS['regularization']
    )
    parser.add_argument(
        '--unobs-weight',
        default=DEFAULT_HYPERPARAMS['unobs_weight']
    )
    parser.add_argument(
        '--wt-type',
        default=DEFAULT_HYPERPARAMS['wt_type']
    )
    parser.add_argument(
        '--feature_wt_factor',
        default=DEFAULT_HYPERPARAMS['feature_wt_factor']
    )
    parser.add_argument(
        '--feature-wt-exp',
        default=DEFAULT_HYPERPARAMS['feature_wt_exp']
    )

    # generic setup parameters
    parser.add_argument(
        '--predict-batch-size',
        help='Batch size for each prediction',
        type=int,
        default=10
    )
    parser.add_argument(
        '--test-percentage',
        help='Percentage of the data used to valiate the model',
        type=float,
        default=0.2
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

    status = 'ok'
    time_start = datetime.utcnow()
    
    try:
        model.train(args)
    except:
        status = 'error'
   
    time_end = datetime.utcnow()
    time_elapsed = time_end - time_start

    # notify the engine
    urllib2.urlopen(args.callback + '&status=' + status + "&t={}".format(time_elapsed.total_seconds())).read()
    