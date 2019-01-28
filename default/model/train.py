#!/usr/bin/env python

import argparse
import urllib2
from datetime import datetime
import model


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
        '--source-bucket',
        help='Bucket where training data is expected',
        default='models.stufff.review'
    )
    parser.add_argument(
        '--model-bucket',
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
    