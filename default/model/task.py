#!/usr/bin/env python

import os
import argparse
import pandas as pd
import numpy as np

from datetime import datetime
from google.cloud import storage
from io import BytesIO

import tensorflow as tf

import metadata
import model
import wals
import util
import prediction as pre

def main():

    # Run the train and evaluate experiment
    time_start = datetime.utcnow()
    print('')
    print(" --> Training started at {}".format(time_start.strftime("%H:%M:%S")))
    print(PARAMS)
    print('')

    source_name = PARAMS.model_id + '/' + PARAMS.model_id + '.' + PARAMS.model_rev + '.csv'
    client = storage.Client()
    source_bucket = client.get_bucket(PARAMS.source_bucket)

    # Load the training data as a dataframe
    print(' --> Import data and prepare training set')
    print("")

    source_blob = source_bucket.get_blob(source_name)
    content = source_blob.download_as_string()

    headers = ['event','entity_type','entity_id','target_entity_type','target_entity_id','timestamp','properties']
    header_types = {'entity_id':np.int32, 'target_entity_id':np.int32, 'timestamp':np.int32 }
    data = pd.read_csv(BytesIO(content), names=headers, header=None, dtype=header_types)

    entity_map, target_entity_map, training_sparse, test_sparse = model.create_data_sets(data, PARAMS.test_percentage)

    print('')
    print(" --> Build the model")
    print('')

    output_row, output_col = model.train_model(model.DEFAULT_HYPERPARAMS, training_sparse)

    print('')
    print(" --> Batch predictions")
    print('')

    predictions = pre.batch_predictions(data, output_row, output_col, PARAMS.predict_batch_size)

    print('')
    print(" --> Save the model")
    print('')

    util.export_model(PARAMS.model_id, PARAMS.model_rev, PARAMS.job_dir, entity_map, target_entity_map, output_row, output_col)
    util.export_predictions(PARAMS.model_id, PARAMS.model_rev, PARAMS.job_dir, predictions)

    train_rmse = wals.get_rmse(output_row, output_col, training_sparse)
    test_rmse = wals.get_rmse(output_row, output_col, test_sparse)
    
    tf.logging.info('Train RMSE = %.2f' % train_rmse)
    tf.logging.info('Test RMSE = %.2f' % test_rmse)

    time_end = datetime.utcnow()
    
    print('')
    print(" --> Training finished at {}".format(time_end.strftime("%H:%M:%S")))
    print('')
    time_elapsed = time_end - time_start
    print(" --> Training elapsed time: {} seconds".format(time_elapsed.total_seconds()))
    print('')


args_parser = argparse.ArgumentParser()
PARAMS = metadata.initialise_params(args_parser)

if __name__ == '__main__':
    main()
