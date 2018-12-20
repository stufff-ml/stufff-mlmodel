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

def main():

    source_name = PARAMS.model_id + '.' + PARAMS.model_rev + '.csv'

    headers = ['event','entity_type','entity_id','target_entity_type','target_entity_id','timestamp','properties']
    header_types = {'entity_id':np.int32, 'target_entity_id':np.int32, 'timestamp':np.int32 }
    data = pd.read_csv(PARAMS.job_dir + '/' + source_name, names=headers, header=None, dtype=header_types)

    # build the model
    entity_map, target_entity_map, training_sparse, test_sparse = model.create_test_and_train_sets(data, PARAMS.test_percentage)
    output_row, output_col = model.train_model(model.DEFAULT_HYPERPARAMS, training_sparse)

    # save trained model to job directory
    model.save_model(PARAMS.model_id, PARAMS.model_rev, PARAMS.job_dir, entity_map, target_entity_map, output_row, output_col)

    # validate results
    train_rmse = wals.get_rmse(output_row, output_col, training_sparse)
    test_rmse = wals.get_rmse(output_row, output_col, test_sparse)

    print('')
    print(data)
    print(' -->')
    print(entity_map)
    print(' -->')
    print(target_entity_map)
    print(' -->')

    entity = data.entity_id.values
    unique_entity = np.unique(entity)
    target_entity = data.target_entity_id.values
    unique_target_entity= np.unique(target_entity)
    print(unique_target_entity)
    print(' -->')

    user_id = 99
    num_recs = 3
    item_recommendations = None
    user_idx = np.searchsorted(unique_entity, user_id)

    if user_idx:
      already_rated = data[data.entity_id.isin([user_id])].target_entity_id
      already_rated_idx = [np.searchsorted(unique_target_entity, i)
                           for i in already_rated]

      print(already_rated)
      print(' -->')
      print(already_rated_idx)
      print(' -->')

      recommendations = model.generate_recommendations(user_idx, already_rated_idx,output_row, output_col,num_recs)
      item_recommendations = [unique_target_entity[i] for i in recommendations]

      print('Rec:')
      print(recommendations)
      print(' -->')
      print(item_recommendations)






args_parser = argparse.ArgumentParser()
PARAMS = metadata.initialise_params(args_parser)

if __name__ == '__main__':
    main()