#!/usr/bin/env python

import pandas as pd
import numpy as np


def single_prediction(entity_index, exclude, row_factor, col_factor, k):
  
  # bounds checking for args
  assert (row_factor.shape[0] - len(exclude)) >= k

  # retrieve entity factor
  entity_f = row_factor[entity_index]

  # dot product of item factors with user factor gives predicted ratings
  pred_ratings = col_factor.dot(entity_f)

  # find candidate recommended item indexes sorted by predicted rating
  k_r = k + len(exclude)
  candidate_items = np.argsort(pred_ratings)[-k_r:]

  # remove previously rated items and take top k
  recommended_items = [i for i in candidate_items if i not in exclude]
  recommended_items = recommended_items[-k:]

  # flip to sort highest rated first
  recommended_items.reverse()

  return recommended_items


def batch_predictions(data, row_factor, col_factor, k):

  unique_entity = np.unique(data.entity_id.values)
  unique_target_entity= np.unique(data.target_entity_id.values)

  #recommended_items = pd.DataFrame(index=unique_entity, columns=['r'])
  recommended_items = pd.DataFrame(columns=np.zeros(shape=k,dtype=np.int32))

  for id in unique_entity:
    item_idx = np.searchsorted(unique_entity, id)
    already_rated = data[data.entity_id.isin([id])].target_entity_id
    already_rated_idx = [np.searchsorted(unique_target_entity, i)
                          for i in already_rated]

    recs_idx = single_prediction(item_idx, already_rated_idx, row_factor, col_factor, k)
    recs = [unique_target_entity[i] for i in recs_idx]

    recommended_items.loc[id] = recs

  return recommended_items