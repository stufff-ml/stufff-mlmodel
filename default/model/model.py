import argparse
import pandas as pd
import numpy as np

from util import read_dataframe, write_dataframe

from surprise import Dataset, Reader, SVD, SVDpp
from surprise.model_selection import GridSearchCV


MAX_RATING=1.0
PRECISION=5
FILTER_THRESHOLD=0.6
MAX_PREDICTION=50


def train(params):
    
    # Load the training data and create a df for training
    temp_df = read_dataframe(params.client_id, params.source_bucket, 'buy.csv')
    raw_df = pd.DataFrame(data={'entity_id': temp_df['entity_id'], 'target_entity_id': temp_df['target_entity_id']})
    raw_df['rating'] = MAX_RATING

    # create the training set
    reader = Reader(rating_scale=(0, MAX_RATING))
    data = Dataset.load_from_df(raw_df[['entity_id', 'target_entity_id', 'rating']], reader)
    training_data = data.build_full_trainset()

    # Find optimal parameters
    print(' --> fitting the model')

    param_grid = {'n_epochs': [10, 30], 'lr_all': [0.002, 0.005], 'reg_all': [0.2, 0.6]}
    gs = GridSearchCV(SVDpp, param_grid, measures=['rmse', 'mae'], cv=3)
    gs.fit(data)

    print(gs.best_score['rmse'])
    print(gs.best_params['rmse'])

    # Build an model, and train it
    print(' --> build the model')

    #model = SVD()
    model = gs.best_estimator['rmse']
    model.fit(training_data)

    # Batch predictions
    print(' --> batch predictions')

    unique_entity = np.unique(raw_df.entity_id.values)
    unique_target_entity= np.unique(raw_df.target_entity_id.values)

    px = pd.DataFrame(-1.0, index=unique_entity, columns=unique_target_entity,dtype=np.float64)
    predx = training_data.build_anti_testset(fill=0)

    for p in predx:
      pred = model.predict(training_data.to_inner_uid(p[0]), training_data.to_inner_iid(p[1]))
      px.at[p[0], p[1]] = round(pred.est, PRECISION)

    # create the export
    print(' --> create export')
    ex = pd.DataFrame(index=unique_entity, columns={'entity_type','target_entity_type','values'})

    for id in unique_entity:
      p1 = px.loc[ id , : ]
      p2 = p1.sort_values(ascending=False)
      p3a = p2[p2 > FILTER_THRESHOLD]
      p3 = p3a[p3a < 1.0].head(MAX_PREDICTION)
      t = zip(p3.index.tolist(), p3.values)
      tf = [item for sublist in t for item in sublist]

      ex.at[id, 'entity_type'] = 'user'
      ex.at[id, 'target_entity_type'] = 'item'
      ex.at[id, 'values'] = tf

    write_dataframe(params.job_id, params.job_dir, 'pred_user.csv', ['entity_type','target_entity_type','values'], 'entity_id', ex)
