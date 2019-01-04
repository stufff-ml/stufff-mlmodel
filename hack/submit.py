from googleapiclient import discovery
from googleapiclient import errors

def submit_job(project_id, job_id):

  # Store your full project ID in a variable in the format the API needs.
  parent_project = 'projects/{}'.format(project_id)

  # Build a representation of the Cloud ML API.
  ml = discovery.build('ml', 'v1')

  # Create a dictionary with the fields from the request body.
  training_input = {
    'scaleTier': 'BASIC',
    'packageUris': ['gs://models.stufff.review/packages/default-1/default-1.tar.gz'],
    'pythonModule': 'model.task',
    'args': [
      '--model-id', '26144595808e',
      '--model-rev','1'
    ],
    'region': 'europe-west1',
    "jobDir": 'gs://models.stufff.review/26144595808e/26144595808e/default-1',
    'runtimeVersion': '1.12',
    'pythonVersion': '2.7'
  }
  request_dict = {
    'jobId': job_id,
    'trainingInput': training_input
  }

  # Create a request to submit a model for training
  request = ml.projects().jobs().create(
                parent=parent_project, body=request_dict)

  # Make the call.
  try:
      response = request.execute()
      print(response)
  except errors.HttpError, err:
      # Something went wrong, print out some information.
      print('There was an error submitting the job. Check the details:')
      print(err._get_reason())


submit_job('stufff-review','M26144595808e_default_2')