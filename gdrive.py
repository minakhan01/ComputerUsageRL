import os
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from pydrive.files import GoogleDriveFileList
from google.colab import auth
from oauth2client.client import GoogleCredentials

from getpass import getpass
import urllib

auth.authenticate_user()
gauth = GoogleAuth()
gauth.credentials = GoogleCredentials.get_application_default()
drive = GoogleDrive(gauth)

from google.colab import drive as gdrive 
gdrive.mount('/content/gdrive')

def get_folder_id(drive, parent_folder_id, folder_name):
  """ 
	Check if destination folder exists and return it's ID
	"""
  folder_exists = False

  # Auto-iterate through all files in the parent folder.
  file_list = GoogleDriveFileList()
  try:
    file_list = drive.ListFile(
        {'q': "'{0}' in parents and trashed=false".format(parent_folder_id)}
        ).GetList()
	# Exit if the parent folder doesn't exist
  except googleapiclient.errors.HttpError as err:
  # Parse error message
    message = ast.literal_eval(err.content)['error']['message']
    if message == 'File not found: ':
      exit(1)
  # Exit with stacktrace in case of other error
    else:
      raise

	# Find the the destination folder in the parent folder's files
  for file1 in file_list:
    if file1['title'] == folder_name:
      folder_exists = True
      return file1['id'], folder_exists

  return None, False

experiment_id = "MinaData"

folder_name = experiment_id

# Change parentid to match that of experiments root folder in gdrive
parentid = '1fakqJ81EcE65gUtVfFWBagQfGEI9IC1j'

# Initialize sepcific experiment folder in drive
folderid, folder_exists = get_folder_id(drive, parentid, folder_name)

import json
import io
from redact_data import *
from redact_data import _write_file, _load_data, redact_data_basic, redact_data_custom, set_timezone

def get_files_dataset(drive, folderid):
  file_list = drive.ListFile({'q': "'{}' in parents and trashed=false".format(folderid)}).GetList()
  for i, file1 in enumerate(sorted(file_list, key = lambda x: x['title']), start=1):
    print(file1['title'])
    file1.GetContentFile(file1['title'])
    # yield file1['title']

def redact_file(input_file, output_file):

    data, bucket_key_dict = _load_data(input_file)
    redacted_data, unique_urls, unique_app_names = redact_data_basic(data, bucket_key_dict)

    url_blacklist = set(["docs.google.com", "www.deeplearningbook.org"])
    app_blacklist = set(["Finder"])

    user_selected_timezone = 'America/New_York'

    # Remove apps and urls from user provided black list
    redacted_data = redact_data_custom(redacted_data, url_blacklist, app_blacklist, 3)
    
    # set user selected timezone
    redacted_data = set_timezone(redacted_data, user_selected_timezone)

    _write_file(output_file, redacted_data)

get_files_dataset(drive, folderid)
filepath = "aw-buckets-export.json"
output_file = "redacted-data.json"
redact_file(filepath, output_file)
redacted_data, bucket_key_dict = _load_data(output_file)