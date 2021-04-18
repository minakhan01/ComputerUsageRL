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