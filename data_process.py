
import numpy as np
from datetime import datetime

# for key in redacted_data["aw-watcher-afk"].keys():
afk = redacted_data["aw-watcher-afk"]
window = redacted_data["aw-watcher-window"]
browser = redacted_data["aw-watcher-web"]
stopwatch = redacted_data["aw-watcher-stopwatch"]

# convert to numpy array
window_vocab = {}
for event in window["events"]:
  app_name = event["data"]["app"]
  if app_name not in window_vocab:
    window_vocab[app_name] = 0
  window_vocab[app_name] += 1

url_vocab = {}
for event in browser["events"]:
  url_name = event["data"]["url"]
  if url_name not in url_vocab:
    url_vocab[url_name] = 0
  url_vocab[url_name] += 1

afk_vocab = {}
for event in afk["events"]:
  status_name = event["data"]["status"]
  if status_name not in afk_vocab:
    afk_vocab[status_name] = 0
  afk_vocab[status_name] += 1

filtered_url_vocab = [url for url in url_vocab.keys() if url_vocab[url] > 10] 

sorted_window_vocab = sorted(window_vocab.items(), key= lambda x: x[1], reverse=True)
window_word_to_id = {sorted_window_vocab[i][0]: i for i in range(len(sorted_window_vocab))}

sorted_url_vocab = sorted(url_vocab.items(), key= lambda x: x[1], reverse=True)
url_word_to_id = {sorted_url_vocab[i][0]: i for i in range(len(sorted_url_vocab))}

window_data = []
for event in window["events"]:
  app_id = window_word_to_id[event["data"]["app"]] 
  time = event["timestamp"]
  duration = event["duration"]
  date_time_obj = datetime.fromisoformat(time)
  timestamp = date_time_obj.timestamp()
  item = [timestamp, duration, 0, app_id]
  window_data.append(item)

url_data = []
for event in browser["events"]:
  url_id = url_word_to_id[event["data"]["url"]]
  tab_count = event["data"]["tabCount"]
  audible = event["data"]["audible"]                                               
  incognito = event["data"]["incognito"]                                               
  time = event["timestamp"]
  date_time_obj = datetime.fromisoformat(time)
  timestamp = date_time_obj.timestamp()
  duration = event["duration"]
  item = [timestamp, duration, 1, url_id]
  url_data.append(item)

afk_data = []
for event in afk["events"]:
  afk_id = event["data"]["status"] == "afk"                                             
  time = event["timestamp"]
  date_time_obj = datetime.fromisoformat(time)
  timestamp = date_time_obj.timestamp()
  duration = event["duration"]
  item = [timestamp, duration, 2, afk_id]
  afk_data.append(item)

all_data = np.concatenate([window_data, url_data, afk_data], axis=0)
py_sorted_data = sorted(all_data, key= lambda x: x[0])
sorted_data = np.array(py_sorted_data, dtype=np.int64)
zero_count = 0
zero_window_count = 0
zero_url_count = 0
zero_afk_count = 0

for row in sorted_data:
  if row[1] == 0:
    zero_count += 1 
    if row[2] == 0:
      zero_window_count += 1
    if row[2] == 1:
      zero_url_count += 1
    if row[2] == 2:
      zero_afk_count += 1

import tensorflow as tf
max_afk_vocab_size = 2
max_window_vocab_size = 100
max_url_vocab_size = 5000

combined_data = []
for row in sorted_data:
  if row[2] == 0: # window
    rename_vocab = row[3] + max_afk_vocab_size + 1
  elif row[2] == 1: # url
    rename_vocab = row[3] + max_afk_vocab_size + max_window_vocab_size + 1 
  elif row[2] == 2:
    rename_vocab = row[3] + 1
  else:
    raise
  if row[3] == 1874:
    print("rename_vocab", rename_vocab, row[3], row[2])
  combined_data.append([row[0], row[1], rename_vocab])
np_combined = np.array(combined_data, dtype= np.int64)

np_url = np.array(url_data)
np_all = np.array(sorted_data)

(np_combined[:,1] > 1).mean()

import tensorflow_datasets as tfds

timestamps = np_combined[:, 0]
delays = timestamps[1:] - timestamps[:-1]
zero = np.zeros([1])
all_delay = np.concatenate([zero, delays], axis = 0) 
delay_added = np.concatenate([np_combined, all_delay[:,None]], axis = 1)
tf_data_list = tf.data.Dataset.from_tensor_slices(delay_added).batch(64, drop_remainder=True).repeat().shuffle(1600).batch(16, drop_remainder=True)
tf_data = tf_data_list.map(lambda x: {"activity_obs": tf.cast(x[:,:-1,2], tf.int32), 
                                      "duration_obs": tf.cast(x[:,:-1,1], tf.int32),
                                      "time_obs": tf.cast(x[:,:-1,0], tf.int32),
                                      "delay_obs": tf.cast(x[:,:-1,3], tf.int32),
                                      "activity_target": tf.cast(x[:,1:,2], tf.int32),
                                      "duration_target": tf.cast(x[:,1:,1], tf.int32),
                                      "delay_target": tf.cast(x[:,1:,3], tf.int32)})
train_dataset = iter(tfds.as_numpy(tf_data))

from datetime import datetime

def event_to_tick(events, delta_time, max_simul_events):
  ticks = []
  current_events = []
  start_time = (events[0][0] // delta_time ) * delta_time
  current_time = start_time
  finished = 0
  num_feature = len(events[0])
  while True:
    for event in events[finished:]:
      if event[0] <= current_time:
        current_events.append(event)
        finished += 1
      else:
        break
    current_events_plus_padding = [x for x in current_events]
    # current_events_plus_time = [[current_time]+ event.tolist()
                                # for event in current_events]
    for j in range(max_simul_events-len(current_events)):
      current_events_plus_padding.append(np.zeros((num_feature,), dtype=np.int64))
    tick_events = np.array(current_events_plus_padding)
    assert tick_events.shape[0] == max_simul_events
    tick = dict(time=np.array([current_time]), 
                start_time=tick_events[None, :, 0],
                duration=tick_events[None, :, 1],
                event_type=tick_events[None, :, 2],
                start_time_features=get_time_features(tick_events[:, 0])[None, :],
                time_features=get_time_features(np.array([current_time]))[None, :]
                )
    # print("tick",jax.tree_map(lambda x: (x.shape, x.dtype), tick))
    yield tick
    current_time += delta_time
    current_events = [event for event in current_events if event[0] + event[1] > current_time]
    if not current_events and finished == len(events):
      break

def get_time_features(unix_times):
  return np.array([list(datetime.fromtimestamp(unix_time).timetuple()) for unix_time in unix_times])

time_interval = 60
max_simul_events = 64
all_ticks = list(event_to_tick(np_combined, time_interval, max_simul_events))

d_seq_len = 64
shift = 1
shuffle_buffer = 1600
batch_size = 16
import tree
def dict_batch(ds):
  tuple_ds = tf.data.Dataset.zip(tuple(tree.flatten(ds)))
  return tuple_ds.map(lambda *args: tree.unflatten_as(ds, args))

tfd_ticks_all = tf.data.Dataset.from_tensor_slices(ticks_dict)
tfd_ticks_window_ds = tfd_ticks_all.window(d_seq_len, shift=shift, drop_remainder=True)
tfd_ticks_window = tfd_ticks_window_ds.interleave(dict_batch).batch(d_seq_len, drop_remainder=True)
cardinality = tfd_ticks_window_ds.cardinality().numpy()[()]
num_train = int(0.9*cardinality)
num_test = cardinality - num_train
tfd_ticks_train = tfd_ticks_window.take(num_train).repeat().shuffle(shuffle_buffer).batch(batch_size, drop_remainder=True)
tfd_ticks_test = tfd_ticks_window.skip(num_train).take(num_test).repeat().shuffle(shuffle_buffer).batch(batch_size, drop_remainder=True)


np_train = tfds.as_numpy(tfd_ticks_train)
np_test = tfds.as_numpy(tfd_ticks_test)