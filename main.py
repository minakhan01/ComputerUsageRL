import time
import cpc
import data_process
import haiku as kh
import optax

get_files_dataset(drive, folderid)
filepath = "aw-buckets-export.json"

def cpc_loss_fn(inputs: ArrayNest):
  return cpc.CPC.Model()(inputs, is_training=True)

cpc_model = hk.transform(cpc_loss_fn)
train_iter = iter(np_train)
test_iter = iter(np_test)

grad_clip_value = 0.25
learning_rate = 2e-4

optimizer = optax.chain(
    optax.clip_by_global_norm(grad_clip_value),
    optax.adam(learning_rate, b1=0.9, b2=0.99))

updater = Updater(cpc_model.init, cpc_model.apply, optimizer)
# updater = CheckpointingUpdater(updater, checkpoint_dir)

# Initialize parameters.
logging.info('Initializing parameters...')
rng = jax.random.PRNGKey(428)
data = next(train_iter)
tree_print(data)
state = updater.init(rng, data)

LOG_EVERY = 50
EVAL_EVERY = 1000
MAX_STEPS = 10**6
NUM_EVAL_ITER = 50

logging.info('Starting train loop...')
prev_time = time.time()
for step in range(MAX_STEPS):
  data = next(train_iter)
  state, metrics = updater.update(state, data)
  if step % LOG_EVERY == 0:
    steps_per_sec = LOG_EVERY / (time.time() - prev_time)
    prev_time = time.time()
    metrics.update({'steps_per_sec': steps_per_sec})
    logging.info({k: float(v) for k, v in metrics.items()})
  if step % EVAL_EVERY == 0:
    eval_metrics = None
    for i in range(NUM_EVAL_ITER):
      data_test = next(test_iter)
      eval_metric = updater.eval(state, data_test)
      if eval_metrics is None:
        eval_metrics = eval_metric
      else:
        eval_metrics = {k: eval_metrics[k]+eval_metric[k] for k in eval_metrics}
    eval_metrics = {k: eval_metrics/NUM_EVAL_ITER for k in eval_metrics}
    logging.info({k: float(v) for k, v in eval_metrics.items()})