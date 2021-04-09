
import logging
import numpy as np

from ray.rllib.utils.sgd import standardized
from ray.rllib.agents import with_common_config
from ray.rllib.agents.maml.maml_tf_policy import MAMLTFPolicy
from ray.rllib.agents.maml.maml_torch_policy import MAMLTorchPolicy
from ray.rllib.agents.trainer_template import build_trainer
from ray.rllib.evaluation.metrics import get_learner_stats
from ray.rllib.execution.common import STEPS_SAMPLED_COUNTER, \
    STEPS_TRAINED_COUNTER, LEARNER_INFO, _get_shared_metrics
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.execution.metric_ops import CollectMetrics
from ray.rllib.evaluation.metrics import collect_metrics
from ray.rllib.utils.deprecation import DEPRECATED_VALUE
from ray.util.iter import from_actors
logger = logging.getLogger(__name__)
def set_worker_tasks(workers, use_meta_env):
    if use_meta_env:
        n_tasks = len(workers.remote_workers())
        tasks = workers.local_worker().foreach_env(lambda x: x)[
            0].sample_tasks(n_tasks)
        for i, worker in enumerate(workers.remote_workers()):
            worker.foreach_env.remote(lambda env: env.set_task(tasks[i]))
def post_process_metrics(adapt_iter, workers, metrics):
    # Obtain Current Dataset Metrics and filter out
    name = "_adapt_" + str(adapt_iter) if adapt_iter > 0 else ""

    # Only workers are collecting data
    res = collect_metrics(remote_workers=workers.remote_workers())

    metrics["episode_reward_max" + str(name)] = res["episode_reward_max"]
    metrics["episode_reward_mean" + str(name)] = res["episode_reward_mean"]
    metrics["episode_reward_min" + str(name)] = res["episode_reward_min"]

    return metrics
class MetaUpdate:
    def __init__(self, workers, maml_steps, metric_gen, use_meta_env):
        self.workers = workers
        self.maml_optimizer_steps = maml_steps
        self.metric_gen = metric_gen
        self.use_meta_env = use_meta_env

    def __call__(self, data_tuple):
        # Metaupdate Step
        samples = data_tuple[0]
        adapt_metrics_dict = data_tuple[1]

        # Metric Updating
        metrics = _get_shared_metrics()
        metrics.counters[STEPS_SAMPLED_COUNTER] += samples.count

        # Sync workers with meta policy
        self.workers.sync_weights()

        # Set worker tasks
        set_worker_tasks(self.workers, self.use_meta_env)

        # Update KLS
        def update(pi, pi_id):
            pass

        self.workers.local_worker().foreach_trainable_policy(update)

        # Modify Reporting Metrics
        metrics = _get_shared_metrics()
        #metrics.info[LEARNER_INFO] = fetches
        metrics.counters[STEPS_TRAINED_COUNTER] += samples.count

        res = self.metric_gen.__call__(None)
        res.update(adapt_metrics_dict)

        return res

def inner_adaptation(workers, samples):
    # Each worker performs one gradient descent
    for i, e in enumerate(workers.remote_workers()):
        e.learn_on_batch.remote(samples[i])

def execution_plan(workers, config):
    # Sync workers with meta policy
    workers.sync_weights()

    # Samples and sets worker tasks
    use_meta_env = config["use_meta_env"]
    set_worker_tasks(workers, use_meta_env)

    # Metric Collector
    metric_collect = CollectMetrics(
        workers,
        min_history=config["metrics_smoothing_episodes"],
        timeout_seconds=config["collect_metrics_timeout"])

    # Iterator for Inner Adaptation Data gathering (from pre->post adaptation)
    inner_steps = config["inner_adaptation_steps"]

    def inner_adaptation_steps(itr):
        buf = []
        split = []
        metrics = {}
        for samples in itr:
            print("Restarting inner adaptation loop")
            # Processing Samples (Standardize Advantages)
            split_lst = []
            for sample in samples:
                sample["advantages"] = standardized(sample["advantages"])
                split_lst.append(sample.count)

            buf.extend(samples)
            split.append(split_lst)

            adapt_iter = len(split) - 1
            print("starting to postprocess metrics")
            metrics = post_process_metrics(adapt_iter, workers, metrics)
            if len(split) > inner_steps:
                print("len split is larger than size of inner steps")
                out = SampleBatch.concat_samples(buf)
                out["split"] = np.array(split)
                buf = []
                split = []

                # Reporting Adaptation Rew Diff
                ep_rew_pre = metrics["episode_reward_mean"]
                ep_rew_post = metrics["episode_reward_mean_adapt_" +
                                      str(inner_steps)]
                metrics["adaptation_delta"] = ep_rew_post - ep_rew_pre
                yield out, metrics
                metrics = {}
            else:
                print("Doing an inner adaptation step")
                inner_adaptation(workers, samples)

    rollouts = from_actors(workers.remote_workers())
    rollouts = rollouts.batch_across_shards()
    rollouts = rollouts.transform(inner_adaptation_steps)

    # Metaupdate Step
    train_op = rollouts.for_each(
        MetaUpdate(workers, config["maml_optimizer_steps"], metric_collect,
                   use_meta_env))
    return train_op