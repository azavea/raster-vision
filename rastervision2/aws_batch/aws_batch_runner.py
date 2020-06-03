import copy
import logging
import os
import uuid
from inspect import signature
from typing import List, Optional

from rastervision2.pipeline import rv_config
from rastervision2.pipeline.runner import Runner

log = logging.getLogger(__name__)
AWS_BATCH = 'aws_batch'


def submit_job(cmd: List[str],
               debug: bool = False,
               profile: str = False,
               attempts: int = 5,
               parent_job_ids: List[str] = None,
               num_array_jobs: Optional[int] = None,
               use_gpu: bool = False,
               job_queue: Optional[str] = None,
               job_def: Optional[str] = None) -> str:
    """Submit a job to run on AWS Batch.

    Args:
        cmd: a command to run in the Docker container for the remote job
        debug: if True, run the command using a ptvsd wrapper which sets up a remote
            VS Code Python debugger server
        profile: if True, run the command using kernprof, a line profiler
        attempts: the number of times to try running the command which is useful
            in case of failure.
        parent_job_ids: optional list of parent Batch job ids. The job created by this
            will only run after the parent jobs complete successfully.
        num_array_jobs: if set, make this a Batch array job with size equal to
            num_array_jobs
        use_gpu: if True, run the job in a GPU-enabled queue
        job_queue: if set, use this job queue
        job_def: if set, use this job definition
    """
    batch_config = rv_config.get_namespace_config(AWS_BATCH)

    if job_queue is None:
        if use_gpu:
            job_queue = batch_config('gpu_job_queue')
        else:
            job_queue = batch_config('cpu_job_queue')

    if job_def is None:
        if use_gpu:
            job_def = batch_config('gpu_job_def')
        else:
            job_def = batch_config('cpu_job_def')

    import boto3
    client = boto3.client('batch')
    job_name = 'raster-vision-{}'.format(uuid.uuid4())

    cmd_list = cmd.split(' ')
    if debug:
        cmd_list = [
            'python', '-m', 'ptvsd', '--host', '0.0.0.0', '--port', '6006',
            '--wait', '-m'
        ] + cmd_list

    if profile:
        cmd_list = ['kernprof', '-v', '-l'] + cmd_list

    kwargs = {
        'jobName': job_name,
        'jobQueue': job_queue,
        'jobDefinition': job_def,
        'containerOverrides': {
            'command': cmd_list
        },
        'retryStrategy': {
            'attempts': attempts
        },
    }
    if parent_job_ids:
        kwargs['dependsOn'] = [{'jobId': id} for id in parent_job_ids]
    if num_array_jobs:
        kwargs['arrayProperties'] = {'size': num_array_jobs}

    job_id = client.submit_job(**kwargs)['jobId']
    msg = 'submitted job with jobName={} and jobId={} w/ parent(s)={}'.format(
        job_name, job_id, parent_job_ids)
    log.info(msg)
    log.info(cmd_list)

    return job_id


class AWSBatchRunner(Runner):
    """Runs pipelines remotely using AWS Batch.

    Requires Everett configuration of form:

    ```
    [AWS_BATCH]
    cpu_job_queue=
    cpu_job_def=
    gpu_job_queue=
    gpu_job_def=
    attempts=
    ```
    """

    def run(self, cfg_json_uri, pipeline, commands, num_splits=1):
        parent_job_ids = []

        # pipeline-specific job queue
        if hasattr(pipeline, 'job_queue'):
            pipeline_job_queue = pipeline.job_queue
        else:
            pipeline_job_queue = None

        # pipeline-specific job definition
        if hasattr(pipeline, 'job_def'):
            pipeline_job_def = pipeline.job_def
        else:
            pipeline_job_def = None

        for command in commands:

            # detect external command
            if hasattr(pipeline, command):
                fn = getattr(pipeline, command)
                params = signature(fn).parameters
                external = hasattr(fn, 'external') and len(params) in {0, 1}
                array_job_capable = hasattr(fn, 'array_job_capable') \
                    and fn.array_job_capable
            else:
                external = False
                array_job_capable = False

            # command-specific job queue, job definition
            job_def = pipeline_job_def
            job_queue = pipeline_job_queue
            if hasattr(pipeline, command):
                fn = getattr(pipeline, command)
                if hasattr(fn, 'job_def'):
                    job_def = fn.job_def
                if hasattr(fn, 'job_queue'):
                    job_queue = fn.job_queue

            num_array_jobs = None
            use_gpu = command in pipeline.gpu_commands

            if not external:
                cmd = [
                    'python', '-m', 'rastervision2.pipeline.cli run_command',
                    cfg_json_uri, command, '--runner', AWS_BATCH
                ]
                if command in pipeline.split_commands and num_splits > 1:
                    num_array_jobs = num_splits
                    cmd += ['--num-splits', str(num_splits)]
                job_id = submit_job(
                    ' '.join(cmd),
                    parent_job_ids=parent_job_ids,
                    num_array_jobs=num_array_jobs,
                    use_gpu=use_gpu,
                    job_queue=job_queue,
                    job_def=job_def)
                parent_job_ids = [job_id]
            else:
                if command in pipeline.split_commands and num_splits > 1:
                    if len(params) == 1 and array_job_capable:
                        cmd = fn(-num_splits)
                        num_array_jobs = num_splits
                        job_id = submit_job(
                            ' '.join(cmd),
                            parent_job_ids=parent_job_ids,
                            num_array_jobs=num_array_jobs,
                            use_gpu=use_gpu,
                            job_queue=job_queue,
                            job_def=job_def)
                        parent_job_ids = [job_id]
                    elif len(params) == 1 and not array_job_capable:
                        num_array_jobs = None
                        new_parent_job_ids = []
                        for cmd in fn(num_splits):
                            job_id = submit_job(
                                ' '.join(cmd),
                                parent_job_ids=parent_job_ids,
                                num_array_jobs=num_array_jobs,
                                use_gpu=use_gpu,
                                job_queue=job_queue,
                                job_def=job_def)
                            new_parent_job_ids.append(job_id)
                        parent_job_ids = copy.copy(new_parent_job_ids)
                    elif len(params) == 0:
                        cmd = fn()
                        num_array_jobs = None
                        job_id = submit_job(
                            ' '.join(cmd),
                            parent_job_ids=parent_job_ids,
                            num_array_jobs=num_array_jobs,
                            use_gpu=use_gpu,
                            job_queue=job_queue,
                            job_def=job_def)
                        parent_job_ids = [job_id]
                else:
                    if len(params) == 0:
                        cmd = fn()
                    elif len(params) == 1:
                        cmd = fn(1)[0]
                    num_array_jobs = 1
                    job_id = submit_job(
                        ' '.join(cmd),
                        parent_job_ids=parent_job_ids,
                        num_array_jobs=num_array_jobs,
                        use_gpu=use_gpu,
                        job_queue=job_queue,
                        job_def=job_def)
                    parent_job_ids = [job_id]

            job_queue = None
            job_def = None

    def get_split_ind(self):
        return int(os.environ.get('AWS_BATCH_JOB_ARRAY_INDEX', 0))
