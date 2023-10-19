from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple
import logging
import os
import uuid
from pprint import pformat

from rastervision.pipeline import rv_config_ as rv_config
from rastervision.pipeline.runner import Runner

if TYPE_CHECKING:
    from rastervision.pipeline.pipeline import Pipeline

log = logging.getLogger(__name__)

AWS_BATCH = 'batch'


def submit_job(cmd: List[str],
               job_name: str,
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
        cmd: Command to run in the Docker container for the remote job as list
            of strings.
        debug: If True, run the command using a ptvsd wrapper which sets up a
            remote VS Code Python debugger server.
        profile: If True, run the command using kernprof, a line profiler.
        attempts: The number of times to try running the command which is
            useful in case of failure.
        parent_job_ids: Optional list of parent Batch job ids. The job created
            by this will only run after the parent jobs complete successfully.
        num_array_jobs: If set, make this a Batch array job with size equal to
            num_array_jobs.
        use_gpu: If True, run the job in a GPU-enabled queue.
        job_queue: If set, use this job queue.
        job_def: If set, use this job definition.
    """
    import boto3

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

    if debug:
        cmd = [
            'python', '-m', 'ptvsd', '--host', '0.0.0.0', '--port', '6006',
            '--wait', '-m'
        ] + cmd

    if profile:
        cmd = ['kernprof', '-v', '-l'] + cmd

    kwargs = {
        'jobName': job_name,
        'jobQueue': job_queue,
        'jobDefinition': job_def,
        'containerOverrides': {
            'command': cmd
        },
        'retryStrategy': {
            'attempts': attempts
        },
    }
    if parent_job_ids:
        kwargs['dependsOn'] = [{'jobId': id} for id in parent_job_ids]
    if num_array_jobs:
        kwargs['arrayProperties'] = {'size': num_array_jobs}

    client = boto3.client('batch')
    job_id = client.submit_job(**kwargs)['jobId']
    return job_id


class AWSBatchRunner(Runner):
    """Runs pipelines remotely using AWS Batch.

    Requires Everett configuration of form:

    .. code-block:: ini

        [AWS_BATCH]
        cpu_job_queue=
        cpu_job_def=
        gpu_job_queue=
        gpu_job_def=
        attempts=
    """

    def run(self,
            cfg_json_uri: str,
            pipeline: 'Pipeline',
            commands: List[str],
            num_splits: int = 1,
            pipeline_run_name: str = 'raster-vision'):
        cmd, args = self.build_cmd(
            cfg_json_uri,
            pipeline,
            commands,
            num_splits,
            pipeline_run_name=pipeline_run_name)
        job_id = submit_job(cmd=cmd, **args)

        job_info = dict(
            name=args['job_name'],
            id=job_id,
            parents=args['parent_job_ids'],
            cmd=cmd,
        )
        job_info_str = pformat(job_info, sort_dicts=False)
        msg = (f'Job submitted:\n{job_info_str}')
        log.info(msg)

    def build_cmd(self,
                  cfg_json_uri: str,
                  pipeline: 'Pipeline',
                  commands: List[str],
                  num_splits: int = 1,
                  pipeline_run_name: str = 'raster-vision'
                  ) -> Tuple[List[str], Dict[str, Any]]:
        parent_job_ids = []

        # pipeline-specific job queue and job definition
        pipeline_job_queue = getattr(pipeline, 'job_queue', None)
        pipeline_job_def = getattr(pipeline, 'job_def', None)

        for command in commands:
            # command-specific job queue, job definition
            cmd_obj = getattr(pipeline, command, None)
            job_def = getattr(cmd_obj, 'job_def', pipeline_job_def)
            job_queue = getattr(cmd_obj, 'job_queue', pipeline_job_queue)

            num_array_jobs = None
            use_gpu = command in pipeline.gpu_commands

            job_name = f'{pipeline_run_name}-{command}-{uuid.uuid4()}'

            cmd = ['python', '-m', 'rastervision.pipeline.cli']

            if rv_config.get_verbosity() > 1:
                num_vs = rv_config.get_verbosity() - 1
                # produces a string like "-vvv..."
                verbosity_opt_str = f'-{"v" * num_vs}'
                cmd += [verbosity_opt_str]

            cmd += [
                'run_command', cfg_json_uri, command, '--runner', AWS_BATCH
            ]

            if command in pipeline.split_commands and num_splits > 1:
                num_array_jobs = num_splits
                cmd += ['--num-splits', str(num_splits)]

            args = dict(
                job_name=job_name,
                parent_job_ids=parent_job_ids,
                num_array_jobs=num_array_jobs,
                use_gpu=use_gpu,
                job_queue=job_queue,
                job_def=job_def)

            return cmd, args

    def get_split_ind(self):
        return int(os.environ.get('AWS_BATCH_JOB_ARRAY_INDEX', 0))
