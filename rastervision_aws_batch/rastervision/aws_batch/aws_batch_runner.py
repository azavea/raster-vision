from typing import TYPE_CHECKING, Any
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
            commands: list[str],
            num_splits: int = 1,
            pipeline_run_name: str = 'raster-vision'):  # pragma: no cover
        parent_job_ids = []
        for command in commands:
            cmd, args = self.build_cmd(
                command,
                cfg_json_uri,
                pipeline,
                num_splits,
                pipeline_run_name=pipeline_run_name)
            job_id = self.run_command(
                cmd, parent_job_ids=parent_job_ids, **args)

            job_info = dict(
                name=args['job_name'],
                id=job_id,
                parents=parent_job_ids,
                cmd=cmd,
            )
            job_info_str = pformat(job_info, sort_dicts=False)
            msg = (f'Job submitted:\n{job_info_str}')
            log.info(msg)

            parent_job_ids = [job_id]

    def build_cmd(self,
                  command: str,
                  cfg_json_uri: str,
                  pipeline: 'Pipeline',
                  num_splits: int = 1,
                  pipeline_run_name: str = 'raster-vision'
                  ) -> tuple[list[str], dict[str, Any]]:

        verbosity = rv_config.get_verbosity_cli_opt()

        # pipeline-specific job queue and job definition
        pipeline_job_queue = getattr(pipeline, 'job_queue', None)
        pipeline_job_def = getattr(pipeline, 'job_def', None)

        # command-specific job queue, job definition
        cmd_obj = getattr(pipeline, command, None)
        job_def = getattr(cmd_obj, 'job_def', pipeline_job_def)
        job_queue = getattr(cmd_obj, 'job_queue', pipeline_job_queue)

        num_array_jobs = None
        use_gpu = command in pipeline.gpu_commands

        job_name = f'{pipeline_run_name}-{command}-{uuid.uuid4()}'

        cmd = ['python', '-m', 'rastervision.pipeline.cli']
        if verbosity:
            cmd += [verbosity]
        cmd += ['run_command', cfg_json_uri, command, '--runner', AWS_BATCH]

        if command in pipeline.split_commands and num_splits > 1:
            num_array_jobs = num_splits
            cmd += ['--num-splits', str(num_splits)]

        args = dict(
            job_name=job_name,
            num_array_jobs=num_array_jobs,
            use_gpu=use_gpu,
            job_queue=job_queue,
            job_def=job_def,
        )
        return cmd, args

    def get_split_ind(self) -> int:
        return int(os.environ.get('AWS_BATCH_JOB_ARRAY_INDEX', 0))

    def run_command(self,
                    cmd: list[str],
                    job_name: str | None = None,
                    debug: bool = False,
                    attempts: int = 1,
                    parent_job_ids: list[str] | None = None,
                    num_array_jobs: int | None = None,
                    use_gpu: bool = False,
                    job_queue: str | None = None,
                    job_def: str | None = None,
                    **kwargs) -> str:  # pragma: no cover
        """Submit a command as a job to AWS Batch.

        Args:
            cmd: Command to run in the Docker container for the remote job as
                list of strings.
            job_name: Optional job name. If None, is set to
                "raster-vision-<uuid>".
            debug: If True, run the command using a ptvsd wrapper which sets up
                a remote VS Code Python debugger server. Defaults to False.
            attempts: The number of times to try running the command which is
                useful in case of failure. Defaults to 5.
            parent_job_ids: Optional list of parent Batch job IDs. The job
                created by this will only run after the parent jobs complete
                successfully. Defaults to None.
            num_array_jobs: If set, make this a Batch array job with size equal
                to num_array_jobs. Defaults to None.
            use_gpu: If True, run the job in a GPU-enabled queue. Defaults to
                False.
            job_queue: If set, use this job queue. Default to None.
            job_def: If set, use this job definition. Default to None.
            **kwargs: Any other kwargs to pass to Batch when submitting job.
        """
        import boto3

        batch_config = rv_config.get_namespace_config(AWS_BATCH)
        device = 'gpu' if use_gpu else 'cpu'

        if job_name is None:
            job_name = f'raster-vision-{uuid.uuid4()}'
        if job_queue is None:
            job_queue = batch_config(f'{device}_job_queue')
        if job_def is None:
            job_def = batch_config(f'{device}_job_def')

        if debug:
            cmd = [
                'python', '-m', 'ptvsd', '--host', '0.0.0.0', '--port', '6006',
                '--wait', '-m'
            ] + cmd

        args = {
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
            args['dependsOn'] = [{'jobId': id} for id in parent_job_ids]
        if num_array_jobs:
            args['arrayProperties'] = {'size': num_array_jobs}

        client = boto3.client('batch')
        job_id = client.submit_job(**args, **kwargs)['jobId']
        return job_id
