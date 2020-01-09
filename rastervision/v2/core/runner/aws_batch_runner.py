import uuid
import logging

from rastervision.v2.core import _rv_config

log = logging.getLogger(__name__)
AWS_BATCH = 'aws_batch'


def submit_job(
        cmd,
        debug=False,
        profile=False,
        attempts=5,
        parent_job_ids=None,
        num_array_jobs=None,
        use_gpu=False):
    batch_config = _rv_config.get_subconfig('AWS_BATCH')
    job_queue = batch_config('cpu_job_queue')
    job_def = batch_config('cpu_job_definition')
    if use_gpu:
        job_queue = batch_config('job_queue')
        job_def = batch_config('job_definition')

    import boto3
    client = boto3.client('batch')
    job_name = 'ffda-{}'.format(uuid.uuid4())

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
    msg = 'submitted job with jobName={} and jobId={}'.format(job_name, job_id)
    log.info(msg)
    log.info(cmd_list)

    return job_id


class AWSBatchRunner():
    def run(self, cfg_json_uri, pipeline, commands, num_splits=1):
        parent_job_ids = []
        for command in commands:
            cmd = [
                'python', '-m',
                'rastervision.v2 run_command', cfg_json_uri,
                command
            ]
            num_array_jobs = None
            if command in pipeline.split_commands and num_splits > 1:
                num_array_jobs = num_splits
                if num_splits > 1:
                    cmd += ['--num-splits', str(num_splits)]
            use_gpu = command in pipeline.gpu_commands
            cmd = ' '.join(cmd)

            job_id = submit_job(
                cmd,
                parent_job_ids=parent_job_ids,
                num_array_jobs=num_array_jobs,
                use_gpu=use_gpu)
            parent_job_ids = [job_id]
