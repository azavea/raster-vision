from rastervision.new_version.batch_submit import _batch_submit

AWS_BATCH = 'aws_batch'


class AWSBatchRunner():
    def run(self, cfg_json_uri, pipeline, commands, num_splits=1):
        parent_job_ids = []
        for command in commands:
            cmd = [
                'python', '-m',
                'rastervision.new_version.pipeline.run_command', cfg_json_uri,
                command
            ]
            num_array_jobs = None
            if command in pipeline.split_commands and num_splits > 1:
                num_array_jobs = num_splits
                if num_splits > 1:
                    cmd += ['--num-splits', str(num_splits)]
            use_gpu = command in pipeline.gpu_commands
            cmd = ' '.join(cmd)

            job_id = _batch_submit(
                cmd,
                parent_job_ids=parent_job_ids,
                num_array_jobs=num_array_jobs,
                use_gpu=use_gpu)
            parent_job_ids = [job_id]
