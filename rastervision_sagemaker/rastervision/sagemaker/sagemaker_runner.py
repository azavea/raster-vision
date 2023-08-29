import logging
from typing import List

import boto3
from rastervision.pipeline import rv_config_ as rv_config
from rastervision.pipeline.runner import Runner

from sagemaker.processing import ScriptProcessor
import sagemaker.pytorch
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.pipeline_context import PipelineSession
from sagemaker.workflow.steps import ProcessingStep, TrainingStep

import os
import tempfile

log = logging.getLogger(__name__)
SAGEMAKER = 'sagemaker'


def make_step(
        step_name: str,
        cmd: List[str],
        role: str,
        image_uri: str,
        instance_type: str,
        use_spot_instances: bool,
        sagemaker_session: PipelineSession,
        tempdir: tempfile.TemporaryDirectory,
):

    python_executable = cmd[0]
    script_name = cmd[1]
    script_arguments = cmd[2:]
    assert python_executable == "python" or python_executable == "python3"

    if 'train' in cmd or 'predict' in cmd:
        # For (possibly) GPU-enabled steps, create an "Estimator".
        # Formally this should probably not be used for prediction in
        # this way, but it is expedient (especially given default
        # service quotas, and other stuff).
        random_py_file = tempfile.mktemp(suffix=".py", dir=tempdir)

        code = f'''#!/usr/bin/env python3

        import os

        os.system({" ".join(cmd)})
        '''

        with open(random_py_file, "w") as f:
            f.write(code)

        step_estimator = sagemaker.pytorch.PyTorch(
            entry_point=random_py_file,
            image_uri=image_uri,
            instance_count=1,
            instance_type=instance_type,
            role=role,
            sagemaker_session=sagemaker_session,
            # use_spot_instances=use_spot_instances,
            # wait_time=60,
        )
        step_args = step_estimator.fit(wait=False)
        step = TrainingStep(step_name, step_args=step_args)
    else:
        # For non-GPU-enabled steps, create a ScriptProcessor.
        step_processor = ScriptProcessor(
            role=role,
            image_uri=image_uri,
            instance_count=1,
            instance_type=instance_type,
            sagemaker_session=sagemaker_session,
            command=[python_executable],
        )
        step_args = step_processor.run(
            inputs=[],
            outputs=[],
            code=script_name,
            arguments=script_arguments,
        )
        step = ProcessingStep(step_name, step_args=step_args)

    return step


class SageMakerRunner(Runner):
    """Runs pipelines remotely using AWS Batch.

    Requires Everett configuration of form:

    ```
    [SAGEMAKER]
    exec_role=
    cpu_image=
    cpu_inst_type=
    gpu_image=
    gpu_inst_type=
    ```
    """

    def run(self,
            cfg_json_uri,
            pipeline,
            commands,
            num_splits=1,
            pipeline_run_name: str = 'raster-vision'):
        # parent_job_ids = []

        config = rv_config.get_namespace_config(SAGEMAKER)
        exec_role = config('exec_role')
        cpu_image = config('cpu_image')
        cpu_inst_type = config('cpu_inst_type')
        gpu_image = config('gpu_image')
        gpu_inst_type = config('gpu_inst_type')
        use_spot_instances = config('use_spot_instances').lower() == "yes"
        sagemaker_session = PipelineSession()

        steps = []

        with tempfile.TemporaryDirectory() as tempdir:
            for command in commands:

                use_gpu = command in pipeline.gpu_commands
                job_name = command
                cmd = [
                    'python',
                    '/opt/src/rastervision_pipeline/rastervision/pipeline/cli.py',  # XXX
                ]
                if rv_config.get_verbosity() > 1:
                    cmd.append('-' + 'v' * (rv_config.get_verbosity() - 1))
                cmd.extend(['run_command', cfg_json_uri, command])

                if command in pipeline.split_commands and num_splits > 1:
                    # If the step can be split, then split it into parts
                    # that do not depend on each other (can run in
                    # parallel).
                    _steps = []
                    for i in range(num_splits):
                        cmd += [
                            '--split-ind',
                            str(i), '--num-splits',
                            str(num_splits)
                        ]
                        step = make_step(
                            f'{job_name}_{i+1}of{num_splits}',
                            cmd,
                            exec_role,
                            gpu_image if use_gpu else cpu_image,
                            gpu_inst_type if use_gpu else cpu_inst_type,
                            use_spot_instances,
                            sagemaker_session,
                            tempdir,
                        )
                        step.add_depends_on(steps)
                        _steps.append(step)
                    steps.extend(_steps)
                else:
                    # If the step can not be split, then submit it as-is.
                    step = make_step(
                        job_name,
                        cmd,
                        exec_role,
                        gpu_image if use_gpu else cpu_image,
                        gpu_inst_type if use_gpu else cpu_inst_type,
                        use_spot_instances,
                        sagemaker_session,
                        tempdir,
                    )
                    step.add_depends_on(steps)
                    steps.append(step)

            # Submit the pipeline to SageMaker
            iam_client = boto3.client('iam')
            role_arn = iam_client.get_role(RoleName=exec_role)['Role']['Arn']
            pipeline = Pipeline(
                name=pipeline_run_name,
                steps=steps,
                sagemaker_session=sagemaker_session,
            )
            pipeline.upsert(role_arn=role_arn)
            execution = pipeline.start()

        print(execution.describe())
