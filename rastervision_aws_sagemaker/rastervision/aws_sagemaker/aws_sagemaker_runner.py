from typing import TYPE_CHECKING
from os.path import join, basename
import logging
from pprint import pprint
import tarfile

import boto3
from rastervision.pipeline import rv_config_ as rv_config
from rastervision.pipeline.runner import Runner
from rastervision.pipeline.file_system import FileSystem
from rastervision.pipeline.file_system.utils import (str_to_file, get_tmp_dir,
                                                     upload_or_copy)

if TYPE_CHECKING:
    from rastervision.pipeline.pipeline import Pipeline
    from rastervision.core.rv_pipeline import RVPipeline, RVPipelineConfig
    from sagemaker.workflow.pipeline_context import _JobStepArguments
    from sagemaker import Session
    from sagemaker.workflow.pipeline import Pipeline as SageMakerPipeline
    from sagemaker.workflow.pipeline_context import PipelineSession
    from sagemaker.workflow.steps import ProcessingStep, TrainingStep

log = logging.getLogger(__name__)

AWS_SAGEMAKER = 'sagemaker'

DEFAULT_MAX_RUN_TIME = 24 * 60 * 60

PYTORCH_ESTIMATOR_SCRIPT_FILENAME = 'train.py'
PYTORCH_ESTIMATOR_TAR_FILENAME = 'train.tar.gz'
PYTORCH_ESTIMATOR_SCRIPT_TEMPLATE = """\
import os

from rastervision.pipeline import rv_config_ as rv_config
from rastervision.pipeline.cli import _run_command

if __name__ == '__main__':
    print('WORLD_SIZE', os.environ.get('WORLD_SIZE'))
    print('RANK', os.environ.get('RANK'))
    print('LOCAL_RANK', os.environ.get('LOCAL_RANK'))
    rv_config.set_tmp_dir_root('/opt/data/tmp/rv')
    _run_command('{cfg_json_uri}', '{rv_cmd}')
"""


class AWSSageMakerRunner(Runner):
    """Runs pipelines remotely using AWS SageMaker.

    Requires Everett configuration of form:

    .. code-block:: ini

        [SAGEMAKER]
        role=
        cpu_image=
        cpu_instance_type=
        gpu_image=
        gpu_instance_type=
        train_image=
        train_instance_type=
        train_instance_count=
        use_spot_instances=
        spot_instance_max_wait_time=
        max_run_time=
    """

    def run(self,
            cfg_json_uri: str,
            pipeline: 'Pipeline',
            commands: list[str],
            num_splits: int = 1,
            cmd_prefix: list[str] = [
                'python', '-m', 'rastervision.pipeline.cli'
            ],
            pipeline_run_name: str = 'rv'):
        config = rv_config.get_namespace_config(AWS_SAGEMAKER)
        role = config('role')

        sagemaker_pipeline = self.build_pipeline(
            cfg_json_uri,
            pipeline,
            commands,
            num_splits,
            cmd_prefix=cmd_prefix,
            pipeline_run_name=pipeline_run_name)

        # Submit the pipeline to SageMaker
        iam_client = boto3.client('iam')
        role_arn = iam_client.get_role(RoleName=role)['Role']['Arn']
        sagemaker_pipeline.upsert(role_arn=role_arn)
        execution = sagemaker_pipeline.start()

        pprint(execution.describe())

    def build_pipeline(self,
                       cfg_json_uri: str,
                       pipeline: 'Pipeline',
                       commands: list[str],
                       num_splits: int = 1,
                       cmd_prefix: list[str] = [
                           'python', '-m', 'rastervision.pipeline.cli'
                       ],
                       pipeline_run_name: str = 'rv') -> 'SageMakerPipeline':
        """Build a SageMaker Pipeline with each command as a step within it."""
        from sagemaker.workflow.pipeline_context import PipelineSession
        from sagemaker.workflow.pipeline import Pipeline as SageMakerPipeline
        from sagemaker.workflow.pipeline_definition_config import (
            PipelineDefinitionConfig)

        verbosity = rv_config.get_verbosity_cli_opt()
        config = rv_config.get_namespace_config(AWS_SAGEMAKER)
        role = config('role')
        cpu_image = config('cpu_image')
        cpu_instance_type = config('cpu_instance_type')

        gpu_image = config('gpu_image')
        gpu_instance_type = config('gpu_instance_type')

        train_image = config('train_image', default=gpu_image)
        train_instance_type = config(
            'train_instance_type', default=gpu_instance_type)
        train_instance_count = int(config('train_instance_count', default='1'))

        use_spot_instances = config('use_spot_instances').lower() == 'yes'
        spot_instance_max_wait_time = int(
            config(
                'spot_instance_max_wait_time',
                default=str(DEFAULT_MAX_RUN_TIME)))
        max_run_time = int(
            config('max_run_time', default=str(DEFAULT_MAX_RUN_TIME)))
        sagemaker_session = PipelineSession()

        steps = []

        for command in commands:
            job_name = f'{pipeline_run_name}-{command}'
            cmd = cmd_prefix[:]
            if verbosity:
                cmd += [verbosity]
            cmd.extend(['run_command', cfg_json_uri, command])

            if command.lower() == 'train':
                use_gpu = True
                instance_type = train_instance_type
                instance_count = train_instance_count
                image_uri = train_image
            else:
                use_gpu = command in pipeline.gpu_commands
                image_uri = gpu_image if use_gpu else cpu_image
                instance_type = (gpu_instance_type
                                 if use_gpu else cpu_instance_type)
                instance_count = 1
                use_spot_instances = False

            if command in pipeline.split_commands and num_splits > 1:
                # If the step can be split, then split it into parts
                # that do not depend on each other (can run in
                # parallel).
                step_splits = [None] * num_splits
                for i in range(num_splits):
                    split_cmd = cmd + [
                        '--split-ind',
                        str(i), '--num-splits',
                        str(num_splits)
                    ]
                    split_job_name = f'{job_name}_{i+1}of{num_splits}'
                    step_split = self.build_step(
                        pipeline,
                        step_name=command,
                        job_name=split_job_name,
                        cmd=split_cmd,
                        role=role,
                        image_uri=image_uri,
                        instance_type=instance_type,
                        use_spot_instances=use_spot_instances,
                        sagemaker_session=sagemaker_session,
                        instance_count=instance_count,
                        max_wait=spot_instance_max_wait_time,
                        max_run=max_run_time,
                    )
                    step_split.add_depends_on(steps)
                    step_splits[i] = step_split
                steps.extend(step_splits)
            else:
                # If the step can not be split, then submit it as-is.
                step = self.build_step(
                    pipeline,
                    step_name=command,
                    job_name=job_name,
                    cmd=cmd,
                    role=role,
                    image_uri=image_uri,
                    instance_type=instance_type,
                    use_spot_instances=use_spot_instances,
                    sagemaker_session=sagemaker_session,
                    instance_count=instance_count,
                    max_wait=spot_instance_max_wait_time,
                    max_run=max_run_time,
                )
                step.add_depends_on(steps)
                steps.append(step)

        pipeline_definition_config = PipelineDefinitionConfig(
            use_custom_job_prefix=True)
        sagemaker_pipeline = SageMakerPipeline(
            name=pipeline_run_name,
            steps=steps,
            sagemaker_session=sagemaker_session,
            pipeline_definition_config=pipeline_definition_config)
        return sagemaker_pipeline

    def build_step(self,
                   pipeline: 'RVPipeline',
                   step_name: str,
                   job_name: str,
                   cmd: list[str],
                   role: str,
                   image_uri: str,
                   instance_type: str,
                   use_spot_instances: bool,
                   sagemaker_session: 'PipelineSession',
                   instance_count: int = 1,
                   max_wait: int = DEFAULT_MAX_RUN_TIME,
                   max_run: int = DEFAULT_MAX_RUN_TIME,
                   **kwargs) -> 'TrainingStep | ProcessingStep':
        """Build appropriate SageMaker pipeline step.

        If ``step_name=='train'``, builds a :class:`.TrainingStep`. Otherwise,
        a :class:`.ProcessingStep`.
        """
        if not use_spot_instances:
            max_wait = None

        if step_name.lower() == 'train':
            from sagemaker.workflow.steps import TrainingStep

            estimator = self._build_pytorch_estimator(
                pipeline_cfg=pipeline.config,
                role=role,
                image_uri=image_uri,
                instance_type=instance_type,
                use_spot_instances=use_spot_instances,
                sagemaker_session=sagemaker_session,
                instance_count=instance_count,
                job_name=job_name,
                max_wait=max_wait,
                max_run=max_run,
                **kwargs,
            )
            step_args: '_JobStepArguments | None' = estimator.fit(wait=False)
            step = TrainingStep(job_name, step_args=step_args)
        else:
            from sagemaker.processing import Processor
            from sagemaker.workflow.steps import ProcessingStep

            step_processor = Processor(
                role=role,
                image_uri=image_uri,
                instance_count=1,
                instance_type=instance_type,
                sagemaker_session=sagemaker_session,
                entrypoint=cmd,
                **kwargs,
            )
            step_args: '_JobStepArguments | None' = step_processor.run(
                wait=False)
            step = ProcessingStep(job_name, step_args=step_args)

        return step

    def run_command(self,
                    cmd: list[str],
                    use_gpu: bool = False,
                    image_uri: str | None = None,
                    instance_type: str | None = None,
                    role: str | None = None,
                    job_name: str | None = None,
                    sagemaker_session: 'Session | None' = None) -> None:
        """Run a single command as a SageMaker processing job.

        Args:
            cmd (list[str]): The command to run.
            use_gpu (bool): Use the GPU instance type and image from the
                Everett config. This is ignored if image_uri and instance_type
                are provided. Defaults to False.
            image_uri (str | None): URI of docker image to use. If not
                provided, will be picked up from Everett config.
                Defaults to None.
            instance_type (str | None): AWS instance type to use. If not
                provided, will be picked up from Everett config.
                Defaults to None.
            role (str | None): AWS IAM role with SageMaker permissions. If
                not provided, will be picked up from Everett config.
                Defaults to None.
            job_name (str | None): Optional job name. Defaults to None.
            sagemaker_session (Session | None): SageMaker session.
                Defaults to None.
        """
        from sagemaker.processing import Processor

        config = rv_config.get_namespace_config(AWS_SAGEMAKER)
        device = 'gpu' if use_gpu else 'cpu'

        if role is None:
            role = config('role')
        if image_uri is None:
            image_uri = config(f'{device}_image')
        if instance_type is None:
            instance_type = config(f'{device}_instance_type')
        if sagemaker_session is None:
            from sagemaker import Session
            sagemaker_session = Session()

        processor = Processor(
            role=role,
            image_uri=image_uri,
            instance_count=1,
            instance_type=instance_type,
            sagemaker_session=sagemaker_session,
            entrypoint=cmd,
            base_job_name=job_name,
        )
        processor.run()

    def _build_pytorch_estimator(self,
                                 pipeline_cfg: 'RVPipelineConfig',
                                 role: str,
                                 image_uri: str,
                                 instance_type: str,
                                 sagemaker_session: 'PipelineSession',
                                 use_spot_instances: bool = False,
                                 instance_count: int = 1,
                                 distribution: dict | None = None,
                                 job_name: str | None = None,
                                 **kwargs):
        from sagemaker.pytorch import PyTorch
        from rastervision.aws_s3.s3_file_system import S3FileSystem

        if distribution is None:
            distribution = dict(torch_distributed=dict(enabled=True))

        train_uri = pipeline_cfg.train_uri
        if FileSystem.get_file_system(train_uri) != S3FileSystem:
            raise ValueError('Pipeline\'s train_uri must be an S3 URI.')

        with get_tmp_dir() as source_dir:
            # create script from template
            script_path = join(source_dir, PYTORCH_ESTIMATOR_SCRIPT_FILENAME)
            _write_train_script(
                script_path, cfg_json_uri=pipeline_cfg.get_config_uri())
            # tar and upload to S3
            tar_path = _tar_script(script_path, source_dir)
            tar_path_s3 = join(train_uri, PYTORCH_ESTIMATOR_TAR_FILENAME)
            upload_or_copy(tar_path, tar_path_s3)

        estimator = PyTorch(
            entry_point=PYTORCH_ESTIMATOR_SCRIPT_FILENAME,
            source_dir=tar_path_s3,
            image_uri=image_uri,
            distribution=distribution,
            instance_count=instance_count,
            instance_type=instance_type,
            role=role,
            sagemaker_session=sagemaker_session,
            base_job_name=job_name,
            use_spot_instances=use_spot_instances,
            **kwargs,
        )
        return estimator


def _write_train_script(script_path: str, cfg_json_uri: str):
    script_str = PYTORCH_ESTIMATOR_SCRIPT_TEMPLATE.format(
        cfg_json_uri=cfg_json_uri, rv_cmd='train')
    log.debug(script_path)
    log.debug(script_str)
    str_to_file(script_str, script_path)
    return script_path


def _tar_script(script_path: str, tar_dir: str):
    tar_path = join(tar_dir, PYTORCH_ESTIMATOR_TAR_FILENAME)
    with tarfile.open(tar_path, 'w:gz') as tar:
        tar.add(script_path, arcname=basename(script_path))
    return tar_path
