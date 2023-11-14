from typing import TYPE_CHECKING, List, Optional, Union
import logging
from pprint import pprint

import boto3
from rastervision.pipeline import rv_config_ as rv_config
from rastervision.pipeline.runner import Runner

if TYPE_CHECKING:
    from rastervision.pipeline.pipeline import Pipeline
    from sagemaker.workflow.pipeline_context import _JobStepArguments
    from sagemaker import Session
    from sagemaker.workflow.pipeline import Pipeline as SageMakerPipeline
    from sagemaker.workflow.pipeline_context import PipelineSession
    from sagemaker.workflow.steps import ProcessingStep, TrainingStep

log = logging.getLogger(__name__)

AWS_SAGEMAKER = 'sagemaker'


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
        use_spot_instances=
    """

    def run(self,
            cfg_json_uri: str,
            pipeline: 'Pipeline',
            commands: List[str],
            num_splits: int = 1,
            cmd_prefix: List[str] = [
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
                       commands: List[str],
                       num_splits: int = 1,
                       cmd_prefix: List[str] = [
                           'python', '-m', 'rastervision.pipeline.cli'
                       ],
                       pipeline_run_name: str = 'rv') -> 'SageMakerPipeline':
        """Build a SageMaker Pipeline with each command as a step within it."""
        from sagemaker.workflow.pipeline_context import PipelineSession
        from sagemaker.workflow.pipeline import Pipeline as SageMakerPipeline

        config = rv_config.get_namespace_config(AWS_SAGEMAKER)
        role = config('role')
        cpu_image = config('cpu_image')
        cpu_instance_type = config('cpu_instance_type')
        gpu_image = config('gpu_image')
        gpu_instance_type = config('gpu_instance_type')
        use_spot_instances = config('use_spot_instances').lower() == 'yes'
        sagemaker_session = PipelineSession()

        steps = []

        for command in commands:
            use_gpu = command in pipeline.gpu_commands
            job_name = f'{pipeline_run_name}-{command}'

            cmd = cmd_prefix[:]

            if rv_config.get_verbosity() > 1:
                num_vs = rv_config.get_verbosity() - 1
                # produces a string like "-vvv..."
                verbosity_opt_str = f'-{"v" * num_vs}'
                cmd += [verbosity_opt_str]

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
                    step = self.build_step(
                        step_name=f'{job_name}_{i+1}of{num_splits}',
                        cmd=cmd,
                        role=role,
                        image_uri=gpu_image if use_gpu else cpu_image,
                        instance_type=(gpu_instance_type
                                       if use_gpu else cpu_instance_type),
                        use_spot_instances=use_spot_instances,
                        sagemaker_session=sagemaker_session,
                        use_gpu=use_gpu)
                    step.add_depends_on(steps)
                    _steps.append(step)
                steps.extend(_steps)
            else:
                # If the step can not be split, then submit it as-is.
                step = self.build_step(
                    step_name=job_name,
                    cmd=cmd,
                    role=role,
                    image_uri=gpu_image if use_gpu else cpu_image,
                    instance_type=(gpu_instance_type
                                   if use_gpu else cpu_instance_type),
                    use_spot_instances=use_spot_instances,
                    sagemaker_session=sagemaker_session,
                    use_gpu=use_gpu)
                step.add_depends_on(steps)
                steps.append(step)

        # Submit the pipeline to SageMaker
        sagemaker_pipeline = SageMakerPipeline(
            name=pipeline_run_name,
            steps=steps,
            sagemaker_session=sagemaker_session,
        )
        return sagemaker_pipeline

    def build_step(self, step_name: str, cmd: List[str], role: str,
                   image_uri: str, instance_type: str,
                   use_spot_instances: bool,
                   sagemaker_session: 'PipelineSession',
                   use_gpu: bool) -> Union['TrainingStep', 'ProcessingStep']:
        """Build an TrainingStep if use_gpu=True, otherwise a ProcessingStep.
        """
        if use_gpu:
            # For GPU-enabled steps, create an "Estimator".
            # Formally this should probably not be used for prediction in
            # this way, but it is expedient (especially given default
            # service quotas, and other stuff).
            from sagemaker.estimator import Estimator
            from sagemaker.workflow.steps import TrainingStep
            step_estimator = Estimator(
                container_entry_point=cmd,
                image_uri=image_uri,
                instance_count=1,
                instance_type=instance_type,
                max_retry_attempts=1,
                role=role,
                sagemaker_session=sagemaker_session,
                use_spot=use_spot_instances,
            )
            step_args: Optional['_JobStepArguments'] = step_estimator.fit(
                wait=False)
            step = TrainingStep(step_name, step_args=step_args)
        else:
            # For non-GPU-enabled steps, create a ScriptProcessor.
            from sagemaker.processing import Processor
            from sagemaker.workflow.steps import ProcessingStep
            step_processor = Processor(
                role=role,
                image_uri=image_uri,
                instance_count=1,
                instance_type=instance_type,
                sagemaker_session=sagemaker_session,
                entrypoint=cmd,
            )
            step_args: Optional['_JobStepArguments'] = step_processor.run(
                wait=False)
            step = ProcessingStep(step_name, step_args=step_args)

        return step

    def run_command(self,
                    cmd: List[str],
                    use_gpu: bool = False,
                    image_uri: Optional[str] = None,
                    instance_type: Optional[str] = None,
                    role: Optional[str] = None,
                    job_name: Optional[str] = None,
                    sagemaker_session: Optional['Session'] = None) -> None:
        """Run a single command as a SageMaker processing job.

        Args:
            cmd (List[str]): The command to run.
            use_gpu (bool): Use the GPU instance type and image from the
                Everett config. This is ignored if image_uri and instance_type
                are provided. Defaults to False.
            image_uri (Optional[str]): URI of docker image to use. If not
                provided, will be picked up from Everett config.
                Defaults to None.
            instance_type (Optional[str]): AWS instance type to use. If not
                provided, will be picked up from Everett config.
                Defaults to None.
            role (Optional[str]): AWS IAM role with SageMaker permissions. If
                not provided, will be picked up from Everett config.
                Defaults to None.
            job_name (Optional[str]): Optional job name. Defaults to None.
            sagemaker_session (Optional[Session]): SageMaker session.
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
