from typing import Any, Dict, List, Optional, Union
from genericpath import exists
from pprint import pformat
import subprocess
from os.path import (basename, isdir, isfile, join, relpath, split)
from tempfile import TemporaryDirectory

import click

from rastervision.pipeline.file_system import (
    file_to_json, sync_from_dir, upload_or_copy, download_or_copy, file_exists,
    sync_to_dir, NotReadableError, download_if_needed)

OLD_VERSION = '0.20'
NEW_VERSION = '0.20.1'
NEW_VERSION_MAJOR_MINOR = '0.20'

EXAMPLES_MODULE_ROOT = 'rastervision.pytorch_backend.examples'
EXAMPLES_PATH_ROOT = '/opt/src/rastervision_pytorch_backend/rastervision/pytorch_backend/examples'  # noqa
REMOTE_PROCESSED_ROOT = f's3://raster-vision/examples/{NEW_VERSION}/processed-data'
REMOTE_OUTPUT_ROOT = f's3://raster-vision/examples/{NEW_VERSION}/output'
LOCAL_RAW_ROOT = '/opt/data/raw-data'
LOCAL_PROCESSED_ROOT = '/opt/data/examples/processed-data'
LOCAL_OUTPUT_ROOT = '/opt/data/examples/output'
LOCAL_COLLECT_ROOT = '/opt/data/examples/collect'
ZOO_UPLOAD_ROOT = f's3://azavea-research-public-data/raster-vision/examples/model-zoo-{NEW_VERSION_MAJOR_MINOR}'  # noqa
SAMPLE_IMG_DIR = f's3://azavea-research-public-data/raster-vision/examples/sample_images'  # noqa

######################
# Default configuration for the examples.
# Each key is the name of the example.
######################
cfg = [
    {
        'key': 'spacenet-rio-cc',
        'task': 'cc',
        'pred_ext': '.json',
        'module': f'{EXAMPLES_MODULE_ROOT}.chip_classification.spacenet_rio',
        'local': {
            'raw_uri': f'{LOCAL_RAW_ROOT}/spacenet-dataset',
            'processed_uri': f'{LOCAL_PROCESSED_ROOT}/spacenet-rio-cc',
            'root_uri': f'{LOCAL_OUTPUT_ROOT}/spacenet-rio-cc'
        },
        'remote': {
            'raw_uri': 's3://spacenet-dataset/',
            'processed_uri': f'{REMOTE_PROCESSED_ROOT}/spacenet-rio-cc',
            'root_uri': f'{REMOTE_OUTPUT_ROOT}/spacenet-rio-cc'
        },
    },
    {
        'key': 'isprs-potsdam-ss',
        'task': 'ss',
        'pred_ext': '',
        'module':
        f'{EXAMPLES_MODULE_ROOT}.semantic_segmentation.isprs_potsdam',
        'local': {
            'raw_uri': f'{LOCAL_RAW_ROOT}/isprs-potsdam/',
            'processed_uri': f'{LOCAL_PROCESSED_ROOT}/isprs-potsdam-ss',
            'root_uri': f'{LOCAL_OUTPUT_ROOT}/isprs-potsdam-ss/'
        },
        'remote': {
            'raw_uri': 's3://raster-vision-raw-data/isprs-potsdam',
            'processed_uri': f'{REMOTE_PROCESSED_ROOT}/isprs-potsdam-ss',
            'root_uri': f'{REMOTE_OUTPUT_ROOT}/isprs-potsdam-ss'
        },
    },
    {
        'key': 'spacenet-vegas-buildings-ss',
        'task': 'ss',
        'pred_ext': '',
        'module':
        f'{EXAMPLES_MODULE_ROOT}.semantic_segmentation.spacenet_vegas',
        'local': {
            'raw_uri': 's3://spacenet-dataset/',
            'root_uri': f'{LOCAL_OUTPUT_ROOT}/spacenet-vegas-buildings-ss'
        },
        'remote': {
            'raw_uri': 's3://spacenet-dataset/',
            'root_uri': f'{REMOTE_OUTPUT_ROOT}/spacenet-vegas-buildings-ss'
        },
        'extra_args': [['target', 'buildings']],
    },
    {
        'key': 'spacenet-vegas-roads-ss',
        'task': 'ss',
        'pred_ext': '',
        'module':
        f'{EXAMPLES_MODULE_ROOT}.semantic_segmentation.spacenet_vegas',
        'local': {
            'raw_uri': 's3://spacenet-dataset/',
            'root_uri': f'{LOCAL_OUTPUT_ROOT}/spacenet-vegas-roads-ss'
        },
        'remote': {
            'raw_uri': 's3://spacenet-dataset/',
            'root_uri': f'{REMOTE_OUTPUT_ROOT}/spacenet-vegas-roads-ss'
        },
        'extra_args': [['target', 'roads']],
    },
    {
        'key': 'cowc-potsdam-od',
        'task': 'od',
        'pred_ext': '.json',
        'module': f'{EXAMPLES_MODULE_ROOT}.object_detection.cowc_potsdam',
        'local': {
            'raw_uri': f'{LOCAL_RAW_ROOT}/isprs-potsdam',
            'processed_uri': f'{LOCAL_PROCESSED_ROOT}/cowc-potsdam-od',
            'root_uri': f'{LOCAL_OUTPUT_ROOT}/cowc-potsdam-od'
        },
        'remote': {
            'raw_uri': 's3://raster-vision-raw-data/isprs-potsdam',
            'processed_uri': f'{REMOTE_PROCESSED_ROOT}/cowc-potsdam-od',
            'root_uri': f'{REMOTE_OUTPUT_ROOT}/cowc-potsdam-od'
        },
    },
    {
        'key': 'xview-od',
        'task': 'od',
        'pred_ext': '.json',
        'module': f'{EXAMPLES_MODULE_ROOT}.object_detection.xview',
        'local': {
            'raw_uri': 's3://raster-vision-xview-example/raw-data',
            'processed_uri': f'{LOCAL_PROCESSED_ROOT}/xview-od',
            'root_uri': f'{LOCAL_OUTPUT_ROOT}/xview-od'
        },
        'remote': {
            'raw_uri': 's3://raster-vision-xview-example/raw-data',
            'processed_uri': f'{REMOTE_PROCESSED_ROOT}/xview-od',
            'root_uri': f'{REMOTE_OUTPUT_ROOT}/xview-od'
        },
    },
]


######################
# commands
######################
@click.group()
def test():
    pass


# --------------------
# run
# --------------------
@test.command()
@click.argument('keys', nargs=-1)
@click.option('--test', is_flag=True, help='Do short test run')
@click.option('--remote', is_flag=True, default=False)
@click.option(
    '--commands',
    help='Space-separated string with RV commansd to run.',
    default=None)
@click.option(
    '--overrides',
    '-o',
    type=(str, str),
    multiple=True,
    metavar='KEY VALUE',
    default=[],
    help='Override experiment config.')
def run(keys=[], test=False, remote=False, commands=None, overrides=[]):
    """Run RV on a set of examples.

    Args:
        keys: the names of the examples.
    """
    overrides = dict(overrides)

    run_all = len(keys) == 0
    validate_keys(keys)

    if commands is not None:
        commands = commands.split(' ')
    for exp_cfg in cfg:
        if run_all or exp_cfg['key'] in keys:
            if len(keys) == 1:
                override_cfg(exp_cfg, overrides)
            _run(exp_cfg, test=test, remote=remote, commands=commands)


# --------------------
# collect
# --------------------
@test.command()
@click.argument('keys', nargs=-1)
@click.option('--collect_dir', default=LOCAL_COLLECT_ROOT)
@click.option('--remote', is_flag=True)
@click.option(
    '--paths',
    '-p',
    help='Space-separated string with URIs to files or dirs to collect.',
    default=None)
@click.option(
    '--overrides',
    '-o',
    type=(str, str),
    multiple=True,
    metavar='KEY VALUE',
    default=[],
    help='Override experiment config.')
def collect(keys, collect_dir, remote, paths, overrides=[]):
    """Download outputs of paths for each example.

    By default, only downloads eval and bundle.
    """
    overrides = dict(overrides)
    if paths is None:
        paths = [
            'train/model-bundle.zip',
            'train/last-model.pth',
            'eval',
            'bundle',
        ]
    else:
        paths = paths.split(' ')

    run_all = len(keys) == 0
    validate_keys(keys)

    dirs = {}
    for exp_cfg in cfg:
        key = exp_cfg['key']
        if run_all or key in keys:
            if len(keys) == 1:
                override_cfg(exp_cfg, overrides)
            dirs[key] = {}
            uris = exp_cfg['remote'] if remote else exp_cfg['local']
            root_uri = uris['root_uri']
            for path in paths:
                src_uri = join(root_uri, path)
                console_info(f'{key}: Fetching {path}')
                if file_exists(src_uri, include_dir=False):
                    # is a single file
                    dst_dir = join(collect_dir, key, split(path)[0])
                    dst_dir = split(to_local_uri(src_uri, dst_dir))[0]
                    download_or_copy(src_uri, dst_dir)
                elif file_exists(src_uri, include_dir=True):
                    # is a directory
                    dst_dir = join(collect_dir, key, path)
                    sync_from_dir(src_uri, dst_dir)
                else:
                    # does not exist
                    console_failure(f'File or dir not found: {src_uri}.')
                dirs[key][path] = dst_dir
    console_info(pformat(dirs))


# --------------------
# predict
# --------------------
@test.command()
@click.argument('keys', nargs=-1)
@click.option('--collect_dir', default=LOCAL_COLLECT_ROOT)
@click.option('--remote', is_flag=True)
@click.option(
    '--overrides',
    '-o',
    type=(str, str),
    multiple=True,
    metavar='KEY VALUE',
    default=[],
    help='Override experiment config.')
def predict(keys, collect_dir, remote, overrides=[]):
    """Test model bundles using predict command on output of collect command.
    """
    overrides = dict(overrides)

    run_all = len(keys) == 0
    validate_keys(keys)

    for exp_cfg in cfg:
        key = exp_cfg['key']
        if run_all or key in keys:
            if len(keys) == 1:
                override_cfg(exp_cfg, overrides)
            uris = exp_cfg['remote'] if remote else exp_cfg['local']
            root_uri = uris['root_uri']
            _collect_dir = join(collect_dir, key)
            fetch_cmd_dir(root_uri, 'bundle', _collect_dir)
            _predict(exp_cfg, _collect_dir)


# --------------------
# compare
# --------------------
@test.command()
@click.option('--root_uri_old', default=None)
@click.option('--root_uri_new', default=None)
@click.option('--examples_root_old', default=None)
@click.option('--examples_root_new', default=None)
@click.option('--download_dir', '-d', default=LOCAL_COLLECT_ROOT)
def compare(root_uri_old: Optional[str],
            root_uri_new: Optional[str],
            examples_root_old: Optional[str] = None,
            examples_root_new: Optional[str] = None,
            download_dir: Optional[str] = LOCAL_COLLECT_ROOT) -> None:
    """Compare different runs of the same example."""
    if root_uri_old is None and root_uri_new is None:
        assert examples_root_old is not None and examples_root_new is not None
        for exp_cfg in cfg:
            key = exp_cfg['key']
            root_uri_old = join(examples_root_old, key)
            root_uri_new = join(examples_root_new, key)
            console_info(f'Comparing\n- {root_uri_old}\n- {root_uri_new}')
            _compare(root_uri_old, root_uri_new, download_dir)
        return
    return _compare(root_uri_old, root_uri_new, download_dir)


def _compare(root_uri_old: Optional[str],
             root_uri_new: Optional[str],
             download_dir: Optional[str] = None) -> None:
    """Compare different runs of the same example."""
    if root_uri_old != '/':
        root_uri_old = root_uri_old.rstrip('/')
    if root_uri_new != '/':
        root_uri_new = root_uri_new.rstrip('/')
    if download_dir is not None:
        return _compare_runs(root_uri_old, root_uri_new, download_dir)

    with TemporaryDirectory(dir='/opt/data/tmp') as tmp_dir:
        return _compare_runs(root_uri_old, root_uri_new, tmp_dir)


# --------------------
# upload to model zoo
# --------------------
@test.command()
@click.argument('keys', nargs=-1)
@click.option('--collect_dir', default=LOCAL_COLLECT_ROOT)
@click.option('--upload_dir', default=ZOO_UPLOAD_ROOT)
@click.option(
    '--overrides',
    '-o',
    type=(str, str),
    multiple=True,
    metavar='KEY VALUE',
    default=[],
    help='Override experiment config.')
def upload(keys, collect_dir, upload_dir, overrides=[]):
    """Upload eval, bundle, and sample predictions to the target dir."""
    overrides = dict(overrides)

    run_all = len(keys) == 0
    validate_keys(keys)

    for exp_cfg in cfg:
        key = exp_cfg['key']
        if run_all or key in keys:
            if len(keys) == 1:
                override_cfg(exp_cfg, overrides)
            _collect_dir = join(collect_dir, key)
            _upload_dir = join(upload_dir, key)
            _upload_to_zoo(exp_cfg, _collect_dir, _upload_dir)


######################
# utils
######################
def _run(exp_cfg: dict,
         test: bool = False,
         remote: bool = False,
         commands: List[str] = None) -> None:
    """Builds a command from the params in exp_cfg and other arguments and
    then executes it.
    """
    uris = exp_cfg['remote'] if remote else exp_cfg['local']
    cmd = ['rastervision']
    rv_profile = exp_cfg.get('rv_profile')
    if rv_profile is not None:
        cmd += ['-p', rv_profile]
    cmd += ['run', 'batch' if remote else 'inprocess', exp_cfg['module']]
    if commands is not None:
        cmd += commands
    cmd += ['-a', 'raw_uri', uris['raw_uri']]
    if 'processed_uri' in uris:
        cmd += ['-a', 'processed_uri', uris['processed_uri']]
    cmd += ['-a', 'root_uri', uris['root_uri']]
    cmd += ['-a', 'test', 'True' if test else 'False']
    extra_args = exp_cfg.get('extra_args')
    if extra_args:
        for k, v in extra_args:
            cmd += ['-a', str(k), str(v)]
    if remote:
        cmd += ['--splits', '3']
    cmd += ['--pipeline-run-name', exp_cfg['key']]
    run_command(cmd)


def _predict(exp_cfg: dict, collect_dir: str) -> None:
    """Download sample image and make predictions on it using the model bundle.
    """
    key = exp_cfg['key']
    console_heading(f'Testing model bundle for {key}...')

    model_bundle_uri = join(collect_dir, 'bundle', 'model-bundle.zip')
    if not exists(model_bundle_uri):
        console_failure(
            f'Bundle does not exist: {model_bundle_uri}', bold=True)
        exit(1)

    pred_dir = join(collect_dir, 'sample-predictions')
    sample_filename = f'sample-img-{key}.tif'
    sample_uri_src = join(SAMPLE_IMG_DIR, sample_filename)
    download_or_copy(sample_uri_src, pred_dir, delete_tmp=True)
    sample_uri_dst = join(pred_dir, sample_filename)

    pred_ext = exp_cfg['pred_ext']
    out_uri = join(pred_dir, f'sample-pred-{key}{pred_ext}')
    cmd = [
        'rastervision', 'predict', model_bundle_uri, sample_uri_dst, out_uri
    ]
    run_command(cmd)


def _upload_to_zoo(exp_cfg: dict, collect_dir: str, upload_dir: str) -> None:
    src_uris = {}
    dst_uris = {}

    src_uris['eval'] = join(collect_dir, 'eval', 'validation_scenes',
                            'eval.json')
    src_uris['bundle'] = join(collect_dir, 'bundle', 'model-bundle.zip')
    src_uris['sample_predictions'] = join(collect_dir, 'sample-predictions')
    src_uris['learner_bundle'] = join(collect_dir, 'train', 'model-bundle.zip')
    src_uris['learner_model'] = join(collect_dir, 'train', 'last-model.pth')

    dst_uris['eval'] = join(upload_dir, 'validation_scenes', 'eval.json')
    dst_uris['bundle'] = join(upload_dir, 'model-bundle.zip')
    dst_uris['sample_predictions'] = join(upload_dir, 'sample-predictions')
    dst_uris['learner_bundle'] = join(upload_dir, 'train', 'model-bundle.zip')
    dst_uris['learner_model'] = join(upload_dir, 'model.pth')

    assert len(src_uris) == len(dst_uris)

    for k, src in src_uris.items():
        dst = dst_uris[k]
        if not exists(src):
            console_failure(f'{k}: {src} not found.')
            exit(1)
        if isfile(src):
            console_info(f'Uploading {k} file: {src} to {dst}.')
            upload_or_copy(src, dst)
        elif isdir(src):
            console_info(f'Syncing {k} dir: {src} to {dst}.')
            sync_to_dir(src, dst)
        else:
            raise ValueError(src)


def _compare_runs(root_uri_old: str,
                  root_uri_new: str,
                  download_dir: Optional[str],
                  commands=['eval']) -> None:
    """Compare outputs of commands for two runs of an example.
    Currently only supports eval, but can be extended to include others.
    """
    for cmd in commands:
        key_old = basename(root_uri_old)
        key_new = basename(root_uri_new)
        cmd_root_uri_old_local = fetch_cmd_dir(
            root_uri_old, cmd, join(download_dir, 'old', key_old))
        cmd_root_uri_new_local = fetch_cmd_dir(root_uri_new, cmd,
                                               join(download_dir, key_new))
        if cmd == 'eval':
            _compare_evals(cmd_root_uri_old_local, cmd_root_uri_new_local)


def _compare_evals(root_uri_old: str,
                   root_uri_new: str,
                   float_tol: float = 1e-3,
                   exclude_keys: list = ['conf_mat', 'count',
                                         'per_scene']) -> None:
    """Compare outputs of the eval command for two runs of an example."""
    console_heading('Comparing keys and values in eval.json files...')
    try:
        eval_json_old = join(root_uri_old, 'validation_scenes', 'eval.json')
        eval_old = file_to_json(download_if_needed(eval_json_old))
    except NotReadableError:
        eval_json_old = join(root_uri_old, 'eval.json')
        eval_old = file_to_json(download_if_needed(eval_json_old))
    eval_json_new = join(root_uri_new, 'validation_scenes', 'eval.json')
    eval_new = file_to_json(download_if_needed(eval_json_new))
    _compare_dicts(
        eval_old, eval_new, float_tol=float_tol, exclude_keys=exclude_keys)


def validate_keys(keys: List[str]) -> None:
    exp_keys = [exp_cfg['key'] for exp_cfg in cfg]
    invalid_keys = set(keys).difference(exp_keys)
    if invalid_keys:
        raise ValueError('{} are invalid keys'.format(', '.join(invalid_keys)))


def run_command(cmd: str) -> None:
    """Run a command in a sub-process."""
    cmd_str = ' '.join(cmd)
    console_info(f'Running command:\n{cmd_str}')
    proc = subprocess.run(cmd)
    if proc.returncode != 0:
        console_failure(
            f'Error: process returned {proc.returncode}', bold=True)
        exit()


def override_cfg(cfg: dict, overrides: dict, sep='.') -> None:
    """Recursively update values in cfg with corresponding values in
    overrides. overrides is expected to be a flattened dict.
    """
    for key_path, v in overrides.items():
        _cfg = cfg
        key_crumbs = key_path.split(sep)
        for _k in key_crumbs[:-1]:
            _cfg = _cfg[_k]
        _cfg[key_crumbs[-1]] = v


def to_local_uri(uri: str, local_root: str, full_path: bool = False) -> str:
    """Convert an S3 URI to a suitable local URI. Do nothing if the URI is
    already local.
    """
    if uri.startswith('s3://'):
        if full_path:
            uri = join(local_root, relpath(uri, 's3://'))
        else:
            uri = join(local_root, basename(uri))
    return uri


def fetch_cmd_dir(root_uri: str, cmd: str, download_dir: str) -> str:
    """Download the output directory of a particular RV command, located at
    root_uri/cmd (e.g. root_uri/eval) to the download_dir directory.
    """
    cmd_root_uri = join(root_uri, cmd)
    console_info(f'Fetching {cmd} directory: {cmd_root_uri}')
    cmd_root_uri_local = to_local_uri(
        cmd_root_uri, download_dir, full_path=False)
    sync_from_dir(cmd_root_uri, cmd_root_uri_local)
    return cmd_root_uri_local


def flatten_dict(d: Union[dict, list], sep: str = '.') -> dict:
    """Flatten a dict so that it does not have any nested dicts or lists.
    Nested keys will be concatenated using the separator, sep. For example,
    {'a': {'b': ['x', 10]}} becomes {'a.b.0': 'x', 'a.b.1': 10}.
    This makes it simpler to compare dicts.

    Args:
        d (Union[dict, list]): A dict or list.
        sep (str, optional): Separator to use for concatenating nested keys.
            Defaults to '.'.

    Returns:
        dict: The flattened dict.
    """
    if not isinstance(d, (dict, list)):
        return d

    flat_d = {}

    if isinstance(d, list):
        for i, v in enumerate(d):
            v = flatten_dict(v)
            if isinstance(v, dict):
                for _k, _v in v.items():
                    flat_d[f'{i}{sep}{_k}'] = _v
            else:
                flat_d[i] = v
    else:
        for k, v in d.items():
            v = flatten_dict(v)
            if isinstance(v, dict):
                for _k, _v in v.items():
                    flat_d[f'{k}{sep}{_k}'] = _v
            else:
                flat_d[k] = v
    return flat_d


def _compare_dicts(dict_old: dict,
                   dict_new: dict,
                   float_tol: float = 1e-3,
                   exclude_keys: list = []) -> None:
    """Compare the keys and values of the two dicts.

    Args:
        dict_old (dict): A dict.
        dict_new (dict): A dict.
        float_tol (float, optional): Count float values as different if the
            abs difference exceeds this threshold. Defaults to 1e-3.
        exclude_keys (list, optional): Ignore the following keys when
            comparing values. Defaults to [].
    """
    dict_old = flatten_dict(dict_old)
    dict_new: Dict[str, Any] = flatten_dict(dict_new)
    keys_old, keys_new = set(dict_old.keys()), set(dict_new.keys())
    diff1, diff2 = keys_new - keys_old, keys_old - keys_new
    if len(diff1) > 0:
        console_failure(f'Missing keys in old: {keys_new - keys_old}')
    if len(diff2) > 0:
        console_failure(f'Missing keys in new: {keys_old - keys_new}')
    if len(diff1) + len(diff2) == 0:
        console_success(f'All keys match')
    intersection = keys_old.intersection(keys_new)
    if len(intersection) == 0:
        console_failure('No matching keys found:')
    else:
        console_success(f'{len(intersection)} matching keys found')
    diff_count = 0
    console_info(f'Ignoring keys that contain {exclude_keys}')
    console_info(f'Float comparison tolerance: {float_tol}')
    for k in sorted(intersection):
        if any(_k in k for _k in exclude_keys):
            continue
        v_old, v_new = dict_old[k], dict_new[k]
        if isinstance(v_new, float) and isinstance(v_old, float):
            if v_new - v_old > float_tol:
                diff_count += 1
                _diff = v_new - v_old
                console_success(f'diff: {k}: '
                                f'{v_new:.6f} - {v_old:.6f}  = {_diff:.6f}')
            elif v_old - v_new > float_tol:
                diff_count += 1
                _diff = v_new - v_old
                console_failure(f'diff: {k}: '
                                f'{v_new:.6f} - {v_old:.6f}  = {_diff:.6f}')
        elif isinstance(v_new, int) and isinstance(v_old, int):
            if v_old != v_new:
                diff_count += 1
                _diff = v_new - v_old
                console_failure(f'diff: {k}: {v_new} - {v_old}  = {_diff}')
        else:
            if v_old != v_new:
                diff_count += 1
                console_failure(f'diff: {k}: {v_new} != {v_old}')
    if diff_count > 0:
        console_failure(f'Number of non-matching values: {diff_count}')
    else:
        console_success(f'All values within tolerance')


def console_info(msg: str, **kwargs) -> None:
    click.secho(msg, fg='yellow', **kwargs)


def console_heading(msg: str, **kwargs) -> None:
    click.secho(msg, fg='magenta', bold=True, **kwargs)


def console_warning(msg: str, **kwargs) -> None:
    click.secho(f'Warning: {msg}', fg='red', **kwargs)


def console_failure(msg: str, **kwargs) -> None:
    click.secho(msg, fg='red', err=True, **kwargs)


def console_success(msg: str, **kwargs) -> None:
    click.secho(msg, fg='cyan', **kwargs)


if __name__ == '__main__':
    test()
