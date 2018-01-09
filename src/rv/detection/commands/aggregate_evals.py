import json

import click

from rv.utils.files import make_dir


def _aggregate_evals(eval_paths, output_path):
    label_to_avgs = {}
    nb_projects = len(eval_paths)
    for eval_path in eval_paths:
        with open(eval_path, 'r') as eval_file:
            project_eval = json.load(eval_file)
            for label_eval in project_eval:
                label_name = label_eval['name']
                label_avgs = label_to_avgs.get(label_name, {})
                for key, val in label_eval.items():
                    if key == 'name':
                        label_avgs['name'] = label_name
                    else:
                        label_avgs[key] = \
                            label_avgs.get(key, 0) + (val / nb_projects)
                label_to_avgs[label_name] = label_avgs

    make_dir(output_path, use_dirname=True)
    with open(output_path, 'w') as output_file:
        json.dump(list(label_to_avgs.values()), output_file, indent=4)


@click.command()
@click.argument('eval_paths', nargs=-1)
@click.argument('output_path')
def aggregate_evals(eval_paths, output_path):
    """Aggregate a set of evals (one per project) into a single eval.
    """
    _aggregate_evals(eval_paths, output_path)


if __name__ == '__main__':
    aggregate_evals()
