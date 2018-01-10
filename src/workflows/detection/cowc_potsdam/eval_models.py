from rv.utils.batch import _batch_submit


def eval_models():
    branch_name = 'lf/30cm'
    attempts = 1
    cpu = True

    experiments = [
        {
            'model': '5cm',
            'test': '5cm-test'
        },
        {
            'model': '30cm',
            'test': '30cm-test'
        },
        {
            'model': '5cm-multi',
            'test': '5cm-test'
        },
    ]

    for experiment in experiments:
        command = """
            python -m rv.detection.run eval_model \
                --merge-thresh 0.5 \
                --channel-order 0 1 2 \
                --evals-uri s3://raster-vision-lf-dev/detection/evals/cowc-potsdam/{model}/ \
                --predictions-uri s3://raster-vision-lf-dev/detection/predictions/cowc-potsdam/{model}/ \
                s3://raster-vision-lf-dev/detection/trained-models/cowc-potsdam/{model}/inference-graph.pb \
                s3://raster-vision-lf-dev/detection/configs/projects/cowc-potsdam/remote/{test}.json \
                s3://raster-vision-lf-dev/detection/configs/label-maps/cowc.pbtxt \
                s3://raster-vision-lf-dev/detection/evals/cowc-potsdam/{model}/avg.json
            """.format(**experiment)
        _batch_submit(branch_name, command, attempts=attempts, cpu=cpu)


if __name__ == '__main__':
    eval_models()
