from rv.utils.batch import _batch_submit


def prep_train_data():
    branch_name = 'lf/30cm'
    attempts = 1
    cpu = True

    splits = [
        '5cm-train', '5cm-test', '30cm-train', '30cm-test',
        '5cm-multi-train']

    for split in splits:
        command = """
            python -m rv.detection.run prep_train_data \
                --debug --single-label car --no-partial \
                s3://raster-vision-lf-dev/detection/configs/projects/cowc-potsdam/remote/{split}.json \
                s3://raster-vision-lf-dev/detection/training-data/cowc-potsdam/{split}.zip \
                s3://raster-vision-lf-dev/detection/configs/label-maps/cowc.pbtxt
            """.format(split=split)
        # print(command)
        _batch_submit(branch_name, command, attempts=attempts, cpu=cpu)


if __name__ == '__main__':
    prep_train_data()
