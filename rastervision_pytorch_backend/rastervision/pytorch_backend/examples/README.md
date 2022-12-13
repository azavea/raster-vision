# `test.py`
The `test.py` script provides some useful commands for running the examples and checking their outputs.

It supports the following commands:
- `run` - run an example either remotely or locally
- `compare` - compare the outputs of the `eval` stage of two runs of the same example
- `collect` - dowload the output of one or more commands (like `train`, `eval`, `bundle`, etc.) produced by a run of an example
- `predict` - download the model bundle produced by a run of an example and use it to make predictions on a sample image
- `upload` - upload model bundle, eval, sample image, sample predictions, and training logs to the model zoo

Details such as the URI's of inputs and outputs to the examples are hard-coded into the script, but can be overridden by supplying the `-o` option to any of the commands (only works when running one example at a time currently). For example:
```sh
python \
"rastervision_pytorch_backend/rastervision/pytorch_backend/examples/test.py" \
run "spacenet-rio-cc" \
-o "remote.processed_uri" "s3://raster-vision-ahassan/rv/0.20/examples/processed/spacenet/rio" \
-o "remote.root_uri" "s3://raster-vision-ahassan/rv/0.20/examples/output/cc/spacenet-rio/"
```

## Usage

### Preprocessing

Some examples require preprocessing data. Remember to upload that first. If unchanged from previous version, just copy it over:

```sh
aws s3 cp \
s3://raster-vision/examples/0.13/processed-data/ \
s3://raster-vision/examples/0.20/processed-data/ \
--recursive
```

### Run

**A specific example**:
```sh
python "rastervision_pytorch_backend/rastervision/pytorch_backend/examples/test.py" \
run "spacenet-rio-cc" \
--remote
```

**All examples**:
```sh
python "rastervision_pytorch_backend/rastervision/pytorch_backend/examples/test.py" \
run --remote
```

### Compare runs
This currently only compares the respective `eval.json`'s, but can be extended in the future to compare other properties/outputs of the runs too.

**A specific example**:
```sh
python "rastervision_pytorch_backend/rastervision/pytorch_backend/examples/test.py" \
compare \
--root_uri_old "s3://raster-vision/examples/0.13/output/spacenet-rio-cc/" \
--root_uri_new "s3://raster-vision/examples/0.20/output/spacenet-rio-cc/"
```

**All examples**:
```sh
python "rastervision_pytorch_backend/rastervision/pytorch_backend/examples/test.py" \
compare \
--examples_root_old "s3://raster-vision/examples/0.13/output/" \
--examples_root_new "s3://raster-vision/examples/0.20/output/"
```

### Collect

**A specific example**:
```sh
python "rastervision_pytorch_backend/rastervision/pytorch_backend/examples/test.py" \
collect "spacenet-rio-cc" \
--remote
```

**All examples**:
```sh
python "rastervision_pytorch_backend/rastervision/pytorch_backend/examples/test.py" \
collect --remote
```

### Predict

**A specific example**:
```sh
python "rastervision_pytorch_backend/rastervision/pytorch_backend/examples/test.py" \
predict "spacenet-rio-cc" \
--remote
```

**All examples**:
```sh
python "rastervision_pytorch_backend/rastervision/pytorch_backend/examples/test.py" \
predict \
--remote
```

### Upload

**A specific example**:
```sh
python "rastervision_pytorch_backend/rastervision/pytorch_backend/examples/test.py" \
upload "spacenet-rio-cc"
```

**All examples**:
```sh
python "rastervision_pytorch_backend/rastervision/pytorch_backend/examples/test.py" \
upload
```
