# `test.py`
The `test.py` script provides some useful commands for running the examples and checking their outputs.

It supports the following commands:
- `run` - run an example either remotely or locally
- `compare` - compare the outputs of the `eval` stage of two runs of the same example
- `collect` - dowload the output of one or more commands (like `train`, `eval`, `bundle`, etc.) produced by a run of an example
- `predict` - download the model bundle produced by a run of an example and use it to make predictions on a sample image
- `upload` - upload model bundle, eval, sample image, sample predictions, and training logs to the model zoo

Details such as the URI's of inputs and outputs to the examples are hard-coded into the script, but can be overridden by supplying the `-o` option to any of the commands (only works when running one example at a time currently). For example: 
```bash
python \
"rastervision_pytorch_backend/rastervision/pytorch_backend/examples/test.py" \
run "spacenet-rio-cc" \
-o "remote.processed_uri" "s3://raster-vision-ahassan/rv/0.13/examples/processed/spacenet/rio" \
-o "remote.root_uri" "s3://raster-vision-ahassan/rv/0.13/examples/output/cc/spacenet-rio/" \
```

## Usage

### Run
A specific example:
```bash
python "rastervision_pytorch_backend/rastervision/pytorch_backend/examples/test.py" \
run "spacenet-rio-cc" \
--remote
```
All examples:
```bash
python "rastervision_pytorch_backend/rastervision/pytorch_backend/examples/test.py" \
run --remote
```

### Compare runs
This currently only compares the respective `eval.json`'s, but can be extended in the future to compare other properties/outputs of the runs too.
```bash
python "rastervision_pytorch_backend/rastervision/pytorch_backend/examples/test.py" \
compare \
"s3://raster-vision-lf-dev/examples/0.12/spacenet-rio-cc/output_6_27c/" \
"s3://raster-vision/examples/0.13/output/spacenet-rio-cc/"
```

### Collect
A specific example:
```bash
python "rastervision_pytorch_backend/rastervision/pytorch_backend/examples/test.py" \
collect "spacenet-rio-cc" \
--remote
```
All examples:
```bash
python "rastervision_pytorch_backend/rastervision/pytorch_backend/examples/test.py" \
collect --remote
```

### Predict
A specific example:
```bash
python "rastervision_pytorch_backend/rastervision/pytorch_backend/examples/test.py" \
predict "spacenet-rio-cc" \
--remote
```
All examples:
```bash
python "rastervision_pytorch_backend/rastervision/pytorch_backend/examples/test.py" \
predict --remote
```

### Upload
A specific example:
```bash
python "rastervision_pytorch_backend/rastervision/pytorch_backend/examples/test.py" \
upload "spacenet-rio-cc"
```
All examples:
```bash
python "rastervision_pytorch_backend/rastervision/pytorch_backend/examples/test.py" \
upload
```
