## Pyhton API for experiment definition

### Experiment Builder

There will be fluent builders for the following configuration types:

-Task
-Backend
-Dataset
--Train,Val,Predict (no builders)
--Scene
---RasterSource
----RasterTransformer
---LabelSource
----LabelTransformer {Do this?}
---LabelStore
--Augmentation
-Analyzer
-Experiment

Commands:
-Analyze
--AnalyzeOptions
-Chip
--ChipOptions
-Train
--TrainOptions
-Predict
--PredictOptions
-Eval
--EvalOptions

Predicter

### Backend config flexibility

Backend configs are stored as a Struct in protobuf, from a template.
The user can specify json configuration that overrides configuration
in an easy way - raster vision will search the json representation of
the config for the key, and replace it with the user supplied value.
If there are multiple keys in the config, an error will be thrown.

### Command configs

Command configuration will also have fluent builders. In addition, command options will have builders.
Command options will be able to be set on an Experiment, so that the experiment generates command
configurations with those options set.

### Serialized Experiment as manifest

How do we account for experiments that were not run all at once?
Each command has it's serialized config in it's output.
Experiment config is never actually saved?
No, experiments are saved in another folder.

### Keys to use branch off workflows

-rv_root/
--analyze/default/
--chip/default/
--train/default/
--predict/default/
--eval/default/
--experiment/experiment-id.json

Keep keys concept.

## Label Store tweaks

We have the same type of labels for scenes, and the label store differs by task type.
This is currently tangled in the system.
We should have label stores be based on labels lazily and based on configuration.

There's a difference between labels, which can either be raster or vector,
and the training labels that are required for training.
Labels + Task = TrainigLabels

Saving predictions is almost an entirely different thing. Simply needs to write the raster
or feature data.

Create LabelSource and LabelStore to differentiate.
`prediction_label_store` would be of this type.
A good example of why this is needed is segmentation, where you might have a
geojson label source but have a raster prediction store.

Once we split label source from label store, we can think of modifications
to polygon-based geojson labels as "transformers", much like raster transformers.
So one places a classification transformer on polygon geojson for classificaiton,
you can place an bbox transformer on it for object detection, and a rasterize transformer
for segmentation. Like the 'stats' raster transformer, we can default this based on
task type in an experiment.

## Customizable backend defaults

Allow configuraiton-based defaults for backends based on model key.
Allow this configuration to be overridden by users, so that they can
define their own model defaults for easy reuse.

## Easy unittest-like API for running experiments

For this release, we will only allow the `python script.py` style of running. In the future we will have autosearch of codebases for listing and running eperiments.

### Running from JSON serialized commands

Commands are always run from JSON configuration. There will be configuration that will
allow plugins to be loaded from client code.

## Add Augmentation capability

An augmentor will work much like a RasterTransformer - it will manipulate the data being
generated from the RasterSource. The difference is, this will only happen for the training
data generation (`chip`), and can produce more data than the read data.

has to manipulate labelstore.

Pass the task to Dataset, it can tell if augmentor works for task.

## Plug architecture that allows users to define their own types

Includes:
-tasks
-backends
-augmentors
-transformers
-evaltuation metrics
-label sources
-label stores
-raster sources

## import rastervision as rv

Taking from the numpy and pandas playbook (as well as others), make most if not all API points accessable
through the root package. This will allow us to simply `import rastervision as rv` and call everying off
of `rv._`, e.g. `rv.ExperimentConfig`

## Allow running only commands that are necessary in a set of experiments
When re-running experiments that have results from previous runs, don't re-run if results are already there. The exception is for training, when there are checkpoint files, you might want to "--continue-train". Also allow for re-running with "clobber" to overwrite past results.

Label stores write out labels - how do we know the URIs for labelstore?
Command might not know all output uri, but experiments.

## Allow running from a repository, that holds plugins across experiments and allows repeatability

The ideal scenario for repeatability is that we have a GitHub repository for each ML project.
That project uses the raster vision docker as is  or as a base.
Each command run is tagged with the Experiment it came from, the git commit hash for the repo it's
contained in, and the raster vision docker commit hash for the version of raster vision you are running.
Allowing repositories to be the main place for running raster vision will also allow plugins to be
specified on a per-project bases.

The plugin repository might actually require  a separate repository,
as I see benefit to having a company-wide plugin repository.

## Run with other dependencies, without having to change out docker container

Allow hooks to run pip install on alternate dependenices, as well as running pip install on the target repository requirements.txt, so that other libraries could be installed. Perhaps a bootstrap.sh script can be used to run on every container run. This is a step below creating your own docker container based on RV and then deploying in Batch, etc.

# Analyzer

An Analyzer will take all the raster sources, download them, and run a processing step on them. You can have multiple analyzers for an experiment. Q: What happens when my raster data is too big for an Analyzer running on one machine? A: Use GeoTrellis for your preprossing.

## Renames

Classification -> Chip Classification
MLTask -> Task
MLBackend -> Backend
LabelStore -> (LableSource, LabelStore)

## Package reorg

To make the code easier to navigate, items in core are broken out to their related packages,
along with configuration related to the package. Because packages contain multiple types and
not just implementations of a single type, the convention is to name the packages
as singular words. Therefore `ml_tasks` becomes `task`, `ml_backends` becomes `backend`, etc.

`core` is still used to contain core types, like `Box` and `Config`, that are not concepts that
have their own packages.

## Dry run flag



## Unit Test reorg

I moved unit tests into a `tests` directory at the same level as the root `rastervision` package,
and renamed everyting to `test_*.py`. This is in line with what as far as I can tell is the standard,
allows for using defaults when running `unittest`, and makes excluding unit tests from distribution
packaging much easier.

To run unit tests, get into the docker container and run

```
python -m unittest discover tests
```

## Flexability in backend installations

Users should be able to pip install rastervision without needing any backend library installed.
The imports that happen for backends should be done on-demand as backends are used.
That way you could pip install rastervision locally and utilize other functionality besides
training or running a model, or do so only with the target backend installed.

## DOCS to be added

Concepts
-Dataset
--Train,Val,Predict
--Scene
--Difference between train, validation, test
---RasterSource
----Transform
---LabelSource
---LabelStore
---Augmentation
----Differentiate between RV Augmentation and Backend Augmentation
-Task
-Backend
--Configuring TF object detection:
---https://github.com/tensorflow/models/blob/eef6bb5bd3b3cd5fcf54306bf29750b7f9f9a5ea/research/object_detection/g3doc/configuring_jobs.md
---https://github.com/tensorflow/models/tree/eef6bb5bd3b3cd5fcf54306bf29750b7f9f9a5ea/research/object_detection/samples/configs
-Commands
--Command Options vs Command Configs
--Stats
--Chip
--Train
--Predict
--Eval

Patterns
-Config
Difference between a Config and the actual thing is that, Configs are informational only - they don't read anything, they won't try to access files etc.
--Builder
--Validate
--Proto
-Plugin

Code Layout
-Top level imports via api packages.

Running
-Local
-Remote

Extending
-Plugins

Tutorials
-COWC local
-COWC remote
-Spacenet
-Transfer learning
-Create your own backend (Plugin tutorial)

## Schedule

- Release Sept 18 [NOT GONNA HIT IT]
-- New date: October 15.
- Tutorials: Roll out over time, ending October 7 [NOPE]
-- New date: October 15

# Wishlist

Ensemble prediction
--Augmentation { NEED? }

### NOTES

Thinking through the stats problem - we have a transformer that is dependant on a file that is produced by
a command, that is not expected to exist yet, or should be user supplied. So we want to allow the user
to state the configuration witout having to have the file exist, or the file URI be stated, as it will
default to a path specified by the experiment. This points to that some configuration types will require
property configuration injected into them, and that URIs that don't exist can be satisfied by the
Experiment solver to structure commands such that by the time that URI is needed, it exists.
Another example of how this effects the configuration is, there are requirements for configuraiton
that only exist for certain commands.

What is the right design pattern to account for this? Currently we have configuration that requires all
parameters to be stated and validated up front.

Instead of a configuration validating it's own URIs, it states the URIs that are required,
and, if None, will need to be supplied by injection when needed.

So, StatsTransformerConfig states a URI need, which is passed up through RasterSource. This is where
it gets tricky - the need comes from something some commands don't need. So perhaps the commands
can state what they need? URIs are transmitted up no just by bulk but by dict:

```
{ 'RASTER_SOURCE': [uris],
  'RASTER_TRANFORMER': [uris] }
```

Since we need to plug in URIs, we'll need to key them in a way that allows the config
to later understand how to set the injected URIs into itself:

```
{ 'RASTER_SOURCE': { 'uris': ['s3://some/tif.tif'] },
  'RASTER_TRANSFORMER': [ { 'stats_uri': None } ] }
```

Since the transformers attached to a raster source are a list, the URI statements would
need to be a list. This structure could be gathered by the Experiment, URI's plugged in,
and sent back down to be set.

Then, commands can report what sort of dependencies they require. For instance,
the dependencies of ANALYZE would only state [RASTER_SOURCE], and CHIP would report
[RASTER_SOURCE, RASTER_TRANSFORMER]. So that something generating the command and validating
what URIs exist, and how commands need to be hooked up, can only check for the things a command
needs.

How would an experiment set the stats_uri dependency? The command can state the output, but
that needs to be tied back to the config. It would have to state the output in the
same way as the input states the dependency, so that it could be replaced. Another thing
to think about is, what about pluggable `analyze`? What if users want to create a custom
transformer that depends on a custom analysis step?

Starting from an experiment configuration. It generates commands, feeding in
relavent configuration (such as raster sources, backend config, etc). A command is given
it's working directory, and gives back the dependencies and the outputs. Those dependencies
are used to inject into the configuration, which is then used to generate the command
configuration.

This happens according to a partial sorting of commands based on stated dependencies.
So commands state what commands must be run before them.

```
ANALYZE_COMMAND -> []
CHIP_COMMAND -> [ANALYZE_COMMAND]
TRAIN_COMMAND -> [CHIP_COMMAND]
PREDICT_COMMAND -> [TRAIN_COMMAND]
EVAL_COMMAND -> [PREDICT_COMMAND]
```

For each command, given experiment config and a working directory.
That validates the configuration for that command, and returns
the command's inputs and outputs (D), as well as a list of mutations to the configuration X -> Y.
Mutate X -> Y.
Generate command config from Y.
Pass Y to next iteration.

Structure command execution based off of accumulated D.
Any left over inputs that are not satisfiable by commands are pure inputs.
Validate that those exits before execution.
How to handle unneeded outputs? If a command provides output that no other command needs,
do we not run it? (Analyze based on need for stats).

Validation is on the onus of the commands, not the configuration.
This is true mostly, but there is some command settings that need to be user supplied.
How do we do this two stage validation?
One validation step on config, the other on command.

For, say a custom transformer that relies on a custom analyzer:
- User writes a plugin for Analyzer that supplies specific output
- User writes a plugin for RasterTransformer that uses the result of the custom Analyzer.
- User runs Analyze and Chip. Transformer config has a uri that needs to be injected by
Analyze command. Analyze command can ask it's analyzers what paths will be produced, given
a working directory. Analyze command will also mutate the experiment configuratio for
anything needing that URI. Analyzer has a method that takes in raster transformers
and injects any necessary methods. Or should the config take in an experiment, and
modify itself based on what it needs? StatsTransformerConfig has a "inject_dependencies"
method that takes in an experiment config, and modifies itself? Or it modifies the experiment
tree based on what it knows? Do we need to pass this in to every type? A LabelStore could
generate it's own path if it knew the experiment.

Visitor pattern-ish. Every config has a traverse method that takes in an experiment config
and a command and returns itself or a copy of itself, modified by global values, as well
as the inputs it needs and the outputs it will produce. This way the onus for knowing
what files will be produced and consumed rely on the configs, which can be pluggable, and
not the commands, which cannot be.

For each command, given experiment config (which contains a working directory)
Traverses the experiment to collect modifications and input/outputs.
Generate command config from modified experiment.
Pass modified experiment to next commands.

-- Shifting builders to use dicts instead of config objects

If we shift builders to use dicts, then we can allow for required arguments to be stated in the
object __init__.

## Approach to running commands in order.

After the above process, we have

```
{ command_type: { inputs: [], outputs: [], empty_uris: []}, command_config }
```

Filter out any commands we aren't running.
Check there are no unsatisfied inputs.

```
{ command_type: { inputs: [], outputs: [] }, command_config }
```

Create a set of edges (deduplicate according to command type and uri):

```
{ (input_uri, (command_type, command_config)), ...
    ((command_type, command_config), output_uri) ... }
```

If we are not re-running commands, remove all commands that have existing outputs.
The tricky one here is train - there's training from model checkpoints.
Really we want to get rid of analyze and chip if their outputs are there,
but can't trust if the model is configured for more training.
You can run predict and/or eval only by only calling run with those commands.

Gather these for each command, across all experiments (deduplicate)

Generate a acyclic digraph out of it.

Find all sources who's in edge is 0. These are source inputs, they need to exist.

Modify the graph by collapsing edges {(command, uri), (uri, command)} to ((command, command)}

This gives a list of commands and the order in which to run them.

Use networkx [topological_sort](https://pelegm-networkx.readthedocs.io/en/latest/reference/generated/networkx.algorithms.dag.topological_sort.html#networkx.algorithms.dag.topological_sort) run jobs in order.

The CommandRunner will take a graph of commands to run, that it can call the topological order on.
In the batch case, jobs will be submitted based on parent jobs from the in edge.

# COMMANDS

## Analyze

Analyze computes anything from the raw raster data that is required for the transformers to work.
It works off of raw, untransformed image data.

## Chip

Chip works off of transformed data, and so should be run after Analyze.

# TODO

Sept. week 1
- Eval config (not tested)  X
- Augmentor (done, not tested)  X
- Finish commands and command configs  X

week 2
- Object Detection Integration test completion  X
- Chip Classification X

- Segmentation (Lewis)
- File system (James)
- Unit Tests (Simon + Nathan)

- Predict Package
- Run w/ rv.main (harder than thought)

- Make pip installable
- remote execution (done, not tested - needs config)

- Test with Amaero and xView
- Plugins on local
- Plugins on remote

Week 3
- Flexibility in backend install

October 2 weeks
- Test Test Test
- Docs Docs Docs
- Tutorails Tutorials Tutorials

#### From TODO comments

September next 2 weeks

TODO:
- Why does chip classification sometimes fail?
- Output log file for TF_Object_detection
- Flexibility in backend install - lazy TF loading
- Spacenet sample
- Use specific temporary directory root

- Debug option to save chips etc
- AOI
- LOGGING
- xView sample

- Segmentation
- pip install
- Runner: Rerun specific commands
- Runner: Logic for rerunning commands that have upstream command rerun
- Runner: Dry run
- Runner: Solidify logic around re-running training etc

- Config validation
- Execute Unit Test Plan

- aws batch setup
- Add potsdam sample, rewrite object detection tutorial.





DONE:
- Keras classification - save_best, short_epoch
- Plugins - including runners X
- Application config w/ plugin configuration. X
-- Configuration of AWS batch X
- Read the docs in feature branch X
- Model Defaults configuration X
- Run from rv.main() X
- Predictor X
- aws batch setup X
- aws amaero test X
- Rename scene_id to id X
- AOI X
- Flexibility in backend install - lazy TF loading X
- Config validation X
- Save experiment config X
- Name keys as experiments by default X
- Spacenet chip classification example X


TODO:

Last week in September
- Add potsdam sample, rewrite object detection tutorial.
- Documentation
- Code polish
-- Renames (including DefaultXProvider)
-- class_map vs classes
-- Better explinations in PyDocs
-- Clean up TODOs
- QGIS plugin
-- Make it use raster vision
-- Change logos
-- Fix issue with downloading from s3
-- Prediction
- Publish to Pypi
- Write release blog announcement

Lewis:
- Segmentation
- pip install
- Spacenet sample - segmentation
- Code coverage tooling
- Unit Tests - Test plan via coverage tool

Simon:
- xView example
- Fill out models
- Unit tests (Lewis)

James:
- Use specific temporary directory root
- Runner: Dry run
- LOGGING

Nathan:
- Unit Tests (Lewis)
- Complete set of configuration for backends
- Image augmentation for TF object detection

October:
- Tutorials
- Documentation
- Testing

azavea/rastervision-samples
- Full test COWC chip classification
- Full test SpaceNet Buildings/Roads
- Full test xView object detection

Nice to haves:
- Keras tensorboard
- Output log file for TF_Object_detection

Questionable:
- (Why does chip classification sometimes fail?)


Post Release:
- Monitor: Configure a Monitor which will accept events like when
commands start and stop. - Hook into Vision API
- Vector tile labels


### Source dir structure

rastervision
-analyzer
-augmentor
-backend
-command
-data
--raster_transformer
--raster_source
--label
--label_source
--label_store
-evaluation
-experiment
-protos
-runner
-task
-utils
__init__.py
__main__.py
registry.py
experiment_suite.py

### RV Command line API

```
rastervision run -f experiment.py train predict eval

rastervision run -n train predict eval

rastervision list .
rastervision list x/y/z.py



python experiment.py \
  --plugins plugins.json \
  --profile amaero \
  run aws_batch \
  --rv_branch branch
  train predict eval

python experiment.py run


python experiment.py run -r aws_batch



python experiment.py run

python experiment.py run train predict eval

// AWS Batch config needs to happen in user .rastervision, project .rastervision,
// or environment variables - not passed in on command line.
// Runner gets constructed with an RVConfig.
python experiment.py run -r aws_batch train predict eval
```
