# 0001 - Repeatable Containerized Experiments

## Problem
We would like to have a workflow for creating snapshots of RV and its environment and using them when running experiments locally or remotely. In addition, it should be easy to repeat an experiment again based on a snapshot. Repeatability is important for establishing credibility. Instead of just claiming we got some results, we can very precisely show others how they were obtained. Repeatability is also important for debugging. If an experiment fails, we need to be able to go back and troubleshoot it.

Currently, we (sort of) attempt to do this by creating a branch off of RV and then push it to origin (ie. the public RV repo). When running a Batch job, we use the latest Docker image pushed to ECR, and the job runs a script to check out the specified branch and run a command in it. This approach has many problems associated with it including the following.
* We delete the branches after the experiment is finished running to avoid cluttering up an open-source repo with client-specific information. This makes it impossible to precisely record what code was executed or to repeat the experiment at a later date.
* Branches are mutable, and sometimes we push something to a branch in the middle of a job running, which can be confusing.
* Even if we thought using branches on origin was a good idea, people not at Azavea cannot push to origin, and our scripts do not support specifying alternative Github repos.
* The Docker image that the Batch job uses is whatever image was created the last time the `ecr_publish` command was run. This means that the environment in which an experiment was run is not necessarily captured by the branch.
* There is no easy way to specify an ECR image tag when submitting a Batch job. Therefore, everybody uses the `latest` tag, which leads to conflicts between developers when one developer runs `ecr_publish` without telling the others.

## Proposed Solution

The overall proposed solution is to 1) store snapshots of RV in a more private and less mutable way and never delete them, and 2) make it so that the Batch job for an experiment uses a specific Docker image based on the experiment snapshot.

### Publishing snapshots
There are two potential approaches to creating RV snapshots, and we are leaning towards using the first one.
* We can create a branch, tag it, and then push it to a private repo (perhaps named `raster-vision-experiments`). Using Github makes intuitive sense because we are accustomed to storing and viewing code on Github. By creating a tag, we can avoid issues related to force-pushing or deleting the branch. By using a private repo, we don't have to worry about revealing client-specific information or adding clutter to the public repo.
* We can create a zip file containing the contents of the repo and then upload it to a private S3 bucket using a unique file name. This is simple, but strips out the commit history and makes it somewhat difficult to inspect the contents of the snapshot and compare it to various branches.

In either case, the snapshot should be named according to a convention. One possibility is:
```
<initials>/<project-name>/<year>-<month>-<day>/<experiment-nickname>
```
In addition we can refer to the snapshot using a URI to a zip file. (Github creates a zip file for each tag.)

### Utilizing snapshots
In order to easily utilize RV snapshots (either when running an experiment for the first time, or repeating it later), we need to make various changes and additions to RV scripts.
* The `update` script needs to take an optional reference to the snapshot, and then bake that snapshot of RV into the image. (Currently, the Dockerfile just copies the local copy of RV into the image.) It also needs to tag that image using the name of the snapshot.
* The `run` script needs to take a tag as input, so that we can run a snapshot locally.
* Currently, there is an `ecr_publish` script that runs uploads the latest Docker image to ECR. This needs to be modified so that it takes a tag name, and runs `update` using that tag, publishes it to ECR using that tag name, and then creates a Batch job def using that URI. We need to make a job def based on the image tag because it is not possible to simply pass the image URI when the job is submitted -- it needs to be encoded in a job def. Since this command now does more, perhaps it should be renamed `aws_publish`.
* The `batch_submit` function/script should take the tag name as an option and then append it to the base Batch job def name to generate a fully resolved job def name.

The workflow for running an experiment will be to run `aws_publish` with the Git tag, then `run` with the tag, then `batch_submit` (or `chain_workflow` which calls `batch_submit`) with the tag and command to run in the container remotely. Thus, the experiment is parameterized by the tag, command to run, and implicitly all the data files (on S3) referenced by the command that is run.

### Cloud config file
To make these scripts easy to use, and not require manually passing in configuration each time they are run (or hardcoding configuration), we should start using some sort of cloud configuration file. This file can contain pieces of information such as the `batch_queue`, `batch_job_def_name`, `ecr_repo`, `github_experiment_repo`, and will be used by the various scripts.

### Flexibility for new, non-Azavea users
The proposed approach adds complexity, which is unattractive to new users who just want to run RV with the latest image published to Quay. To allow this, they should be able to skip the "Publishing Snapshot" step, and just utilize the latest Docker image published to Quay. To do this, they will need to create a Batch job def that points to the image on Quay, and then set the appropriate  fields in the cloud config file. When using the `batch_submit` script, they will omit the tag option which will default to using the job def in the cloud config file.

## Unsolved issues

### Seeding random number generators
Another aspect of repeatability is setting random number generator seeds so the same sequence of random numbers is always generated. Unfortunately, this is more difficult than expected due to limitations of the underlying libraries, and will need to be covered in another issue. See https://github.com/keras-team/keras/issues/2280

### Mutability of data
If the data referenced by an experiment is modified, moved, or deleted after it has been run, the experiment is no longer repeatable. Due to the size of the files (many GBs) and the fact that they are shared across experiments, it is not feasible to store them on a git branch. Therefore, we need to resort to a convention that data is not mutated in any way after an experiment is run. Unfortunately, this is difficult to enforce, and can be annoying since data files are usually shared between experiments, and sometimes a better data organization scheme is devised as experimentation progresses. However, I don't see any easy solution to this problem at this time.
