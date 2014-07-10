# CostFunctionLearning #

This repository contains code for performing experiments on synthetic
data with cost learning models described in the documents under the
'paper' directory.  Below is an overview of the organization of the
code, how to run various components, and some notes about possible
future improvements.

## Layout of the project ##

The code is organized into the following packages in the *src* directory:

* *cost.data.annotation* - Classes for loading synthetic datasets into
memory.

* *cost.model* - Classes implementing the cost learning models described
under in the documents under the 'paper' directory.

* *cost.model.factoredcost* - Classes implementing various ways of categorizing
incorrect predictions, and generating vectors representing these 
categorizations for use in the cost learning model.

* *cost.scratch* - Code for performing miscellaneous tasks. The files in
this directory contain the main functions where the code starts running.

* *cost.util* - Miscellaneous utility classes.

*cost.scratch* contains the entry points for the code, so if you're trying
to understand how the code works, then this is a good place to start.

The *experiments/KCVTest* directory contains the configuration files 
for running cross validation experiments for the cost learning models
on synthetic data.  The format of these files is the same as other experiment
configuration files that use the ARKWater library.  See that library for
more documentation.

The *files* directory contains templates for configuration files that should
be filled in when you're setting up the project.

The *paper* directory contains unpublished papers about the cost learning
models implemented in *cost.model*.  There is a *nips2014* paper that contains
details about the most recent *cost.model.SupervisedModelSVMCLN* implementation,
and a *previous-approaches* paper that contains details about the other models
in the *cost.model* package.

The *syntheticDataModels* directory contains models by which to generate 
synthetic data sets using *cost.scratch.ConstructFakeData*.

## How to run things ##

Before running anything, you need to configure the project for your local 
setup.  To configure, do the following:

1. Untar the jars at *files/jars.tgz* into an appropriate location.

2.  Copy *files/build.xml* and *files/cost.properties* to the top-level 
directory of the project. 

3.  Fill out the copied *cost.properties* and *build.xml* files with the 
appropriate settings by replacing the text in those files that is
surrounded by square brackets with the appropriate paths.

## Possible Improvements ##

* The implementations of cost learning models in *cost.model* and the SVM
in *ark.model* use SGD in training.  Since this is an approximate method,
it seems more difficult to debug and ensure convergence (although that may
only be due to my (Bill's) inexperience with these things).  It might be
a good idea to implement non-stochastic versions during
development of further models to make it easier to ensure that things are
working properly.

* The example synthetic data models in *syntheticDataModels* each only 
contain four features, and *cost.scratch.ConstructFakeData* uses these
to create data sets with inconsistent labels (the same combination of
features can have more than one label in training).  It might be good
to create more realistic synthetic data sets with models that have at 
least several hundreds or thousands of features, and not necessarily 
inconsistent labels.  The small number of features in the existing synthetic
data sets seems somewhat detrimental to the models since the feature
weights become super sensitive to each feature, and this is an 
unnecessary extra concern to worry about.
