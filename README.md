# Physics Applications of AI Project

## Overview

Your task is to develop your very own machine learning project for one of three applications (your choice!):
* Jet identification at the ATLAS experiment (classification)
* Incident Angle prediction of astroparticles (regression)
* Galaxy classification (classification/regression)

Each task has its very own dataset and list of objectives. The main criteria for these projects isn't necessarily getting the best performance by some metric, but logical development and trying out different techniques! Everything you try should be defined with functions and we would liek to see the history of how your code evolved using git version control!

We will give a few pointers of things to try out, but top marks will be achieved by experimenting with approaches not covered in the course - just like any programming course there is far too much to cover in a single semester, and the internet has vast amounts of support and ideas - give them a go!

You are also asked to write a report to document what you have tried and why, the results they produced, and challenges you faced.

First you should choose one of the projects detailed below! It is up to you which you work on, but you should only work on one!

Each project has its own docker container which you can pull, which will contain the data and a small notebook showing how to open and work with the data.

You should then create a repository from this project, which is where you will upload your code and results.


## Developing the project

Although you have used notebooks in the examples classes, we would now like to focus on code being written in a repository - this means having a "run" file, which is your python executable, and additional functions/classes spread out over files. Try not to define one long function, but split things up into logical components that take arguments.

To keep the environment consistent please write the code to use the packages contained in the docker container.

You can develop the code in whichever editor or IDE you like and to access it from the container to run you can do the following
```
docker run -v /path/to/project:/home/project python /home/project/run.py --arguments
```
remember when doing this that any data under `/path/to/project/data` will now be at `/home/project/data` in the container!

### Developing with an IDE

Another convenient way to edit and develop is with VS code. It is posible to develop within a container by using the `remote containers` extension. To set this up just add the folder and file
`
.devcontainer/devcontainer.json
`
to your repository with the following content
```
{
    "image": "registry.gitlab.unige.ch/ai-in-physics-course/exercises/base:latest",
    "forwardPorts": [8888],
}
```

Docker contaiers can also be used with other software to make development much easier. There is no right or wrong way to work in this course, so use whichever environments you are most familiar with, even vim.

## Repository setup

You should make a "fork" of this repository into your own personal gitlab area. The button to do this to the top right of the repository page, in line with the repository name.

You can then clone the project (get a local copy) and work on it locally using `git add` `git commit` and `git push`.

Marks will be awarded for using git to manage your code development, and not just adding everything at the end of the project. Towards the end of the project we will announce how you can submit the project through gitlab for grading.

## Working on the project

You should work on the project at your own pace and in your own time. During the rest of the semester we will use the examples class as a chance for you to ask for advice and discuss the projects. We will also happily answer any questions on moodle.

Work on the projects on your laptop (or if you have access to a cluster this is even better!) and develop code to train the models for the aims of the project. You do not need to complete all aims, and you can focus on trying many different approaches on one aim. 

The code should be documented and it should be explained how to run the code.

## Project report

You are asked to write a report on the project as part of the course. This should document the studies you have performed, including figures and tables of results, as well as motivation and description.

This report should be no longer than 20 pages in the main body, and written with Latex.

## Project presentation

You are also asked to prepare a presentation on your project, explaining the approach you have taken and results you have achieved. 

More information about this will come towards the end of the semester.

## Grading details

You will receive marks for the report, the presentation and the project itself.

The minimum requirements for the project are
* Reable code in the repository which loads data, trains a model, and evaluates its performance
* A trained model which is not just random guesses
* A notebook will be accepted as long as functions are defined in external files (e.g. util.py). However, top marks will not be awarded.
  * When working on a project it is strongly advised not to use notebooks beyond testing.

You will receive marks for
* Use of functions and classes
* Using a range of layers, architectures and strategies in the project (either for one aim or across multiple)
* Ease of configurability to change the model (use of config files or options are a bonus!)
* Comparing multiple approaches with several metrics, not just the loss

You will receive bonus marks for
* Using techniques and approaches not covered in the exercises
* Having a flexible framework avoiding hardcoded values and options, and instead using config files

You will not be judged based on the final performance of the model in your grade, but instead the ideas you try and how you structure the project.

# Project details

## Jet identification in ATLAS

### Dataset

A set of events describing large jets which have been recorded by the ATLAS detector. Your task is to identify the origin of the jet:
* QCD (just a quark or gluon)
* W/Z boson (decaying to two quarks)
* Top quark (decaying to a b-quark and a W boson, which then decays into two quarks)

You have access to a set of variables which can help separate the jets, as well its four-vector in the form of its mass, the transverse momentum ($p_T$), and angles in the detector of the jet ($\eta$, $\phi$). These four quantities $(m,p_T,\eta,\phi)$ and can be converted to other four vector representations such as $(E,p_x,p_y,p_y)$.

The other variables are known as "substructure" variables, as they describe properties of the jet not directly related to its momentum, for example how the energy is distributed within the jet or whether it has a multiple-prong structure. They are calculated using "constituents" of the jet, which are the energy deposits recorded by the detector which are grouped (clustered) together to build the jet. 

The dataset also contains all of these constituents per jet. These could be used with more advanced architectures or used to create "image" representations of jets.

### Aims

1. Design and train a classifier to separate QCD jets from either W/Z or top jets in a binary classification task
2. Design and train a classifier to identify all three jet types in a multi-classifier
3. Make the classifier sensitive to the substructure but not to the four-momentum of the jets as best as possible

#### Studies
* In both cases, how do you make an optimal decision on which class each jet belongs to? 
* What is the performance as a function of the jet four momenta components? Look in particular at the mass!
* Can you somehow use the constituents themselves to improve performance?

## Incidence angle prediction of astroparticles passing through a detector

### Dataset


### Aims


#### Studies

## Galaxy identification

### Dataset


### Aims


#### Studies
