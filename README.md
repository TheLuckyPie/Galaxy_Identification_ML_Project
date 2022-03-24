# Physics Applications of AI Project

## Overview

Your task is to develop your very own machine learning project for one of three applications (your choice!):
* Jet identification at the ATLAS experiment (classification)
* Incident Angle prediction of astroparticles (regression)
* Galaxy classification (classification/regression)

Each task has its very own dataset and list of objectives. The main criteria for these projects isn't necessarily getting the best performance by some metric, but logical development and trying out different techniques! Everything you try should be defined with functions and we would liek to see the history of how your code evolved using git version control!

We will give a few pointers of things to try out, but top marks will be achieved by experimenting with approaches not covered in the course - just like any programming course there is far too much to cover in a single semester, and the internet has vast amounts of support and ideas - give them a go!

You are also asked to write a report to document what you have tried and why, the results they produced, and challenges you faced.

## Developing the project

Although you have used notebooks in the examples classes, we would now like to focus on code being written in a repository - this means having a "run" file, which is your python executable, and additional functions/classes spread out over files. Try not to define one long function, but split things up into logical components that take arguments.

To keep the environment consistent please write the code to use the packages contained in the docker container.

You can develop the code in whichever editor or IDE you like and to access it from the container to run you can do the following
```
docker run -v /path/to/project:/home/project python /home/project/run.py --arguments
```
remember when doing this that any data under `/path/to/project/data` will now be at `/home/project/data` in the container!

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

## Repository setup

You should make a "fork" of this repository into your own personal gitlab area. The button to do this to the top right of the repository page, in line with the repository name.

You can then clone the project (get a local copy) and work on it locally using `git add` `git commit` and `git push`.

Marks will be awarded for using git to manage your code development, and not just adding everything at the end of the project. Towards the end of the project we will announce how you can submit the project through gitlab for grading.

# Projects

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
