# Physics Applications of AI Project

<details>
<summary>Show/Hide Table of Contents</summary>

[[_TOC_]]

</details>

## Overview

Your task is to develop your very own machine learning project for one of three applications (your choice!):
* Jet identification at the ATLAS experiment (classification)
* Incident Angle prediction of astroparticles (regression)
* Galaxy classification (classification/regression)

Each task has its very own dataset and list of objectives. The main criteria for these projects isn't necessarily getting the best performance by some metric, but logical development and trying out different techniques! Everything you try should be defined with functions and we would like to see the history of how your code evolved using git version control!

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
* Making sure your results are reproducible (set random seeds and save configurations!)

You will receive bonus marks for
* Using techniques and approaches not covered in the exercises
* Having a flexible framework avoiding hardcoded values and options, and instead using config files

You will not be judged based on the final performance of the model in your grade, but instead the ideas you try and how you structure the project.

---

# Project details

## Jet identification in ATLAS

Location: `registry.gitlab.unige.ch/ai-in-physics-course/exercises/project-jettag:latest`

Classification task to identify the origin of large showers of particles recorded in the ATLAS detector!

Identifying heavy particles decaying into particles which are close to oneanother is a key tool when hunting for new physics processes! And is the focus of lots of ML interest, both [within the collaboration](https://arxiv.org/abs/1808.07858) and the [wider community](https://arxiv.org/abs/1902.09914)!

### Dataset

A set of events describing large jets which have been recorded by the ATLAS detector. Your task is to identify the origin of the jet:
* QCD (just a quark or gluon)
* W/Z boson (decaying to two quarks)
* Top quark (decaying to a b-quark and a W boson, which then decays into two quarks)

You have access to a set of variables which can help separate the jets, as well its four-vector in the form of its mass, the transverse momentum ($`p_T`$), and angles in the detector of the jet ($`\eta`$, $`\phi`$). These four quantities $`(m,p_T,\eta,\phi)`$ and can be converted to other four vector representations such as $`(E,p_x,p_y,p_y)`$.

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

Location: `registry.gitlab.unige.ch/ai-in-physics-course/exercises/project-dampe:latest`

With the DAMPE detector we hope to find evidence of Dark Matter in space. The detector is a satelite in space with a calorimeter, but being in space particles could come from any direction, perfect for a regression task!

### Dataset
In this project you will be looking at calorimeter images and energies measured in the [DAMPE](http://dpnc.unige.ch/dampe/) space telescope calorimeter. DAMPE is a satellite based experiment designed to search for dark matter signatures.

This dataset consists of (simulated) hits in the DAMPE calorimeter, which can be visualised as an image! Each pixel in the image is one readout-cell of the calorimeter, think of it as a camera with a very coarse granularity, but which is sensitive to particles not light. 

As the data is simulated, the true location of the incoming particle is known, and it's trajectory can be inferred from the x-y coordinates at the top and bottom of the calorimeter.

### Aims
Infer the x-y coordinates of the incident particle at the top and bottom of the calorimeter using the image and scalar values.

#### Studies
* Check if your model is biased to certain areas of the calorimeter.
* Can all four coordinates sensibly be regressed together? Or are there better ways of regressing the target?
* How accurately can the true trajectory be inferred from the values your model predicts? How can you measure this?

## Galaxy identification

Location: `registry.gitlab.unige.ch/ai-in-physics-course/exercises/project-galaxies:latest`

In this project you will be looking at images of galaxies and classifiying them into categories based on their shape.

This is a very challenging task for machine learning, as even for humans we can't be 100% sure exactly what type of galaxy we see in an image - if we only see it side on how do we know if it is a spiral or uniform disk! 

### Dataset

The dataset for this challenge was generated using crowdsourcing, where non-experts were asked to assign images following a specific taxonomy. Each image was then assigned a float between zero and one based on the fraction of participants who assigned the image to a given class. 
This means that, instead of having binary labels, the targets of this dataset are floating point numbers. 
The full project is then a regression problem, but here we break it down into a much simpler classification and regression tasks.
These do not include the full complexity of the original challenge, if you want to try and tackle this and submit this as your project, you are welcome to.

The challenge in this project has been adapted from a [kaggle challenge](https://www.kaggle.com/c/galaxy-zoo-the-galaxy-challenge/data). The tasks you will pursue in this project are simpler than those in the original challenge, which was defined as follows.

Note that the corresponding kaggle challenge is out of date now, and the techniques that won this challenge will not be as interesting for you to consider. For image classification, there are other challenges on kaggle that are more recent, and will be much more interesting to look at for finding useful techniques.


### Aims
This project is broken down into three subtasks as follows. For this project it is not important how many of these you work on, if you focus on one task that will not be a problem.

## Exercise 1
In the first exercise, we only consider the top-level question (3 answers) and samples where at least 80 % of participants gave the same answer.
We assumed this to be a certain classification and transformed them to one-hot encoded labels.
The task for this part of the exercise is to use the image to predict the corresponding label.

## Exercise 2

In the second exercise, we consider the second layer of questions from the original kaggle challenge, i.e. Q2 and Q7.
Here, we do not use one-hot encoded labels, but the original floats that range between 0 and 1, thus making the classification problem a regression problem.
Here we can use all images in the dataset (no separate copy is produced).
You should try to make sure that the output of the classifier matches the hierarchical structure of the questions, e.g. the the summed values for Q2 equal the value for answer Q1.1.

## Exercise 3

In the third exercise, we further include questions Q6 and Q8 regarding oddities.
You will have to improve your architecture in order to correctly classify rare object classes.
Again, all images can be used.


#### Studies
* Does the same architecture (with a different output layer) perform well for all three tasks?
* Can you use augmentations to improve the classification performance? (Especially at test time).
* Can you use the output of a model from one task to inform the prediction of the next
