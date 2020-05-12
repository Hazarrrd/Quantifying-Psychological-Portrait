# Quantifying Psychological Portrait
Authors: Michał Pogoda, Jan Sieradzki

## Description
### About Big Five personality 
>In psychological trait theory, the Big Five personality traits, also known as the five-factor model (FFM) and the OCEAN model, is a suggested taxonomy, or grouping, for personality traits, developed from the 1980s onwards. When factor analysis (a statistical technique) is applied to personality survey data, some words used to describe aspects of personality are often applied to the same person. For example, someone described as conscientious is more likely to be described as "always prepared" rather than "messy". This theory is based therefore on semantic associations between words and not on neuropsychological experiments. This theory uses descriptors of common language and suggests five broad dimensions commonly used to describe the human personality and psyche. -- <cite>https://en.wikipedia.org/wiki/Big_Five_personality_traits</cite>

### Dataset
The questionare consists of 50 statements, scaled from 1 to 5 based on how much respondant agrees with the statement. It's explicitly defined, that 1 means "strong disagree", 3 means "neutral" and 5 means "strongly agree". There are 50 questions in total. Questions are divided into 5 groups, each composed of 10 questions. Questions from one group are tailored to measure 1 particular trait at the time. 

In the raw dataset there are additional metadata like time users spent anwsering each question and parameters of device on wich questionare was completed (ip origin, resolution etc.). Where reaseach into these metadata might be insightfull (eg. how country affects trait distribution, are people with larger self-esteem more likely to have high resolutions?), we explicitly focused on questionare data.

### Goals
The aim of the project is to measure, in a probabilistic reasoning, how well does Big Five personality traits explain questionare anwsers from Open Psychometrics. The dataset was kindly provided by Bojan Tunguz on Keggle. We also compare probabilistic interpetation of Big Five model with a fully connected model with latents that fully deduced during inference. Last goal of the project is to use Dirichlet Process Mixture Model to explore the dataset cluster people into groups of people with similar psychological portrait.

## Project notebooks
* `5 traits Model.ipynb` - Bayeasian interpretation of Big Five personality traits model
* `Custom Traits.ipynb` - Generalized bayesian interpretation of Big Five personality traits, with categorical probabilities described as general function of latents
* `DPMM.ipynb` - Dirichlet Process Mixture Model
* `Dataset Analisis.ipynb` - Initial data exploration

## Project configuration
All of the analisis is done in Jupyter notebooks. To ease the launch of project, we provided a docker container.
To launch a project [install docker](https://docs.docker.com/engine/install/ubuntu/).
1. Install docker-compose
```
sudo apt update && sudo apt install docker-compose
```
2. Install git large file storage
```
curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash
```
```
sudo apt install git-lfs
```
3. Clone repository
```
git clone git@github.com:puma-wust/project-1-sieradzki-pogoda.git 
```
4. Navigate to project root
```
cd project-1-sieradzki-pogoda
```
5. Build container
```
docker-compose build
```
6. Launch jupyter in container
```
docker-compose up
```
