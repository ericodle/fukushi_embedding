# fukushi_embedding
NLP study on Japanese adverb embedding and clustering

<p align="center">
  <img src="img/gd_logo.png" width="350" title="logo">
</p>


## Getting Started

## Prerequisite

Install [Python3](https://www.python.org/downloads/) on your computer.

Enter this into your computer's command line interface (terminal, control panel, etc.) to check the version:

  ```sh
  python --version
  ```

If the first number is not a 3, update to Python3.

## Setup

Here is an easy way to use our GitHub repository.

### Step 1: Clone the repository


Open the command line interface and run:
  ```sh
  git clone https://github.com/ericodle/fukushi_embedding.git
  ```

### Step 2: Navigate to the project directory
Find where your computer saved the project, then enter:

  ```sh
  cd /path/to/project/directory
  ```

If performed correctly, your command line interface should resemble

```
user@user:~/fukushi_embedding-main$
```

### Step 3: Create a virtual environment: 
I like to use a **virtual environment**.
Let's make one called "fe-env"


```sh
python3 -m venv fe-env
```

A virtual environment named "fe-env" has been created. 
Let's enter the environment to do our work:


```sh
source fe-env/bin/activate
```

When performed correctly, your command line interface prompt should look like 

```
(fe-env) user@user:~/fukushi_embedding-main$
```

### Step 3: Install requirements.txt

Next, let's install specific software versions so everything works properly.

  ```sh
pip3 install -r requirements.txt
  ```

### Step 4: Run GenreDiscern

This project has a GUI for easy use.
Activate the GUI by running the following terminal command:

  ```sh
python3 xxx.py
  ```

#### Pre-process sorted music dataset

Simply click "MFCC Extraction" from the Hub Window.

### Train a model

Simply click "Train Model" from the Hub Window.

#### Sort music using trained model

(Feature coming soon)

## Repository Files

- [ ] xxx.py

This script is xxx.


## Citing Our Research

Our research paper provides a comprehensive overview of the methodology, results, and insights derived from this repository. You can access the full paper by following this link: []()

<!-- LICENSE -->

## License
This project is open-source and is released under the [MIT License](LICENSE). Feel free to use and build upon our work while giving appropriate credit.


