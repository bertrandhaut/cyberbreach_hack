# Cyber Breach Hack

What ?

A tool which finds the optimal solution of the breach protocol mini games of Cyberpunk 2077.

Why?

- These mini games are quite repetitive on the long run.
- Hacking an hacking game seemed so elegant.

How does it work ?
- Add a screenshot of the breach game in a dedicated directory. I'm using the screenshot functionality of the geforce 
  experience.
- This directory is monitored by a watchdog which triggers the "hack" function each time a new file is detected.
- The "keyboard" and "objective" symbols are extracted from the image.
- A brute force approach is testing all combinations and printing the shortest optimal solution in the terminal.


## Install

Just download the files and ensure to have the dependencies listed in requirements.yml in your python 
environment.

Alternatively create a dedicated environment:

~~~
conda env create -f requirements.yml
~~~

## Run

Modify the cyberbreach\main.py to specify the directory in which the screenshots will be generated.
 
Start the script in a console and ensure to have the cyberbreach directory in your PYTHONPATH environment variable. 
Or use the start.bat script.

When a new screenshot is added in the monitored path, the following output will be display in the console:

~~~
2020-12-27 11:29:31,158 [INFO] __main__: M:
[['55' '7a' '55' '1c' '1c' '1c' 'FF']
 ['55' 'FF' 'FF' 'e9' '1c' 'e9' '1c']
 ['1c' '55' '55' '7a' '1c' '55' 'bd']
 ['55' 'bd' 'bd' '1c' 'FF' 'bd' '7a']
 ['7a' 'bd' '1c' '55' '1c' 'e9' '1c']
 ['FF' 'e9' 'bd' 'bd' 'e9' '1c' 'e9']
 ['7a' 'FF' 'bd' '1c' '1c' '1c' '7a']]
2020-12-27 11:29:31,712 [INFO] __main__: T:
[['e9' 'FF' '1c' None]
 ['1c' 'FF' '7a' None]
 ['1c' 'e9' '1c' 'e9']]
2020-12-27 11:29:31,712 [INFO] __main__: n_buffer: 6
2020-12-27 11:29:32,349 [INFO] __main__: C4, R2, C5, R6, C1, R3
2020-12-27 11:29:32,349 [INFO] __main__: Symbol selection: ['1c', 'e9', '1c', 'e9', 'FF', '1c']
2020-12-27 11:29:32,349 [INFO] __main__: gain: 9.994
2020-12-27 11:29:32,349 [INFO] __main__: Solution: [4 2 5 6 1 3]
~~~

M is the keyboard matrix.

T is the objective matrix

C4, R2, C5, R6, C1, R3 is the "optimal" solution. Should be read as:
- select the 4th column of the selected row
- select the 2nd row of the selected column
- ...


## Optimality

The solution proposed is the one maximizing the gain computed as
sum_i i^2 * s_i  with

 - s_i = 1 if i-th objective is satisfied
 - s_i = 0 otherwise

## Limitations

This code was written with minimal effort so that "it works on my machine". This implies some limitations:

- **The code is valid only for one particular resolution: 1920 x 1080.** Its probably easy to adapt it for other resolutions
  by scaling hardcoded values and replacing some strict equality with approximate equalities.
- Many hardcoded logics, will certainly break if these minigames are modified by a future patch.
- Badly structured and limited (or even wrongs) comments.
- Brute force approach not efficient (but shouldn't be an issue if your machine is able to run cyberpunk 2077).

## What's next?

Nothing. If you want something more feel free to fork it and/or submit Pull Request.