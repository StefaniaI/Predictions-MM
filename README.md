# Strategic Behavior in Two-sided Matching Markets with Prediction-enhanced Preference-formation
This repository contains an agent-based simulation for repeated school choice. Its goal is to investigate the effects of adversarial interaction attacks on both schools and students.

For detailed information on the model please see the paper titled "Strategic Behavior in Two-sided Matching Markets with Prediction-enhanced Preference-formation" (- working paper).

## Programming language
The simulation is coded in Python - version 3.8.10.

## Files
Before running the code, please make sure all the required libraries are installed. For each *.py file, the used libraries are imported at the top of the code.

In this repository, you will find the following files:

### config.csv
A file with all the relevant parameters, some example values, and a short explanation of what each parameter controls. Note, that there are more parameters than described in the paper. For simplicity, within the paper, we kept some of them (e.g. the number of attributes) fixed.

Config files of this type can be used to run different simulations. You can either change this one, or create a new following the structure of config.csv.

### Market.py
This defines classes for attributes, students, schools, and market.

### Simulation.py
Contains classes for simulating the market. There is a simulation class and functions for finding the equilibria. Finally, it has a function to generate the config files necessary for running the experiments reported in the paper. To create the *.csv files and run the simulations, you can use the following commands:

```python
import simulation as sim
x, y, z = sim.configs_ve()
sim.generate_config_csvs(x, y, z, no_folders = 1)
sim.run_sims(file_name_configs='Simulation_results\\file_names1.txt')
```

By increasing the number of folders one can paralelise the simulation. Running the simulation for the parameters in config XXXXX.csv produces the file runXXXXX.csv with the respective results. Moreover, depending on the operating system, you might need to create a Simulation_results folder first. You can also create alternatives for the configs_ve function to test for other parameter combinations.

Finally, the find_equilibria function performs the best-response analysis for the configuration specified in config.csv:

```python
sim.find_equilibria(0.25) 
```


### Data_analysis.py
The file contains code for visualising the simulartion results. However, this is not part of the core simulation, but could be useful as an example for visualising the data.