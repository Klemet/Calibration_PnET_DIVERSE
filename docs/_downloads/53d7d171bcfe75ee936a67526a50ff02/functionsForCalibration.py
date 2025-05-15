# -*- coding: utf-8 -*-
"""
@author: Klemet for the DIVERSE project.

This file contains functions used in the jupyter notebook to calibrate
PnET Succession for LANDIS-II.

They have many different used to make the calibration process much easier and
quicker, by allowing a lot of interactions between the python scripts LANDIS-II
parameter files, simulations and outputs.

There are :
    
- Functions to parse the parameters of a simulation into python dictionnaries
  (for example, reading some template scenario files to then modify the
   parameters)
- Functions to write python dictionnaries containing LANDIS-II parameters into text files
  (for example, to write the parameters that were read with the parsing functions,
   and then modified back into text files for a future simulation)
- Functions to launch a LANDIS-II simulation
- Functions to parse the PnET outputs
  (these outputs can be raster files of .csv files)
- Functions to plot the PnET outputs


Most functions come with a snippet of code to test them, that can be put in the
Jupyter notebook if you want to test them quickly.

All functions here were written with the help of AI to make things quicker. I
acknowledge the ethical issues raised by AI, and do not want to dismiss them;
I made this choice as the amount of work needed for calibrating PnET is enormous,
and I wanted this to be done as quickly as possible.
"""

#%% PACKAGES NEEDED FOR THE FUNCTIONS

import os, re, glob
import matplotlib.pyplot as plt
import pandas as pd
import subprocess
import rasterio
import numpy as np
import shutil
import sqlite3
import time
import sys
import pexpect
import nbformat
from shapely.geometry import Point
import xarray as xr
from datetime import datetime
import geopandas as gpd
import cftime
import requests
from tqdm import tqdm
import matplotlib.dates as mdates
from matplotlib.colors import to_rgba
from matplotlib.gridspec import GridSpec
from rasterio.windows import Window
from scipy.spatial.distance import cdist
from siphon.catalog import TDSCatalog
from clisops.core import subset
import warnings

#%% FUNCTIONS TO PARSE PARAMETERS

def parse_PnET_ComplexTableParameterfile(file_path):
    """
    Function used to parse a LANDIS-II parameter file that as the same
    format as the PnET species parameter file or the PnET Ecoregions parameter
    file; meaning :
        
        - A first line that indicates the type of the parameter file
          (i.e. "LandisData PnETSpeciesParameters")
        - A second line that lists all parameters for a certain entity
        - Many lines that lists the values of the parameters for several
          entities (species, ecoregions, etc.)

    Parameters
    ----------
    file_path : String
        Path to the LANDIS-II parameter file.

    Returns
    -------
    result_dict : Dictionnary
        Dictionnary containing the parameter values.
        The first entry is "LandisData":"Value"
        The second entry is another dictionnary whose key is the first word
        on the line that describe all of the parameters for the entities (i.e. PnETSpeciesParameters)
        This nested dictionnary then contains other nested dictionnaries, one
        for each entity (e.g. one per ecoregion), with the associated key of the
        name of the entity (e.g. "eco1"). These dictionnaries contain the value
        of all of the parameters for the entity,a assiciated with the name
        of the parameter as a key.
        
        Here is an example of an output dictionnary for a simple PnET ecoregion
        parameter file with one ecoregion (eco0).
        
        {'LandisData': 'EcoregionParameters',
        'EcoregionParameters':
            {'eco0':
             {'RootingDepth': '1000',
              'SoilType': 'SAND',
              'LeakageFrac': '1',
              'PrecLossFrac': '0',
              'SnowSublimFrac': '0.15',
              'ClimateFileName': 'climate.txt'}
            }
        }
            
        For example, to get parameter "SoilType" for "eco0", use :
            
            outputDict["EcoregionParameters"]["eco0"]["SoilType"].

    """
    
    # Initialize the main dictionary
    result_dict = {}

    with open(file_path, 'r') as file:
        # Read the first line and skip if it's empty or a comment
        first_line = file.readline().strip()
        if not first_line or first_line[0:2] == ">>":
            while(not first_line or first_line[0:2] == ">>"):
                first_line = file.readline().strip()
                
        first_line_parts = first_line.split()
        result_dict[first_line_parts[0]] = first_line_parts[1]  # LandisData: PnETSpeciesParameters or EcoregionParameters

        # Read the second line for keys of nested dictionaries and skip if it's empty or a comment
        keys_line = file.readline().strip()
        if not keys_line or keys_line[0:2] == ">>":
            while(not keys_line or keys_line[0:2] == ">>"):
                keys_line = file.readline().strip()
                
        keys = keys_line.split()[1:]  # Ignore first word as it is the parameter of the table

        # Initialize the nested dictionary
        result_dict[first_line_parts[1]] = {}

        # Read each subsequent line for species/ecoregion/disturbance reduction data
        for line in file:
            line = line.strip()  # Remove leading/trailing whitespace
            if not line or line[0:2] == ">>":  # Skip empty lines or comment lines
                continue
            
            parts = line.split()
            species_name = parts[0]  # First part is the species name
            values = parts[1:]  # Remaining parts are values

            # We remove any trailing space or empty character in the values
            values = [value for value in values if (value is not None and value != "None" and value != "" and value != " " and value != "\t")]

            # Create a nested dictionary for this species
            species_dict = {}
            for i, value in enumerate(values):
                if '<<' in value:
                    value = value.split('<<')[0]  # Ignore everything after "<<"
                try:
                    species_dict[keys[i]] = value
                except (ValueError, IndexError):
                    continue  # Handle any conversion errors or index out of range

            # Add the species dictionary to the main dictionary
            result_dict[first_line_parts[1]][species_name] = species_dict

    return result_dict

# Example usage:
# Assuming 'data.txt' is your input file with the specified format.
# data_dict = parse_landis_data('data.txt')
# print(data_dict)

def parse_LANDIS_SpeciesCoreFile(file_path):
    """
    Function used to parse the LANDIS-II "Core" parameter file
    for the species simulated. It has a structure that is a bit different
    from other parameter files, so it has it own function.

    Parameters
    ----------
    file_path : String
        Path to the LANDIS-II core species parameter file.

    Returns
    -------
    result_dict : Dictionnary
        Dictionnary containing the parameter values.
        The first entry is "LandisData":"Species"
        The other keys are for the different species, which then 
        leads to a nested dictionnary where each key is a parameter name,
        and is associated with the value of this parameter for this species.
        
        Here is an example of an output dictionnary for a simple core parameter
        file with two species.
        
        {'LandisData': 'Species',
         'querrubr': {
              'Longevity': '100',
              'Sexual Maturity': '50',
              'Shade Tolerance': '3',
              'Fire Tolerance': '2',
              'Seed Dispersal Distance - Effective': '30',
              'Seed Dispersal Distance - Maximum': '3000',
              'Vegetative Reproduction Probability': '0.9',
              'Sprout Age - Min': '20',
              'Sprout Age - Max': '100',
              'Post Fire Regen': 'resprout'},
         'pinubank':
             {'Longevity': '100',
              'Sexual Maturity': '15',
              'Shade Tolerance': '1',
              'Fire Tolerance': '3',
              'Seed Dispersal Distance - Effective': '20',
              'Seed Dispersal Distance - Maximum': '100',
              'Vegetative Reproduction Probability': '0',
              'Sprout Age - Min': '0',
              'Sprout Age - Max': '0',
              'Post Fire Regen': 'serotiny'}
        }

        For example, to get parameter "Longevity" for "pinubank", use :
            
            outputDict["pinubank"]["Longevity"].

    """
    
    # Define the keys for the nested dictionary
    # Edited for v8 to remove shade tolerance and fire tolerance
    keys = [
        "Longevity",
        "Sexual Maturity",
        "Seed Dispersal Distance - Effective",
        "Seed Dispersal Distance - Maximum",
        "Vegetative Reproduction Probability",
        "Sprout Age - Min",
        "Sprout Age - Max",
        "Post Fire Regen"
    ]
    
    # Initialize an empty dictionary to store the data
    data_dict = {}

    with open(file_path, 'r') as file:
        # Read the first line for main key-value pair
        first_line = file.readline().strip()
        main_key, main_value = first_line.split('\t')
        data_dict[main_key] = main_value
        
        # Read the next lines
        for line in file:
            line = line.strip()
            # Ignore lines starting with '>>'
            if line.startswith('>>') or not line:
                continue
            
            # Split the line into components
            components = line.split()
            species_name = components[0]
            values = components[1:]

            # Create a nested dictionary for each species
            nested_dict = {}
            for i, key in enumerate(keys):
                if i < len(values):
                    nested_dict[key] = values[i]
                else:
                    nested_dict[key] = None  # Handle missing values gracefully

            # Add the species data to the main dictionary
            data_dict[species_name] = nested_dict

    return data_dict

# # Example usage:
# file_path = r'D:\OneDrive - UQAM\1 - Projets\Post-Doc - Parametrer LANDIS-II pour DIVERSE\4 - Beginning PnET Calibration\Calibration_PnET_DIVERSE\StartingCalibrationTest\SimulationFiles\PnETGitHub_OneCellSim\species.txt'  # Replace with your actual file path
# landis_data_dict = parse_LANDIS_SpeciesCoreFile(file_path)
# print(landis_data_dict)


def parse_LANDIS_SimpleParameterFile(file_path):
    """
    Function used to parse the all LANDIS-II parameter files that have
    a relatively simple structure where each parameter is associated to
    one value. Most LANDIS-II parameter files follow this format.

    Parameters
    ----------
    file_path : String
        Path to the LANDIS-II parameter file.

    Returns
    -------
    result_dict : Dictionnary
        Dictionnary containing the parameter values.
        Each key is the name of the parameter, with the value being the value
        of the parameter.
    """
    
    result = {}
    
    with open(file_path, 'r') as file:
        for line in file:
            # Skip lines starting with '>>'
            if line.strip().startswith('>>'):
                continue
            
            # Remove everything after '<<' (including '<<')
            line = re.split('<<', line)[0].strip()
            
            # Skip empty lines
            if not line:
                continue
            
            # Split the line into words
            words = line.split()
            
            if len(words) >= 2:
                key = words[0]
                value = words[1]
                if len(words) > 2:
                    for wordIndex in range(2, len(words)):
                        value = value + " " + words[wordIndex]
                
                # Handle quoted keys
                # if key.startswith('"') and key.endswith('"'):
                    # key = f"'{key}'"
                
                result[key] = value
    
    return result

# # Example Usage
# file_path = r'calibrationFolder/StartingCalibrationTest/SimulationFiles/PnETGitHub_OneCellSim/scenario.txt'  # Replace with your actual file path
# config_dict = parse_LANDIS_ScenarioCoreFile(file_path)
# print(config_dict)


def parse_ecoregions_file(file_path):
    """
    Function used to parse the LANDIS-II core ecoregion file,
    which is a bit different from the others - and so gets its own function.

    Parameters
    ----------
    file_path : String
        Path to the LANDIS-II core ecoregion parameter file.

    Returns
    -------
    result_dict : Dictionnary
        Dictionnary containing the parameter values.
        The first entry is "LandisData":"Ecoregions"
        The other keys are for the different ecoregions, which then 
        leads to a nested dictionnary where each key is a parameter name,
        and is associated with the value of this parameter for this ecoregion.
        
        Example of output dictionnary for a simple file with two ecoregions :
        {'LandisData': 'Ecoregions',
         'eco0': {'active': 'no', 'Map Code': '0', 'Description': '"inactive"'},
         'eco1': {'active': 'yes',
          'Map Code': '1',
          'Description': '"well-drained rocky slopes - SW aspect"'}}
    """
    
    result = {}
    
    with open(file_path, 'r') as file:
        lines = file.readlines()
        
    # Process the first line
    if lines and not lines[0].startswith('>>'):
        key, value = lines[0].strip().split(None, 1)
        result[key] = value
    
    # Process the remaining lines
    for line in lines[1:]:
        line = line.strip()
        if not line.startswith('>>'):
            # Split the line into a list of 4 items
            items = line.split(None, 3)
            if len(items) == 4:
                active, code, name, description = items
                # Create nested dictionary for each ecoregion
                result[name] = {
                    "active": active,
                    "Map Code": code,
                    "Description": description
                }
    
    return result


def parse_LANDIS_ClimateDataFile(file_path):
    """
    Function used to parse the LANDIS-II climate file.

    Parameters
    ----------
    file_path : String
        Path to the LANDIS-II climate file.

    Returns
    -------
    result_dict : Dictionnary
        Dictionnary containing the parameter values.
        Contains nested dictionnaries to first call :
            - The year
            - Then the month
            - Then the parameter name
        
        
        Example of structure :
            {'1600-1996': 
             {'1': {'TMax': '2.1',
               'TMin': '-6.1',
               'PAR': '350.9959848079',
               'Prec': '12.14',
               'CO2': '313.065'},
              '2': {'TMax': '4.1',
               'TMin': '-5',
               'PAR': '422.8764365948',
               'Prec': '27.27',
               'CO2': '313.065'},
              ...
    """
    
    
    result = {}
    
    with open(file_path, 'r') as file:
        # Skip the header line
        next(file)
        
        for line in file:
            # Split the line into values
            values = line.strip().split('\t')
            
            # Extract values
            year, month, tmax, tmin, par, prec, co2 = values
            
            # Create or update the nested dictionary structure
            if year not in result:
                result[year] = {}
            
            result[year][month] = {
                "TMax": tmax,
                "TMin": tmin,
                "PAR": par,
                "Prec": prec,
                "CO2": co2
            }
    
    return result

# # Example usage:
# file_path = rcalibrationFolder/StartingCalibrationTest/SimulationFiles/PnETGitHub_OneCellSim/Climate.txt'  # Replace with your actual file path
# climate_data = read_climate_data(file_path)
# print(climate_data)



def parse_All_LANDIS_PnET_Files(folder_path):
    """
    The function uses the different functions defined in this file
    (parse_LANDIS_SimpleParameterFile, parse_LANDIS_SpeciesCoreFile,
     parse_ecoregions_file, parse_LANDIS_ClimateDataFile,
     parse_PnET_ComplexTableParameterfile) to parse all of the files
    necessary for a simple simulation with PnET Succession :
        - The LANDIS-II main scenario file, core species and core ecoregion file
        - The PnET parameter files, including the PnET Output and site output files
        - All other files (e.g. raster files) are not parsed; but their path
          is recording to facilitate copy/paste operations.

    Parameters
    ----------
    file_path : String
        Path to the folder where the LANDIS-II parameter files are.

    Returns
    -------
    result_dict : Dictionnary
        Dictionnary containing the parameter values.
        Contain nested dictionnaries; each key of the main dictionnary has
        the name of a file (e.g. scenario.txt), which is then associated by the
        dictionnary containing the parameter values (see docstring of the
        other functions).
    """
    
    
    results = {}  # Initialize an empty dictionary to store results

    # List all files in the specified directory
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        # Check if the file has a .txt extension
        if filename.endswith('.txt') or filename.endswith('.csv'):
            with open(file_path, 'r') as file:
                content = file.read()
                # We make a version of the list of lines with the words separated
                # so as to make the recognition of the file through the "LandisData"
                # keyword more robust (can be recognized despite trailing spaces, etc.)
                content_separated = [line.split() for line in content.splitlines()]
                # print(content_separated)

                # Now, we do things differently in the case of each parameter file found

                # Main scenario file
                if ["LandisData", "Scenario"] in content_separated:
                    print("Found : Main scenario file : " + str(filename))
                    results[filename] = parse_LANDIS_SimpleParameterFile(file_path)

                # Main species parameter core file
                elif ["LandisData", "Species"] in content_separated:
                    print("Found : Main species parameter file : " + str(filename))
                    results[filename] = parse_LANDIS_SpeciesCoreFile(file_path)

                # Ecoregions file
                elif ["LandisData", "Ecoregions"] in content_separated:
                    print("Found : Main ecoregions file : " + str(filename))
                    results[filename] = parse_ecoregions_file(file_path)

                # Initial Communities file
                # We add the detection of the .csv file header/columns
                elif ["LandisData", "Initial Communities"] in content_separated or ["LandisData", '"Initial Communities"'] in content_separated or ["LandisData", "Initial", "Communities"] in content_separated or ['LandisData', '"Initial', 'Communities"'] in content_separated or ["MapCode,SpeciesName,CohortAge,CohortBiomass"] in content_separated:
                    print("Found : Initial Communities file : " + str(filename))
                    # Now that initial communities are a .csv file, we can load this file in a panda dataframe for easy editing
                    results[filename] = pd.read_csv(file_path)

                # Climate data file
                elif ["Year", "Month", "TMax", "TMin", "PAR", "Prec", "CO2"] in content_separated:
                    print("Found : Climate file : " + str(filename))
                    results[filename] = parse_LANDIS_ClimateDataFile(file_path)

                # PnET Ecoregion parameter files
                elif ["LandisData", "EcoregionParameters"] in content_separated:
                    print("Found : PnET Ecoregion parameter file : " + str(filename))
                    results[filename] = parse_PnET_ComplexTableParameterfile(file_path)

                # PnET Succession Main file
                elif ["LandisData", "PnET-Succession"] in content_separated or ["LandisData", '"PnET-Succession"'] in content_separated:
                    print("Found : Main PnET parameter file : " + str(filename))
                    results[filename] = parse_LANDIS_SimpleParameterFile(file_path)  

                # PnET Generic parameters file
                elif ["LandisData", "PnETGenericParameters"] in content_separated:
                    print("Found : PnET generic parameter file : " + str(filename))
                    results[filename] = parse_LANDIS_SimpleParameterFile(file_path)
                    
                # PnET Species Parameter file
                elif ["LandisData", "PnETSpeciesParameters"] in content_separated:
                    print("Found : PnET species parameters file : " + str(filename))
                    results[filename] = parse_PnET_ComplexTableParameterfile(file_path)

                # PnET Output Biomass file
                elif ["LandisData", "Output-PnET"] in content_separated or ["LandisData", '"Output-PnET"'] in content_separated: 
                    print("Found : PnET OutputBiomass parameter file : " + str(filename))
                    results[filename] = parse_LANDIS_SimpleParameterFile(file_path)

                # PnET Output Sites File
                elif ["LandisData", "PNEToutputsites"] in content_separated:
                    print("Found : PnET OutputSites parameter file : " + str(filename))
                    results[filename] = parse_LANDIS_SimpleParameterFile(file_path)

                # PnET Disturbance Reduction File
                elif ["LandisData", "DisturbanceReductions"] in content_separated:
                    print("Found : Disturbance Reduction File : " + str(filename))
                    results[filename] = parse_PnET_ComplexTableParameterfile(file_path)
                
                # If it's not a parameter file, we record its path
                # Can be used to copy/paste any other file
                else:
                    print("Found : Additional file : " + str(filename))
                    results[filename] = file_path
                    # If it's not a parameter file, we keep its path to copy it
                    
        else:
            print("Found : Additional file : " + str(filename))
            results[filename] = file_path
                    
    return results

#%% FUNCTIONS TO WRITE THE PARAMETERS INTO TEXTE FILES

def write_PnET_ComplexTableParameterfile(outputFilePath, PnETSpeciesParametersDict):
    """
    This function is used so that the dictionnary obtained with parse_PnET_SpeciesParameters_file,
    containing the PnET species parameters,
    can be put back into a text file after being modified.

    Parameters
    ----------
    outputFilePath : String
        Path of the file to write the parameters into.
    data_dict : Dictionnary
        Dictionnary obtained with PnETSpeciesParametersDict.

    Returns
    -------
    None.

    """
    
    with open(outputFilePath, 'w') as file:
        # Write the first line: LandisData PnETSpeciesParameters
        landis_data_key = list(PnETSpeciesParametersDict.keys())[0]
        species_parameters_key = PnETSpeciesParametersDict[landis_data_key]
        file.write(f"{landis_data_key}\t{species_parameters_key}\n\n")
        
        # Write the second line: PnETSpeciesParameters followed by keys
        keys = list(PnETSpeciesParametersDict[species_parameters_key].values())[0].keys()
        keys_line = "\t".join([list(PnETSpeciesParametersDict.keys())[1]] + list(keys))
        file.write(f"{keys_line}\n")
        
        # Write each species and its corresponding values
        for species_name, parameters in PnETSpeciesParametersDict[species_parameters_key].items():
            # Format each value to a string without trailing zeros
            values_line = "\t".join([species_name] + [f"{value:.5f}".rstrip('0').rstrip('.') if isinstance(value, float) else str(value) for value in parameters.values()])
            file.write(f"{values_line}\n")

# # Example usage:
# PnETSpeciesParametersDict = {'LandisData': 'DisturbanceReductions', 'DisturbanceReductions': {'WoodReduction': {'fire': 0.33, 'wind': 0.0, 'harvest': 0.7, 'bda': 0.0}, 'FolReduction': {'fire': 1.0, 'wind': 0.0, 'harvest': 0.0, 'bda': 0.0}, 'RootReduction': {'fire': 0.0, 'wind': 0.0, 'harvest': 0.0, 'bda': 0.0}, 'DeadWoodReduction': {'fire': 0.7, 'wind': 0.0, 'harvest': 0.0, 'bda': 0.0}, 'LitterReduction': {'fire': 0.9, 'wind': 0.0, 'harvest': 0.1, 'bda': 0.0}}}
# write_PnET_ComplexTableParameterfile('D:\Klemet\Desktop\TEMP\outputTESTPNET.txt', PnETSpeciesParametersDict)

def write_LANDIS_SpeciesCoreFile(file_path, data_dict):
    """
    This function is used so that the dictionnary obtained with parse_LANDIS_SpeciesCoreFile,
    containing the PnET species parameters,
    can be put back into a text file after being modified.

    Parameters
    ----------
    outputFilePath : String
        Path of the file to write the parameters into.
    data_dict : Dictionnary
        Dictionnary obtained with parse_LANDIS_SpeciesCoreFile.

    Returns
    -------
    None.

    """
    
    with open(file_path, 'w') as file:
        # Write the fixed header lines
        file.write("LandisData\tSpecies\n")
        file.write("\n")
        file.write(">>                      Sexual    Shade  Fire  Seed Disperal Dist  Vegetative   Sprout Age  Post-Fire\n")
        file.write(">> Name      Longevity  Maturity  Tol.   Tol.  Effective  Maximum  Reprod Prob  Min    Max  Regen\n")
        file.write(">> ----      ---------  --------  -----  ----  ---------  -------  -----------  ----------  --------\n")

        # Define fixed widths for each column
        widths = {
            'species': 16,
            'longevity': 11,
            'sexual_maturity': 9,
            'shade_tolerance': 7,
            'fire_tolerance': 8,
            'seed_disp_effective': 8,
            'seed_disp_maximum': 12,
            'veg_repro_prob': 10,
            'sprout_age_min': 6,
            'sprout_age_max': 6,
            'post_fire_regen': 12
        }

        # Write species data
        for species_name, attributes in data_dict.items():
            if species_name == 'LandisData':
                continue  # Skip the main key
            
            # Extract values from the nested dictionary
            longevity = attributes.get("Longevity", "").ljust(widths['longevity'])
            sexual_maturity = attributes.get("Sexual Maturity", "").ljust(widths['sexual_maturity'])
            shade_tolerance = attributes.get("Shade Tolerance", "").ljust(widths['shade_tolerance'])
            fire_tolerance = attributes.get("Fire Tolerance", "").ljust(widths['fire_tolerance'])
            seed_disp_effective = attributes.get("Seed Dispersal Distance - Effective", "").ljust(widths['seed_disp_effective'])
            seed_disp_maximum = attributes.get("Seed Dispersal Distance - Maximum", "").ljust(widths['seed_disp_maximum'])
            veg_repro_prob = attributes.get("Vegetative Reproduction Probability", "").ljust(widths['veg_repro_prob'])
            sprout_age_min = attributes.get("Sprout Age - Min", "").ljust(widths['sprout_age_min'])
            sprout_age_max = attributes.get("Sprout Age - Max", "").ljust(widths['sprout_age_max'])
            post_fire_regen = attributes.get("Post Fire Regen", "").ljust(widths['post_fire_regen'])

            # Format the line for this species with fixed widths
            line = (f"{species_name:<{widths['species']}}"
                    f"{longevity}{sexual_maturity}{shade_tolerance}{fire_tolerance}"
                    f"{seed_disp_effective}{seed_disp_maximum}{veg_repro_prob}"
                    f"{sprout_age_min}{sprout_age_max}{post_fire_regen}\n")
            
            # Write the line to the file
            file.write(line)

# # Example usage:
# data_dict = {
#     'LandisData': 'Species',
#     'querrubr': {
#         'Longevity': '100',
#         'Sexual Maturity': '50',
#         'Shade Tolerance': '3',
#         'Fire Tolerance': '2',
#         'Seed Dispersal Distance - Effective': '30',
#         'Seed Dispersal Distance - Maximum': '3000',
#         'Vegetative Reproduction Probability': '0.9',
#         'Sprout Age - Min': '20',
#         'Sprout Age - Max': '100',
#         'Post Fire Regen': 'resprout'
#     },
#     'pinubank': {
#         'Longevity': '100',
#         'Sexual Maturity': '15',
#         'Shade Tolerance': '1',
#         'Fire Tolerance': '3',
#         'Seed Dispersal Distance - Effective': '20',
#         'Seed Dispersal Distance - Maximum': '100',
#         'Vegetative Reproduction Probability': '0',
#         'Sprout Age - Min': '0',
#         'Sprout Age - Max': '0',
#         'Post Fire Regen': 'serotiny'
#     }
# }

# file_path = r"D:\Klemet\Desktop\TEMP\output_landis_data.txt"  # Replace with your desired output file path
# write_LANDIS_SpeciesCoreFile(file_path, data_dict)


def write_LANDIS_SimpleParameterFile(file_path, config_dict):
    """
    This function is used so that the dictionnary obtained with parse_LANDIS_SimpleParameterFile,
    containing the PnET species parameters,
    can be put back into a text file after being modified.

    Parameters
    ----------
    outputFilePath : String
        Path of the file to write the parameters into.
    data_dict : Dictionnary
        Dictionnary obtained with parse_LANDIS_SimpleParameterFile.

    Returns
    -------
    None.

    """
    
    with open(file_path, 'w') as file:
        for key, value in config_dict.items():
            
            # Convert value to string if it's not already
            value = str(value)
            
            # Write the line to the file
            file.write(f"{key}  {value}\n")

# # Usage example
# config_dict = {
#     "LandisData": "Scenario",
#     "Duration": 50,
#     "Species": "species.txt",
#     "Ecoregions": "ecoregion.txt",
#     "EcoregionsMap": "ecoregion.img",
#     "CellLength": 30,
#     '"PnET-Succession"': "pnetsuccession.txt",
#     '"Output-PnET"': "biomass.outputPnET.txt",
#     "RandomNumberSeed": 1111
# }

# write_config_file(config_dict, 'D:\Klemet\Desktop\TEMP\output_config.txt')

def write_LANDIS_MainEcoregionsFile(file_path, data_dict):
    """
    This function is used so that the dictionnary obtained with parse_ecoregions_file,
    containing the PnET species parameters,
    can be put back into a text file after being modified.

    Parameters
    ----------
    outputFilePath : String
        Path of the file to write the parameters into.
    data_dict : Dictionnary
        Dictionnary obtained with parse_ecoregions_file.

    Returns
    -------
    None.

    """
    
    with open(file_path, 'w') as file:
        # Write the fixed header lines
        file.write("LandisData Ecoregions\n")
        file.write("\n")
        file.write(">>         Map\n")
        file.write(">> Active  Code  Name                            Description\n")
        file.write(">> ------  ----  ------------------------------  -----------\n")

        # Define fixed widths for each column
        widths = {
            'active': 8,
            'Map code': 4,
            'name': 30,
            'description': 20,
        }

        # Write ecoregion data
        for ecoregion_name, attributes in data_dict.items():
            if ecoregion_name == 'LandisData':
                continue  # Skip the main key
            
            # Extract values from the nested dictionary
            activeStatus = attributes.get("active", "").ljust(widths['active'])
            MapCode = attributes.get("Map Code", "").ljust(widths['Map code'])
            name = ecoregion_name.ljust(widths['name'])
            description = attributes.get("Description", "").ljust(widths['description'])

            # Format the line for this species with fixed widths
            line = (f"     {activeStatus}{MapCode}{name}{description}\n")
            
            # Write the line to the file
            file.write(line)

# # Example usage:
# data_dict = {'LandisData': 'Ecoregions', 'eco0': {'active': 'no', 'Map Code': '0', 'Description': '"inactive"'}, 'eco1': {'active': 'yes', 'Map Code': '1', 'Description': '"well-drained rocky slopes - SW aspect"'}}

# file_path = r"D:\Klemet\Desktop\TEMP\output_landis_data.txt"  # Replace with your desired output file path
# write_LANDIS_MainEcoregionsFile(file_path, data_dict)


def write_climate_data(file_path, data):
    """
    This function is used so that the dictionnary obtained with parse_LANDIS_ClimateDataFile,
    containing the PnET species parameters,
    can be put back into a text file after being modified.

    Parameters
    ----------
    outputFilePath : String
        Path of the file to write the parameters into.
    data_dict : Dictionnary
        Dictionnary obtained with parse_LANDIS_ClimateDataFile.

    Returns
    -------
    None.

    """
    
    with open(file_path, 'w') as file:
        # Write the header
        file.write("Year\tMonth\tTMax\tTMin\tPAR\tPrec\tCO2\n")
        
        # Iterate through the dictionary and write data
        for year, months in data.items():
            for month, values in months.items():
                line = f"{year}\t{month}\t{values['TMax']}\t{values['TMin']}\t{values['PAR']}\t{values['Prec']}\t{values['CO2']}\n"
                file.write(line)

# # Example usage:
# climate_data = {'1600-1996': {'1': {'TMax': '2.1', 'TMin': '-6.1', 'PAR': '350.9959848079', 'Prec': '12.14', 'CO2': '313.065'}, '2': {'TMax': '4.1', 'TMin': '-5', 'PAR': '422.8764365948', 'Prec': '27.27', 'CO2': '313.065'}, '3': {'TMax': '9.6', 'TMin': '-1.1', 'PAR': '681.9568657618', 'Prec': '13.93', 'CO2': '313.065'}, '4': {'TMax': '16.5', 'TMin': '4.6', 'PAR': '829.6802172436', 'Prec': '50.08', 'CO2': '313.065'}, '5': {'TMax': '21.6', 'TMin': '9.6', 'PAR': '814.6535078721', 'Prec': '93.63', 'CO2': '313.065'}, '6': {'TMax': '26.2', 'TMin': '14.7', 'PAR': '722.9276678308', 'Prec': '160.56', 'CO2': '313.065'}, '7': {'TMax': '28.1', 'TMin': '17.1', 'PAR': '872.8925973439', 'Prec': '89.87', 'CO2': '313.065'}, '8': {'TMax': '27.4', 'TMin': '16.4', 'PAR': '722.1271241014', 'Prec': '80.46', 'CO2': '313.065'}, '9': {'TMax': '23.5', 'TMin': '12.2', 'PAR': '538.6992001016', 'Prec': '101.9', 'CO2': '313.065'}, '10': {'TMax': '17', 'TMin': '6.1', 'PAR': '398.4617125821', 'Prec': '92.88', 'CO2': '313.065'}, '11': {'TMax': '10.7', 'TMin': '1.5', 'PAR': '286.6469988492', 'Prec': '51.52', 'CO2': '313.065'}, '12': {'TMax': '4.1', 'TMin': '-3.7', 'PAR': '256.5194056339', 'Prec': '18.33', 'CO2': '313.065'}}, '1997': {'1': {'TMax': '2.1', 'TMin': '-6.1', 'PAR': '297.6606941234', 'Prec': '26.416', 'CO2': '313.065'}, '2': {'TMax': '4.1', 'TMin': '-5', 'PAR': '459.687347277', 'Prec': '29.972', 'CO2': '313.065'}, '3': {'TMax': '9.6', 'TMin': '-1.1', 'PAR': '639.2213468298', 'Prec': '123.19', 'CO2': '313.065'}, '4': {'TMax': '16.5', 'TMin': '4.6', 'PAR': '757.6523826222', 'Prec': '36.576', 'CO2': '313.065'}, '5': {'TMax': '21.6', 'TMin': '9.6', 'PAR': '820.4509999536', 'Prec': '54.864', 'CO2': '313.065'}, '6': {'TMax': '26.2', 'TMin': '14.7', 'PAR': '718.279288454', 'Prec': '38.862', 'CO2': '313.065'}, '7': {'TMax': '28.1', 'TMin': '17.1', 'PAR': '813.0007719273', 'Prec': '39.37', 'CO2': '313.065'}, '8': {'TMax': '27.4', 'TMin': '16.4', 'PAR': '745.4156095801', 'Prec': '54.864', 'CO2': '313.065'}, '9': {'TMax': '23.5', 'TMin': '12.2', 'PAR': '585.7956123241', 'Prec': '136.652', 'CO2': '313.065'}, '10': {'TMax': '17', 'TMin': '6.1', 'PAR': '422.2229098247', 'Prec': '105.156', 'CO2': '313.065'}, '11': {'TMax': '10.7', 'TMin': '1.5', 'PAR': '301.6793240297', 'Prec': '160.528', 'CO2': '313.065'}, '12': {'TMax': '4.1', 'TMin': '-3.7', 'PAR': '277.9706075513', 'Prec': '28.702', 'CO2': '313.065'}}, '1998': {'1': {'TMax': '2.1', 'TMin': '-6.1', 'PAR': '308.6888052882', 'Prec': '190.5', 'CO2': '313.065'}, '2': {'TMax': '4.1', 'TMin': '-5', 'PAR': '513.1942832055', 'Prec': '144.018', 'CO2': '313.065'}, '3': {'TMax': '9.6', 'TMin': '-1.1', 'PAR': '564.5587479666', 'Prec': '154.94', 'CO2': '313.065'}, '4': {'TMax': '16.5', 'TMin': '4.6', 'PAR': '801.777901273', 'Prec': '118.618', 'CO2': '313.065'}, '5': {'TMax': '21.6', 'TMin': '9.6', 'PAR': '809.1306271837', 'Prec': '116.84', 'CO2': '313.065'}, '6': {'TMax': '26.2', 'TMin': '14.7', 'PAR': '785.4460606177', 'Prec': '130.048', 'CO2': '313.065'}, '7': {'TMax': '28.1', 'TMin': '17.1', 'PAR': '709.353639689', 'Prec': '42.926', 'CO2': '313.065'}, '8': {'TMax': '27.4', 'TMin': '16.4', 'PAR': '714.033867377', 'Prec': '63.246', 'CO2': '313.065'}, '9': {'TMax': '23.5', 'TMin': '12.2', 'PAR': '618.1866783245', 'Prec': '22.606', 'CO2': '313.065'}, '10': {'TMax': '17', 'TMin': '6.1', 'PAR': '476.891657399', 'Prec': '142.748', 'CO2': '313.065'}, '11': {'TMax': '10.7', 'TMin': '1.5', 'PAR': '289.9886518071', 'Prec': '1.778', 'CO2': '313.065'}, '12': {'TMax': '4.1', 'TMin': '-3.7', 'PAR': '248.2078448468', 'Prec': '20.574', 'CO2': '313.065'}}, '1999-2300': {'1': {'TMax': '2.1', 'TMin': '-6.1', 'PAR': '350.9959848079', 'Prec': '12.14', 'CO2': '313.065'}, '2': {'TMax': '4.1', 'TMin': '-5', 'PAR': '422.8764365948', 'Prec': '27.27', 'CO2': '313.065'}, '3': {'TMax': '9.6', 'TMin': '-1.1', 'PAR': '681.9568657618', 'Prec': '13.93', 'CO2': '313.065'}, '4': {'TMax': '16.5', 'TMin': '4.6', 'PAR': '829.6802172436', 'Prec': '50.08', 'CO2': '313.065'}, '5': {'TMax': '21.6', 'TMin': '9.6', 'PAR': '814.6535078721', 'Prec': '93.63', 'CO2': '313.065'}, '6': {'TMax': '26.2', 'TMin': '14.7', 'PAR': '722.9276678308', 'Prec': '160.56', 'CO2': '313.065'}, '7': {'TMax': '28.1', 'TMin': '17.1', 'PAR': '872.8925973439', 'Prec': '89.87', 'CO2': '313.065'}, '8': {'TMax': '27.4', 'TMin': '16.4', 'PAR': '722.1271241014', 'Prec': '80.46', 'CO2': '313.065'}, '9': {'TMax': '23.5', 'TMin': '12.2', 'PAR': '538.6992001016', 'Prec': '101.9', 'CO2': '313.065'}, '10': {'TMax': '17', 'TMin': '6.1', 'PAR': '398.4617125821', 'Prec': '92.88', 'CO2': '313.065'}, '11': {'TMax': '10.7', 'TMin': '1.5', 'PAR': '286.6469988492', 'Prec': '51.52', 'CO2': '313.065'}, '12': {'TMax': '4.1', 'TMin': '-3.7', 'PAR': '256.5194056339', 'Prec': '18.33', 'CO2': '313.065'}}}
# output_file = 'D:\Klemet\Desktop\TEMP\climate_data_output.txt'
# write_climate_data(climate_data, output_file)

def write_all_LANDIS_files(outputFolder, dataDict, copyNonParsedFiles = True):
    """
    Does the opposite of parse_All_LANDIS_PnET_Files : uses the dictionnary of
    LANDIS-II parameters produced by parse_All_LANDIS_PnET_Files to write down
    all of the files back into parameter files for LANDIS-II. Useful to read
    a "template" simulation, edit the parameters, and then write them back in
    another folder.
    
    The type of writing function to use is found by looking at which parameter
    file we're dealing with (through the LandisData parameter at the beginning
    of each file, which is kept in the dictionnary generated by parse_All_LANDIS_PnET_Files).
              
    ⚠ Files that were not parsed by parse_All_LANDIS_PnET_Files (and whose path
       is recorded in the output dict of parse_All_LANDIS_PnET_Files) are copied
       to the new output folder by default. Switch copyNonParsedFiles to false
       to disable.

    Parameters
    ----------
    outputFolder : String
        Folder where to write/copy the files to.
    dataDict : Dictionnary
        Data dict made by parse_All_LANDIS_PnET_Files (and which can be modified
        before using this function).
    copyNonParsedFiles : Boolean, default to True.
        Enables the copy of the files that were not parsed by parse_All_LANDIS_PnET_Files
        into the output folder.

    Returns
    -------
    None.

    """
    
    for filename in list(dataDict.keys()):
        
        if isinstance(dataDict[filename], dict):
            
            if "LandisData" in dataDict[filename]:

                # Main scenario file
                if dataDict[filename]["LandisData"] == "Scenario":
                    print("Found : Main scenario file : " + str(filename))
                    write_LANDIS_SimpleParameterFile(outputFolder + str(filename), dataDict[filename])
        
                # Main species parameter core file
                elif dataDict[filename]["LandisData"] == "Species":
                    print("Found : Main species parameter file : " + str(filename))
                    write_LANDIS_SpeciesCoreFile(outputFolder + str(filename), dataDict[filename])
        
                # Ecoregions file
                elif dataDict[filename]["LandisData"] == "Ecoregions":
                    print("Found : Main ecoregions file : " + str(filename))
                    write_LANDIS_MainEcoregionsFile(outputFolder + "ecoregion.txt", dataDict[filename])
        
                # Initial Communities file
                # Dealing with the .csv file is out of it if statement, see below
                elif dataDict[filename]["LandisData"] == "Initial Communities" or dataDict[filename]["LandisData"] == '"Initial Communities"':
                    print("Found : Initial Communities file : " + str(filename))
                    write_LANDIS_SimpleParameterFile(outputFolder + str(filename), dataDict[filename])
        
                # PnET Ecoregion parameter files
                elif dataDict[filename]["LandisData"] == "EcoregionParameters":
                    print("Found : PnET Ecoregion parameter file : " + str(filename))
                    write_PnET_ComplexTableParameterfile(outputFolder + str(filename), dataDict[filename])
        
                # PnET Succession Main file
                elif dataDict[filename]["LandisData"] == "PnET-Succession" or dataDict[filename]["LandisData"] == '"PnET-Succession"': 
                    print("Found : Main PnET parameter file : " + str(filename))
                    write_LANDIS_SimpleParameterFile(outputFolder + str(filename), dataDict[filename])
        
                # PnET Generic parameters file
                elif dataDict[filename]["LandisData"] == "PnETGenericParameters": 
                    print("Found : PnET generic parameter file : " + str(filename))
                    write_LANDIS_SimpleParameterFile(outputFolder + str(filename), dataDict[filename])
                    
                # PnET Species Parameter file
                elif dataDict[filename]["LandisData"] == "PnETSpeciesParameters":
                    print("Found : PnET species parameters file : " + str(filename))
                    write_PnET_ComplexTableParameterfile(outputFolder + str(filename), dataDict[filename])
        
                # PnET Output Biomass file
                elif dataDict[filename]["LandisData"] == "Output-PnET" or dataDict[filename]["LandisData"] == '"Output-PnET"':
                    print("Found : PnET OutputBiomass parameter file : " + str(filename))
                    write_LANDIS_SimpleParameterFile(outputFolder + str(filename), dataDict[filename])
        
                # PnET Output Sites File
                elif dataDict[filename]["LandisData"] == "PNEToutputsites": 
                    print("Found : PnET OutputSites parameter file : " + str(filename))
                    write_LANDIS_SimpleParameterFile(outputFolder + str(filename), dataDict[filename])
        
                # PnET Disturbance Reduction File
                elif dataDict[filename]["LandisData"] == "DisturbanceReductions":
                    print("Found : Disturbance Reduction File : " + str(filename))
                    write_PnET_ComplexTableParameterfile(outputFolder + str(filename), dataDict[filename])
            
            else: # If it's a dict but there is no LandisData parameter, it must be the climate file
                
                # Climate data file - identified by looking at the keys of the first month of the first year in the dict
                if  any(item in list(list(dataDict[filename].values())[0].values())[0].keys() for item in ["Year", "Month", "TMax", "TMin", "PAR", "Prec", "CO2"]):
                    print("Found : Climate file : " + str(filename))
                    write_climate_data(outputFolder + str(filename), dataDict[filename])
                    
                else:
                    print("WARNING : file type not recognized for the following dictionnary :" + str(dataDict[filename]))

        elif "DataFrame" in str(type(dataDict[filename])): # Dealing with dataframe (often loaded from .csv files) to output them back as .csv
                print("Found : dataframe : " + str(filename) + ", exporting to .csv")
                dataDict[filename].to_csv(outputFolder + str(filename), index=False)
            
        # If it's not a parameter file, we record its path
        # Can be used to copy/paste any other file
        else:
            print("Found : Additional file : " + str(filename))
            if os.path.isfile(dataDict[filename]):
                shutil.copy(dataDict[filename], outputFolder)
            elif os.path.isdir(dataDict[filename]):
                shutil.copytree(dataDict[filename], outputFolder + "/" + (os.path.basename(dataDict[filename])), dirs_exist_ok  = True)
            else:
                print("WARNING : " + str(filename) + " is neither a file or a folder, can't copy to " + str(outputFolder))
    

#%% FUNCTIONS TO LAUNCH A LANDIS-II SIMULATION

def runLANDIS_Simulation(simulationFolder, scenarioFileName, printSim = False, timeout = None):
    """
    Function used to run a LANDIS-II simulation from inside Jupyter notebook.
    ⚠ THIS WILL ONLY WORK IF JUPYTER NOTEBOOK/JUPYTER LAB HAS BEEN LAUNCHED
       FROM INSIDE LINUX. It is made to be used with the Docker image recommanded
       in the Readme of the repository.

    💡 There seems to be a bug in PnET where simulations can freeze when all cohorts
    have died, which sometimes happen during calibration runs. To bypass this and prevent
    the whole Python kernel from hanging, we use the Pexpect packages which allows us
    to create a "Timeout" error if LANDIS-II doesn't output any more characters for a while
    - as is the case for this type of situation. By default, the function will wait patiently
    for LANDIS-II to run.

    Parameters
    ----------
    simulationFolder : String
        The folder containing the simulation parameter files.
    scenarioFileName : TYPE
        The name of the String file containing the "main scenario parameters" for
        the LANDIS-II simulation (the scenario file).
    printSim : Boolean, optional
        Used to print the outputs of the sim (the LANDIS-II log). The default is False.
    timeout: int, optional
        Used to stop the sim if it's stuck. Time in second to wait for an update to the LANDIS-II output in the terminal. "None" will wait forever.

    Returns
    -------
    None.

    """
    
    # This command runs a LANDIS-II simulation based on the LANDIS-II installation of the docker image 
    # (located in /bin/LANDIS_Linux). printSim can be use so that the command prints things or not.
    
    # command = "cd " + simulationFolder + "; dotnet /bin/LANDIS_Linux/build/Release/Landis.Console.dll " + scenarioFileName
    command = '/bin/bash -c "cd '+ str(simulationFolder) + ' && dotnet $LANDIS_CONSOLE '+ str(scenarioFileName) + '"'
    # if printSim:
        # result = subprocess.run(command, shell=True)
        # print(result.stdout)
    # else:
        # subprocess.run(command, shell=True, stdout=subprocess.DEVNULL)

    try:
        # Spawn the command
        print("Launching LANDIS-II simulation with command : " + str(command))
        child = pexpect.spawn(command)
        
        # Read output until EOF or timeout
        output = ""
        foundError = False
        
        # print("Entering while loop")
        while True:
            try:
                # Read non-blocking with specified timeout
                data = child.read_nonblocking(size=1024, timeout=timeout)
                output += data.decode('utf-8')  # Decode bytes to string
                # print("While loop : output is " + str(output))
            except pexpect.exceptions.TIMEOUT:
                print("Timeout occurred while waiting for LANDIS-II output.")
                break  # Exit loop on timeout
            except pexpect.EOF:
                # Check the exit status
                child.close()
                if child.exitstatus != 0 or foundError:
                    print("The LANDIS-II simulation seem to have encountered an error; please check the landis log file for more details.")
                else:
                    print("The LANDIS-II simulation has finished properly!")
                break
        
            # Prints the last line
            if printSim:
                chunkText = output.splitlines()[-1]
                print(chunkText)
                if "ERROR" in chunkText.upper() or "EXCEPTION" in chunkText.upper() or "FAILED" in chunkText.upper():
                    foundError = True

    
    except Exception as e:
        return f"An error occurred: {str(e)}"


        

# runLANDIS_Simulation(testScenarioPath,
#                      "scenario.txt")

#%% FUNCTION TO PARSE OUTPUTS

def list_immediate_folders(directory):
    """
    Small function used in parse_outputRasters_PnET to return the folders in
    a given directory (but without going into subfolders).

    Parameters
    ----------
    directory : String
        Directory where the folders should be read.

    Returns
    -------
    list
        A list of directories directly inside (not recursive) the folder given as an argument.
    """
    return [folder for folder in os.listdir(directory) if os.path.isdir(os.path.join(directory, folder))]

def parse_outputRasters_PnET(base_folder, species_names, outputsUnitDict, cellLength, ecoregionsRasterPath):
    """
    Function used to parse the output rasters made with the PnET Output
    extension.
    
    ⚠ The function will consider that all rasters for one variable (e.g. AETAvg,
        LeafAreaIndex, etc.) will be in their own folder. Csv files will be ignored.
        Sorting of the raster file (and thus the order of the values) is based
        on the time step number in the name of each raster file.
    ⚠ The function expects that all raster files of a given variable are in their
       separate folder, and that all of the folders for the different variables 
       are inside one bigger folder, like this :
           
       output
         |
         |----FoliageBiomass
                  |
                  |---- pinubank
                  |       |
                  |       |------ FoliageBiomass-0.img
                  |       |------ FoliageBiomass-10.img
                  |       |------ ....
                  |       |------ FoliageBiomass-100.img
                  |
                  |---- querrubr
                  |       |
                  |       |------ FoliageBiomass-0.img
                  |       |------ FoliageBiomass-10.img
                  |       |------ ....
                  |       |------ FoliageBiomass-100.img
                  

    Parameters
    ----------
    base_folder : String
        Folder where to find the different folders where the variables are.
    species_names : List of strings
        List of species names/codes used for the simulation. Used to recognize
        the folders containing the outputs by species.
    outputsUnitDict : Dictionnary
        A dictionnary associating the name of an output variable (e.g. LeafAreaIndex)
        with its units (used for plotting, see plot_TimeSeries_RasterPnETOutputs)
        and also a keyword that tells us if the output should be summed (sum), averaged (average)
        converted in absolute biomass in metric tons (biomassSum) or skipped (any other value).
        Here is 3 entries for an example example of this dictionnary :
        outputsUnitDict = {"AETAvg":["average","mm of water"], # AETAvg is average monthly actual evapotranspiration across the last 12 months of the timestep
               "AgeDistribution":["sum","Number of cohorts in the landscape"], # Unique age cohorts in each cell
               "AnnualPsn":["average","Photosynthesis (g/m2)"], # Sum of monthly net photosynthesis)
               ...}
            
    cellLength : Int
        Length of a cell size used in the simulation from which the outputs have beem made.
        Used to compute the absolute biomass values in the landscape (as the original outputs
        are always expressed in g/m2 in LANDIS-II).
    ecoregionsRasterPath : String
        Path of the ecoregion raster used in the simulation. Necessary to identify the inactive
        cells (0). If other ecoregions IDs are inactive (never saw that before, but you never know),
        simply reclassify your raster before givng it to the function.

    Returns
    -------
    results : Dictionnary
        A dictionnary containing the outputs for each variable name (keys) in the
        form of a list of value (time series). Can be used directly with plot_TimeSeries_RasterPnETOutputs
        to plot the results.

    """
    
    results = {}

    # We load the ecoregion raster and reclassify it
    # Everything at 0 will be inactive
    # This is used for the averaging of some outputs
    with rasterio.open(ecoregionsRasterPath) as src:
        dataEcoregions = src.read(1)
        dataEcoregions = np.where(dataEcoregions > 0, 1, 0)

    # Walk through the base folder
    for subfolder in list_immediate_folders(base_folder):
        # print("Current subfolder : " + str(subfolder))
        
        files = glob.glob(base_folder + "/" + subfolder + "/*")  # Lists all files
        # print("Files detected by glob : " + str(files))
        
        raster_files = [f for f in files if f.endswith(('.img', '.tif', '.tiff'))]
        # print("Raster files found : " + str(raster_files))
        
        if raster_files:
            # If there are raster files in the subfolder
            time_series_sum = []
            for raster_file in sorted(raster_files, key=lambda x: int(''.join(filter(str.isdigit, x)))):
                with rasterio.open(raster_file) as src:
                    data = src.read(1)  # Read the first band
                    # We sum, we average, we do the biomass transform, or we skip depending on the output.
                    if subfolder not in outputsUnitDict:
                        print("Error with output subfolder " + subfolder + " : output name wasn't found in outputsUnitDict. Please check the dictionnary.")
                    elif outputsUnitDict[subfolder][0] == "average":
                        time_series_sum.append(np.nanmean(np.where(dataEcoregions == 1, data, np.nan)))  # Sum values, ignoring NaNs and every inactive cells (ecoregions = 0)
                    elif outputsUnitDict[subfolder][0] == "sum":
                        time_series_sum.append(np.nansum(data))  # Sum values, ignoring NaNs
                    elif outputsUnitDict[subfolder][0] == "biomassSum":
                        # Total biomass is to be expressed in Mega grams (metric tons) and for the whole landscape.
                        # Since ouputs are in g/m2, we convert all of the cells of the landscape to metric tons by multiplying them by the cell
                        # size (in m2, with is CellLength multiplied by itself) and by dividing by a million (to go from grams to tons).
                        time_series_sum.append(np.sum((data * (cellLength*cellLength))/1000000))
                                            
                    # Else (categorial, monthyl) : we skip it and simply do nothing.

            # If no values in the time series list, we skip
            if len(time_series_sum) > 0:
                results[os.path.basename(subfolder)] = time_series_sum
        
        else:
            # Check for subfolders and species names
            foundSpeciesName = False
            inner_subfolders = list_immediate_folders(base_folder + "/" + subfolder)
            # print("Subfolders found : " + str(inner_subfolders))
            
            for inner_subfolder in inner_subfolders:  # Get immediate subfolders
                if inner_subfolder in species_names:
                    foundSpeciesName = True

            if foundSpeciesName:
                species_results = dict()
                
                for inner_subfolder in inner_subfolders:  # Get immediate subfolders
                    # print("Current inner_subfolder : " + str(inner_subfolder))
                    
                    if inner_subfolder in species_names:
                        filesSubfolder = glob.glob(base_folder + "/" + subfolder + "/" + inner_subfolder + "/*")
                        inner_raster_files = [f for f in filesSubfolder if f.endswith(('.img', '.tif', '.tiff'))]
                        
                        if inner_raster_files:
                            species_time_series_sum = []
                            for raster_file in sorted(inner_raster_files, key=lambda x: int(''.join(filter(str.isdigit, x)))):
                                with rasterio.open(raster_file) as src:
                                    data = src.read(1)  # Read the first band
                                    # We sum, we average, we do the biomass transform, or we skip depending on the output.
                                    if subfolder not in outputsUnitDict:
                                        print("Error with output subfolder " + subfolder + " : output name wasn't found in outputsUnitDict. Please check the dictionnary.")
                                    elif outputsUnitDict[subfolder][0] == "average":
                                        species_time_series_sum.append(np.nanmean(np.where(dataEcoregions == 1, data, np.nan)))  # Sum values, ignoring NaNs and every inactive cells (ecoregions = 0)
                                    elif outputsUnitDict[subfolder][0] == "sum":
                                        species_time_series_sum.append(np.nansum(data))  # Sum values, ignoring NaNs
                                    elif outputsUnitDict[subfolder][0] == "biomassSum":
                                        # Total biomass is to be expressed in Mega grams (metric tons) and for the whole landscape.
                                        # Since ouputs are in g/m2, we convert all of the cells of the landscape to metric tons by multiplying them by the cell
                                        # size (in m2, with is CellLength multiplied by itself) and by dividing by a million (to go from grams to tons).
                                        species_time_series_sum.append(np.sum((data * (cellLength*cellLength))/1000000))

                            # If no value recorded, we skip.
                            if len(species_time_series_sum) > 0:
                                species_results[inner_subfolder] = species_time_series_sum

                # If no value recorded for any species, we skip.
                if species_results != {}:
                    results[os.path.basename(subfolder)] = species_results

    return results

# # Example usage
# species_list = ['AllSpecies', 'pinubank', 'querrubr']
# result_dict = parse_outputRasters_PnET("/calibrationFolder/StartingCalibrationTest/SimulationFiles/PnETGitHub_OneCellSim/output",
#                               species_list,
#                              outputsUnitDict,
#                              int(PnETGitHub_OneCellSim["scenario.txt"]["CellLength"]),
#                              PnETGitHub_OneCellSim["ecoregion.img"])
# print(result_dict)


def parse_CSVFiles_PnET_SitesOutput(folder_path, start_year):
    """
    This function parses the outputs generated with the extension PnET Sites Output.
    
    Only the files corresponding to cohorts (e.g. Cohort_pinubank_1931) are read;
    the other csv files (Establishment, Site) are skipped.
    
    💡 Can be used with plot_TimeSeries_CSV_PnETSitesOutputs to plot the outputs quickly.

    Parameters
    ----------
    folder_path : String
        The path of a folder that contains all of the .csv file for a given site
        generated by PnET Sites output (e.g. the folder in which the csv files
        for the cohorts are, lile Cohort_pinubank_1931.csv).
    start_year : int
        The starting year of the simulation. Used to compute the initial age
        of the cohorts..

    Returns
    -------
    dataframes : TYPE
        A dictionnary associating the identification of cohorts to the content
        of the csv file (as a panda dataframe) associated with the cohort.

    """
    
    # Initialize an empty dictionary to hold dataframes
    dataframes = {}
    
    # Use glob to find all CSV files in the specified folder
    csv_files = glob.glob(os.path.join(folder_path, "*.csv"))
    
    if len(csv_files) == 0:
        print("WARNING : No file found in " + str(folder_path) + ". Please check your path.")
    
    else:
        # Loop through the list of CSV files
        for file in csv_files:
            # Extract the filename without the path
            filename = os.path.basename(file)
            
            # Skip files named "Establishment.csv" or "Site.csv"
            if filename in ["Establishment.csv", "Site.csv"]:
                continue
            
            # Extract the year from the filename
            parts = filename.split('_')
            if len(parts) == 3 and parts[0] == "Cohort":
                cohort_name = parts[1]
                year = int(parts[2].split('.')[0])  # Get the year before the file extension
                
                # Calculate years at start
                years_at_start = start_year - year 
                
                # Create a key for the dictionary
                key = f"{cohort_name} - {years_at_start} years old at start"
                
                # Read the CSV file into a DataFrame
                df = pd.read_csv(file)
                
                # Store the DataFrame in the dictionary with the constructed key
                dataframes[key] = df
    
    return dataframes

# Example usage:
# dataframes_dict = parse_CSVFiles_PnET_SitesOutput("/calibrationFolder/StartingCalibrationTest/SimulationFiles/PnETGitHub_OneCellSim/output/Site1",
#                                  int(PnETGitHub_OneCellSim["pnetsuccession.txt"]["StartYear"]))

# print(dataframes_dict)

#%% FUNCTIONS TO PLOT THE VARIABLES

def plot_TimeSeries_RasterPnETOutputs(data_dict, outputs_unit_dict, timestep, simulationDuration):
    """
    Plots the summarized outputs made by the extension PnET output and read
    by the function parse_outputRasters_PnET as Matplotlib plots.

    Parameters
    ----------
    data_dict : Dictionnary
        Dictionnary generated with parse_outputRasters_PnET.
    outputs_unit_dict : Dictionnary
        A dictionnary associating the name of an output variable (e.g. LeafAreaIndex)
        with its units (used for plotting, see plot_TimeSeries_RasterPnETOutputs)
        and also a keyword that tells us if the output should be summed (sum), averaged (average)
        converted in absolute biomass in metric tons (biomassSum) or skipped (any other value).
        Here is 3 entries for an example example of this dictionnary :
        outputsUnitDict = {"AETAvg":["average","mm of water"], # AETAvg is average monthly actual evapotranspiration across the last 12 months of the timestep
               "AgeDistribution":["sum","Number of cohorts in the landscape"], # Unique age cohorts in each cell
               "AnnualPsn":["average","Photosynthesis (g/m2)"], # Sum of monthly net photosynthesis)
               ...}
        This dictionnary is also used with parse_outputRasters_PnET. See help(parse_outputRasters_PnET).
    timestep : int
        Time step between values in the outputs, in years.
    simulationDuration : int
        Duration in years of the simulation.

    Returns
    -------
    None.

    """
    
    
    # Define time steps (0 to 50 years)
    years = np.arange(0, simulationDuration + timestep, timestep)

    for output_key, (operation, description) in outputs_unit_dict.items():
        if output_key in data_dict:
            output_data = data_dict[output_key]
            
            # Check if the data is a dictionary (nested structure)
            if isinstance(output_data, dict):
                for species_key, species_data in output_data.items():
                    plt.figure(figsize=(10, 5))
                    plt.plot(years, species_data, marker='o', label=species_key)
                    plt.title(f"{output_key} for {species_key}")
                    plt.xlabel("Years")
                    plt.ylabel(description)
                    plt.xticks(years)
                    plt.ylim(0)
                    plt.grid()
                    plt.legend()
                    plt.show()
            else:
                # If it's not a nested dictionary
                plt.figure(figsize=(10, 5))
                plt.plot(years, output_data, marker='o')
                plt.title(output_key)
                plt.xlabel("Years")
                plt.ylabel(description)
                plt.xticks(years)
                plt.ylim(0)
                plt.grid()
                plt.show()

# Example usage:
# plot_TimeSeries_RasterPnETOutputs(result_dict, outputsUnitDict,
#                 int(PnETGitHub_OneCellSim["pnetsuccession.txt"]["Timestep"]),
#                 int(PnETGitHub_OneCellSim["scenario.txt"]["Duration"]))


def plot_TimeSeries_CSV_PnETSitesOutputs(df, referenceDict = {}, columnToPlotSelector = [], trueTime = False, realBiomass = True, cellLength = 30, referenceLabel = "Reference - FVS", labelOfFirstCurve = ""):
    """
    Plots the outputs made by the extension PnET Sites output and read
    by the function parse_CSVFiles_PnET_SitesOutput as Matplotlib plots.

    Parameters
    ----------
    df : Dataframe
        One of the dataframe that is inside the dictionnary generated by
        parse_CSVFiles_PnET_SitesOutput, and corresponding to age cohort of
        interest.
    referenceDict : TYPE, optional
        A dictionnary containing "reference values" to plot for some
        variable. The keys should be variable names (corresponding to columns in
        df) which brings to a nested dictionnary with 2 entries : "Time" contains
        the time for each reference value, and "Values" contains the values corresponding
        to each time. The default is {}.
        Example : {"Wood(gDW)":{"Time":range(0, 70, 10), "Values":[1, 3, 3, 3, 3, 4, 5]}}
        UPDATE : Can now also be a list to display several curves.
    columnToPlotSelector : List of strings, optional
        Contains column names from df that should be plotted; the rest is then ignored.
        The default is [], which plots all columns.
    trueTime : Booelan, optional
        If True, then the time on the plot will be the years of the simulation - 
        starting, for example, in 1992. The default is False, which will lead to the
        years in the plot being from 0 to the last year of the simulation.
    realBiomass : Booelan, optional
        If true, the measures of biomass in df (which will be Wood(gWD), Root(gWD) and Fol(gWD))
        will be changed from g/m2 into absolute values in Mg (tons) for the site. The default is True.
    cellLength : Int, optional
        The length of a cell size. MUST BE PROVIDED IF realBiomass = True. The default is 30.
    referenceLabel: String, optional
        Labels the reference curve. If referenceDict is a list with several reference curves,
        then referenceLabel must be a list too.
    labelOfFirstCurve: String, optional
        Label of the first curve (made with data from object df). If left empty, then the name of the variable given in columnToPlotSelector is used.

    Returns
    -------
    None.

    """
    
    # Convert 'Time' to numeric type
    df['Time'] = pd.to_numeric(df['Time'])

    # Get the list of columns to plot (excluding Time, Year, and Month)
    if len(columnToPlotSelector) > 0:
        columns_to_plot = columnToPlotSelector
    else: # If not selector is given to the function, then we plot all of the columns
        columns_to_plot = [col for col in df.columns if col not in ['Time', 'Year', 'Month', 'Unnamed', 'Age(yr)']]

    # Create a plot for each variable
    for column in columns_to_plot:
        plt.figure(figsize=(12, 6))

        columnName = column

        # We edit the measures of biomass to put them not as g/m2, but tons in total for the site
        if realBiomass and (column == "Wood(gDW)" or column == "Root(gDW)" or column == "Fol(gDW)" or column == "WoodAndLeaves(gDW)"):
            columnData = df[column] * (cellLength*cellLength)
            columnData = columnData / 1000000
            columnName = column[:-5] + " (Mg or Metric tons)"
        else:
            columnData = df[column]

        # We prepare the label of the first curve if it is given
        if labelOfFirstCurve == "":
            labelFirstCurve = "PnET Succession - " + str(column)
        else:
            labelFirstCurve = labelOfFirstCurve
        
        if trueTime:
            plt.plot(df['Time'], columnData, label = labelFirstCurve)
        else: #We edit the time to remove the years (e.g. 2000, 2001, etc.) and just use 0 as starting year. Makes things easier for the reference curve.
            timeNormalized = df['Time'] - min(df['Time'])
            plt.plot(timeNormalized, columnData, label = labelFirstCurve)
            
        # If reference curve exists for the variable, we display it on the curve

        # If we gave a list or dictionnary, we'll display them
        if type(referenceDict) is list:
            if type(referenceLabel) is not list or (type(referenceLabel) is list and len(referenceDict) != len (referenceLabel)):
                raise Exception("If referenceDict contains a list of curves, then referenceLabel must contain a list of label of the same size") 
            cmap = plt.get_cmap('viridis', len(referenceDict))
            coloursForCurve = cmap(np.linspace(0, 1, len(referenceDict)))
            color_iterator = iter(coloursForCurve)
            labelIterator = iter(referenceLabel)
            for listRefDict in referenceDict:
                if column in listRefDict:
                    colourForCurve = next(color_iterator)
                    labelForCurve = next(labelIterator)
                    plt.plot(listRefDict[column]["Time"], listRefDict[column]["Values"], color=colourForCurve, label =  labelForCurve)
        else:
            if column in referenceDict:
                plt.plot(referenceDict[column]["Time"], referenceDict[column]["Values"], color='#ebcb8b', label = referenceLabel)
        
        # Set the x-axis ticks to increment by 10 years
        if trueTime:
            start_year = int(df['Time'].min())
            end_year = int(df['Time'].max()) + 1
            plt.xticks(np.arange(start_year, end_year, 10))
        else:
            start_year = int(min(timeNormalized))
            end_year = int(max(timeNormalized)) + 1
            plt.xticks(np.arange(start_year, end_year, 10))
        
        # Set the y-axis to start at 0
        plt.ylim(bottom=0)
        
        # Set labels and title
        plt.xlabel('Time (Years)')
        plt.ylabel(columnName)
        plt.title(f'Time Series of {columnName}')
        
        # Add grid for better readability
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45)

        # Puts a legend
        plt.legend(loc='upper right')
        
        # Adjust layout to prevent cutting off labels
        plt.tight_layout()


    print("All plots have been generated and saved.")

# # Call the function
# referenceBiomassDict = {"Wood(gDW)":{"Time":range(0, 70, 10), "Values":[1, 3, 3, 3, 3, 4, 5]}}
# plot_TimeSeries_CSV_PnETSitesOutputs(dataframes_dict["querrubr - 9 years old at start"],
#                     referenceBiomassDict,
#                     ["Wood(gDW)"],
#                     trueTime = False,
#                     realBiomass = True,
#                     cellLength = int(PnETGitHub_OneCellSim["scenario.txt"]["CellLength"]))

#%% FUNCTIONS TO LAUNCH AND PARSE A FVS (FOREST VEGETATION SIMULATOR) SIMULATION

def FVS_on_simulationOnSingleEmptyStand(Latitude,
                                        Longitude,
                                        Slope,
                                        Elevation,
                                        treeSpeciesCode,
                                        treesPerHectares,
                                        siteIndex,
                                        variant = "FVSon",
                                        Max_BA = "",
                                        timestep = 10,
                                        numberOfTimesteps = 12,
                                        outputFormatBiomass = "Metric tons per hectares",
                                        outputFormatYears = "Real date",
                                        folderForFiles = "/tmp/FVS_SingleEmptyStandRun",
                                        clearFiles = True,
                                        printOutput = False):


    # Check if the directory exists
    if os.path.exists(folderForFiles):
        # Remove the directory and all its contents
        shutil.rmtree(folderForFiles)
        print(f"The directory '{folderForFiles}' has been deleted.")
    
    # Create the directory
    os.makedirs(folderForFiles)
    print(f"The directory '{folderForFiles}' has been created.")

    print("Creating Database with stand and tree ini values")
    # Connect to the SQLite database (or create it if it doesn't exist)
    conn = sqlite3.connect(folderForFiles + "/FVSData.db")
    cursor = conn.cursor()
    
    # Create table FVS_StandInit
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS "FVS_StandInit" (
            	"Stand_CN"	TEXT,
            	"Stand_ID"	TEXT,
            	"Variant"	TEXT,
            	"Inv_Year"	INTEGER,
            	"Groups"	REAL,
            	"AddFiles"	TEXT,
            	"FVSKeywords"	TEXT,
            	"Latitude"	REAL,
            	"Longitude"	REAL,
            	"Region"	INTEGER,
            	"Forest"	INTEGER,
            	"District"	INTEGER,
            	"Compartment"	INTEGER,
            	"Location"	INTEGER,
            	"Ecoregion"	TEXT,
            	"BEC"	TEXT,
            	"PV_Code"	TEXT,
            	"PV_Ref_Code"	INTEGER,
            	"Age"	INTEGER,
            	"Aspect"	REAL,
            	"Slope"	REAL,
            	"Elevation"	REAL,
            	"ElevFt"	REAL,
            	"Basal_Area_Factor"	REAL,
            	"Inv_Plot_Size"	REAL,
            	"Brk_DBH"	TEXT,
            	"Num_Plots"	INTEGER,
            	"NonStk_Plots"	INTEGER,
            	"Sam_Wt"	REAL,
            	"Stk_Pcnt"	REAL,
            	"DG_Trans"	INTEGER,
            	"DG_Measure"	INTEGER,
            	"HTG_Trans"	INTEGER,
            	"HTG_Measure"	INTEGER,
            	"Mort_Measure"	INTEGER,
            	"Max_BA"	REAL,
            	"Max_SDI"	REAL,
            	"Site_Species"	TEXT,
            	"Site_Index"	REAL,
            	"Model_Type"	INTEGER,
            	"Physio_Region"	INTEGER,
            	"Forest_Type"	INTEGER,
            	"State"	INTEGER,
            	"County"	INTEGER,
            	"Fuel_Model"	INTEGER,
            	"Fuel_0_25_H"	REAL,
            	"Fuel_25_1_H"	REAL,
            	"Fuel_1_3_H"	REAL,
            	"Fuel_3_6_H"	REAL,
            	"Fuel_6_12_H"	REAL,
            	"Fuel_12_20_H"	REAL,
            	"Fuel_20_35_H"	REAL,
            	"Fuel_35_50_H"	REAL,
            	"Fuel_gt_50_H"	REAL,
            	"Fuel_0_25_S"	REAL,
            	"Fuel_25_1_S"	REAL,
            	"Fuel_1_3_S"	REAL,
            	"Fuel_3_6_S"	REAL,
            	"Fuel_6_12_S"	REAL,
            	"Fuel_12_20_S"	REAL,
            	"Fuel_20_35_S"	REAL,
            	"Fuel_35_50_S"	REAL,
            	"Fuel_gt_50_S"	REAL,
            	"Fuel_Litter"	REAL,
            	"Fuel_Duff"	REAL,
            	"Fuel_0_06_H"	REAL,
            	"Fuel_06_25_H"	REAL,
            	"Fuel_25_76_H"	REAL,
            	"Fuel_76_152_H"	REAL,
            	"Fuel_152_305_H"	REAL,
            	"Fuel_305_508_H"	BLOB,
            	"Fuel_508_889_H"	REAL,
            	"Fuel_889_1270_H"	REAL,
            	"Fuel_gt_1270_H"	REAL,
            	"Fuel_0_06_S"	REAL,
            	"Fuel_06_25_S"	REAL,
            	"Fuel_25_76_S"	REAL,
            	"Fuel_76_152_S"	REAL,
            	"Fuel_152_305_S"	REAL,
            	"Fuel_305_508_S"	REAL,
            	"Fuel_508_889_S"	REAL,
            	"Fuel_889_1270_S"	REAL,
            	"Fuel_gt_1270_S"	REAL,
            	"Photo_Ref"	INTEGER,
            	"Photo_code"	TEXT,
            	"Moisture"	REAL
            )
    ''')
    
    # Create table FVS_TreeInit
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS "FVS_TreeInit" (
                        	"Stand_CN"	TEXT,
                        	"Stand_ID"	TEXT,
                        	"StandPlot_CN"	TEXT,
                        	"StandPlot_ID"	TEXT,
                        	"Plot_ID"	INTEGER,
                        	"Tree_ID"	INTEGER,
                        	"Tree_Count"	REAL,
                        	"History"	INTEGER,
                        	"Species"	TEXT,
                        	"DBH"	REAL,
                        	"DG"	REAL,
                        	"Ht"	REAL,
                        	"HTG"	REAL,
                        	"HtTopK"	REAL,
                        	"CrRatio"	INTEGER,
                        	"Damage1"	INTEGER,
                        	"Severity1"	INTEGER,
                        	"Damage2"	INTEGER,
                        	"Severity2"	INTEGER,
                        	"Damage3"	INTEGER,
                        	"Severity3"	INTEGER,
                        	"TreeValue"	INTEGER,
                        	"Prescription"	INTEGER,
                        	"Age"	INTEGER,
                        	"Slope"	INTEGER,
                        	"Aspect"	INTEGER,
                        	"PV_Code"	INTEGER,
                        	"TopoCode"	INTEGER,
                        	"SitePrep"	INTEGER
                        )
    ''')
    
    # Inserting Stand Row
    data_to_insert = {
    'Stand_CN': 'STAND_EMPTY',
    'Stand_ID': 'STAND_EMPTY',
    'Variant': str(variant[-2:]),
    'Inv_Year': 2023,
    'Groups': 'All_Stands',
    'Latitude': str(Latitude),
    'Longitude': str(Longitude),
    'Age': 0,
    'Aspect': 0,
    'Slope': str(Slope),
    'Elevation': str(Elevation),
    'Basal_Area_Factor': -1.0, # Used to control number of tree in stands.
    'Inv_Plot_Size': 1.0, # Used to control number of trees in stand.
    'Num_Plots': 0,
    'Site_Species': str(treeSpeciesCode),
    'Site_Index': str(siteIndex),
    'Max_BA': str(Max_BA)
    }

    columns = ', '.join(data_to_insert.keys())
    values = ', '.join(['%s'] * len(data_to_insert))
    sql = "INSERT INTO FVS_StandInit (" + str(columns) + ") VALUES" + str(tuple(data_to_insert.values()))

    # print(sql)

    cursor.execute(sql)
    # print(f"{cursor.rowcount} record inserted in database.")

    # WARNING : It looks like the tree density is calculated differently from one variant to the other.
    # In particular, it seems like the american variants of FVS will use the number of trees given in the stand
    # as the density/acre, while FVS Ontario will use it as density/hectare. I tested almost all american variants,
    # and this is to be that every time.
    # We make the change here.
    if variant != "FVSon" and variant != "FVSbc" : # The two canadian variants seem to only deal in Ha
        numberOfTrees = treesPerHectares * 0.4046856422
    else:
        numberOfTrees = treesPerHectares
    
    # Inserting Tree row
    data_to_insert = {
    'Stand_CN': 'STAND_EMPTY',
    'Stand_ID': 'STAND_EMPTY',
    'Plot_ID': 1,
    'Tree_ID': 1,
    'Tree_Count': numberOfTrees,
    'History': 1, # Values 1-5 doesn't seem to change anything. We'll use that.
    'Species' : str(treeSpeciesCode),
    'DBH' : 0.5 # Used to initialize young trees
    }

    columns = ', '.join(data_to_insert.keys())
    values = ', '.join(['%s'] * len(data_to_insert))
    sql = "INSERT INTO FVS_TreeInit (" + str(columns) + ") VALUES" + str(tuple(data_to_insert.values()))

    # print(sql)

    cursor.execute(sql)
    # print(f"{cursor.rowcount} record inserted in database.")

    # Commit changes to SQL database and close the connection
    conn.commit()
    conn.close()

    # Creating Keyword file
    print("Creating Keyword file")
    
    Keywordfile_content = """STDIDENT
STAND_EMPTY

ECHOSUM
SCREEN

DATABASE
DSNin
FVSData.db
StandSQL
SELECT * FROM FVS_StandInit WHERE Stand_ID = 'STAND_EMPTY'
EndSQL
TreeSQL
SELECT * FROM FVS_TreeInit WHERE Stand_ID = 'STAND_EMPTY'
EndSQL
End

TIMEINT           0      INSERT_TIMEINT
NUMCYCLE          INSERT_NUMCYCLE

ESTAB           2023
STOCKADJ          -1
END

TREELIST           0
CUTLIST            0

COMMENT
The following lines are used to produce a "Carbon report" with the Fire and Fuel extension of FVS.
See user guide for more : https://www.fs.usda.gov/fmsc/ftp/fvs/docs/gtr/FFEguide.pdf
CARBCALC 1 1 is used to say that we want the report in metric tons/ha (not us tons), and that a more refined algorithm for biomass estimation be used (Jenkins algorithm, which estimates bark biomass in contrast to the regular algorithm).
CARBREPT is used to output the report in the main output (.out) file of the simulation.
END

FMIN
CARBCALC 1 1
CARBREPT
END

PROCESS
STOP
"""
    Keywordfile_content = Keywordfile_content.replace("INSERT_TIMEINT", str(timestep))
    Keywordfile_content = Keywordfile_content.replace("INSERT_NUMCYCLE", str(numberOfTimesteps))
    
    with open(folderForFiles + "/SingleStandSim_Keywords.key", 'w', encoding='utf-8') as file:
        file.write(Keywordfile_content)

    # Launching the sim
    print("Launching FVS sim")
    result = subprocess.run([variant, '--keywordfile=SingleStandSim_Keywords.key'], cwd=folderForFiles, capture_output=True, text=True)
    if printOutput:
        print(result.stdout)

    # print("WARNING : the content of the output dictionnary of this function can differ slightly from what is printed above (console outputs of FVS). It seems that the console output can round certain variables differently than what is in the .out files (that will be read to create the dictionnary). Differences should be minimal.\n\n")

    # Reading the Gross Total Volume as Output - outdated
    # print("Reading Outputs")
    # mapping = {}
    # ignoreFirstLine = True
    # with open(folderForFiles + "/SingleStandSim_Keywords.sum", 'r') as file:
        # for line in file:
            # if ignoreFirstLine:
                # ignoreFirstLine = False
            # else:
                # Split the line into parts based on whitespace
                # parts = line.split()
                
                # Check if there are enough parts to avoid IndexError
                # if len(parts) > 8:
                    # Get the number from the first column (index 0)
                    # key = int(parts[0])
                    # Get the number from the ninth column (index 8)
                    # value = int(parts[8])
                    
                    # Add to dictionary
                    # mapping[key] = value

    # Reading the total stand carbon output and converting it into biomass
    # The carbon outputs can only be outputed in the .out main report file. Not ideal, but we can get it there.
    mapping = {}
    
    # Define the target string to search for
    target_string_carbonReport = "******  CARBON REPORT VERSION 1.0 ******"
    target_string_carbonUnits = "ALL VARIABLES ARE REPORTED IN"
    target_string_lastLineBeforeValues = "YEAR    Total    Merch     Live     Dead     Dead      DDW    Floor  Shb/Hrb   Carbon   Carbon  from Fire"
    target_dashesLine = "--------------------------------------------------------------------------------------------------------------"

    with open(folderForFiles + "/SingleStandSim_Keywords.out", 'r') as file:
        lines = file.readlines()

    # Initialize variables to track whether we've found the target string and to collect report lines
    found_target_carbonReport = False
    found_target_carbonUnits = False
    found_target_lastLineBeforeValues = False
    found_target_lastDashesLine = False
    report_lines = []

    # The loops will look in every lines until we find the mention of the carbon report,
    # and the lines indicating the beginning of the value table.
    for line in lines:
        if not found_target_carbonReport:
            if target_string_carbonReport in line:
                found_target_carbonReport = True
        else:
            if not found_target_carbonUnits:
                if target_string_carbonUnits in line:
                    found_target_carbonUnits = True
                    if "TONS/ACRE" in line :
                        carbonUnits = "TONS/ACRE"
                        print("Detected unit in carbon outputs is US Tons/Acre. Will transform into Metric ton / hectare.")
                    elif "METRIC TONS/HECTARE" in line:
                        carbonUnits = "METRIC TONS/HECTARE"
                    else:
                        raise Exception("Carbon units not properly detected in .out file of FVS. Please check the file; should be TONS/ACRES or METRIC TONS/HECTARE.")
            else:
                if not found_target_lastLineBeforeValues:
                    if target_string_lastLineBeforeValues in line:
                        found_target_lastLineBeforeValues = True
                else:
                    if not found_target_lastDashesLine:
                        if target_dashesLine in line:
                            found_target_lastDashesLine = True
                    else:
                        if re.match(r'^\s*-\s*$', line):
                            break
                        else:
                            # Split the line into parts based on whitespace
                            parts = line.split()
                            
                            # Check if there are enough parts to avoid IndexError
                            if len(parts) > 8:
                                # Get the number from the first column (index 0)
                                key = int(parts[0])
                                # Get the number from the second column (index 1), stand Aboveground live carbon
                                value = float(parts[1])
                                
                                # Add to dictionary
                                mapping[key] = value    

    # To convert carbon back to biomass : The algorithm of the Fire and Fuel extension computes biomass, but only outputs carbon (no option for outputting biomass).
    # However, indicates the convertion factor. Quote :
    # "Biomass, expressed as dry weight, is assumed to be 50 percent carbon (Penman et al. 2003) for all pools except forest floor, which is estimated as 37 percent carbon (Smith and Heath 2002)"
    # This is confirmed in the source code; see https://github.com/USDAForestService/ForestVegetationSimulator/blob/5c29887e4168fd8182c1b2bad762f900b7d7e90c/archive/bgc/src/binitial.f#L28
    # Therefore, to get biomass from carbon outputs, we just have to multiply it by 2.
    for key in mapping.keys():
        mapping[key] = mapping[key]*2

    # There is a big issue when choosing the different variants : sometimes, the keyword CARBCALC 1 1 - which indicates among other things that
    # we want an output of carbon in Metric tons per hectare rather than US tons / acre - does not work ! 
    # So here, we check and make the transformation to Metric tons per hectares if needed.
    if carbonUnits == "TONS/ACRE":
        for key in mapping.keys():
            # Metric Tons per Hectare = US Tons per Acre×2.24127
            mapping[key] = mapping[key]*2.24127
            
    # Delete the folder created for the inputs and outputs if specified
    if clearFiles:
        print("Clearing files")
        shutil.rmtree(folderForFiles)
        print(f"The directory '{folderForFiles}' has been deleted.")

    # We end up by converting things according to the arguments of the function
    if outputFormatBiomass != "Metric tons per hectares" and outputFormatBiomass != "Metric gram per meter squared":
        raise Exception("outputFormatBiomass must be either 'Metric tons per hectares' or 'Metric gram per meter squared'.") 
    if outputFormatYears != "Real date" and outputFormatYears != "Start at 0":
        raise Exception("outputFormatYears must be either 'Real date' or 'Start at 0'.")

    if outputFormatYears == "Start at 0":
        initialKeys = list(mapping.keys())
        beginningYear = min(initialKeys)
        for key in initialKeys:
            mapping[key - beginningYear] = mapping[key]
            del mapping[key]

    if outputFormatBiomass == "Metric gram per meter squared":
        for key in mapping.keys():
            # Mg to gram
            mapping[key] = mapping[key] * 1000000
            # Hectare to m2
            mapping[key] = mapping[key]/10000

    return(mapping)

#%% MISC FUNCTIONS

def read_markdown_cell(notebook_path, markdownCellNumber):
    """Reads the markdown from a given cell in a jupyter notebook.
    Usefull to get the tables of parameters, so that we can transform
    them back into Python objects to deal with.
    
    The markdownCellNumber parameter starts at 1 for the first markdown cell
    of the notebook. Code cells or raw cells are not counted."""
    
    # Load the notebook
    with open(notebook_path, 'r') as f:
        notebook = nbformat.read(f, as_version=4)

    # Iterate through cells to find the last markdown cell
    testMarkdown = 1
    for i in range(0, len(notebook.cells) - 1):
        if notebook.cells[i].cell_type == 'markdown':
            if testMarkdown == markdownCellNumber:
                return notebook.cells[i].source  # Return the markdown content
            else:
                testMarkdown += 1
    return None  # Return None if no markdown cell is found


def extract_table(text):
    """Extract markdown table from text.

    WARNING : The function is pretty simple, and will just take anything
    between the first | and last | character. As such, don't put several
    markdwon table in the string of text you want to analyze, or don't use |
    for other things in your string."""
    
    # Find the index of the first "|" character
    start_index = text.find("|")
    # Find the index of the last "|" character
    end_index = text.rfind("|")
    
    # Check if both indices are valid
    if start_index != -1 and end_index != -1 and start_index < end_index:
        # Extract and return the substring between the two indices
        return text[start_index:end_index + 1].strip()
    else:
        return "No valid table found."

def parseTableSpeciesParameters(markdown_table):
    """Parse a markdown table like the ones used in 3.Initial_Species_Parameters.ipynb
    into a Python Dictionnary we can then use for the previous functions that write LANDIS-II
    scenario files (see above)."""
    
    # Split the input into lines
    lines = markdown_table.strip().split('\n')
    
    # Extract headers
    headers = [header.strip('* ').strip("`") for header in lines[0].strip('| ').split('|')]
    headers = [header for header in headers if header]  # Remove empty headers
    
    # Initialize an empty dictionary to hold the results
    species_data = {}
    
    # Iterate through the rows of the table, starting from the second row
    for line in lines[2:]:
        # Split the line into columns
        columns = [col.strip('* ') for col in line.strip('| ').split('|')]
        species_name = re.sub(r'\s*\[\^.*?\]', '', columns[0]).strip()  # Remove footnotes and asterisks
        
        # Create a dictionary for the species
        species_info = {}
        
        # Iterate over the remaining columns and associate them with headers
        for header, value in zip(headers[1:], columns[1:]):
            # clean_value = re.sub(r'\s*\[\^.*?\]', '', value).strip()  # Remove footnotes
            cleaned_value = re.match(r'(\S+)', value) # Clean the value to extract only the number at the beginning
            if cleaned_value:
                species_info[header] = cleaned_value.group(1)
        
        # Add the species info to the main dictionary
        species_data[species_name] = species_info
    
    return species_data

def parseTableGenericParameters(markdown_table: str) -> dict:
    """Same as parse_markdown_table_core_species_parameters,
    but for tables where we have one value per parameter, for
    the generic parameters, where there are no variations by species."""
    
    # Split the input string into lines
    lines = markdown_table.strip().split('\n')
    
    # Extract headers from the first line (the one with backticks)
    headers = [value.strip("`") for value in re.split(r'\s*\|\s*', lines[0]) if value.strip()]

    # Initialize a dictionary to hold the results
    result = {}
    
    # Process each row of data (starting from the second line)
    for line in lines[2:]:
        # Extract values from the row using a regex that captures all columns
        values = [value.strip() for value in re.split(r'\s*\|\s*', line) if value.strip()]
        
        # If there are values, process them
        if len(values) == len(headers):  # Ensure we have the same number of values as headers
            for i, value in enumerate(values):
                # Clean the value to extract only the number at the beginning
                cleaned_value = re.match(r'(\S+)', value)
                if cleaned_value:
                    result[headers[i]] = cleaned_value.group(1)
        else:
            print("ERROR : Mismatch between number of headers of table and values")

    return result

def replace_in_dict(d, old_str, new_str):
    """Function to replace every key or value string in a dictionnary
    with another. Used to change the name of species to their species code"""
    if isinstance(d, dict):
        return {replace_in_dict(k, old_str, new_str): replace_in_dict(v, old_str, new_str) for k, v in d.items()}
    elif isinstance(d, list):
        return [replace_in_dict(item, old_str, new_str) for item in d]
    elif isinstance(d, str):
        return d.replace(old_str, new_str)
    else:
        return d

# Functions to transform kelvins to celciuses
def kelvin_to_celsius(kelvin):
    if isinstance(kelvin, list):
        return [temp - 273.15 for temp in kelvin]
    else:
        return kelvin - 273.15

# Function to load and filter data in a .nc file by polygon
def load_and_filter_by_polygon(file_path, shapefile_path):
    # Load the shapefile
    gdf = gpd.read_file(shapefile_path)
    # Ensure the shapefile is in the same CRS as the climate data (usually EPSG:4326)
    if gdf.crs != 'EPSG:4326':
        gdf = gdf.to_crs('EPSG:4326')
    # Get the polygon from the shapefile (assuming first polygon if multiple)
    polygon = gdf.geometry.iloc[0]
    # Load the climate data
    ds = xr.open_dataset(file_path)
    # Create a mask for points inside the polygon
    # We try with different attribute names
    try:
        lon_grid, lat_grid = np.meshgrid(ds.lon.values, ds.lat.values)
        mask = np.zeros((len(ds.lat), len(ds.lon)), dtype=bool)
        for i in range(len(ds.lat)):
            for j in range(len(ds.lon)):
                point = Point(lon_grid[i, j], lat_grid[i, j])
                mask[i, j] = polygon.contains(point)
        # Apply the mask to the dataset
        # First, we need to convert the mask to have the same dimensions as the dataset
        mask_da = xr.DataArray(mask, coords=[ds.lat, ds.lon], dims=['lat', 'lon'])

    # In some cases, lat and lon are not the names used for the dimensions. We try another.
    except:
        # Code to handle the exception
        print("lat and lon not found as attributes of ds. trying other")
        lon_grid, lat_grid = np.meshgrid(ds.LonDim.values, ds.LatDim.values)
        mask = np.zeros((len(ds.LatDim), len(ds.LonDim)), dtype=bool)
        for i in range(len(ds.LatDim)):
            for j in range(len(ds.LonDim)):
                point = Point(lon_grid[i, j], lat_grid[i, j])
                mask[i, j] = polygon.contains(point)
        # Apply the mask to the dataset
        # First, we need to convert the mask to have the same dimensions as the dataset
        mask_da = xr.DataArray(mask, coords=[ds.LatDim, ds.LonDim], dims=['LatDim', 'LonDim'])
        print(mask_da)

    # Apply the mask
    ds_filtered = ds.where(mask_da, drop=True)
    return ds_filtered

# Progress bar for downloads of some files with urllib.request
def progress_hook(count, block_size, total_size):
    """
    A callback function for urlretrieve that displays a progress bar.

    Args:
        count: The count of blocks transferred so far
        block_size: The size of each block in bytes
        total_size: The total size of the file in bytes
    """
    downloaded = count * block_size
    percent = min(int(downloaded * 100 / total_size), 100)

    # Create a simple progress bar
    bar_length = 50
    filled_length = int(bar_length * percent // 100)
    bar = '█' * filled_length + '-' * (bar_length - filled_length)

    # Print the progress bar
    sys.stdout.write(f'\r|{bar}| {percent}% Complete ({downloaded}/{total_size} bytes)')
    sys.stdout.flush()

    # Add a newline when download completes
    if downloaded >= total_size:
        sys.stdout.write('\n')

# Function to process CO2 data and fill the dataframe
# Based on datasets from https://zenodo.org/records/5021361
def process_co2_data(historical_ds, shapefile, df):
    # Add CO2_Concentration column to dataframe
    df['CO2_Concentration'] = np.nan

    # Load the shapefile
    gdf = gpd.read_file(shapefile)
    # Ensure the shapefile is in the same CRS as the climate data (usually EPSG:4326)
    if gdf.crs != 'EPSG:4326':
        gdf = gdf.to_crs('EPSG:4326')
    # Get the polygon from the shapefile (assuming first polygon if multiple)
    polygon = gdf.geometry.iloc[0]
    # Get the centroid of the polygon
    polygon_centroid = polygon.centroid

    # Process historical data (1950-2013)
    # Create a mask for points inside the polygon for historical data
    lon_values = historical_ds.Longitude.values
    lat_values = historical_ds.Latitude.values
    lon_grid, lat_grid = np.meshgrid(lon_values, lat_values)
    hist_mask = np.zeros((len(lat_values), len(lon_values)), dtype=bool)

    # Check each point if it's inside the polygon
    for i in range(len(lat_values)):
        for j in range(len(lon_values)):
            point = Point(lon_grid[i, j], lat_grid[i, j])
            hist_mask[i, j] = polygon.contains(point)

    # If no points are inside the polygon, find the closest point to the polygon centroid
    if not np.any(hist_mask):
        print("No points found inside the polygon for historical data. Finding closest point...")
        min_dist = float('inf')
        closest_i, closest_j = 0, 0

        for i in range(len(lat_values)):
            for j in range(len(lon_values)):
                point = Point(lon_grid[i, j], lat_grid[i, j])
                dist = point.distance(polygon_centroid)
                if dist < min_dist:
                    min_dist = dist
                    closest_i, closest_j = i, j

        hist_mask[closest_i, closest_j] = True
        print(f"Closest point to polygon centroid for historical data: Lon={lon_values[closest_j]}, Lat={lat_values[closest_i]}")

    # Fill in CO2 concentration for each year and month in the dataframe
    for index, row in df.iterrows():
        year = int(row['Year'])
        month = int(row['Month'])

        if 1950 <= year <= 2013:
            # Find the corresponding time in the historical dataset
            time_str = f"{year}-{month:02d}-01"
            try:
                # Find the time index
                time_idx = np.where(historical_ds.Times == np.datetime64(time_str))[0][0]

                # Extract CO2 values for this time and apply the mask
                co2_slice = historical_ds.value.isel(Times=time_idx).values
                masked_values = co2_slice[hist_mask]

                # Calculate the average if there are valid values
                if len(masked_values) > 0 and not np.all(np.isnan(masked_values)):
                    df.at[index, 'CO2_Concentration'] = np.nanmean(masked_values)
            except (IndexError, KeyError):
                print(f"Time {time_str} not found in historical dataset")

        elif year == 2014:
            print("You are outside of historical data ! Use process_co2_data_withFutureInterpolation to deal with future data from https://zenodo.org/records/5021361")

    return df

# Function to process CO2 data and fill the dataframe
# Based on datasets from https://zenodo.org/records/5021361
# But this time, takes into account future data coming from
# https://zenodo.org/records/5021361; and does an interpolation
# because there is one missing year (2014) between their historical
# and future datasets
def process_co2_data_withFutureInterpolation(historical_ds, future_ds, shapefile, df):
    # Add CO2_Concentration column to dataframe
    df['CO2_Concentration'] = np.nan

    # Load the shapefile
    gdf = gpd.read_file(shapefile)
    # Ensure the shapefile is in the same CRS as the climate data (usually EPSG:4326)
    if gdf.crs != 'EPSG:4326':
        gdf = gdf.to_crs('EPSG:4326')
    # Get the polygon from the shapefile (assuming first polygon if multiple)
    polygon = gdf.geometry.iloc[0]
    # Get the centroid of the polygon
    polygon_centroid = polygon.centroid

    # Process historical data (1950-2013)
    # Create a mask for points inside the polygon for historical data
    lon_values = historical_ds.Longitude.values
    lat_values = historical_ds.Latitude.values
    lon_grid, lat_grid = np.meshgrid(lon_values, lat_values)
    hist_mask = np.zeros((len(lat_values), len(lon_values)), dtype=bool)

    # Check each point if it's inside the polygon
    for i in range(len(lat_values)):
        for j in range(len(lon_values)):
            point = Point(lon_grid[i, j], lat_grid[i, j])
            hist_mask[i, j] = polygon.contains(point)

    # If no points are inside the polygon, find the closest point to the polygon centroid
    if not np.any(hist_mask):
        print("No points found inside the polygon for historical data. Finding closest point...")
        min_dist = float('inf')
        closest_i, closest_j = 0, 0

        for i in range(len(lat_values)):
            for j in range(len(lon_values)):
                point = Point(lon_grid[i, j], lat_grid[i, j])
                dist = point.distance(polygon_centroid)
                if dist < min_dist:
                    min_dist = dist
                    closest_i, closest_j = i, j

        hist_mask[closest_i, closest_j] = True
        print(f"Closest point to polygon centroid for historical data: Lon={lon_values[closest_j]}, Lat={lat_values[closest_i]}")

    # Process future data (2014-2100)
    # Create a mask for points inside the polygon for future data
    lon_values = future_ds.longitude.values
    lat_values = future_ds.latitude.values
    lon_grid, lat_grid = np.meshgrid(lon_values, lat_values)
    future_mask = np.zeros((len(lat_values), len(lon_values)), dtype=bool)

    # Check each point if it's inside the polygon
    for i in range(len(lat_values)):
        for j in range(len(lon_values)):
            point = Point(lon_grid[i, j], lat_grid[i, j])
            future_mask[i, j] = polygon.contains(point)

    # If no points are inside the polygon, find the closest point to the polygon centroid
    if not np.any(future_mask):
        print("No points found inside the polygon for future data. Finding closest point...")
        min_dist = float('inf')
        closest_i, closest_j = 0, 0

        for i in range(len(lat_values)):
            for j in range(len(lon_values)):
                point = Point(lon_grid[i, j], lat_grid[i, j])
                dist = point.distance(polygon_centroid)
                if dist < min_dist:
                    min_dist = dist
                    closest_i, closest_j = i, j

        future_mask[closest_i, closest_j] = True
        print(f"Closest point to polygon centroid for future data: Lon={lon_values[closest_j]}, Lat={lat_values[closest_i]}")

    # Extract monthly CO2 values for 2013 to capture seasonal pattern
    monthly_values_2013 = {}
    for month in range(1, 13):
        time_str = f"2013-{month:02d}-01"
        try:
            time_idx = np.where(historical_ds.Times == np.datetime64(time_str))[0][0]
            co2_slice = historical_ds.value.isel(Times=time_idx).values
            masked_values = co2_slice[hist_mask]
            if len(masked_values) > 0 and not np.all(np.isnan(masked_values)):
                monthly_values_2013[month] = np.nanmean(masked_values)
        except (IndexError, KeyError):
            print(f"Time {time_str} not found in historical dataset")

    # Extract monthly CO2 values for 2015 to capture seasonal pattern
    monthly_values_2015 = {}
    for month in range(1, 13):
        time_str = f"2015-{month:02d}-01"
        try:
            time_idx = np.where(future_ds.time == cftime.DatetimeNoLeap(2015, month, 15, 0, 0, 0, 0))[0][0]
            co2_slice = future_ds.CO2.isel(time=time_idx).values
            masked_values = co2_slice[future_mask]
            if len(masked_values) > 0 and not np.all(np.isnan(masked_values)):
                monthly_values_2015[month] = np.nanmean(masked_values)
        except (IndexError, KeyError):
            print(f"Time {time_str} not found in future dataset")

    # Calculate annual averages for 2013 and 2015
    if len(monthly_values_2013) == 12 and len(monthly_values_2015) == 12:
        annual_avg_2013 = sum(monthly_values_2013.values()) / 12
        annual_avg_2015 = sum(monthly_values_2015.values()) / 12

        # Calculate seasonal anomalies (deviations from annual mean)
        seasonal_anomaly_2013 = {month: value - annual_avg_2013 for month, value in monthly_values_2013.items()}
        seasonal_anomaly_2015 = {month: value - annual_avg_2015 for month, value in monthly_values_2015.items()}

        # Average the seasonal anomalies from 2013 and 2015
        avg_seasonal_anomaly = {month: (seasonal_anomaly_2013[month] + seasonal_anomaly_2015[month]) / 2 
                               for month in range(1, 13)}

        # Calculate the expected annual average for 2014 (linear interpolation)
        annual_avg_2014 = annual_avg_2013 + (annual_avg_2015 - annual_avg_2013) / 2

        # Calculate the expected monthly values for 2014
        monthly_values_2014 = {month: annual_avg_2014 + avg_seasonal_anomaly[month] for month in range(1, 13)}

        print("Successfully calculated seasonal pattern for 2014")
    else:
        print("Cannot calculate seasonal pattern, missing monthly data for 2013 or 2015")
        monthly_values_2014 = None

    # Fill in CO2 concentration for each year and month in the dataframe
    for index, row in df.iterrows():
        year = int(row['Year'])
        month = int(row['Month'])

        if 1950 <= year <= 2013:
            # Find the corresponding time in the historical dataset
            time_str = f"{year}-{month:02d}-01"
            try:
                # Find the time index
                time_idx = np.where(historical_ds.Times == np.datetime64(time_str))[0][0]

                # Extract CO2 values for this time and apply the mask
                co2_slice = historical_ds.value.isel(Times=time_idx).values
                masked_values = co2_slice[hist_mask]

                # Calculate the average if there are valid values
                if len(masked_values) > 0 and not np.all(np.isnan(masked_values)):
                    df.at[index, 'CO2_Concentration'] = np.nanmean(masked_values)
            except (IndexError, KeyError):
                print(f"Time {time_str} not found in historical dataset")

        elif year == 2014:
            # Use the seasonally adjusted values for 2014
            if monthly_values_2014 and month in monthly_values_2014:
                df.at[index, 'CO2_Concentration'] = monthly_values_2014[month]
            else:
                # Fallback to simple linear interpolation if seasonal adjustment failed
                if 'value' in monthly_values_2013.get(month, {}) and 'value' in monthly_values_2015.get(month, {}):
                    interpolated_value = (monthly_values_2013[month] + monthly_values_2015[month]) / 2
                    df.at[index, 'CO2_Concentration'] = interpolated_value
                else:
                    print(f"Cannot interpolate for 2014-{month:02d}, missing data")

        elif 2014 <= year <= 2100:
            # Find the corresponding time in the future dataset
            time_str = f"{year}-{month:02d}-01"
            try:
                # Find the time index
                # Different here, as this file uses cftime format.
                # Plus, don't ask me why, they put the day for the timestep at 15
                time_idx = np.where(future_ds.time == cftime.DatetimeNoLeap(year, month, 15, 0, 0, 0, 0))[0][0]

                # Extract CO2 values for this time and apply the mask
                # Here again, dimension names and all are different
                co2_slice = future_ds.CO2.isel(time=time_idx).values
                masked_values = co2_slice[future_mask]

                # Calculate the average if there are valid values
                if len(masked_values) > 0 and not np.all(np.isnan(masked_values)):
                    df.at[index, 'CO2_Concentration'] = np.nanmean(masked_values)
            except (IndexError, KeyError):
                print(f"Time {time_str} not found in future dataset")

    return df



def standardize_xarray_dataset(ds):
    """
    Transform an xarray Dataset by converting longitude and latitude from variables to coordinates
    when they exist as variables and the dataset only has a time coordinate.

    Parameters:
    -----------
    ds : xarray.Dataset
        The input dataset, typically loaded from a .nc file

    Returns:
    --------
    xarray.Dataset
        Transformed dataset with longitude and latitude as coordinates if applicable
    """

    # Change dimension name if it's not "time"
    if "Times" in set(ds.coords):
        ds = ds.rename_dims(dims_dict={"Times":"time"})
        
    # Check if longitude and latitude exist as variables
    lon_var = None
    lat_var = None

    # Look for common longitude and latitude variable names
    lon_names = ['lon', 'longitude', 'LON', 'LONGITUDE', 'Longitude', "LonDim"]
    lat_names = ['lat', 'latitude', 'LAT', 'LATITUDE', 'Latitude', "LatDim"]

    for name in lon_names:
        if name in ds.variables:
            lon_var = name
            break

    for name in lat_names:
        if name in ds.variables:
            lat_var = name
            break

    # If both longitude and latitude variables exist, convert them to coordinates
    if lon_var is not None and lat_var is not None:
        # Create a new dataset with longitude and latitude as coordinates
        lon_values = ds[lon_var].values
        lat_values = ds[lat_var].values

        # Create meshgrid if longitude and latitude are 1D
        if lon_values.ndim == 1 and lat_values.ndim == 1:
            lon_grid, lat_grid = np.meshgrid(lon_values, lat_values)

            # Create new dataset with proper coordinates
            new_ds = xr.Dataset(
                coords={
                    'time': (["time"], ds.coords["time"].values),
                    'latitude': (['latitude'], lat_values),
                    'longitude': (['longitude'], lon_values)
                }
            )

            # Copy all variables except longitude and latitude
            for var_name, var in ds.variables.items():
                if var_name not in [lon_var, lat_var, 'time']:
                    # Reshape the data if needed based on dimensions
                    if var.ndim == 1:
                        # Handle 1D variables
                        new_ds[var_name] = var
                    else:
                        # Assume the variable has dimensions that match the grid
                        new_dims = ['time', 'latitude', 'longitude'][:var.ndim]
                        new_ds[var_name] = (new_dims, var.values)

            return new_ds
        else:
            # If longitude and latitude are already 2D, use them directly as coordinates
            ds = ds.assign_coords({
                'longitude': ds[lon_var],
                'latitude': ds[lat_var]
            })

            # Drop the original variables to avoid duplication
            ds = ds.drop_vars([lon_var, lat_var])

            return ds

    # If conditions are not met, return the original dataset
    return ds

# Downloads a file quickly using the axel package for linux, and display a progress bar properly in Jupyter notebook
def download_file(url, save_path):
    # Start the process
    process = subprocess.Popen(['axel', '-n', '10', '--output=' + str(save_path), url],
                              stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT,
                              universal_newlines=True,
                              bufsize=1)
    
    last_line = ""
    # Read and display output in real-time
    for line in iter(process.stdout.readline, ''):
        # Store the last line
        last_line = line.rstrip()
    
        # Clear the current line and print the new one
        # This creates the "updating" effect
        sys.stdout.write('\r' + ' ' * 100 + '\r')  # Clear line
        sys.stdout.write(last_line)
        sys.stdout.flush()
    
    # Wait for the process to complete
    return_code = process.wait()
    
    # Print a newline at the end
    print()
    
    # Create a result object similar to subprocess.run for compatibility
    result = type('', (), {})()
    result.returncode = return_code
    result.stdout = last_line
    result.stderr = ""

# Similar toload_and_filter_by_polygon (see above), but adapted for CanLEADv1 datasets
def load_and_filter_by_polygon_canLEADv1(file_path, shapefile_path):
    # Load the shapefile
    gdf = gpd.read_file(shapefile_path)
    # Ensure the shapefile is in the same CRS as the climate data (usually EPSG:4326)
    if gdf.crs != 'EPSG:4326':
        gdf = gdf.to_crs('EPSG:4326')
    # Get the polygon from the shapefile (assuming first polygon if multiple)
    polygon = gdf.geometry.iloc[0]
    # Load the climate data
    ds = xr.open_dataset(file_path)
    # Create a mask for points inside the polygon
    lon_grid, lat_grid = np.meshgrid(ds.lon.values, ds.lat.values)
    mask = np.zeros((len(ds.lat), len(ds.lon)), dtype=bool)
    # Check each point if it's inside the polygon
    for i in range(len(ds.lat)):
        for j in range(len(ds.lon)):
            point = Point(lon_grid[i, j], lat_grid[i, j])
            mask[i, j] = polygon.contains(point)
    # Apply the mask to the dataset
    # First, we need to convert the mask to have the same dimensions as the dataset
    mask_da = xr.DataArray(mask, coords=[ds.lat, ds.lon], dims=['lat', 'lon'])
    # Apply the mask
    ds_filtered = ds.where(mask_da, drop=True)
    return ds_filtered

# Used for conversion of time format when converting CanLEADv1 data into a panda dataframe
def convert_cftime_to_datetime(cftime_obj):
	if pd.isna(cftime_obj):
		return pd.NaT
	return datetime(cftime_obj.year, cftime_obj.month, cftime_obj.day, 
                   cftime_obj.hour, cftime_obj.minute, cftime_obj.second)


def plot_climateVariable_time_series_with_moving_average(variable_dict, dates_time, variableName, window_years=20, highlightModel = ""):
        """
        Plot precipitation time series with a 20-year moving average for multiple models,
        compute the median of the averaged data, and highlight the model closest to the median.
    
        Parameters:
        -----------
        variable_dict : dict
            Dictionary with model names as keys and variable time series as values
        dates_time : list
            List of date strings with the same length as the precipitation time series
        window_years : int
            Size of the moving average window in years (default: 20)
        """
        # Convert dates to datetime objects
        dates = [datetime.strptime(date_str, "%Y-%m-%d") for date_str in dates_time]
    
        # Create a DataFrame with dates as index for easier manipulation
        df = pd.DataFrame(variable_dict, index=dates)
    
        # Calculate the window size in terms of data points
        # Assuming data is annual, monthly, or daily
        # Data is monthly, so windows is mutliplied by 12
        window_size = 12 * window_years
    
        # Apply moving average to each model's data
        df_avg = df.rolling(window=window_size, center=True, min_periods=1).mean()
    
        # print(df)
        # print(df_avg)
    
        # Compute the median across all models for the averaged data
        df_avg['median'] = df_avg.drop(columns=['median'] if 'median' in df_avg.columns else []).median(axis=1)
    
        # Calculate which model is closest to the median
        # Using mean squared error as the distance metric
        mse_values = {}
        for model in variable_dict.keys():
            # Filter out NaN values that might appear at the edges due to the rolling window
            valid_idx = ~np.isnan(df_avg[model]) & ~np.isnan(df_avg['median'])
            if valid_idx.sum() > 0:
                mse_values[model] = np.mean((df_avg[model][valid_idx] - df_avg['median'][valid_idx])**2)
            else:
                mse_values[model] = float('inf')
    
        closest_model = min(mse_values, key=mse_values.get)
    
        # Create the plot
        plt.figure(figsize=(14, 8))
    
        # Generate a colormap with 26 distinct colors
        colors = plt.cm.tab20.colors + plt.cm.tab20b.colors + plt.cm.tab20c.colors
    
        # Plot each model's time series with transparency
        for i, model in enumerate(variable_dict.keys()):
            if model != closest_model:
                color = colors[i % len(colors)]
                plt.plot(df_avg.index, df_avg[model], color=color, alpha=0.5, linewidth=1, label=model)
    
        # Plot the median with less transparency
        plt.plot(df_avg.index, df_avg['median'], color='black', alpha=0.8, linewidth=2, label='Median')
    
        # Plot the closest model on top with full opacity and thicker line
        plt.plot(df_avg.index, df_avg[closest_model], color='red', alpha=1.0, 
                 linewidth=3, label=f'{closest_model} (Closest to Median)')

        # Plot an additional model if we want
        if highlightModel != "":
            plt.plot(df_avg.index, df_avg[highlightModel], color='yellow', alpha=1.0, 
            linewidth=3, label=f'{highlightModel}')
    
        # Format the x-axis to show dates properly
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
        plt.gcf().autofmt_xdate()  # Rotate date labels
    
        plt.title(f'{variableName} Time Series by Model ({window_years}-Year Moving Average)')
        plt.xlabel('Date')
        plt.ylabel(str(variableName) + ' (Moving Average)')
        plt.grid(True, alpha=0.3)
        # plt.ylim(0, 100)
    
        # Create a legend with a reasonable size
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
    
        plt.tight_layout()
        return plt

def plot_climate_variables_with_moving_average(precip_dict, tmax_dict, tmin_dict, dates_time, window_years=20, highlightModel = ""):
    """
    Plot time series with a moving average for precipitation, max temperature, and min temperature.
    Compute the median for each variable and highlight the model closest to the median across all variables.

    Parameters:
    -----------
    precip_dict : dict
        Dictionary with model names as keys and precipitation time series as values
    tmax_dict : dict
        Dictionary with model names as keys and maximum temperature time series as values
    tmin_dict : dict
        Dictionary with model names as keys and minimum temperature time series as values
    dates_time : list
        List of date strings with the same length as the time series
    window_years : int
        Size of the moving average window in years (default: 20)
    """
    # Convert dates to datetime objects
    dates = [datetime.strptime(date_str, "%Y-%m-%d") for date_str in dates_time]

    # Create DataFrames with dates as index for easier manipulation
    df_precip = pd.DataFrame(precip_dict, index=dates)
    df_tmax = pd.DataFrame(tmax_dict, index=dates)
    df_tmin = pd.DataFrame(tmin_dict, index=dates)

    # Calculate the window size in terms of data points
    # Assuming data is monthly (most common for climate data)
    window_size = 12 * window_years

    # Apply moving average to each model's data for all variables
    df_precip_avg = df_precip.rolling(window=window_size, center=True, min_periods=1).mean()
    df_tmax_avg = df_tmax.rolling(window=window_size, center=True, min_periods=1).mean()
    df_tmin_avg = df_tmin.rolling(window=window_size, center=True, min_periods=1).mean()

    # Compute the median across all models for each variable
    df_precip_avg['median'] = df_precip_avg.median(axis=1)
    df_tmax_avg['median'] = df_tmax_avg.median(axis=1)
    df_tmin_avg['median'] = df_tmin_avg.median(axis=1)

    # Calculate standardized MSE for each model across all variables
    models = list(precip_dict.keys())
    mse_values = {model: {'precip': 0, 'tmax': 0, 'tmin': 0} for model in models}

    # Calculate MSE for each variable
    for model in models:
        # Filter out NaN values that might appear at the edges due to the rolling window
        valid_idx_precip = ~np.isnan(df_precip_avg[model]) & ~np.isnan(df_precip_avg['median'])
        valid_idx_tmax = ~np.isnan(df_tmax_avg[model]) & ~np.isnan(df_tmax_avg['median'])
        valid_idx_tmin = ~np.isnan(df_tmin_avg[model]) & ~np.isnan(df_tmin_avg['median'])

        if valid_idx_precip.sum() > 0:
            mse_values[model]['precip'] = np.mean((df_precip_avg[model][valid_idx_precip] - df_precip_avg['median'][valid_idx_precip])**2)
        else:
            mse_values[model]['precip'] = float('inf')

        if valid_idx_tmax.sum() > 0:
            mse_values[model]['tmax'] = np.mean((df_tmax_avg[model][valid_idx_tmax] - df_tmax_avg['median'][valid_idx_tmax])**2)
        else:
            mse_values[model]['tmax'] = float('inf')

        if valid_idx_tmin.sum() > 0:
            mse_values[model]['tmin'] = np.mean((df_tmin_avg[model][valid_idx_tmin] - df_tmin_avg['median'][valid_idx_tmin])**2)
        else:
            mse_values[model]['tmin'] = float('inf')

    # Standardize MSE values for each variable
    precip_mse_values = np.array([mse_values[model]['precip'] for model in models])
    tmax_mse_values = np.array([mse_values[model]['tmax'] for model in models])
    tmin_mse_values = np.array([mse_values[model]['tmin'] for model in models])

    # Remove infinite values for standardization
    precip_mse_finite = precip_mse_values[np.isfinite(precip_mse_values)]
    tmax_mse_finite = tmax_mse_values[np.isfinite(tmax_mse_values)]
    tmin_mse_finite = tmin_mse_values[np.isfinite(tmin_mse_values)]

    # Standardize each variable's MSE (z-score)
    precip_mean, precip_std = np.mean(precip_mse_finite), np.std(precip_mse_finite)
    tmax_mean, tmax_std = np.mean(tmax_mse_finite), np.std(tmax_mse_finite)
    tmin_mean, tmin_std = np.mean(tmin_mse_finite), np.std(tmin_mse_finite)

    # Calculate standardized MSE and combined score
    combined_scores = {}
    for i, model in enumerate(models):
        precip_score = (mse_values[model]['precip'] - precip_mean) / precip_std if np.isfinite(mse_values[model]['precip']) else float('inf')
        tmax_score = (mse_values[model]['tmax'] - tmax_mean) / tmax_std if np.isfinite(mse_values[model]['tmax']) else float('inf')
        tmin_score = (mse_values[model]['tmin'] - tmin_mean) / tmin_std if np.isfinite(mse_values[model]['tmin']) else float('inf')

        # Average of standardized scores (lower is better)
        if np.isfinite(precip_score) and np.isfinite(tmax_score) and np.isfinite(tmin_score):
            combined_scores[model] = (precip_score + tmax_score + tmin_score) / 3
        else:
            combined_scores[model] = float('inf')

    # Find the model with the lowest combined standardized MSE
    closest_model = min(combined_scores, key=combined_scores.get)

    # Create the plot with 3 subplots and extra space at the top for the main title
    fig = plt.figure(figsize=(16, 16))  # Increased height
    gs = GridSpec(3, 1, figure=fig, height_ratios=[1, 1, 1])

    # Add a main title with more space
    fig.suptitle(f'Climate Variables with {window_years}-Year Moving Average\nBest Overall Model: {closest_model}', 
                fontsize=16, y=0.98)  # Positioned higher

    # Generate a colormap with 26 distinct colors
    colors = plt.cm.tab20.colors + plt.cm.tab20b.colors + plt.cm.tab20c.colors

    # Create a color mapping for models
    color_map = {model: colors[i % len(colors)] for i, model in enumerate(models)}

    # Create empty lists to collect all line objects and labels for the legend
    all_lines = []
    all_labels = []

    # Plot precipitation
    ax1 = fig.add_subplot(gs[0])
    for model in models:
        if model != closest_model:
            line, = ax1.plot(df_precip_avg.index, df_precip_avg[model], color=color_map[model], 
                           alpha=0.5, linewidth=1)
            all_lines.append(line)
            all_labels.append(model)

    median_line, = ax1.plot(df_precip_avg.index, df_precip_avg['median'], color='black', 
                          alpha=0.8, linewidth=2)
    closest_line, = ax1.plot(df_precip_avg.index, df_precip_avg[closest_model], color='red', 
                           alpha=1.0, linewidth=3)
    if highlightModel != "":
        highlighted_line = ax1.plot(df_precip_avg.index, df_precip_avg[highlightModel], color='yellow', 
                             alpha=1.0, linewidth=3)
        

    # Add median and closest model to the legend collection
    all_lines.append(median_line)
    all_labels.append('Median')
    all_lines.append(closest_line)
    all_labels.append(f'{closest_model} (Best Overall)')
    if highlightModel != "":
        all_lines.append(highlighted_line)
        all_labels.append(f'{highlightModel} (Highlighted)')
    

    ax1.set_title(f'Precipitation ({window_years}-Year Moving Average)')
    ax1.set_ylabel('Precipitation')
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax1.xaxis.set_major_locator(mdates.AutoDateLocator())

    # Plot max temperature
    ax2 = fig.add_subplot(gs[1])
    for model in models:
        if model != closest_model:
            ax2.plot(df_tmax_avg.index, df_tmax_avg[model], color=color_map[model], 
                    alpha=0.5, linewidth=1)

    ax2.plot(df_tmax_avg.index, df_tmax_avg['median'], color='black', 
             alpha=0.8, linewidth=2)
    ax2.plot(df_tmax_avg.index, df_tmax_avg[closest_model], color='red', 
             alpha=1.0, linewidth=3)
    if highlightModel != "":
        ax2.plot(df_tmax_avg.index, df_tmax_avg[highlightModel], color='yellow', 
             alpha=1.0, linewidth=3)

    ax2.set_title(f'Maximum Temperature ({window_years}-Year Moving Average)')
    ax2.set_ylabel('Maximum Temperature')
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax2.xaxis.set_major_locator(mdates.AutoDateLocator())

    # Plot min temperature
    ax3 = fig.add_subplot(gs[2])
    for model in models:
        if model != closest_model:
            ax3.plot(df_tmin_avg.index, df_tmin_avg[model], color=color_map[model], 
                    alpha=0.5, linewidth=1)

    ax3.plot(df_tmin_avg.index, df_tmin_avg['median'], color='black', 
             alpha=0.8, linewidth=2)
    ax3.plot(df_tmin_avg.index, df_tmin_avg[closest_model], color='red', 
             alpha=1.0, linewidth=3)
    if highlightModel != "":
        ax3.plot(df_tmin_avg.index, df_tmin_avg[highlightModel], color='yellow', 
             alpha=1.0, linewidth=3)

    ax3.set_title(f'Minimum Temperature ({window_years}-Year Moving Average)')
    ax3.set_xlabel('Date')
    ax3.set_ylabel('Minimum Temperature')
    ax3.grid(True, alpha=0.3)
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax3.xaxis.set_major_locator(mdates.AutoDateLocator())

    # Format dates on x-axis
    plt.gcf().autofmt_xdate()

    # Create a separate legend figure for all models
    # Place the legend outside the plot area
    fig.subplots_adjust(right=0.8, top=0.9)  # Make room for the legend and title

    # Create the legend with all models
    legend = fig.legend(all_lines, all_labels, loc='center right', 
                       bbox_to_anchor=(0.98, 0.5), fontsize='small', 
                       title="Models", ncol=1)

    # Adjust the legend to make it more readable if there are many models
    if len(models) > 15:
        legend._ncol = 2  # Use 2 columns for the legend if there are many models

    plt.tight_layout(rect=[0, 0, 0.8, 0.97])  # Adjust layout but leave space for legend and title

    return fig, closest_model, combined_scores

def save_average_raster(output_path, average_array, metadata):
    """
    Save the average array as a new raster.

    Args:
        output_path: Path where to save the output raster
        average_array: Numpy array containing the average values
        metadata: Metadata for the output raster
    """
    # Update metadata for the output file
    metadata.update({
        'dtype': 'float32',
        'driver': 'GTiff',
    })

    with rasterio.open(output_path, 'w', **metadata) as dst:
        # Write all bands
        if average_array.ndim == 3:
            for i in range(average_array.shape[0]):
                dst.write(average_array[i].astype(np.float32), i+1)
        else:
            dst.write(average_array.astype(np.float32), 1)

# Used to average soil textures rasters

def calculate_raster_average(raster_paths, weights=None):
    """
    Calculate the weighted average of multiple rasters with the same dimensions,
    given as a list of paths to the rasters and corresponding weights.

    Args:
        raster_paths: List of paths to .tif raster files
        weights: List of weights corresponding to each raster. If None, equal weights are used.

    Returns:
        tuple: (weighted_average_array, metadata) where weighted_average_array is a numpy array
               containing the weighted average values and metadata is from the first raster
    """
    if not raster_paths:
        raise ValueError("No raster paths provided")

    # Set default weights if none provided
    if weights is None:
        weights = np.ones(len(raster_paths))

    # Validate weights
    if len(weights) != len(raster_paths):
        raise ValueError("Number of weights must match number of rasters")

    # Convert weights to numpy array
    weights = np.array(weights, dtype=np.float64)

    # Open the first raster to get metadata and initialize the sum arrays
    with rasterio.open(raster_paths[0]) as src:
        # Get metadata from the first raster
        metadata = src.meta.copy()
        # Read all bands
        first_raster = src.read()
        # Initialize sum array with zeros of the same shape
        weighted_sum_array = np.zeros_like(first_raster, dtype=np.float64)
        # Add the first raster to the weighted sum
        weighted_sum_array += first_raster * weights[0]

    # Process the remaining rasters
    for i, path in enumerate(raster_paths[1:], 1):
        with rasterio.open(path) as src:
            # Check dimensions match
            if src.shape != (metadata['height'], metadata['width']):
                raise ValueError(f"Raster {path} has different dimensions than the first raster")
            # Add to the weighted sum
            weighted_sum_array += src.read() * weights[i]

    # Calculate the weighted average
    weights_sum = np.sum(weights)
    if weights_sum == 0:
        raise ValueError("Sum of weights cannot be zero")

    weighted_average_array = weighted_sum_array / weights_sum

    return weighted_average_array, metadata


# Takes the path to two averaged sand and clay % rasters, 
# and generates a re-classified raster with codes corresponding to
# the 12 soils types from the SaxtonAndRawls default parameters in PnET Succession
# using the euclidian distance in the dimension of % of sand and % of clay for each cell
# and the soil types
def classify_soil_types_chunked(sand_path, clay_path, output_path, chunk_size=1024):
    """
    Classify soil types based on sand, clay, and silt percentages using a chunked approach.

    Parameters:
    -----------
    sand_path : str
        Path to the .tif file containing sand percentage values
    clay_path : str
        Path to the .tif file containing clay percentage values
    silt_path : str
        Path to the .tif file containing silt percentage values
    output_path : str
        Path to save the output classification .tif file
    chunk_size : int
        Size of chunks to process at once (default: 1024)
    """
    # Define soil types with their sand and clay percentages
    # Comes from the Saxton and Rawls default parameter file of
    # PnET Succession 5.1, in C:\Program Files\LANDIS-II-v7\extensions\Defaults\SaxtonAndRawlsParameters.txt
    soil_types = {
        "SAND": (0.85, 0.04),
        "LOSA": (0.80, 0.05),
        "SALO": (0.63, 0.10),
        "LOAM": (0.41, 0.19),
        "SILO": (0.15, 0.18),
        "SILT": (0.05, 0.10),
        "SNCL": (0.53, 0.26),
        "CLLO": (0.29, 0.32),
        "SLCL": (0.09, 0.32),
        "SACL": (0.50, 0.40),
        "SICL": (0.08, 0.47),
        "CLAY": (0.13, 0.55),
        "BEDR": (0.10, 0.10)
    }

    # Convert soil types to a numpy array for efficient distance calculation
    soil_types_array = np.array(list(soil_types.values()))
    soil_type_names = list(soil_types.keys())

    # Open all input files
    print("Opening all files")
    with rasterio.open(sand_path) as sand_src, \
         rasterio.open(clay_path) as clay_src:

        # Get dimensions and profile from sand raster
        height = sand_src.height
        width = sand_src.width
        profile = sand_src.profile

        # Update profile for output
        profile.update(
            dtype=rasterio.uint8,
            count=1,
            nodata=0
        )

        # Create output file
        print("Creating output")
        with rasterio.open(output_path, 'w', **profile) as dst:
            # Process the raster in chunks
            for row in range(0, height, chunk_size):
                row_end = min(row + chunk_size, height)
                chunk_height = row_end - row

                for col in range(0, width, chunk_size):
                    col_end = min(col + chunk_size, width)
                    chunk_width = col_end - col

                    # Define the window for the current chunk
                    window = Window(col, row, chunk_width, chunk_height)

                    # print("reading data for current chunk")
                    # Read data for the current window
                    sand_chunk = sand_src.read(1, window=window)
                    clay_chunk = clay_src.read(1, window=window)

                    # Create a mask for valid data
                    valid_mask = ~(np.isnan(sand_chunk) | np.isnan(clay_chunk) )

                    # Initialize output chunk with zeros
                    soil_classification_chunk = np.zeros_like(sand_chunk, dtype=np.uint8)

                    # Process only valid cells in the chunk
                    if np.any(valid_mask):
                        # Get valid cell coordinates
                        valid_indices = np.where(valid_mask)

                        
                        # Stack sand and clay percentages for valid cells
                        # We divide by 100 as the values in the raster are in %,
                        # but the values in the soil matrix from saxton and rawls parameters
                        # are in decimals
                        soil_composition = np.vstack((
                            sand_chunk[valid_indices]/100,
                            clay_chunk[valid_indices]/100
                        )).T
                        # print("Soil composition object :")
                        # print(soil_composition)

                        # Calculate distances and find closest soil type
                        distances = cdist(soil_composition, soil_types_array)
                        # print("Distances object :")
                        # print(distances)
                        # The +1 here is because the function returns the index of
                        # the soil texture from the data matrix that is closest to 
                        # the given cell. However, indexes go from 0 to 11 (as there
                        # are 12 soils textures), but the script creates codes ranging
                        # from 1 to 12 to put in the raster. Hence th + 1.
                        # It's also because 0 is often the nodata value in rasters.
                        closest_soil_indices = np.argmin(distances, axis=1) + 1
                        # print("Closets soil indices object :")
                        # print(closest_soil_indices)

                        # Assign soil type indices to output chunk
                        soil_classification_chunk[valid_indices] = closest_soil_indices
                        
                        # input("Press Enter to continue...")

                    # Write the chunk to the output file
                    dst.write(soil_classification_chunk, 1, window=window)

                    # print(f"Processed chunk: Row {row}-{row_end}, Col {col}-{col_end}")

    # Print soil type codes for reference
    print("\nSoil Type Codes:")
    for i, soil_type in enumerate(soil_type_names, 1):
        print(f"{i}: {soil_type}")

    print(f"\nClassification complete. Output saved to: {output_path}")

def safe_convert_to_datetime(t):
    # Check if date is within pandas' supported range
    try:
        # Create a datetime object directly
        if hasattr(t, 'year') and hasattr(t, 'month') and hasattr(t, 'day'):
            dt = datetime(t.year, t.month, t.day, 
                          getattr(t, 'hour', 0), 
                          getattr(t, 'minute', 0), 
                          getattr(t, 'second', 0))

            # Check if within pandas supported range (roughly)
            if dt.year < 1678 or dt.year > 2261:
                # For dates outside pandas range, return the datetime object
                # but note these won't work with pandas time operations
                return dt
            else:
                return pd.Timestamp(dt)
        else:
            # Fallback for other types
            return pd.Timestamp(str(t))
    except (ValueError, TypeError, OverflowError):
        # If conversion fails, return the original object
        # You might want to handle this differently based on your needs
        print(f"Warning: Could not convert {t} to datetime")
        return t

import suncalc
from datetime import datetime, timedelta
import pytz

def daylight_hour_average(rsds_24h_avg, latitude, longitude, year, month, day):
    """
    Converts a 24-hour average rsds (W/m² or other unit) to a daylight-hour average.

    Args:
        swd_24h_avg (float): 24-hour average SWD (W/m² or other unit of input)
        latitude (float): Latitude in degrees
        longitude (float): Longitude in degrees
        year (int): Year (e.g., 2025)
        month (int): Month (1-12)
        day (int): Day (1-31)

    Returns:
        float: Daylight-hour average SWD (W/m²)
    """
    # Create a date object (local time, converted to UTC for suncalc)
    date_local = datetime(year, month, day)
    timezone = pytz.timezone('UTC')  # Adjust to your local timezone if needed
    date_utc = timezone.localize(date_local)

    # Get sunrise and sunset times
    times = suncalc.get_times(date_utc, longitude, latitude)
    sunrise = times['sunrise']
    sunset = times['sunset']

    # Handle polar day/night edge cases
    if sunset < sunrise:
        daylight_hours = (sunset + timedelta(days=1) - sunrise).seconds / 3600
    else:
        daylight_hours = (sunset - sunrise).seconds / 3600

    # Calculate total daily energy (Wh/m²)
    total_energy = rsds_24h_avg * 24

    # Compute daylight-hour average (avoid division by zero)
    if daylight_hours == 0:
        return 0.0  # Polar night: no sunlight
    else:
        return total_energy / daylight_hours

def GetDataFromESGFdatasets(catalogURL,
                            yearStart,
                            yearStop,
                            pathOfShapefileForSubsetting,
                            nameOfVariable,
                            enableWarnings = True):
    """
    Used to read a data catalog from a ESGF node, loop around the .nc files in the catalog,
    and for each file, keep the data only for the years of interest, subset the data in a polygon,
    and get the values for the variables to then put them in a panda data frame.
    """
    
    with warnings.catch_warnings():
        if not enableWarnings:
            warnings.simplefilter("ignore")  # Ignore all warnings
        else:
            warnings.simplefilter("default") # Use the default warning behavior
    
        # We load the ESGF catalog from the URL
        cat = TDSCatalog(catalogURL)
    
        # We initialize the time coder to help xarray decode time using the "cftime" format
        # It's the recommand approach based on the warning messages we get without it
        print("Loading catalog")
        time_coder = xr.coders.CFDatetimeCoder(use_cftime=True)
        
        # Create an empty DataFrame to store all results
        all_data = pd.DataFrame(columns=["lat", "lon", "year", "month", "day", nameOfVariable])
        
        # Loop through each dataset
        # We skip the 0 because it corresponds here to an entry that is not an .nc file
        for datasetID in range(0, len(cat.datasets)):
        
            # There are sometimes other files in the dataset that do not correspond to .nc files;
            # we don't deal with those.
            if ".nc" in str(cat.datasets[datasetID]):
                
                print(f"Processing {cat.datasets[datasetID]}...")
            
                # Open the dataset
                try:
                    ds = xr.open_dataset(cat.datasets[datasetID].access_urls["OPENDAP"], decode_times=time_coder)
            
                    # print("Checking dataset " + str(os.path.basename(url)))
                    # Handle time values properly
                    # First, convert the time values to strings and then to datetime objects
                    time_values = ds.time.values
            
                    # Check the type of time values and convert accordingly
                    # Function to safely convert a cftime object to datetime
            
                    # Convert time values based on their type
                    if hasattr(time_values[0], 'calendar'):
                        # These are cftime objects
                        time_dates = [safe_convert_to_datetime(t) for t in time_values]
                    elif isinstance(time_values[0], (str, np.str_)):
                        # If they're already strings, parse them directly
                        time_dates = [pd.to_datetime(t) for t in time_values]
                    elif isinstance(time_values[0], np.datetime64):
                        # If they're numpy datetime64
                        time_dates = pd.DatetimeIndex(time_values).to_pydatetime().tolist()
                    else:
                        # Fallback for other types
                        time_dates = [safe_convert_to_datetime(t) for t in time_values]
            
                   # Get min and max dates
                    time_min = min(time_dates)
                    time_max = max(time_dates)
            
                    # Stopping if this file is not in the right range
                    if time_max.year < yearStart or time_min.year > yearStop:
                        print("Cancelling; dataset is before " + str(yearStart) + " or after " + str(yearStop))
                    else:
                        # Subset to the year range if needed
                        if time_min.year < yearStart or time_max.year > yearStop:
                            ds = ds.sel(time=slice(str(yearStart) + '-01-01', str(yearStop) + '-12-31'))
                
                        # print("Subsetting to polygon")
                        # Subset with the polygon shape
                        ds1 = subset.subset_shape(
                            ds, shape=pathOfShapefileForSubsetting, buffer=1)
                
                        # print("Processing time")
                        # Process the data
                        time_values = ds1.time.values
                        time_dates = []
                        for t in time_values:
                            # Handle cftime objects
                            if hasattr(t, 'year') and hasattr(t, 'month') and hasattr(t, 'day'):
                                time_dates.append(datetime(t.year, t.month, t.day))
                            else:
                                # Fallback for other formats
                                time_dates.append(pd.to_datetime(str(t)))
                
                        # Create a list to store data rows
                        data_rows = []
                
                        # Iterate through each grid cell
                        for lat_val in ds1.lat.values:
                            for lon_val in ds1.lon.values:
                
                                # print("Looking at grid cell " + str(lat_val) + ", " + str(lon_val))
                                # Extract data for this grid cell
                                cell_data = ds1.sel(lat=lat_val, lon=lon_val, method='nearest')
                
                                # print("Extracting values")
                                # Extract rsds values for all times at this location
                                variable_values = cell_data[nameOfVariable].values
                
                                # print("Creating rows in dataframe")
                                # Create a row for each time point
                                for i, time in enumerate(time_dates):
                                    data_rows.append({
                                        "lat": lat_val,
                                        "lon": lon_val,
                                        "year": time.year,
                                        "month": time.month,
                                        "day": time.day,
                                        # Transforming downwelling shortwave radiation from W/m2 to umol.m2/s-1
                                        # There was a misleading quote in the PnET user Guide that made me believe that PAR = Downwelling Shortwave Radiation.
                                        # But that's not true at all. In fact, Downwelling shortwave radiation is often refered as global solar radiation.
                                        # See https://library.wmo.int/viewer/68695/?offset=3#page=298&viewer=picture&o=search&n=0&q=shortwave . But other references exist.
                                        # So, Downwelling shortave radiation is for wavelengths of 0.2–4.0 μm; PAR is for 0.4–0.7 μm.
                                        # As such, to convert our Downwelling Shortwave Radiation in W/m2 to PAR in umol.m2/s-1, 
                                        # we must multiply it by 2.02 as indicated in the user guide of PnET Succession.
                                        nameOfVariable: variable_values[i]
                                    })
                
                        # Convert to DataFrame and append to main DataFrame
                        period_df = pd.DataFrame(data_rows)
                        all_data = pd.concat([all_data, period_df], ignore_index=True)
            
                except Exception as e:
                    print(f"Error processing {datasetID}: {e}")
                    continue
            else:
                print("Not processing file " + str(cat.datasets[datasetID]) + " in dataset because it is not a .nc file")
    return(all_data)

def GetDataFromESGFdataset(datasetURL,
                            yearStart,
                            yearStop,
                            pathOfShapefileForSubsetting,
                            nameOfVariable,
                            enableWarnings = True):
    """
    Used to read a data catalog from a ESGF node, but only for a single .nc file instead of a whole .gn catalog,
    check if the data is only for the years of interest, subset the data in a polygon,
    and get the values for the variables to then put them in a panda data frame.
    """
    
    with warnings.catch_warnings():
        if not enableWarnings:
            warnings.simplefilter("ignore")  # Ignore all warnings
        else:
            warnings.simplefilter("default") # Use the default warning behavior
    
        # We initialize the time coder to help xarray decode time using the "cftime" format
        # It's the recommand approach based on the warning messages we get without it
        print("Loading catalog")
        time_coder = xr.coders.CFDatetimeCoder(use_cftime=True)
        
        # Create an empty DataFrame to store all results
        all_data = pd.DataFrame(columns=["lat", "lon", "year", "month", "day", nameOfVariable])
        

        
        print(f"Processing {os.path.basename(datasetURL)}...")
    
        # Open the dataset
        try:
            ds = xr.open_dataset(datasetURL, decode_times=time_coder)
    
            # print("Checking dataset " + str(os.path.basename(url)))
            # Handle time values properly
            # First, convert the time values to strings and then to datetime objects
            time_values = ds.time.values
    
            # Check the type of time values and convert accordingly
            # Function to safely convert a cftime object to datetime
    
            # Convert time values based on their type
            print(time_values[1])
            if hasattr(time_values[0], 'calendar'):
                print("CFTtime object detected")
                # These are cftime objects
                time_dates = [safe_convert_to_datetime(t) for t in time_values]
            elif isinstance(time_values[0], (str, np.str_)):
                print("Date string detected")
                # If they're already strings, parse them directly
                time_dates = [pd.to_datetime(t) for t in time_values]
            elif isinstance(time_values[0], np.datetime64):
                print("Numpy datetime64 detected")
                # If they're numpy datetime64
                time_dates = pd.DatetimeIndex(time_values).to_pydatetime().tolist()
            else:
                # Fallback for other types
                time_dates = [safe_convert_to_datetime(t) for t in time_values]
    
           # Get min and max dates
            time_min = min(time_dates)
            time_max = max(time_dates)
    
            # Stopping if this file is not in the right range
            if time_max.year < yearStart or time_min.year > yearStop:
                print("Cancelling; dataset is before " + str(yearStart) + " or after " + str(yearStop))
            else:
                # Subset to the year range if needed
                if time_min.year < yearStart or time_max.year > yearStop:
                    ds = ds.sel(time=slice(str(yearStart) + '-01-01', str(yearStop) + '-12-31'))
        
                # print("Subsetting to polygon")
                # Subset with the polygon shape
                ds1 = subset.subset_shape(
                    ds, shape=pathOfShapefileForSubsetting, buffer=1)
        
                # print("Processing time")
                # Process the data
                time_values = ds1.time.values
                time_dates = []
                for t in time_values:
                    # Handle cftime objects
                    if hasattr(t, 'year') and hasattr(t, 'month') and hasattr(t, 'day'):
                        time_dates.append(datetime(t.year, t.month, t.day))
                    else:
                        # Fallback for other formats
                        time_dates.append(pd.to_datetime(str(t)))
        
                # Create a list to store data rows
                data_rows = []
        
                # Iterate through each grid cell
                for lat_val in ds1.lat.values:
                    for lon_val in ds1.lon.values:
        
                        # print("Looking at grid cell " + str(lat_val) + ", " + str(lon_val))
                        # Extract data for this grid cell
                        cell_data = ds1.sel(lat=lat_val, lon=lon_val, method='nearest')
        
                        # print("Extracting values")
                        # Extract rsds values for all times at this location
                        variable_values = cell_data[nameOfVariable].values
        
                        # print("Creating rows in dataframe")
                        # Create a row for each time point
                        for i, time in enumerate(time_dates):
                            data_rows.append({
                                "lat": lat_val,
                                "lon": lon_val,
                                "year": time.year,
                                "month": time.month,
                                "day": time.day,
                                # Transforming downwelling shortwave radiation from W/m2 to umol.m2/s-1
                                # There was a misleading quote in the PnET user Guide that made me believe that PAR = Downwelling Shortwave Radiation.
                                # But that's not true at all. In fact, Downwelling shortwave radiation is often refered as global solar radiation.
                                # See https://library.wmo.int/viewer/68695/?offset=3#page=298&viewer=picture&o=search&n=0&q=shortwave . But other references exist.
                                # So, Downwelling shortave radiation is for wavelengths of 0.2–4.0 μm; PAR is for 0.4–0.7 μm.
                                # As such, to convert our Downwelling Shortwave Radiation in W/m2 to PAR in umol.m2/s-1, 
                                # we must multiply it by 2.02 as indicated in the user guide of PnET Succession.
                                nameOfVariable: variable_values[i]
                            })
        
                # Convert to DataFrame and append to main DataFrame
                period_df = pd.DataFrame(data_rows)
                all_data = pd.concat([all_data, period_df], ignore_index=True)
            
        except Exception as e:
            print(f"Error processing {datasetID}: {e}")
    return(all_data)