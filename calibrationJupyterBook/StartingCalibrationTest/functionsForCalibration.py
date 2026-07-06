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
import os
import re
import glob
import matplotlib.pyplot as plt
import pandas as pd
import subprocess
import rasterio
from rasterio.mask import mask
import numpy as np
import shutil
import sqlite3
import time
import sys
import pexpect
import nbformat
from shapely.geometry import Point
import xarray as xr
from datetime import datetime, timedelta
import geopandas as gpd
import cftime
import requests
from tqdm import tqdm
import matplotlib.dates as mdates
from matplotlib.colors import to_rgba, LinearSegmentedColormap
from matplotlib.gridspec import GridSpec
from rasterio.windows import Window
from rasterio.warp import transform
from scipy.spatial.distance import cdist
from siphon.catalog import TDSCatalog
import clisops.core.subset
import warnings
import math
from scipy import ndimage
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d
from itertools import cycle
from pygam import LinearGAM, s, te
from collections import defaultdict
import calendar
from datetime import date
from dateutil.relativedelta import relativedelta
import pytz
import suncalc
from suncalc import get_times
import json
from rasterio.transform import Affine
import csv
import random

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



def parse_All_LANDIS_PnET_Files(folder_path, printFiles = False):
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
                # if printFiles: print(content_separated)

                # Now, we do things differently in the case of each parameter file found

                # Main scenario file
                if ["LandisData", "Scenario"] in content_separated:
                    if printFiles: print("Found : Main scenario file : " + str(filename))
                    results[filename] = parse_LANDIS_SimpleParameterFile(file_path)

                # Main species parameter core file
                elif ["LandisData", "Species"] in content_separated:
                    if printFiles: print("Found : Main species parameter file : " + str(filename))
                    results[filename] = parse_LANDIS_SpeciesCoreFile(file_path)

                # Ecoregions file
                elif ["LandisData", "Ecoregions"] in content_separated:
                    if printFiles: print("Found : Main ecoregions file : " + str(filename))
                    results[filename] = parse_ecoregions_file(file_path)

                # Initial Communities file
                # We add the detection of the .csv file header/columns
                elif ["LandisData", "Initial Communities"] in content_separated or ["LandisData", '"Initial Communities"'] in content_separated or ["LandisData", "Initial", "Communities"] in content_separated or ['LandisData', '"Initial', 'Communities"'] in content_separated or ["MapCode,SpeciesName,CohortAge,CohortBiomass"] in content_separated:
                    if printFiles: print("Found : Initial Communities file : " + str(filename))
                    # Now that initial communities are a .csv file, we can load this file in a panda dataframe for easy editing
                    results[filename] = pd.read_csv(file_path)

                # Climate data file
                elif ["Year", "Month", "TMax", "TMin", "PAR", "Prec", "CO2"] in content_separated:
                    if printFiles: print("Found : Climate file : " + str(filename))
                    results[filename] = parse_LANDIS_ClimateDataFile(file_path)

                # PnET Ecoregion parameter files
                elif ["LandisData", "EcoregionParameters"] in content_separated:
                    if printFiles: print("Found : PnET Ecoregion parameter file : " + str(filename))
                    results[filename] = parse_PnET_ComplexTableParameterfile(file_path)

                # PnET Succession Main file
                elif ["LandisData", "PnET-Succession"] in content_separated or ["LandisData", '"PnET-Succession"'] in content_separated:
                    if printFiles: print("Found : Main PnET parameter file : " + str(filename))
                    results[filename] = parse_LANDIS_SimpleParameterFile(file_path)  

                # PnET Generic parameters file
                elif ["LandisData", "PnETGenericParameters"] in content_separated:
                    if printFiles: print("Found : PnET generic parameter file : " + str(filename))
                    results[filename] = parse_LANDIS_SimpleParameterFile(file_path)
                    
                # PnET Species Parameter file
                elif ["LandisData", "PnETSpeciesParameters"] in content_separated:
                    if printFiles: print("Found : PnET species parameters file : " + str(filename))
                    results[filename] = parse_PnET_ComplexTableParameterfile(file_path)

                # PnET Output Biomass file
                elif ["LandisData", "Output-PnET"] in content_separated or ["LandisData", '"Output-PnET"'] in content_separated: 
                    if printFiles: print("Found : PnET OutputBiomass parameter file : " + str(filename))
                    results[filename] = parse_LANDIS_SimpleParameterFile(file_path)

                # PnET Output Sites File
                elif ["LandisData", "PNEToutputsites"] in content_separated:
                    if printFiles: print("Found : PnET OutputSites parameter file : " + str(filename))
                    results[filename] = parse_LANDIS_SimpleParameterFile(file_path)

                # PnET Disturbance Reduction File
                elif ["LandisData", "DisturbanceReductions"] in content_separated:
                    if printFiles: print("Found : Disturbance Reduction File : " + str(filename))
                    results[filename] = parse_PnET_ComplexTableParameterfile(file_path)
                
                # If it's not a parameter file, we record its path
                # Can be used to copy/paste any other file
                else:
                    if printFiles: print("Found : Additional file : " + str(filename))
                    results[filename] = file_path
                    # If it's not a parameter file, we keep its path to copy it
                    
        else:
            if printFiles: print("Found : Additional file : " + str(filename))
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
            values_line = "\t".join([species_name] + [f"{value}".rstrip('0').rstrip('.') if isinstance(value, float) else str(value) for value in parameters.values()])
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

def write_all_LANDIS_files(outputFolder, dataDict, copyNonParsedFiles = True, printFiles = False):
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
                    if printFiles: print("Found : Main scenario file : " + str(filename))
                    write_LANDIS_SimpleParameterFile(outputFolder + str(filename), dataDict[filename])
        
                # Main species parameter core file
                elif dataDict[filename]["LandisData"] == "Species":
                    if printFiles:print("Found : Main species parameter file : " + str(filename))
                    write_LANDIS_SpeciesCoreFile(outputFolder + str(filename), dataDict[filename])
        
                # Ecoregions file
                elif dataDict[filename]["LandisData"] == "Ecoregions":
                    if printFiles:print("Found : Main ecoregions file : " + str(filename))
                    write_LANDIS_MainEcoregionsFile(outputFolder + "ecoregion.txt", dataDict[filename])
        
                # Initial Communities file
                # Dealing with the .csv file is out of it if statement, see below
                elif dataDict[filename]["LandisData"] == "Initial Communities" or dataDict[filename]["LandisData"] == '"Initial Communities"':
                    if printFiles:print("Found : Initial Communities file : " + str(filename))
                    write_LANDIS_SimpleParameterFile(outputFolder + str(filename), dataDict[filename])
        
                # PnET Ecoregion parameter files
                elif dataDict[filename]["LandisData"] == "EcoregionParameters":
                    if printFiles:print("Found : PnET Ecoregion parameter file : " + str(filename))
                    write_PnET_ComplexTableParameterfile(outputFolder + str(filename), dataDict[filename])
        
                # PnET Succession Main file
                elif dataDict[filename]["LandisData"] == "PnET-Succession" or dataDict[filename]["LandisData"] == '"PnET-Succession"': 
                    if printFiles:print("Found : Main PnET parameter file : " + str(filename))
                    write_LANDIS_SimpleParameterFile(outputFolder + str(filename), dataDict[filename])
        
                # PnET Generic parameters file
                elif dataDict[filename]["LandisData"] == "PnETGenericParameters": 
                    if printFiles:print("Found : PnET generic parameter file : " + str(filename))
                    write_LANDIS_SimpleParameterFile(outputFolder + str(filename), dataDict[filename])
                    
                # PnET Species Parameter file
                elif dataDict[filename]["LandisData"] == "PnETSpeciesParameters":
                    if printFiles:print("Found : PnET species parameters file : " + str(filename))
                    write_PnET_ComplexTableParameterfile(outputFolder + str(filename), dataDict[filename])
        
                # PnET Output Biomass file
                elif dataDict[filename]["LandisData"] == "Output-PnET" or dataDict[filename]["LandisData"] == '"Output-PnET"':
                    if printFiles:print("Found : PnET OutputBiomass parameter file : " + str(filename))
                    write_LANDIS_SimpleParameterFile(outputFolder + str(filename), dataDict[filename])
        
                # PnET Output Sites File
                elif dataDict[filename]["LandisData"] == "PNEToutputsites": 
                    if printFiles:print("Found : PnET OutputSites parameter file : " + str(filename))
                    write_LANDIS_SimpleParameterFile(outputFolder + str(filename), dataDict[filename])
        
                # PnET Disturbance Reduction File
                elif dataDict[filename]["LandisData"] == "DisturbanceReductions":
                    if printFiles:print("Found : Disturbance Reduction File : " + str(filename))
                    write_PnET_ComplexTableParameterfile(outputFolder + str(filename), dataDict[filename])
            
            else: # If it's a dict but there is no LandisData parameter, it must be the climate file
                
                # Climate data file - identified by looking at the keys of the first month of the first year in the dict
                if  any(item in list(list(dataDict[filename].values())[0].values())[0].keys() for item in ["Year", "Month", "TMax", "TMin", "PAR", "Prec", "CO2"]):
                    if printFiles:print("Found : Climate file : " + str(filename))
                    write_climate_data(outputFolder + str(filename), dataDict[filename])
                    
                else:
                    print("WARNING : file type not recognized for the following dictionnary :" + str(dataDict[filename]))

        elif "DataFrame" in str(type(dataDict[filename])): # Dealing with dataframe (often loaded from .csv files) to output them back as .csv
                if printFiles: print("Found : dataframe : " + str(filename) + ", exporting to .csv")
                dataDict[filename].to_csv(outputFolder + str(filename), index=False)
            
        # If it's not a parameter file, we record its path
        # Can be used to copy/paste any other file
        else:
            if printFiles: print("Found : Additional file : " + str(filename))
            if os.path.isfile(dataDict[filename]):
                shutil.copy(dataDict[filename], outputFolder)
            elif os.path.isdir(dataDict[filename]):
                shutil.copytree(dataDict[filename], outputFolder + "/" + (os.path.basename(dataDict[filename])), dirs_exist_ok  = True)
            else:
                print("WARNING : " + str(filename) + " is neither a file or a folder, can't copy to " + str(outputFolder))
    

#%% FUNCTIONS TO LAUNCH A LANDIS-II SIMULATION

def runLANDIS_Simulation(simulationFolder, scenarioFileName, printSim = False, timeout = None, printRunning = True):
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
        if printRunning:
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
                    if printRunning:
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


def plot_TimeSeries_CSV_PnETSitesOutputs(df, referenceDict = {}, columnToPlotSelector = [], trueTime = False, realBiomass = True, cellLength = 30, referenceLabel = "Reference curve", labelOfFirstCurve = ""):
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
    # To speed up : If we're in same year/month, we avoid re-reading the data
    year_previous = "-99"
    month_previous = "-99"
    value_previous = 0
    for index, row in df.iterrows():
        year = int(row['Year'])
        month = int(row['Month'])

        if year_previous == year and month_previous == month:
            df.at[index, 'CO2_Concentration'] = value_previous
        
        else:
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
    
                    # We record this measurement to go faster next time
                    year_previous = year
                    month_previous = month
                    value_previous = np.nanmean(masked_values)
                except (IndexError, KeyError):
                    print(f"Time {time_str} not found in historical dataset")
    
            elif year == 2014:
                print("You are outside of historical data ! Use process_co2_data_withFutureInterpolation to deal with future data from https://zenodo.org/records/5021361")

    return df


def process_co2_data_monthly(historical_ds, shapefile, df):
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
            # Get all days in this month from the historical dataset
            start_date = f"{year}-{month:02d}-01"
            # Determine last day of month
            if month == 12:
                end_date = f"{year}-{month:02d}-31"
            else:
                next_month = month + 1
                end_date = f"{year}-{next_month:02d}-01"

            try:
                # Select all time steps for this month
                month_data = historical_ds.sel(Times=slice(start_date, end_date))

                # Extract CO2 values for all days in the month and apply the mask
                co2_month = month_data.value.values

                # Apply spatial mask and calculate mean across space and time
                monthly_values = []
                for time_idx in range(co2_month.shape[0]):
                    masked_values = co2_month[time_idx][hist_mask]
                    if len(masked_values) > 0 and not np.all(np.isnan(masked_values)):
                        monthly_values.append(np.nanmean(masked_values))

                # Calculate monthly average
                if len(monthly_values) > 0:
                    df.at[index, 'CO2_Concentration'] = np.mean(monthly_values)

            except (IndexError, KeyError, ValueError) as e:
                print(f"Error processing {year}-{month:02d}: {e}")

        elif year == 2014:
            print("You are outside of historical data! Use process_co2_data_withFutureInterpolation to deal with future data from https://zenodo.org/records/5021361")
            print("For now, will use values from the latest year to deal with this row")
            # Get all days in this month from the historical dataset
            start_date = f"{2013}-{month:02d}-01"
            # Determine last day of month
            if month == 12:
                end_date = f"{2013}-{month:02d}-31"
            else:
                next_month = month + 1
                end_date = f"{2013}-{next_month:02d}-01"

            try:
                # Select all time steps for this month
                month_data = historical_ds.sel(Times=slice(start_date, end_date))

                # Extract CO2 values for all days in the month and apply the mask
                co2_month = month_data.value.values

                # Apply spatial mask and calculate mean across space and time
                monthly_values = []
                for time_idx in range(co2_month.shape[0]):
                    masked_values = co2_month[time_idx][hist_mask]
                    if len(masked_values) > 0 and not np.all(np.isnan(masked_values)):
                        monthly_values.append(np.nanmean(masked_values))

                # Calculate monthly average
                if len(monthly_values) > 0:
                    df.at[index, 'CO2_Concentration'] = np.mean(monthly_values)

            except (IndexError, KeyError, ValueError) as e:
                print(f"Error processing {year}-{month:02d}: {e}")

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
    # To speed up : If we're in same year/month, we avoid re-reading the data
    year_previous = "-99"
    month_previous = "-99"
    value_previous = 0
    for index, row in df.iterrows():
        year = int(row['Year'])
        month = int(row['Month'])

        if year_previous == year and month_previous == month:
            df.at[index, 'CO2_Concentration'] = value_previous

        else:
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

                    # We record this measurement to go faster next time
                    year_previous = year
                    month_previous = month
                    value_previous = np.nanmean(masked_values)
                    
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

                    # We record this measurement to go faster next time
                    year_previous = year
                    month_previous = month
                    value_previous = np.nanmean(masked_values)
                        
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

def convert_to_daylight_average(ds, variable_name):
    """
    Converts monthly average values (over all hours) to daylight-hour averages for a given variable.

    Args:
        ds (xr.Dataset): xarray Dataset containing the variable to convert
        variable_name (str): Name of the variable to convert (e.g., 'rsds')

    Returns:
        xr.Dataset: Original dataset with new variable '{variable_name}_MonthlyDaytime' added
    """
    da = ds[variable_name]

    time_coord = da.time.values
    lat_coord = da.rlat.values
    lon_coord = da.rlon.values

    n_time = len(time_coord)
    n_lat = len(lat_coord)
    n_lon = len(lon_coord)

    print(f"Processing {n_time} time steps, {n_lat} latitudes, {n_lon} longitudes")

    daylight_hours_array = np.zeros((n_time, n_lat, n_lon))
    days_in_month_array = np.zeros(n_time)

    # Create meshgrid of all lat/lon combinations once
    lon_grid, lat_grid = np.meshgrid(lon_coord, lat_coord)
    lon_flat = lon_grid.flatten()
    lat_flat = lat_grid.flatten()

    for t_idx, time_val in enumerate(tqdm(time_coord, desc="Processing months")):
        dt = np.datetime64(time_val, 'D').astype(datetime)
        year = dt.year
        month = dt.month
        days_in_month = calendar.monthrange(year, month)[1]
        days_in_month_array[t_idx] = days_in_month

        # Accumulate daylight hours for all days in month
        monthly_daylight = np.zeros((n_lat, n_lon))

        for day in range(1, days_in_month + 1):
            date = datetime(year, month, day)

            # Vectorized call: pass arrays of dates, lons, lats
            dates = np.full(len(lon_flat), date)
            times_df = get_times(dates, lon_flat, lat_flat)

            # Calculate daylight hours
            sunrise = times_df['sunrise'].values
            sunset = times_df['sunset'].values

            # Handle cases where sunset < sunrise (crosses midnight)
            daylight_seconds = (sunset - sunrise).astype('timedelta64[s]').astype(float)
            daylight_hours = daylight_seconds / 3600.0

            # Reshape back to grid
            daylight_grid = daylight_hours.reshape(n_lat, n_lon)
            monthly_daylight += daylight_grid

        daylight_hours_array[t_idx, :, :] = monthly_daylight

    print("\nCalculating daylight averages...")

    # Vectorized calculation
    total_hours = (days_in_month_array * 24).reshape(-1, 1, 1)
    total_energy = da.values * total_hours

    output_data = np.where(
        daylight_hours_array > 0,
        total_energy / daylight_hours_array,
        0.0
    )

    output_data = np.where(np.isnan(da.values), np.nan, output_data)

    # Create new DataArray
    new_var_name = f"{variable_name}_MonthlyDaytime"
    ds[new_var_name] = xr.DataArray(
        output_data,
        coords=da.coords,
        dims=da.dims,
        attrs={**da.attrs, 'description': f'Daylight-hour average of {variable_name}'}
    )

    print("Done!")
    return ds


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
        # print("Loading catalog")
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
                            verbose = False,
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
        # print("Loading catalog")
        time_coder = xr.coders.CFDatetimeCoder(use_cftime=True)
        
        # Create an empty DataFrame to store all results
        all_data = pd.DataFrame(columns=["lat", "lon", "year", "month", "day", nameOfVariable])
        

        if verbose:
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
            # print(time_values[1])
            if hasattr(time_values[0], 'calendar'):
                # print("CFTtime object detected")
                # These are cftime objects
                time_dates = [safe_convert_to_datetime(t) for t in time_values]
            elif isinstance(time_values[0], (str, np.str_)):
                # print("Date string detected")
                # If they're already strings, parse them directly
                time_dates = [pd.to_datetime(t) for t in time_values]
            elif isinstance(time_values[0], np.datetime64):
                # print("Numpy datetime64 detected")
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
                if verbose:
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
                # print("Lat values : " + str(ds1.lat.values))
                # print("Lon values : " + str(ds1.lon.values))
                for lat_val in ds1.lat.values:
                    for lon_val in ds1.lon.values:
        
                        # print("Looking at grid cell " + str(lat_val) + ", " + str(lon_val))
                        # Extract data for this grid cell
                        cell_data = ds1.sel(lat=lat_val, lon=lon_val, method='nearest')
        
                        # print("Extracting values")
                        # Extract rsds values for all times at this location
                        variable_values = cell_data[nameOfVariable].values
                        # print("Variable values :" + str(variable_values))
        
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
            print(f"Error processing {os.path.basename(datasetURL)}: {e}")
    return(all_data)


def extract_unique_coordinates(dataframe):
    """
    Extract unique (latitude, longitude) pairs from a DataFrame with 'lat' and 'lon' columns.

    Args:
        dataframe (pd.DataFrame): DataFrame containing 'lat' and 'lon' columns

    Returns:
        list: List of unique (latitude, longitude) tuples
    """
    # Check if required columns exist
    if 'lat' not in dataframe.columns or 'lon' not in dataframe.columns:
        raise ValueError("DataFrame must contain 'lat' and 'lon' columns")

    # Extract unique coordinate pairs
    unique_coords = dataframe[['lat', 'lon']].drop_duplicates().values.tolist()

    # Convert to list of tuples
    return [tuple(coord) for coord in unique_coords]

def find_closest_coordinate_to_polygon_center(shapefile_path, coordinates_list):
    """
    Find the coordinate from a list that is closest to the center of a polygon in a shapefile.

    Args:
        shapefile_path (str): Path to the shapefile containing the polygon
        coordinates_list (list): List of (latitude, longitude) tuples

    Returns:
        tuple: The (latitude, longitude) coordinate closest to the polygon center
    """
    # Read the shapefile
    gdf = gpd.read_file(shapefile_path)

    # Ensure the shapefile has at least one polygon
    if len(gdf) == 0:
        raise ValueError("Shapefile contains no polygons")

    # Get the first polygon (or you could iterate through multiple polygons)
    polygon = gdf.geometry.iloc[0]

    # Calculate the centroid of the polygon
    centroid = polygon.centroid
    centroid_lat, centroid_lon = centroid.y, centroid.x

    # Function to calculate Haversine distance (in km) between two coordinates
    def haversine_distance(lat1, lon1, lat2, lon2):
        # Convert decimal degrees to radians
        lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])

        # Haversine formula
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
        c = 2 * math.asin(math.sqrt(a))
        r = 6371  # Radius of Earth in kilometers
        return c * r

    # Find the closest coordinate to the centroid
    min_distance = float('inf')
    closest_coordinate = None

    for coord in coordinates_list:
        lat, lon = coord
        distance = haversine_distance(lat, lon, centroid_lat, centroid_lon)

        if distance < min_distance:
            min_distance = distance
            closest_coordinate = coord

    return closest_coordinate

def transform_to_historical_averages_vectorized(df):
    # Group by month, day, and variable name to calculate historical averages
    historical_averages = df.groupby(['month', 'day', 'Variable'])['eco1'].transform('mean')

    # Create a new dataframe with the transformed values
    df_transformed = df.copy()
    df_transformed['eco1'] = historical_averages

    return df_transformed


def plottingNFISpeciesUpperBiomassByProvince(species_path,
                 age_path,
                 biomass_path,
                 speciesName,
                 dictOfMasks):
    """
    This function is used to plot data from the National Forest Inventory (NFI) of Canada.
    It reads a raster containing the age of the forest pixels; another with the aboveground biomass
    of the forest pixels; and another with the % of biomass abundance of a species of interest.
    It then reads "mask" rasters corresponding to the canadian province.
    The function then use the other function "process_NFI_rasters_to_arraysForPlot" to select
    the points that have the most biomass for a set of age windows, removing any point with low values
    for better visibility. It then plots the selected points on an easy-to-read plot.
    
    Parameters:
    -----------
    species_path : str
        Path to the species prevalence raster file (values 0-100%).
    age_path : str
        Path to the forest age raster file (years).
    biomass_path : str
        Path to the total biomass raster file (tons/ha).
    speciesName : str
        Name of the species being analyzed (used for labeling and output).
    dictOfMasks : dict
        Dictionnary containing the path to the different provinces, associated to the code for each province (e.g. QC, BC, etc.)
    """
    # We create the dictionnary in which we will temporarely put the selected points
    dictOfPointsSelected = {key: "" for key in dictOfMasks}
    
    # We read the rasters and extract the points thanks to a special function
    with rasterio.open(species_path) as src_species:
        with rasterio.open(age_path) as src_age:
            with rasterio.open(biomass_path) as src_biomass:
                print("Reading species raster...")
                species_data = src_species.read(1)
                profile = src_species.profile
                print("Reading age raster...")
                age_data = src_age.read(1)
                print("Reading biomass raster...")
                biomass_data = src_biomass.read(1)
            
                for province in dictOfMasks:
                    # Process the rasters
                    print("Dealing with "+ str(province))
                    age_sample, abies_biomass_sample = process_NFI_rasters_to_arraysForPlot(
                        species_data, age_data, biomass_data, dictOfMasks[province], thresholdMaximumBiomass = 0.90, thresholdMinimumPercentBiomass = 0, verbose = False
                    )
                    # Convert the tons per hectares of the values into g/m2 for easy comparison with LANDIS-II outputs
                    abies_biomass_sample = abies_biomass_sample * 100
                    # Save the results
                    dictOfPointsSelected[province] = [age_sample, abies_biomass_sample]
    
    # Now, we plot the selected points on a graph, with colors for each province
    # We create the scatter plot
    # First, we create the color cycler
    colours = cycle(['#a3be8c', '#5e81ac', '#bf616a', '#d08770', '#ebcb8b', '#2e3440', '#8fbcbb'])
    plt.figure(figsize=(10, 8))
    # Then, we plot the points
    for province in dictOfPointsSelected:
        plt.scatter(dictOfPointsSelected[province][0], dictOfPointsSelected[province][1], alpha=0.6, s=8, c=next(colours), label = province)
    plt.xlabel('Forest Age (years)')
    plt.ylabel(str(speciesName) + ' Biomass (g/m2)')
    plt.title('Relationship between Forest Age and ' + str(speciesName) + ' Biomass')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    # plt.savefig('age_vs_abies_biomass.png', dpi=300)
    plt.show()

def process_NFI_rasters_to_arraysForPlot(species_data, age_data, biomass_data, mask_path=None, thresholdMaximumBiomass=0.90, thresholdMinimumPercentBiomass = 0, verbose = False):
    """
    Takes the rasters from the National Forest Inventory of Canada (see https://open.canada.ca/data/en/dataset/ec9e2659-1c29-4ddb-87a2-6aced147a990
    ; forest age, aboveground biomass, and % of aboveground biomass for a given species), to then return flattenned arrays
    containing data only from pixels in a mask array, and only the pixels with the most biomass
    for the species in 3 years age intervalls. Can also remove any pixel where the species is not
    very present.

    thresholdMaximumBiomass keeps only points (pixels) who's biomass is above thresholdMaximumBiomass*(maximum-minimum) for each 3 year window.
    This allows us to keep only the points with the largest biomass relative to the window.

    thresholdMinimumPercentBiomass keeps only points where the species has a certain amount of relative biomass in the cell (> thresholdMinimumPercentBiomass).
    This allows us to keep only points where the species is sufficiently abundant.

    Returns : age_sample, species_biomass_sample
    Both age_sample and species_biomass_sample can be used to easily plot the values for the filtered pixels.
    """
    
    # # Step 1: Read the rasters
    # with rasterio.open(species_path) as src:
    #     species_data = src.read(1)
    #     profile = src.profile

    # with rasterio.open(age_path) as src:
    #     age_data = src.read(1)

    # with rasterio.open(biomass_path) as src:
    #     biomass_data = src.read(1)

    # Step 2: Mask values where the species below the threshold of biomass percentage
    mask_no_species = species_data <= thresholdMinimumPercentBiomass

    # Step 3: Apply additional mask if provided
    if mask_path:
        if isinstance(mask_path, str):
            with rasterio.open(mask_path) as src:
                additional_mask = src.read(1)
                additional_mask_bool = additional_mask == 0
                combined_mask = np.logical_or(mask_no_species, additional_mask_bool)
        elif isinstance(mask_path, list):
            for maskRasterPath in mask_path:
                try:
                    with rasterio.open(maskRasterPath) as src:
                        additional_mask_toadd = src.read(1)
                        additional_mask += additional_mask_toadd
                except:
                    with rasterio.open(maskRasterPath) as src:
                        additional_mask = src.read(1)
            additional_mask_bool = additional_mask == 0
            combined_mask = np.logical_or(mask_no_species, additional_mask_bool)
                    

    else:
        combined_mask = mask_no_species

    # Create masked arrays
    species_masked = np.ma.array(species_data, mask=combined_mask)
    age_masked = np.ma.array(age_data, mask=combined_mask)
    biomass_masked = np.ma.array(biomass_data, mask=combined_mask)

    # Step 4: Calculate biomass of species of interest
    species_proportion = species_masked / 100.0
    species_biomass = species_proportion * biomass_masked

    # Step 5: Prepare data for plotting
    age_flat = age_masked.compressed()
    species_biomass_flat = species_biomass.compressed()

    # Remove pixels with zero biomass for the species
    non_zero_mask = species_biomass_flat > 0
    age_filtered = age_flat[non_zero_mask]
    species_biomass_filtered = species_biomass_flat[non_zero_mask]

    # Filter points above value of biomass per 3-year age range
    if len(age_filtered) > 0:
        # Create age bins (vectorized)
        age_bins = (age_filtered // 3).astype(int)
        unique_bins = np.unique(age_bins)

        # Pre-allocate arrays for efficiency
        sample_indices = []

        for bin_val in unique_bins:
            bin_mask = age_bins == bin_val
            bin_indices = np.where(bin_mask)[0]

            if len(bin_indices) > 0:
                # Get biomass values for this bin
                bin_biomass = species_biomass_filtered[bin_indices]

                # Calculate threshold based on maximum value
                percent_Maximum = thresholdMaximumBiomass*(np.max(bin_biomass)-np.min(bin_biomass))

                # Get indices where biomass is above a percentage of the maximum
                above_percentile_mask = bin_biomass >= percent_Maximum
                selected_local_indices = np.where(above_percentile_mask)[0]

                # Convert back to global indices
                sample_indices.extend(bin_indices[selected_local_indices])

        # Extract sampled data
        sample_indices = np.array(sample_indices)
        age_sample = age_filtered[sample_indices]
        species_biomass_sample = species_biomass_filtered[sample_indices]
    else:
        age_sample = np.array([])
        species_biomass_sample = np.array([])

    if verbose:
        print(f"Total pixels after initial masking: {len(age_flat):,}")
        print(f"Pixels with non-zero species biomass: {len(age_filtered):,}")
        print(f"Age range: {int(np.min(age_filtered)) if len(age_filtered) > 0 else 0} - {int(np.max(age_filtered)) if len(age_filtered) > 0 else 0} years")
        # print(f"Sample size (top {points_per_interval} per 3-year range): {len(age_sample):,}")

    return age_sample, species_biomass_sample


def processNFI_RastersIntoDataFrameForGAM(age_raster_path, biomass_raster_path, abundance_raster_path, mask_raster_path="None"):
    """
    Process forest rasters and create dataframe with biomass analysis using pandas groupby optimization
    """

    # Read rasters into numpy arrays
    with rasterio.open(age_raster_path) as src:
        age_data = src.read(1).astype(np.float32)

    with rasterio.open(biomass_raster_path) as src:
        biomass_data = src.read(1).astype(np.float32)

    with rasterio.open(abundance_raster_path) as src:
        abundance_data = src.read(1).astype(np.float32)

    if mask_raster_path != "None":
        with rasterio.open(mask_raster_path) as src:
            mask_data = src.read(1).astype(np.int32)

    # Apply mask - only process pixels where mask = 1
    if mask_raster_path != "None":
        valid_mask = mask_data == 1
    else:
        valid_mask = age_data != -9999

    # Round age values to 3-year windows (0-3 -> 0, 3-6 -> 3, etc.)
    age_rounded = np.floor(age_data / 3) * 3
    age_rounded = age_rounded.astype(np.int32)

    # Round abundance to nearest percentage
    abundance_rounded = np.round(abundance_data).astype(np.int32)

    # Calculate aboveground biomass for species (biomass * abundance%)
    species_biomass = biomass_data * (abundance_data / 100.0)

    # Get province name from mask raster filename
    if mask_raster_path != "None":
        province_name = os.path.splitext(os.path.basename(mask_raster_path))[0]
    else:
        province_name = "All of Canada"

    # Create valid data mask
    biomass_valid = (
        valid_mask & 
        ~np.isnan(species_biomass) & 
        (species_biomass >= 0) &
        ~np.isnan(age_rounded) &
        ~np.isnan(abundance_rounded) &
        (abundance_rounded >= 0)
    )

    print(f"Processing {np.sum(biomass_valid)} valid pixels out of {biomass_valid.size} total pixels")

    # Create DataFrame with valid pixels only
    df_pixels = pd.DataFrame({
        'age': age_rounded[biomass_valid],
        'abundance': abundance_rounded[biomass_valid],
        'biomass': species_biomass[biomass_valid]
    })

    print(f"Created pixel dataframe with {len(df_pixels)} rows")

    results = []

    # Group by age and abundance combinations
    grouped = df_pixels.groupby(['age', 'abundance'])['biomass']

    print(f"Found {len(grouped)} unique age-abundance combinations")

    for (age_val, abundance_val), group in tqdm(grouped, desc="Processing groups"):
        selected_biomass = group.values

        # Handle single pixel case
        if len(selected_biomass) == 1:
            upper_biomass_value = selected_biomass[0]
        else:
            # Remove pixels above 99.5th percentile
            percentile_99_5 = np.percentile(selected_biomass, 99.5)
            filtered_biomass = selected_biomass[selected_biomass <= percentile_99_5]

            # Skip if no pixels remain after filtering
            if len(filtered_biomass) == 0:
                continue

            # Calculate 95% of maximum value from filtered pixels (without the top outliers)
            # This is the value we will put in the row
            max_value = np.max(filtered_biomass)
            upper_biomass_value = max_value * 0.95

        # Add row to results
        results.append({
            'Age': int(age_val),
            'Province': province_name,
            '% Biomass of species': int(abundance_val),
            'Upper Biomass value': upper_biomass_value
        })

    # Create final dataframe from results
    result_df = pd.DataFrame(results)

    print(f"Generated final dataframe with {len(result_df)} rows")

    return result_df

def analyze_species_growth_curves(
    prevalence_raster_path,
    age_raster_path,
    biomass_raster_path,
    species_name,
    output_csv_path,
    prevalence_thresholds=[100, 80, 60, 40, 20],
    peak_age_min=0,
    peak_age_max=10,
    senescence_age_window=5,
    outlier_age_window=5,
    outlier_percentile=99.80,
    smoothing_window=21,
    smoothing_polyorder=3,
    n_sample_plot=2000000,
    create_plot=True,
    plot_figsize=(20, 8),
    maskPath="NONE"
):
    """
    Analyze species growth curves across different prevalence thresholds using raster data.

    This function processes forest raster data to construct growth curves showing how biomass 
    changes with forest age for a given species. It creates multiple curves based on different 
    prevalence thresholds to understand how species dominance affects growth patterns. The 
    analysis includes outlier removal, curve construction with growth and senescence phases, 
    interpolation to yearly intervals, and smoothing. Results are saved as CSV and optionally 
    visualized in plots.

    Parameters:
    -----------
    prevalence_raster_path : str
        Path to the species prevalence raster file (values 0-100%).
    age_raster_path : str
        Path to the forest age raster file (years).
    biomass_raster_path : str
        Path to the total biomass raster file (tons/ha).
    species_name : str
        Name of the species being analyzed (used for labeling and output).
    output_csv_path : str
        Path where the output CSV file will be saved.
    prevalence_thresholds : list of float, default=[100, 80, 50, 30, 20]
        List of prevalence thresholds (%) to analyze. For each threshold, only pixels 
        with prevalence ≤ threshold will be included in the analysis.
    peak_age_min : float, default=0
        Minimum age (years) to search for peak biomass.
    peak_age_max : float, default=10
        Maximum age (years) to search for peak biomass.
    senescence_age_window : float, default=5
        Age window (years) for selecting points in the senescence phase.
    outlier_age_window : float, default=5
        Age window (years) for outlier removal using discrete age bins.
    outlier_percentile : float, default=99.80
        Percentile threshold for outlier removal within each age window.
    smoothing_window : int, default=21
        Window length for Savitzky-Golay smoothing (must be odd).
    smoothing_polyorder : int, default=3
        Polynomial order for Savitzky-Golay smoothing.
    n_sample_plot : int, default=2000000
        Maximum number of points to sample for visualization.
    create_plot : bool, default=True
        Whether to create and display the visualization plots.
    plot_figsize : tuple, default=(20, 8)
        Figure size for the plots (width, height).

    Returns:
    --------
    dict
        Dictionary containing the analysis results with keys:
        - 'curves_data': Dictionary with curve data for each threshold
        - 'summary_stats': Summary statistics for each curve
        - 'processing_time': Total processing time in seconds
        - 'n_pixels_processed': Number of pixels processed
        - 'output_csv_path': Path to the saved CSV file

    Output CSV Structure:
    --------------------
    The CSV file contains columns:
    - 'age': Forest age in years (integer values)
    - For each threshold T: 'raw_curve_T%' and 'smoothed_curve_T%' columns
      containing biomass values (tons/ha) for that threshold
    """

    def read_rasters():
        """Read the three raster files and return their data arrays"""
        print("Reading raster files...")

        with rasterio.open(prevalence_raster_path) as src:
            prevalence = src.read(1).astype(np.float32)
            prevalence[prevalence < 0] = np.nan

        with rasterio.open(age_raster_path) as src:
            age = src.read(1).astype(np.float32)
            age[age <= 0] = np.nan

        with rasterio.open(biomass_raster_path) as src:
            biomass = src.read(1).astype(np.float32)
            biomass[biomass < 0] = np.nan

        if maskPath != "NONE": 
            with rasterio.open(maskPath) as src:
                maskRaster = src.read(1).astype(np.float32)
        else:
            maskRaster = "NONE"

        print("✓ Rasters loaded successfully")
        return prevalence, age, biomass, maskRaster

    def process_data(prevalence, age, biomass, maskRaster, maskPath):
        """Process the data and create masks"""
        print("Processing data and applying masks...")

        # Mask pixels with no species presence
        mask = (prevalence > 0) & (~np.isnan(prevalence)) & (~np.isnan(age)) & (~np.isnan(biomass))

        # Calculate species biomass (prevalence as proportion * total biomass)
        species_biomass = (prevalence / 100.0) * biomass

        # We further mask the data with a raster mask
        if maskPath != "NONE":
            # Create combined mask
            # print("maskRaster :")
            # print(str(maskRaster))
            # print("Mask :")
            # print(str(mask))
            mask = (maskRaster > 0) & (mask > 0)
        
        # Flatten arrays and apply mask
        age_flat = age[mask]
        species_biomass_flat = species_biomass[mask]
        prevalence_flat = prevalence[mask]
        
        print(f"✓ {len(age_flat)} valid pixels found")
        return age_flat, species_biomass_flat, prevalence_flat

    def remove_outliers_fast(age_data, biomass_data, prevalence_data, age_window, percentile):
        """Remove outliers using discrete age windows - FAST METHOD"""
        print(f"Removing outliers (age window: {age_window} years, percentile: {percentile}%)...")

        df = pd.DataFrame({
            'age': age_data,
            'biomass': biomass_data,
            'prevalence': prevalence_data
        })

        # Create discrete age bins
        min_age = df['age'].min()
        max_age = df['age'].max()
        age_bins = np.arange(min_age, max_age + age_window, age_window)

        print(f"Processing {len(age_bins)-1} age windows...")

        # Assign each point to an age bin
        df['age_bin'] = pd.cut(df['age'], bins=age_bins, include_lowest=True, right=False)

        outlier_mask = np.ones(len(df), dtype=bool)

        # Process each age bin
        for bin_label in tqdm(df['age_bin'].cat.categories, desc="Processing age windows"):
            bin_mask = df['age_bin'] == bin_label

            if bin_mask.sum() > 0:
                bin_data = df[bin_mask]
                threshold = np.percentile(bin_data['biomass'], percentile)

                # Mark outliers in this bin
                outlier_indices = bin_data[bin_data['biomass'] > threshold].index
                outlier_mask[outlier_indices] = False

        filtered_df = df[outlier_mask]
        print(f"✓ {len(filtered_df)} pixels remaining after outlier removal")

        return filtered_df['age'].values, filtered_df['biomass'].values, filtered_df['prevalence'].values

    def filter_by_prevalence_threshold(age_data, biomass_data, prevalence_data, threshold):
        """Filter data to only include points below the prevalence threshold"""
        mask = prevalence_data <= threshold
        filtered_age = age_data[mask]
        filtered_biomass = biomass_data[mask]
        filtered_prevalence = prevalence_data[mask]

        print(f"  Threshold {threshold}%: {len(filtered_age)} points remaining ({len(filtered_age)/len(age_data)*100:.1f}% of original)")

        return filtered_age, filtered_biomass, filtered_prevalence

    def find_peak_in_age_range(age_data, biomass_data, age_min, age_max):
        """Find peak by looking in a specific age range"""
        # Filter data to specified age range
        age_mask = (age_data >= age_min) & (age_data <= age_max)

        if not np.any(age_mask):
            print(f"    Warning: No data found in age range {age_min}-{age_max}. Using global maximum.")
            peak_idx = np.argmax(biomass_data)
            peak_age = age_data[peak_idx]
            peak_biomass = biomass_data[peak_idx]
        else:
            range_biomass = biomass_data[age_mask]
            range_ages = age_data[age_mask]

            peak_idx = np.argmax(range_biomass)
            peak_age = range_ages[peak_idx]
            peak_biomass = range_biomass[peak_idx]

        return peak_age, peak_biomass

    def construct_growth_curve(age_data, biomass_data, peak_age_min, peak_age_max, senescence_age_window):
        """Construct the raw growth curve with three phases"""

        if len(age_data) == 0:
            print("    Warning: No data available for curve construction")
            return np.array([[0, 0]])

        df = pd.DataFrame({'age': age_data, 'biomass': biomass_data})
        df = df.sort_values('age').reset_index(drop=True)

        # Find peak using specified age range
        peak_age, peak_biomass = find_peak_in_age_range(age_data, biomass_data, peak_age_min, peak_age_max)

        curve_points = []

        # Phase 1: Initial growth (0,0 to peak)
        curve_points.append((0, 0))

        current_biomass = 0
        current_age = 0

        # Pre-filter data for growth phase
        growth_data = df[df['age'] <= peak_age].copy()

        while current_age < peak_age:
            # Find next point: closest age but higher biomass
            candidates = growth_data[(growth_data['age'] > current_age) & 
                                    (growth_data['biomass'] > current_biomass)]

            if len(candidates) == 0:
                break

            # Select closest in age
            next_idx = candidates['age'].idxmin()
            next_point = candidates.loc[next_idx]
            curve_points.append((next_point['age'], next_point['biomass']))
            current_age = next_point['age']
            current_biomass = next_point['biomass']

        # Add peak if not already included
        if current_age != peak_age:
            curve_points.append((peak_age, peak_biomass))

        # Phase 2: Senescence with age window constraint
        current_biomass = peak_biomass
        current_age = peak_age

        # Pre-filter data for senescence phase
        senescence_data = df[df['age'] >= peak_age].copy()

        iteration_count = 0
        max_iterations = 1000  # Safety limit

        while iteration_count < max_iterations:
            # Find next point: lower biomass, within age window, closest to current biomass
            candidates = senescence_data[
                (senescence_data['age'] > current_age) & 
                (senescence_data['age'] <= current_age + senescence_age_window) &  # Age window constraint
                (senescence_data['biomass'] < current_biomass)
            ]

            if len(candidates) == 0:
                break

            # Select point closest in biomass (but lower)
            candidates = candidates.copy()
            candidates['biomass_diff'] = abs(candidates['biomass'] - current_biomass)

            # Find the point with biomass closest to current biomass (but lower)
            next_idx = candidates['biomass_diff'].idxmin()
            next_point = candidates.loc[next_idx]
            curve_points.append((next_point['age'], next_point['biomass']))
            current_age = next_point['age']
            current_biomass = next_point['biomass']

            iteration_count += 1

        curve_array = np.array(curve_points)

        return curve_array

    def interpolate_curve_yearly(curve_points):
        """Interpolate the curve to have one point per year"""

        if len(curve_points) < 2:
            return curve_points

        # Sort by age to ensure proper interpolation
        sorted_indices = np.argsort(curve_points[:, 0])
        sorted_curve = curve_points[sorted_indices]

        # Define age range for interpolation (integer years)
        min_age = int(np.floor(sorted_curve[0, 0]))
        max_age = int(np.ceil(sorted_curve[-1, 0]))

        # Create yearly age points
        yearly_ages = np.arange(min_age, max_age + 1)

        # Interpolate biomass values
        interp_func = interp1d(sorted_curve[:, 0], sorted_curve[:, 1], 
                              kind='linear', bounds_error=False, fill_value='extrapolate')
        yearly_biomass = interp_func(yearly_ages)

        # Combine into interpolated curve
        interpolated_curve = np.column_stack([yearly_ages, yearly_biomass])

        return interpolated_curve

    def smooth_curve_savgol(curve_points, window_length, polyorder):
        """Apply Savitzky-Golay filtering to smooth the curve"""

        if len(curve_points) < 4:
            return curve_points

        if len(curve_points) < window_length:
            # Use maximum possible window length (must be odd)
            window_length = max(3, (len(curve_points) // 2) * 2 - 1)

        # Ensure window_length is odd
        if window_length % 2 == 0:
            window_length += 1

        # Ensure polyorder is less than window_length
        polyorder = min(polyorder, window_length - 1)

        # Apply Savitzky-Golay filter to biomass values
        smoothed_biomass = savgol_filter(curve_points[:, 1], window_length, polyorder)

        # Keep original ages
        smoothed_points = np.column_stack([curve_points[:, 0], smoothed_biomass])

        return smoothed_points

    def weighted_sample_points(age_data, biomass_data, prevalence_data, n_sample):
        """Sample points with probability weights based on prevalence"""
        if len(age_data) <= n_sample:
            print(f"Using all {len(age_data)} points for visualization")
            return age_data, biomass_data, prevalence_data

        print(f"Sampling {n_sample} points with prevalence-based weighting...")

        # Use prevalence as probability weights
        weights = prevalence_data / prevalence_data.sum()

        # Sample indices based on weights
        sample_idx = np.random.choice(len(age_data), n_sample, replace=False, p=weights)

        age_sample = age_data[sample_idx]
        biomass_sample = biomass_data[sample_idx]
        prevalence_sample = prevalence_data[sample_idx]

        print(f"✓ Sampled {len(age_sample)} points (avg prevalence: {prevalence_sample.mean():.1f}%)")

        return age_sample, biomass_sample, prevalence_sample

    # Main execution
    print(f"=== {species_name} Growth Curve Analysis with Prevalence Thresholds ===")
    start_time = time.time()

    print(f"Parameters:")
    print(f"  Peak search range: {peak_age_min}-{peak_age_max} years")
    print(f"  Senescence age window: {senescence_age_window} years")
    print(f"  Outlier removal: {outlier_age_window}-year windows, {outlier_percentile}% threshold")
    print(f"  Smoothing window: {smoothing_window} points")
    print(f"  Prevalence thresholds: {prevalence_thresholds}")

    # Read data
    prevalence, age, biomass, maskRaster = read_rasters()

    # Process data
    age_flat, species_biomass_flat, prevalence_flat = process_data(prevalence, age, biomass, maskRaster, maskPath)

    # Remove outliers using fast method
    age_clean, biomass_clean, prevalence_clean = remove_outliers_fast(
        age_flat, species_biomass_flat, prevalence_flat, 
        age_window=outlier_age_window, 
        percentile=outlier_percentile
    )

    # Store curves for each threshold
    curves_data = {}
    colors = plt.cm.viridis(np.linspace(0, 1, len(prevalence_thresholds)))

    print(f"\nProcessing curves for each prevalence threshold...")

    for i, threshold in enumerate(prevalence_thresholds):
        print(f"\n--- Processing threshold {threshold}% ---")

        # Filter data by prevalence threshold
        age_thresh, biomass_thresh, prevalence_thresh = filter_by_prevalence_threshold(
            age_clean, biomass_clean, prevalence_clean, threshold
        )

        if len(age_thresh) == 0:
            print(f"  No data available for threshold {threshold}%. Skipping.")
            continue

        # Construct growth curve
        print(f"  Constructing growth curve...")
        raw_curve = construct_growth_curve(
            age_thresh, biomass_thresh, 
            peak_age_min=peak_age_min, 
            peak_age_max=peak_age_max, 
            senescence_age_window=senescence_age_window
        )

        # Interpolate curve to yearly intervals
        print(f"  Interpolating to yearly intervals...")
        interpolated_curve = interpolate_curve_yearly(raw_curve)

        # Smooth the interpolated curve
        print(f"  Smoothing curve...")
        smoothed_curve = smooth_curve_savgol(
            interpolated_curve, 
            window_length=smoothing_window, 
            polyorder=smoothing_polyorder
        )

        # Store results
        curves_data[threshold] = {
            'raw_curve': raw_curve,
            'interpolated_curve': interpolated_curve,
            'smoothed_curve': smoothed_curve,
            'color': colors[i],
            'n_points': len(age_thresh)
        }

        print(f"  ✓ Threshold {threshold}%: {len(raw_curve)} raw points, {len(smoothed_curve)} smoothed points")

    # Create CSV output
    print(f"\nCreating CSV output...")

    # Find the full age range across all curves
    all_ages = set()
    for threshold, data in curves_data.items():
        all_ages.update(data['smoothed_curve'][:, 0].astype(int))

    age_range = sorted(all_ages)

    # Create DataFrame
    csv_data = {'age': age_range}

    for threshold in sorted(curves_data.keys()):
        data = curves_data[threshold]

        # Create dictionaries for quick lookup
        raw_dict = {int(age): biomass for age, biomass in data['interpolated_curve']}
        smooth_dict = {int(age): biomass for age, biomass in data['smoothed_curve']}

        # Fill columns
        csv_data[f'raw_curve_{threshold}%'] = [raw_dict.get(age, np.nan) for age in age_range]
        csv_data[f'smoothed_curve_{threshold}%'] = [smooth_dict.get(age, np.nan) for age in age_range]

    df_output = pd.DataFrame(csv_data)

    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)

    # Save CSV
    df_output.to_csv(output_csv_path, index=False)
    print(f"✓ CSV saved to: {output_csv_path}")

    # Create visualization if requested
    if create_plot:
        # Sample points for visualization (using all data)
        age_plot, biomass_plot, prevalence_plot = weighted_sample_points(
            age_clean, biomass_clean, prevalence_clean, n_sample=n_sample_plot
        )

        print("\nCreating visualization...")
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=plot_figsize)

        # Left plot: Raw curves
        scatter1 = ax1.scatter(age_plot, biomass_plot, c=prevalence_plot, 
                              cmap='RdYlGn', alpha=0.4, s=0.5, vmin=0, vmax=100)

        for threshold, data in curves_data.items():
            ax1.plot(data['interpolated_curve'][:, 0], data['interpolated_curve'][:, 1], 
                     color=data['color'], linewidth=2, alpha=0.8,
                     label=f'Raw curve ≤{threshold}% (n={data["n_points"]})')

        ax1.set_xlabel('Forest Age (years)')
        ax1.set_ylabel(f'{species_name} Biomass (tons/ha)')
        ax1.set_title(f'{species_name} - Raw Growth Curves by Prevalence Threshold')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(True, alpha=0.3)

        # Right plot: Smoothed curves
        scatter2 = ax2.scatter(age_plot, biomass_plot, c=prevalence_plot, 
                              cmap='RdYlGn', alpha=0.4, s=0.5, vmin=0, vmax=100)

        for threshold, data in curves_data.items():
            ax2.plot(data['smoothed_curve'][:, 0], data['smoothed_curve'][:, 1], 
                     color=data['color'], linewidth=3, alpha=0.9,
                     label=f'Smoothed curve ≤{threshold}% (n={data["n_points"]})')

        ax2.set_xlabel('Forest Age (years)')
        ax2.set_ylabel(f'{species_name} Biomass (tons/ha)')
        ax2.set_title(f'{species_name} - Smoothed Growth Curves by Prevalence Threshold')
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax2.grid(True, alpha=0.3)

        # Add colorbars
        cbar1 = plt.colorbar(scatter1, ax=ax1)
        cbar1.set_label(f'{species_name} Prevalence (%)')
        cbar2 = plt.colorbar(scatter2, ax=ax2)
        cbar2.set_label(f'{species_name} Prevalence (%)')

        plt.tight_layout()
        plt.show()

    # Calculate summary statistics
    summary_stats = {}
    for threshold, data in curves_data.items():
        peak_idx = np.argmax(data['smoothed_curve'][:, 1])
        peak_biomass = data['smoothed_curve'][peak_idx, 1]
        peak_age = data['smoothed_curve'][peak_idx, 0]

        summary_stats[threshold] = {
            'peak_biomass': peak_biomass,
            'peak_age': peak_age,
            'n_points': data['n_points'],
            'age_range': (data['smoothed_curve'][0, 0], data['smoothed_curve'][-1, 0])
        }

    # Summary output
    end_time = time.time()
    processing_time = end_time - start_time

    print(f"\n=== Analysis Complete ===")
    print(f"Total processing time: {processing_time:.1f} seconds")
    print(f"Processed {len(age_clean)} pixels total")
    print(f"CSV saved to: {output_csv_path}")

    print(f"\nCurve Summary:")
    for threshold in sorted(summary_stats.keys()):
        stats = summary_stats[threshold]
        print(f"  Threshold ≤{threshold}%: Peak {stats['peak_biomass']:.2f} tons/ha at age {stats['peak_age']:.1f} ({stats['n_points']} points)")

    # Return results
    return {
        'curves_data': curves_data,
        'summary_stats': summary_stats,
        'processing_time': processing_time,
        'n_pixels_processed': len(age_clean),
        'output_csv_path': output_csv_path
    }

def prepareNFIDataForGAMs(abundance_path,
                          age_path,
                          biomass_path,
                          speciesName,
                          age_window = 5,
                          abundance_window = 10,
                          percentile = 100,
                          mask_path = None,
                         printPlot = False):
    """
    Prepare Canadian National Forest Inventory (NFI) raster data for Generalized Additive Model (GAM) analysis.

    This function processes three co-registered NFI rasters (species abundance, forest age, and total forest aboveground biomass)
    to create a windowed dataset suitable for GAM modeling. The function applies spatial masking,
    calculates a new raster of species biomass using the 3 NFI raster,
    creates discrete age and abundance windows and computes the highest percentile of biomass for each of them while enforcing
    monotonic constraints to ensure biological realism (and to simplify the fitting of GAMs afterward).

    Parameters
    ----------
    abundance_path : str
        Path to the species abundance raster file (0-100% values representing species abundance
        relative to other tree species in each pixel) from the NFI.
    age_path : str
        Path to the forest age raster file (values indicating forest age in years, 0 or nodata
        for non-forest pixels) from the NFI.
    biomass_path : str
        Path to the total aboveground biomass raster file (values in tons per hectare) from the NFI.
    speciesName : str
        Name of the target species for labeling plots and outputs.
    age_window : int, optional
        Size of discrete, non-overlapping age windows in years (default: 5). Used to find the high percentile in each window.
    abundance_window : int, optional
        Size of discrete, non-overlapping abundance windows in percentage points (default: 10). Used to find the high percentile in each window.
    percentile : int or float, optional
        Percentile value (0-100) to calculate for biomass within each age-abundance window
        (default: 100, i.e., maximum value).
    mask_path : str or None, optional
        Path to a mask raster file. Pixels with values < 1 are kept, pixels with values >= 1
        are excluded from analysis (default: None, no masking applied). WARNING : It's an inverted mask to exclude provinces.

    Returns
    -------
    None
        The function generates plots and prints processing information.
        The function then returns a panda dataframe with the processed (windowed) data.

    Processing Steps
    ----------------
    1. **Raster Reading**: Loads abundance, age, biomass, and optional mask rasters using rasterio.
    2. **Data Validation**: Handles nodata values and applies validity masks.
    3. **Species Biomass Calculation**: Computes species-specific biomass by multiplying 
       abundance percentage with total biomass.
    4. **Spatial Masking**: Applies optional mask to exclude unwanted pixels.
    5. **Windowing**: Creates discrete, non-overlapping windows for age and abundance ranges.
    6. **Statistical Aggregation**: Calculates specified percentile of biomass within each window.
    7. **Monotonic Constraint**: Ensures biomass values increase (or remain constant) with 
       increasing abundance within each age window.
    8. **Zero-Point Addition**: Adds artificial data points at age 0 with biomass 0 for all 
       abundance levels to enforce biological realism. This is use to push the GAMs done afterward
       to pass through 0 at age 0 (intercept).
    9. **Visualization**: Generates scatter plots showing age vs. biomass relationships.

    Notes
    -----
    - All input rasters must have the same projection, extent, and resolution.
    - Pixels with zero species abundance are excluded from analysis.
    - The monotonic constraint prevents biologically unrealistic scenarios where higher
      species abundance results in lower species biomass.
    - Zero-point addition helps constrain GAM models to pass through the origin, reflecting
      the biological reality that forests of age 0 have zero biomass.
    - The function includes diagnostic prints showing data processing statistics.
    """

    # Read the three rasters
    def read_rasters(abundance_path,
                    age_path,
                    biomass_path,
                    mask_path=None):
        """Read the three raster files and return their data arrays"""
    
        # Read abundance raster
        with rasterio.open(abundance_path) as src:
            abundance = src.read(1).astype(np.float32)
            abundance[abundance < 0] = np.nan  # Handle nodata values
    
        # Read age raster
        with rasterio.open(age_path) as src:
            age = src.read(1).astype(np.float32)
            age[age <= 0] = np.nan  # Handle nodata values
    
        # Read biomass raster
        with rasterio.open(biomass_path) as src:
            biomass = src.read(1).astype(np.float32)
            biomass[biomass < 0] = np.nan  # Handle nodata values
    
        # Read mask raster if provided
        mask = None
        if mask_path is not None:
            with rasterio.open(mask_path) as src:
                mask = src.read(1).astype(np.float32)
    
        return abundance, age, biomass, mask
    
    # Process data and create initial dataframe
    def create_initial_dataframe(abundance, age, biomass, mask=None):
        """Create initial dataframe with pixel data"""
    
        # Calculate species biomass
        abies_biomass = (abundance / 100.0) * biomass
    
        # Create masks for valid data
        valid_mask = (~np.isnan(abundance)) & (~np.isnan(age)) & (~np.isnan(biomass)) & (abundance > 0)
    
        # Apply additional mask if provided (keep pixels with value 1, exclude pixels with value 0)
        if mask is not None:
            print("Mask has been detected. Masking...")
            values, counts = np.unique(mask, return_counts=True)
            print("Values:", values)
            print("Counts:", counts)
            valid_mask = (~np.isnan(abundance)) & (~np.isnan(age)) & (~np.isnan(biomass)) & (abundance > 0) & (mask > 0)
            values, counts = np.unique((mask > 0), return_counts=True)
            print("Values:", values)
            print("Counts:", counts)
        else:
            valid_mask = (~np.isnan(abundance)) & (~np.isnan(age)) & (~np.isnan(biomass)) & (abundance > 0)
    
    
        # Extract valid pixels
        valid_abundance = abundance[valid_mask]
        valid_age = age[valid_mask]
        valid_abies_biomass = abies_biomass[valid_mask]
    
        # Create dataframe
        df = pd.DataFrame({
            'Age': valid_age,
            'Abundance': valid_abundance,
            'Biomass_Abies': valid_abies_biomass
        })
    
        return df
    
    # Create windowed statistics dataframe with monotonic constraint
    def create_windowed_dataframe(df, age_window=5, abundance_window=5, percentile=99):
        """Create windowed statistics dataframe with monotonic abundance constraint"""
    
        # Define age windows
        min_age = int(df['Age'].min())
        max_age = int(df['Age'].max())
        age_bins = range(min_age, max_age + age_window, age_window)
    
        # Define abundance windows
        abundance_bins = range(0, 101, abundance_window)
    
        windowed_data = []
    
        for i in range(len(age_bins) - 1):
            age_start = age_bins[i]
            age_end = age_bins[i + 1]
            age_middle = (age_start + age_end) / 2
    
            # Filter data for current age window
            age_mask = (df['Age'] >= age_start) & (df['Age'] < age_end)
            age_subset = df[age_mask]
    
            if len(age_subset) == 0:
                continue
    
            # Track previous biomass value for monotonic constraint
            previous_biomass = None
    
            for j in range(len(abundance_bins) - 1):
                abundance_start = abundance_bins[j]
                abundance_end = abundance_bins[j + 1]
                abundance_middle = (abundance_start + abundance_end) / 2
    
                # Filter data for current abundance window
                abundance_mask = (age_subset['Abundance'] >= abundance_start) & (age_subset['Abundance'] < abundance_end)
                window_subset = age_subset[abundance_mask]
    
                if len(window_subset) == 0:
                    continue
    
                # Calculate percentile
                biomass_percentile = np.percentile(window_subset['Biomass_Abies'], percentile)
    
                # Check monotonic constraint: current biomass should be >= previous biomass
                if previous_biomass is not None and biomass_percentile < previous_biomass:
                    # print(f"Skipping entry - Age: {age_middle:.1f}, Abundance: {abundance_middle:.1f}% "
                          # f"(Biomass {biomass_percentile:.2f} < previous {previous_biomass:.2f})")
                    continue  # Skip this entry as it violates monotonic constraint
    
                # Add entry to windowed data
                windowed_data.append({
                    'Age': age_middle,
                    'Abundance': abundance_middle,
                    'Biomass_Abies': biomass_percentile
                })
    
                # Update previous biomass for next iteration
                previous_biomass = biomass_percentile
    
        # We end up by creating an entry that will represent a 0 point for all abondance window
        # This will later force the GAM to pass by 0 as much as possible for an intercept, and represents
        # a biological reality : a forest of age 0 must have a biomass of 0, no matter its composition
        for j in range(len(abundance_bins) - 1):
            abundance_start = abundance_bins[j]
            abundance_end = abundance_bins[j + 1]
            abundance_middle = (abundance_start + abundance_end) / 2
    
            windowed_data.append({
                'Age': 0,
                'Abundance': abundance_middle,
                'Biomass_Abies': 0
            })

            # We do the same to add an "enpoint", approximatly after the closest point to 0
            # windowed_data.append({
            #     'Age': df["Age"].max() * 1.1,
            #     'Abundance': abundance_middle,
            #     'Biomass_Abies': 0
            # })
    
    
        print(f"Applied monotonic abundance constraint - kept {len(windowed_data)} entries")
        return pd.DataFrame(windowed_data)
    
    # Plot the results
    def plot_results(windowed_df):
        """Create the Age vs Biomass plot with abundance-based coloring"""
    
        plt.figure(figsize=(12, 8))
    
        # Create custom colormap (red to green)
        colors = ['red', 'yellow', 'green']
        n_bins = 100
        cmap = LinearSegmentedColormap.from_list('abundance', colors, N=n_bins)
    
        # Create scatter plot without edge colors
        scatter = plt.scatter(windowed_df['Age'], windowed_df['Biomass_Abies'], 
                             c=windowed_df['Abundance'], cmap=cmap, 
                             s=60, alpha=0.7)
    
        # Add colorbar
        cbar = plt.colorbar(scatter)
        cbar.set_label('Abundance of ' + str(speciesName) + ' (%)', fontsize=12)
    
        # Set labels and title
        plt.xlabel('Age (years)', fontsize=12)
        plt.ylabel('Biomass of ' + str(speciesName) + ' (tons/ha)', fontsize=12)
        plt.title('Age vs Biomass of ' + str(speciesName) + '\n(Colored by Abundance)', fontsize=14)
    
        # Add grid
        plt.grid(True, alpha=0.3)
    
        # Adjust layout
        plt.tight_layout()
        plt.show()
    
    # Plot the initial dataframe results
    def plot_initial_results(df):
        """Create the Age vs Biomass plot for initial dataframe with abundance-based coloring"""
    
        plt.figure(figsize=(12, 8))
    
        # Create custom colormap (red to green)
        colors = ['red', 'yellow', 'green']
        n_bins = 100
        cmap = LinearSegmentedColormap.from_list('abundance', colors, N=n_bins)
    
        # Create scatter plot without edge colors
        scatter = plt.scatter(df['Age'], df['Biomass_Abies'], 
                             c=df['Abundance'], cmap=cmap, 
                             s=60, alpha=0.7)
    
        # Add colorbar
        cbar = plt.colorbar(scatter)
        cbar.set_label('Abundance of ' + str(speciesName) + ' (%)', fontsize=12)
    
        # Set labels and title
        plt.xlabel('Age (years)', fontsize=12)
        plt.ylabel('Biomass of ' + str(speciesName)+ ' (tons/ha)', fontsize=12)
        plt.title('Age vs Biomass of ' + str(speciesName) + ' - All Pixels\n(Colored by Abundance)', fontsize=14)
    
        # Add grid
        plt.grid(True, alpha=0.3)
    
        # Adjust layout
        plt.tight_layout()
        plt.show()
    
        
    # Execute the workflow
    print("Reading raster data...")
    abundance, age, biomass, mask = read_rasters(abundance_path,
                                                 age_path,
                                                 biomass_path,
                                                 mask_path)
    
    print("Creating initial dataframe...")
    df = create_initial_dataframe(abundance, age, biomass, mask)
    print(f"Initial dataframe shape: {df.shape}")
    print(f"Age range: {df['Age'].min():.1f} - {df['Age'].max():.1f}")
    print(f"Abundance range: {df['Abundance'].min():.1f} - {df['Abundance'].max():.1f}")
    
    print(f"\nProcessing with parameters:")
    print(f"Age window: {age_window}")
    print(f"Abundance window: {abundance_window}")
    print(f"Percentile: {percentile}")
    
    print("\nCreating windowed dataframe...")
    windowed_df = create_windowed_dataframe(df, age_window, abundance_window, percentile)
    print(f"Windowed dataframe shape: {windowed_df.shape}")
    
    # Plot the initial dataframe
    # if len(df) > 0:
    #     print("\nCreating initial dataframe plot...")
    #     plot_initial_results(df)
    
    if len(windowed_df) > 0 and printPlot:
        print("\nCreating plot...")
        plot_results(windowed_df)
    
        # Display sample of windowed data
        # print("\nSample of windowed data:")
        # print(windowed_df.head(10))
    else:
        print("No data points found in the specified windows!")

    return(windowed_df)


def createGAMsAndPredictionsForNFIDataGrowthCurves(windowed_df,
                                                   csvOutputName,
                                                   speciesName,
                                                   cutoff_age = 150,
                                                   abundance_levels = [100, 80, 60, 40, 20],
                                                   printPlot = False):
    """
    Create an Generalized Additive Model (GAM) on the Canadian National Forest Inventory (NFI) processed
    by the function prepareNFIDataForGAMs in order to generate predicted growth curves for different
    levels of species abundance. It then exports the curves to a CSV file so that they can be re-used
    to calibrate PnET-Succession later.

    Parameters
    ----------
    windowed_df : pandas.DataFrame
        Input dataframe generated by the function prepareNFIDataForGAMscontaining windowed forest data with columns:
        - 'Age': Forest age in years
        - 'Abundance': Species abundance percentage (0-100%)
        - 'Biomass_Abies': Species biomass in tons per hectare
    csvOutputName : str
        Filename for the CSV output containing prediction curves.
    speciesName : str
        Name of the target species for labeling plots and outputs.
    cutoff_age : int or float, optional
        Maximum age threshold for model fitting. Only data points with age < cutoff_age
        are used for GAM training (default: 150 years).
    abundance_levels : list of int or float, optional
        List of abundance percentages for which to generate prediction curves
        (default: [100, 80, 60, 40, 20]).
    printPlot : bool, optional
        Whether to display the GAM results plot (default: False).

    Returns
    -------
    None
        The function generates plots (if requested), exports CSV files, and prints processing
        information but does not return values.

    Model Specifications
    --------------------
    The GAM model includes:
    - **Age Effect**: Smooth spline with concave constraint (s(0, constraints='concave'))
    - **Interaction Effect**: Tensor product of age and abundance with concave constraint
    - **Abundance Weighting**: Observations weighted by (abundance/100)² to emphasize
      high abundance data points

    You can edit the GAM structure to your liking to better fit your data; this is just what
    worked best in my case after trying several structures.

    Output CSV Structure
    --------------------
    The exported CSV contains:
    - 'Age' column: Age values from 0 to cutoff_age (100 points)
    - 'GAM_Prediction_X%Abundance' columns: Predicted biomass for each abundance level

    Notes
    -----
    - The model uses pygam's LinearGAM with tensor product interactions
    - Abundance weighting helps ensure a better fit for high abundance predictions (for which we have less points)
    - The concave constraint reflects typical forest growth patterns
    - Model diagnostics including R-squared and constraint information are printed
    - Age range for predictions spans from 0 to the specified cutoff age
    """
    
    # Filter data based on age cutoff
    def filter_data_by_age(windowed_df, cutoff_age):
        """Filter dataframe to keep only pixels younger than cutoff age"""
    
        filtered_df = windowed_df[windowed_df['Age'] < cutoff_age].copy()
    
        print(f"Original dataframe shape: {windowed_df.shape}")
        print(f"Filtered dataframe shape (Age < {cutoff_age}): {filtered_df.shape}")
    
        if len(filtered_df) > 0:
            print(f"Filtered age range: {filtered_df['Age'].min():.1f} - {filtered_df['Age'].max():.1f}")
            print(f"Filtered abundance range: {filtered_df['Abundance'].min():.1f} - {filtered_df['Abundance'].max():.1f}")
    
        return filtered_df
    
    # Fit GAM model with constraints and weights
    def fit_gam_model(df):
        """Fit a Generalized Additive Model with constraints and abundance weighting"""
    
        if len(df) == 0:
            print("No data available for GAM fitting!")
            return None
    
        # Prepare data
        X = df[['Age', 'Abundance']].values
        y = df['Biomass_Abies'].values
    
        # Create weights that emphasize higher abundance observations
        weights = (df['Abundance'] / 100.0) ** 2  # Square to really emphasize high abundance
    
        # Fit GAM with:
        # - Smooth term for age with concave constraint
        # - Tensor product interaction with concave constraint
        gam = LinearGAM(
            s(0, n_splines=10, constraints='concave') +  # Age: concave
            te(0, 1, constraints='concave'),  # Tensor product interaction: concave
            fit_intercept=False  
        )

        # gam = LinearGAM(
        #     s(0, n_splines=10) +  # Age: concave
        #     te(0, 1),  # Tensor product interaction: concave
        #     fit_intercept=False  
        # )

        # gam = LinearGAM(
        # te(0, 1, n_splines=50),
        # fit_intercept=True  # Add intercept back
        # )
    
    
        # Fit the model with weights
        gam.fit(X, y, weights=weights)

        # Create separate plots for each term
        # for i, term in enumerate(gam.terms):
        #     if not term.isintercept:
        #         fig, ax = plt.subplots(figsize=(8, 6))
        #         XX = gam.generate_X_grid(term=i)
        #         pdep, confi = gam.partial_dependence(term=i, X=XX, width=0.95)
        
        #         ax.plot(XX[:, term.feature], pdep, label='Partial Dependence')
        #         ax.fill_between(XX[:, term.feature], confi[:, 0], confi[:, 1], 
        #                         alpha=0.3, label='95% CI')
        #         ax.set_title(f'Partial Dependence - Feature {term.feature}')
        #         ax.set_xlabel(f'Feature {term.feature}')
        #         ax.set_ylabel('Partial Dependence')
        #         ax.legend()
        #         plt.tight_layout()
        #         plt.show()


        print(f"GAM fitted successfully with constraints and abundance weighting!")
        print(f"GAM summary:")
        print(f"  - Number of observations: {len(y)}")
        print(f"  - R-squared (pseudo): {gam.statistics_['pseudo_r2']['explained_deviance']:.3f}")
        print(f"  - Constraints applied:")
        print(f"    * Age effect: concave")
        print(f"    * Tensor product interaction: concave")
        print(f"    * Intercept forced to 0")
        print(f"    * Weighted toward high abundance observations")
    
        return gam
    
    # Generate predictions for multiple abundance levels
    def generate_multiple_predictions(gam, cutoff_age, abundance_levels):
        """Generate GAM predictions for multiple abundance levels across age range"""
    
        if gam is None:
            return None, None
    
        # Create age range from 0 to cutoff
        age_range = np.linspace(0, cutoff_age, 100)
    
        predictions_dict = {}
    
        for abundance_level in abundance_levels:
            # Create prediction data with constant abundance
            pred_data = np.column_stack([age_range, np.full(len(age_range), abundance_level)])
    
            # Generate predictions
            predictions = gam.predict(pred_data)
    
            predictions_dict[abundance_level] = predictions
    
            print(f"Generated predictions for {abundance_level}% abundance:")
            print(f"  - Biomass range: {predictions.min():.2f} - {predictions.max():.2f}")
            print(f"  - Value at age 0: {predictions[0]:.6f}")
    
        return age_range, predictions_dict
    
    # Plot results with multiple GAM curves
    def plot_gam_results(df, age_range=None, predictions_dict=None, abundance_levels=None):
        """Plot the filtered data points with multiple GAM prediction curves"""
    
        plt.figure(figsize=(16, 10))
    
        # Create custom colormap (red to green)
        colors = ['red', 'yellow', 'green']
        n_bins = 100
        cmap = LinearSegmentedColormap.from_list('abundance', colors, N=n_bins)
    
        # Plot data points
        scatter = plt.scatter(df['Age'], df['Biomass_Abies'], 
                             c=df['Abundance'], cmap=cmap, 
                             s=60, alpha=0.7, label='Observed data')
    
        # Plot GAM prediction curves if available
        if age_range is not None and predictions_dict is not None and abundance_levels is not None:
            # Create color palette for curves
            curve_colors = plt.cm.viridis(np.linspace(0, 1, len(abundance_levels)))
    
            for i, abundance_level in enumerate(abundance_levels):
                if abundance_level in predictions_dict:
                    plt.plot(age_range, predictions_dict[abundance_level], 
                            color=curve_colors[i], linewidth=3, 
                            label=f'GAM prediction ({abundance_level}% abundance)')
    
        # Add colorbar positioned in the bottom half
        cbar = plt.colorbar(scatter, shrink=0.5, pad=0.12, anchor=(0.0, 0.0))
        cbar.set_label('Abundance of ' + str(speciesName) + ' (%)', fontsize=12)
    
        # Set labels and title
        plt.xlabel('Age (years)', fontsize=12)
        plt.ylabel('Biomass of ' + str(speciesName)+ ' (tons/ha)', fontsize=12)
        plt.title('Weighted GAM Model: Age vs Biomass of ' + str(speciesName) + '\n(Multiple Abundance Levels)', fontsize=14)
    
        # Add legend positioned in the top half
        plt.legend(fontsize=10, bbox_to_anchor=(1.02, 1.0), loc='upper left')
    
        # Add grid
        plt.grid(True, alpha=0.3)
    
        # Adjust layout to prevent overlap
        plt.tight_layout()
        plt.subplots_adjust(right=0.75)  # Make room for colorbar and legend
        plt.show()
    
    def export_predictions_to_csv(age_range, predictions_dict, abundance_levels, filename='gam_predictions.csv'):
        """Export GAM prediction curves to CSV file"""
    
        if age_range is None or predictions_dict is None:
            print("No predictions available to export!")
            return
    
        # Create dataframe with age as first column
        export_df = pd.DataFrame({'Age': age_range})
    
        # Add prediction columns for each abundance level
        for abundance_level in abundance_levels:
            if abundance_level in predictions_dict:
                column_name = f'GAM_Prediction_{abundance_level}%Abundance'
                export_df[column_name] = predictions_dict[abundance_level]
    
        # Export to CSV
        export_df.to_csv(filename, index=False)
    
        print(f"Predictions exported to {filename}")
        print(f"Shape: {export_df.shape}")
        print(f"Columns: {list(export_df.columns)}")
        print(f"Age range: {export_df['Age'].min():.1f} - {export_df['Age'].max():.1f}")
    
        # Display first few rows as preview
        print("\nPreview of exported data:")
        print(export_df.head())
    
        return export_df
    
    # Execute GAM workflow
    print("Filtering data by age cutoff...")
    filtered_df = filter_data_by_age(windowed_df, cutoff_age)
    
    if len(filtered_df) > 0:
        print("\nFitting weighted GAM model...")
        gam_model = fit_gam_model(filtered_df)
    
        if gam_model is not None:
            print(f"\nGenerating predictions for abundance levels: {abundance_levels}")
            pred_ages, pred_biomass_dict = generate_multiple_predictions(gam_model, cutoff_age, abundance_levels)

            if printPlot:
                print("\nCreating plot with multiple GAM predictions...")
                plot_gam_results(filtered_df, pred_ages, pred_biomass_dict, abundance_levels)
    
            # Display model statistics
            print("\nGAM Model Information:")
            print(f"Cutoff age used: {cutoff_age} years")
            print(f"Data points used for fitting: {len(filtered_df)}")
            print(f"Prediction curves: 0-{cutoff_age} years for {len(abundance_levels)} abundance levels")
            print(f"Model constraints and weighting applied successfully")
        else:
            print("GAM fitting failed!")
    else:
        print("No data available after age filtering!")

    # Export predictions to CSV
    if pred_ages is not None and pred_biomass_dict is not None:
        export_df = export_predictions_to_csv(pred_ages, pred_biomass_dict, abundance_levels, 
                                             filename= csvOutputName)

def plotNFICurvesFromGAMAndFromAlgorithmTogether(pathOfCsvFileWithGAMPredictions,
                                                 pathOfCsvFileWithAlgorithmPrediction):
    """
    Compare and visualize forest growth curves from GAM predictions (generated by the function
    createGAMsAndPredictionsForNFIData) and algorithm predictions (generated by analyze_species_growth_curves).

    This function creates a comparative plot displaying biomass growth curves generated by two
    different methods: Generalized Additive Models (GAM) and an alternative algorithm. The plot
    allows visual comparison of prediction methods for forest biomass estimation across different
    abundance levels and age ranges.

    Parameters
    ----------
    pathOfCsvFileWithGAMPredictions : str
        File path to CSV containing GAM prediction curves. Expected structure:
        - First column: Age values (years)
        - Subsequent columns: Biomass predictions for different abundance levels
          (e.g., 'GAM_Prediction_100%Abundance', 'GAM_Prediction_80%Abundance', etc.)
    pathOfCsvFileWithAlgorithmPrediction : str
        File path to CSV containing algorithm prediction curves. Expected structure:
        - First column: Age values (years)
        - Subsequent columns: Various prediction outputs, including columns with 'smoothed'
          in their names which will be plotted

    Returns
    -------
    None
        The function generates and displays a matplotlib plot but does not return values.

    Plot Features
    -------------
    - **GAM Curves**: Plotted as solid lines using viridis colormap (purple to yellow)
    - **Algorithm Curves**: Plotted as dashed lines using inverted viridis colormap (yellow to purple;
    inverted as the column names are inverted for this one, because of how previous functions are written.)
    - **Selective Plotting**: Only plots 'smoothed' columns from the algorithm predictions, avoiding the raw curves. 
    - **Grid**: Light grid overlay for easier value reading
    """

    # Read the two CSV files
    pathOfCsvFileWithGAMPredictions = pd.read_csv(pathOfCsvFileWithGAMPredictions)
    pathOfCsvFileWithAlgorithmPrediction = pd.read_csv(pathOfCsvFileWithAlgorithmPrediction)
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    
    # Get age column (assuming first column is age)
    age_col1 = pathOfCsvFileWithGAMPredictions.iloc[:, 0]
    age_col2 = pathOfCsvFileWithAlgorithmPrediction.iloc[:, 0]
    
    # Plot curves from first CSV file
    biomass_cols1 = pathOfCsvFileWithGAMPredictions.columns[1:]  # All columns except age
    n_curves1 = len(biomass_cols1)
    viridis_colors1 = plt.cm.viridis(np.linspace(0, 1, n_curves1))
    
    for i, col in enumerate(biomass_cols1):
        plt.plot(age_col1, pathOfCsvFileWithGAMPredictions[col], color=viridis_colors1[i], label=col, linewidth=2)
    
    # Plot curves from second CSV file (only "Smoothed" columns)
    smoothed_cols = [col for col in pathOfCsvFileWithAlgorithmPrediction.columns[1:] if 'smoothed' in col]
    n_curves2 = len(smoothed_cols)
    
    if n_curves2 > 0:
        # Inverted viridis palette (yellow to purple)
        viridis_colors2 = plt.cm.viridis(np.linspace(1, 0, n_curves2))
    
        for i, col in enumerate(smoothed_cols):
            plt.plot(age_col2, pathOfCsvFileWithAlgorithmPrediction[col], color=viridis_colors2[i], 
                    label=col, linestyle='--', linewidth=2)
    
    # Customize the plot
    plt.xlabel('Age')
    plt.ylabel('Biomass')
    plt.title('Biomass vs Age Curves from GAM (solid) and from algorithm (Dashed) - Abies balsamea')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()



def plot_species_biomass_age_comparison(prevalence_path, age_path, biomass_path, 
                                        mask1_path, mask2_path, species_name):
    """
    Plot species-specific biomass vs forest age for two different spatial masks.

    This function creates a scatter plot comparing the relationship between forest age
    and species-specific biomass across two different spatial regions. Each point is
    color-coded by species prevalence (red=0%, green=100%), with mask1 points appearing
    lighter than mask2 points for visual distinction.

    Parameters
    ----------
    prevalence_path : str
        Path to raster file containing species prevalence values (0-100%).
        Values represent the percentage of the species in each pixel.
    age_path : str
        Path to raster file containing forest stand age values (years).
        Non-forest pixels should be 0 or nodata.
    biomass_path : str
        Path to raster file containing total live aboveground biomass (tons/ha).
    mask1_path : str
        Path to first spatial mask raster. Points within this mask will be
        plotted with lighter colors.
    mask2_path : str
        Path to second spatial mask raster. Points within this mask will be
        plotted with darker colors for comparison.
    species_name : str
        Name of the species for plot labels and title (e.g., 'Abies balsamea').

    Returns
    -------
    tuple
        (n_points_mask1, n_points_mask2) - Number of valid points in each mask.

    Notes
    -----
    - All input rasters must have the same projection, extent, and resolution.
    - Species-specific biomass is calculated as: (prevalence/100) * total_biomass
    - Only pixels with prevalence > 0 and age > 0 are included in the analysis.
    - If more than 3000 points exist per mask, sampling is performed with bias toward
      higher prevalence values.
    """

    def read_raster(filepath):
        """Read raster and return data array"""
        with rasterio.open(filepath) as src:
            return src.read(1)

    def lighten_color(prevalence, factor=0.5):
        """Convert prevalence (0-100) to RGB with lightening factor"""
        norm_prev = prevalence / 100.0
        r = 1.0 - norm_prev + (norm_prev) * factor
        g = norm_prev + (1.0 - norm_prev) * factor
        b = 0.0 + (1.0) * factor
        return (r, g, b)

    def weighted_sample_points(age_data, biomass_data, prevalence_data, n_sample):
        """Sample points with probability weights based on prevalence"""
        if len(age_data) <= n_sample:
            print(f"Using all {len(age_data)} points for visualization")
            return age_data, biomass_data, prevalence_data

        print(f"Sampling {n_sample} points with prevalence-based weighting...")

        # Use prevalence as probability weights
        weights = prevalence_data / prevalence_data.sum()

        # Sample indices based on weights
        sample_idx = np.random.choice(len(age_data), n_sample, replace=False, p=weights)

        age_sample = age_data[sample_idx]
        biomass_sample = biomass_data[sample_idx]
        prevalence_sample = prevalence_data[sample_idx]

        print(f"✓ Sampled {len(age_sample)} points (avg prevalence: {prevalence_sample.mean():.1f}%)")

        return age_sample, biomass_sample, prevalence_sample

    def get_filename_without_extension(filepath):
        """Extract filename without path and extension"""
        return os.path.splitext(os.path.basename(filepath))[0]

    # Read input rasters
    print("Reading rasters...")
    prevalence = read_raster(prevalence_path)
    age = read_raster(age_path)
    biomass = read_raster(biomass_path)
    mask1 = read_raster(mask1_path)
    mask2 = read_raster(mask2_path)

    # Get mask labels from filenames
    print("Masking rasters...")
    mask1_label = get_filename_without_extension(mask1_path)
    mask2_label = get_filename_without_extension(mask2_path)

    # Mask pixels with no species presence
    valid_species = prevalence > 0

    # Calculate species-specific biomass
    species_biomass = (prevalence / 100.0) * biomass

    # Create combined masks
    mask1_combined = valid_species & (age > 0) & (mask1 > 0)
    mask2_combined = valid_species & (age > 0) & (mask2 > 0)

    # Extract data for mask1
    age_mask1 = age[mask1_combined]
    biomass_mask1 = species_biomass[mask1_combined]
    prevalence_mask1 = prevalence[mask1_combined]

    # Extract data for mask2
    age_mask2 = age[mask2_combined]
    biomass_mask2 = species_biomass[mask2_combined]
    prevalence_mask2 = prevalence[mask2_combined]

    # Sample points if needed
    print("Sampling points if needed...")
    print(f"\n{mask1_label}:")
    age_mask1, biomass_mask1, prevalence_mask1 = weighted_sample_points(
        age_mask1, biomass_mask1, prevalence_mask1, 500000
    )

    print(f"\n{mask2_label}:")
    age_mask2, biomass_mask2, prevalence_mask2 = weighted_sample_points(
        age_mask2, biomass_mask2, prevalence_mask2, 500000
    )

    # Create figure
    print("Creating figure...")
    fig, ax = plt.subplots(figsize=(12, 8))

    # Plot mask1 (lighter colors)
    colors_mask1 = [lighten_color(p, factor=0.6) for p in prevalence_mask1]
    ax.scatter(age_mask1, biomass_mask1, c=colors_mask1, s=1, alpha=0.6, label=mask1_label)

    # Plot mask2 (darker colors)
    colors_mask2 = [lighten_color(p, factor=0.0) for p in prevalence_mask2]
    ax.scatter(age_mask2, biomass_mask2, c=colors_mask2, s=1, alpha=0.6, label=mask2_label)

    # Labels and title
    ax.set_xlabel('Forest Age (years)', fontsize=12)
    ax.set_ylabel(f'{species_name} Biomass (tons/ha)', fontsize=12)
    ax.set_title(f'{species_name} Biomass vs Age\n(Color: Red=0% prevalence, Green=100% prevalence)', 
                 fontsize=14)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    # Create colorbar
    cmap = LinearSegmentedColormap.from_list('prevalence', ['red', 'green'])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=100))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, label=f'{species_name} Prevalence (%)')

    plt.tight_layout()
    plt.show()

    n_mask1 = len(age_mask1)
    n_mask2 = len(age_mask2)
    print(f"\nFinal point counts:")
    print(f"{mask1_label}: {n_mask1} points")
    print(f"{mask2_label}: {n_mask2} points")

    return n_mask1, n_mask2


def extract_try_traits(filepath, encoding='latin-1'):
    """
    Extract plant functional trait data from TRY database file into a nested dictionary.

    Parameters
    ----------
    filepath : str
        Path to the TRY database text file (tab-separated format).
    encoding : str, optional
        Character encoding of the file. Default is 'latin-1'.
        Common alternatives: 'utf-8', 'iso-8859-1', 'cp1252'.

    Returns
    -------
    dict
        Nested dictionary with structure:
        {species_name: {trait_name: [list of trait values]}}
        where species_name is from AccSpeciesName column,
        trait_name is from TraitName column,
        and values are from OrigValueStr column.

    Notes
    -----
    The string "none" is preserved as a valid trait value and not treated as missing data.

    Examples
    --------
    >>> data = extract_try_traits('try_data.txt')
    >>> data['Quercus robur']['Leaf area']
    ['45.2', '52.1', '48.7']

    >>> data = extract_try_traits('try_data.txt', encoding='utf-8')
    """
    # Read tab-separated file with specified encoding
    # keep_default_na=False prevents "none" from being converted to NaN
    df = pd.read_csv(filepath, sep='\t', encoding=encoding, 
                     keep_default_na=False, na_values=['nan', 'NaN'], low_memory=False)

    # Remove rows with missing TraitName (empty strings after our na_values setting)
    df = df[df['TraitName'] != '']

    # Initialize nested dictionary with automatic list creation
    trait_dict = defaultdict(lambda: defaultdict(list))

    # Populate the nested dictionary
    for _, row in df.iterrows():
        species = row['AccSpeciesName']
        trait = row['TraitName']
        value = row['OrigValueStr']

        trait_dict[species][trait].append(value)

    # Convert to regular dict for cleaner output
    return {species: dict(traits) for species, traits in trait_dict.items()}

def average_tolerance_traits(trait_dict):
    """
    Calculate average tolerance trait values for each species from TRY trait dictionary.

    Parameters
    ----------
    trait_dict : dict
        Nested dictionary from extract_try_traits() with structure:
        {species_name: {trait_name: [list of trait values]}}
        Made by extract_try_traits().

    Returns
    -------
    dict
        Dictionary with structure:
        {species_name: {trait_name: average_value}}
        containing only tolerance traits with numeric averages.

    Notes
    -----
    Conversion rules for trait values:
    - Numeric 1-5: kept as is
    - 0 or "none"/"no": converted to 1
    - "low": converted to 1.5
    - "medium": converted to 2.5
    - "tolerant": converted to 3.5
    - Values > 5: capped at 5
    - Unrecognized values: ignored and reported

    Examples
    --------
    >>> raw_data = extract_try_traits('try_data.txt')
    >>> avg_data = average_tolerance_traits(raw_data)
    >>> avg_data['Quercus robur']['Waterlogging tolerance']
    2.75
    """
    # Mapping for text values to numeric scores
    value_mapping = {
        'none': 1,
        'no': 1,
        'low': 1.5,
        'medium': 2.5,
        'tolerant': 3.5,
        'intolerant': 1.5,
        'late-successional': 3.5
    }

    result_dict = {}
    ignored_values = set()

    for species, traits in trait_dict.items():
        species_tolerances = {}

        for trait_name, values in traits.items():
            # Only process traits with "tolerance" in the name
            if 'tolerance' not in trait_name.lower():
                continue

            numeric_values = []

            for value in values:
                # Convert to string and lowercase for consistent processing
                value_str = str(value).strip().lower()

                # Try to convert to float first
                try:
                    num_value = float(value_str)

                    # Apply conversion rules
                    if num_value == 0:
                        numeric_values.append(1)
                    elif num_value > 5:
                        numeric_values.append(5)
                    else:
                        numeric_values.append(num_value)

                except ValueError:
                    # Handle text values
                    if value_str in value_mapping:
                        numeric_values.append(value_mapping[value_str])
                    else:
                        # Ignore and track unrecognized values
                        ignored_values.add(value)

            # Calculate average if we have valid numeric values
            if numeric_values:
                species_tolerances[trait_name] = sum(numeric_values) / len(numeric_values)

        # Only add species if they have tolerance traits
        if species_tolerances:
            result_dict[species] = species_tolerances

    # Print ignored values if any
    if ignored_values:
        print("Ignored values (unrecognized):")
        for val in sorted(ignored_values):
            print(f"  - {val}")

    return result_dict

def plot_competition_matrices(avg_trait_dict):
    """
    Plot competition probability matrices for each tolerance trait.

    Creates one heatmap per trait showing the probability that each species
    (challenger, rows) wins against each other species (opponent, columns)
    based on their tolerance values.

    Parameters
    ----------
    avg_trait_dict : dict
        Dictionary from average_tolerance_traits() with structure:
        {species_name: {trait_name: average_value}}

    Returns
    -------
    None
        Displays matplotlib figures with competition matrices.

    Notes
    -----
    Competition probability calculation:
    - Challenger tolerance 1 vs Opponent tolerance 5: 0% win probability
    - Challenger tolerance 5 vs Opponent tolerance 1: 100% win probability
    - Equal tolerances: 50% win probability
    - Other combinations: linear interpolation
    - Diagonal (species vs itself): displayed as white with no probability

    Formula: P(win) = 0.5 + 0.125 * (challenger_tolerance - opponent_tolerance)

    Examples
    --------
    >>> avg_data = average_tolerance_traits(raw_data)
    >>> plot_competition_matrices(avg_data)
    """
    # Organize data by trait
    trait_data = {}
    for species, traits in avg_trait_dict.items():
        for trait_name, trait_value in traits.items():
            if trait_name not in trait_data:
                trait_data[trait_name] = {}
            trait_data[trait_name][species] = trait_value

    # Create one plot per trait
    for trait_name, species_values in trait_data.items():
        # Get sorted species list for consistent ordering
        species_list = sorted(species_values.keys())
        n_species = len(species_list)

        # Initialize probability matrix with NaN for diagonal
        prob_matrix = np.zeros((n_species, n_species))

        # Calculate win probabilities
        for i, challenger in enumerate(species_list):
            for j, opponent in enumerate(species_list):
                if i == j:
                    # Set diagonal to NaN for masking
                    prob_matrix[i, j] = np.nan
                    continue

                challenger_tol = species_values[challenger]
                opponent_tol = species_values[opponent]

                # Linear interpolation formula
                diff = challenger_tol - opponent_tol
                prob_win = 0.5 + 0.125 * diff

                # Ensure probability is within [0, 1]
                prob_win = np.clip(prob_win, 0, 1)

                prob_matrix[i, j] = prob_win

        # Create figure
        fig, ax = plt.subplots(figsize=(max(5, n_species * 0.5), max(4, n_species * 0.4)))

        # Create masked array to handle NaN values (diagonal)
        masked_matrix = np.ma.masked_invalid(prob_matrix)

        # Create heatmap with red-to-blue colormap
        im = ax.imshow(masked_matrix, cmap='RdBu', aspect='auto', vmin=0, vmax=1)

        # Set background color for masked cells (diagonal) to white
        ax.set_facecolor('white')

        # Set ticks and labels
        ax.set_xticks(np.arange(n_species))
        ax.set_yticks(np.arange(n_species))
        ax.set_xticklabels(species_list, rotation=45, ha='right', fontsize=8)
        ax.set_yticklabels(species_list, fontsize=8)

        # Add labels
        ax.set_xlabel('Opponent', fontsize=12, fontweight='bold')
        ax.set_ylabel('Challenger', fontsize=12, fontweight='bold')
        ax.set_title(f'Competition Matrix: {trait_name}', fontsize=14, fontweight='bold', pad=20)

        # Add probability values in cells (skip diagonal)
        for i in range(n_species):
            for j in range(n_species):
                if i != j:  # Skip diagonal
                    ax.text(j, i, f'{prob_matrix[i, j]:.2f}',
                           ha='center', va='center', color='black', fontsize=9)

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Win Probability', rotation=270, labelpad=20, fontsize=11)

        plt.tight_layout()
        plt.show()

def plot_competition_matrices_with_shade(avg_trait_dict, shade_weight=1.0):
    """
    Plot competition probability matrices for each tolerance trait, incorporating shade tolerance.

    Creates one heatmap per trait showing the probability that each species
    (challenger, rows) wins against each other species (opponent, columns).
    For non-shade tolerances, the probability is a weighted average of the
    specific tolerance and shade tolerance probabilities.

    Parameters
    ----------
    avg_trait_dict : dict
        Dictionary from average_tolerance_traits() with structure:
        {species_name: {trait_name: average_value}}
    shade_weight : float, optional
        Weight for shade tolerance in the weighted average. Default is 1.0.
        - 1.0: equal weight (1:1 ratio)
        - 2.0: shade tolerance weighted twice as much (2:1 ratio)
        - 0.5: shade tolerance weighted half as much (1:2 ratio)

    Returns
    -------
    None
        Displays matplotlib figures with competition matrices.

    Notes
    -----
    Competition probability calculation:
    - For shade tolerance: standard calculation only
    - For other tolerances: weighted average of trait probability and shade probability
    - Formula: P(win) = (shade_weight × P_shade + P_trait) / (shade_weight + 1)
    - Diagonal (species vs itself): displayed as white with no probability

    Examples
    --------
    >>> avg_data = average_tolerance_traits(raw_data)
    >>> plot_competition_matrices_with_shade(avg_data, shade_weight=2.0)
    """
    # Organize data by trait
    trait_data = {}
    for species, traits in avg_trait_dict.items():
        for trait_name, trait_value in traits.items():
            if trait_name not in trait_data:
                trait_data[trait_name] = {}
            trait_data[trait_name][species] = trait_value

    # Find shade tolerance trait
    shade_trait_name = None
    for trait_name in trait_data.keys():
        if 'shade' in trait_name.lower():
            shade_trait_name = trait_name
            break

    if shade_trait_name is None:
        print("Warning: No shade tolerance trait found. Using standard calculation for all traits.")

    # Calculate shade probability matrix once if it exists
    shade_prob_matrix = None
    species_list = None

    if shade_trait_name:
        species_list = sorted(trait_data[shade_trait_name].keys())
        n_species = len(species_list)
        shade_prob_matrix = np.zeros((n_species, n_species))

        for i, challenger in enumerate(species_list):
            for j, opponent in enumerate(species_list):
                if i == j:
                    shade_prob_matrix[i, j] = np.nan
                    continue

                challenger_tol = trait_data[shade_trait_name][challenger]
                opponent_tol = trait_data[shade_trait_name][opponent]
                diff = challenger_tol - opponent_tol
                prob_win = 0.5 + 0.125 * diff
                shade_prob_matrix[i, j] = np.clip(prob_win, 0, 1)

    # Create one plot per trait
    for trait_name, species_values in trait_data.items():
        # Get sorted species list for consistent ordering
        if species_list is None:
            species_list = sorted(species_values.keys())

        # Filter to only species present in current trait
        current_species = sorted(species_values.keys())
        n_species = len(current_species)

        # Initialize probability matrix
        prob_matrix = np.zeros((n_species, n_species))

        # Initialize output dictionnary
        competitionDict = dict()

        # Calculate win probabilities
        for i, challenger in enumerate(current_species):
            for j, opponent in enumerate(current_species):
                if i == j:
                    prob_matrix[i, j] = np.nan
                    continue

                challenger_tol = species_values[challenger]
                opponent_tol = species_values[opponent]
                diff = challenger_tol - opponent_tol
                prob_win = 0.5 + 0.125 * diff
                prob_win = np.clip(prob_win, 0, 1)

                # If not shade tolerance and shade data exists, apply weighted average
                if trait_name != shade_trait_name and shade_prob_matrix is not None:
                    # Find indices in shade matrix
                    if challenger in trait_data[shade_trait_name] and opponent in trait_data[shade_trait_name]:
                        shade_i = species_list.index(challenger)
                        shade_j = species_list.index(opponent)
                        shade_prob = shade_prob_matrix[shade_i, shade_j]

                        if not np.isnan(shade_prob):
                            # Weighted average: (shade_weight * P_shade + P_trait) / (shade_weight + 1)
                            prob_win = (shade_weight * shade_prob + prob_win) / (shade_weight + 1)

                prob_matrix[i, j] = prob_win
                if challenger not in competitionDict.keys():
                    competitionDict[challenger] = dict()
                competitionDict[challenger][opponent] = prob_win

        # Create figure
        fig, ax = plt.subplots(figsize=(max(5, n_species * 0.5), max(4, n_species * 0.4)))

        # Create masked array to handle NaN values (diagonal)
        masked_matrix = np.ma.masked_invalid(prob_matrix)

        # Create heatmap with red-to-blue colormap
        im = ax.imshow(masked_matrix, cmap='RdBu', aspect='auto', vmin=0, vmax=1)

        # Set background color for masked cells (diagonal) to white
        ax.set_facecolor('white')

        # Set ticks and labels
        ax.set_xticks(np.arange(n_species))
        ax.set_yticks(np.arange(n_species))
        ax.set_xticklabels(current_species, rotation=45, ha='right', fontsize=8)
        ax.set_yticklabels(current_species, fontsize=8)

        # Add labels
        ax.set_xlabel('Opponent', fontsize=12, fontweight='bold')
        ax.set_ylabel('Challenger', fontsize=12, fontweight='bold')

        # Add shade weight info to title for non-shade traits
        title = f'Competition Matrix: {trait_name}'
        if trait_name != shade_trait_name and shade_trait_name:
            title += f' (Shade weight: {shade_weight})'
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)

        # Add probability values in cells (skip diagonal)
        for i in range(n_species):
            for j in range(n_species):
                if i != j:
                    ax.text(j, i, f'{prob_matrix[i, j]:.2f}',
                           ha='center', va='center', color='black', fontsize=9)

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Win Probability', rotation=270, labelpad=20, fontsize=11)

        plt.tight_layout()
        plt.show()

    return competitionDict

def convert_monthly_precip_to_cm(precip_series, start_date, end_date):
    """
    Convert monthly precipitation time series from kg m-2 s-1 to cm.

    Parameters:
    -----------
    precip_series : list
        List of precipitation values in kg m-2 s-1 (monthly averages)
    start_date : tuple
        (year, month) of first value in series
    end_date : tuple
        (year, month) of last value in series

    Returns:
    --------
    list
        Precipitation values in cm (monthly totals)
    """

    # Generate list of (year, month) tuples for the entire period
    start = date(start_date[0], start_date[1], 1)
    end = date(end_date[0], end_date[1], 1)

    months = []
    current = start
    while current <= end:
        months.append((current.year, current.month))
        current += relativedelta(months=1)

    # Verify length matches
    if len(precip_series) != len(months):
        raise ValueError(f"Series length ({len(precip_series)}) doesn't match date range ({len(months)} months)")

    # Convert each value
    precip_cm = []
    for precip_rate, (year, month) in zip(precip_series, months):
        days_in_month = calendar.monthrange(year, month)[1]
        seconds_in_month = days_in_month * 24 * 3600

        # kg m-2 s-1 * seconds = kg m-2 = mm, then / 10 = cm
        monthly_cm = (precip_rate * seconds_in_month) / 10
        precip_cm.append(monthly_cm)

    return precip_cm


def average_climate_data_by_month(input_filepath):
    """Used to create monthly averaged files for the climate
    from monthly measurements. In the monthly averaged version,
    the value for each month is the same accross all years, and is
    the average of all values for this month and variable accross all years.

    We do this because while we could use the Monthly_AverageAllYears statement
    from the climate library to do this, this seems to create a problem in PnET-Cohort
    which is not adapted to deal with the year values that the climate library
    return in that case. See https://github.com/LANDIS-II-Foundation/Library-PnET-Cohort/issues/7
    Instead of fixing the code, it's simpler to just modify
    our input files."""
    # Read the CSV file
    df = pd.read_csv(input_filepath)

    # Group by Month and Variable, calculate mean of eco1
    monthly_avg = df.groupby(['Month', 'Variable'], as_index=False)['eco1'].mean()

    # Create a complete dataframe with all Year/Month/Variable combinations
    # using the original years but with averaged values
    years = df['Year'].unique()
    result = []

    for year in years:
        temp_df = monthly_avg.copy()
        temp_df['Year'] = year
        result.append(temp_df)

    # Concatenate and reorder columns
    df_averaged = pd.concat(result, ignore_index=True)
    df_averaged = df_averaged[['Year', 'Month', 'Variable', 'eco1']]

    # Sort by Year, Month, Variable for consistency
    df_averaged = df_averaged.sort_values(['Year', 'Month', 'Variable']).reset_index(drop=True)
    
    # We sort the rows
    df_averaged = df_averaged.sort_values(by=['Variable', 'Year', 'Month'], ascending=[True, True, True])
    
    # Generate output filename
    output_filepath = input_filepath.replace('.csv', '_MonthlyAveraged.csv')

    # Save to CSV
    df_averaged.to_csv(output_filepath, index=False)

    return output_filepath


import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
from collections import defaultdict

def plot_all_cohort_results(output_folder, species_colors):
    """
    Plot vegetation model results for multiple cohorts.

    Parameters:
    -----------
    output_folder : str or Path
        Path to folder containing cohort CSV files
    """

    # Step 1: Load all cohort files
    output_path = Path(output_folder)
    cohort_files = list(output_path.glob("Cohort_*.csv"))

    if not cohort_files:
        print(f"No cohort files found in {output_folder}")
        return

    # Step 2: Parse filenames and load data
    cohorts_data = []
    species_cohorts = defaultdict(list)

    for file in cohort_files:
        # Extract species and implant year from filename
        parts = file.stem.split('_')
        species = parts[1]
        implant_year = int(parts[2])
        label = f"{species}_{implant_year}"

        # Load data
        df = pd.read_csv(file)
        df['Cohort'] = label
        df['Species'] = species
        df['ImplantYear'] = implant_year
        df['AbovegroundBiomass_InSite'] = df['SiteWood(gDW)'] + df['SiteFol(gDW)']

        cohorts_data.append(df)
        species_cohorts[species].append((implant_year, label))

    # Step 3: Combine all data
    all_data = pd.concat(cohorts_data, ignore_index=True)

    # Step 4: Determine time range
    min_year = all_data['Year'].min()
    max_year = all_data['Year'].max()

    # Step 5: Assign colors to species and cohorts
    species_list = sorted(species_cohorts.keys())
    # base_colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Blue, Orange, Green
    # species_colors = {species: base_colors[idx % 3] for idx, species in enumerate(species_list)}

    cohort_colors = {}
    for idx, species in enumerate(species_list):
        cohorts = sorted(species_cohorts[species])
        n_cohorts = len(cohorts)

        # Create gradient for this species
        # base_color = base_colors[idx % 3]
        base_color = species_colors[species]
        base_rgb = plt.matplotlib.colors.to_rgb(base_color)

        for i, (implant_year, label) in enumerate(cohorts):
            # Vary brightness from 0.4 to 1.0
            brightness = 0.4 + 0.6 * (i / max(1, n_cohorts - 1))
            cohort_colors[label] = tuple(c * brightness for c in base_rgb)

    # Step 6: Prepare data for stackplots
    # Aggregate aboveground biomass by species and year
    species_biomass = all_data.groupby(['Year', 'Species'])['AbovegroundBiomass_InSite'].sum().unstack(fill_value=0)
    years = species_biomass.index.values

    # Prepare data arrays for stackplot
    biomass_arrays = [species_biomass[species].values if species in species_biomass.columns else np.zeros(len(years)) 
                      for species in species_list]

    # Calculate relative proportions
    total_biomass = np.sum(biomass_arrays, axis=0)
    total_biomass[total_biomass == 0] = 1  # Avoid division by zero
    proportion_arrays = [(biomass / total_biomass * 100) for biomass in biomass_arrays]

    # Step 7: Create plots
    variables = ['SiteWood(gDW)', 'Fol(gDW)', 'LAI(m2)', 'SiteLAI(m2)', 'NSC(gC)']
    fig, axes = plt.subplots(5, 1, figsize=(12, 16))

    # Individual cohort plots
    for ax, var in zip(axes[:3], variables):
        for cohort_label in sorted(cohort_colors.keys()):
            cohort_df = all_data[all_data['Cohort'] == cohort_label]
            ax.plot(cohort_df['Year'], cohort_df[var], 
                   label=cohort_label, 
                   color=cohort_colors[cohort_label],
                   linewidth=1.5,
                   alpha=0.6)

        ax.set_xlabel('Year')
        ax.set_ylabel(var)
        ax.set_title(f'{var} over time')
        ax.set_xlim(min_year, max_year)
        ax.grid(True, alpha=0.3)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)

    # Absolute aboveground biomass stackplot
    axes[3].stackplot(years, *biomass_arrays, 
                      labels=species_list,
                      colors=[species_colors[sp] for sp in species_list],
                      alpha=0.7)
    axes[3].set_xlabel('Year')
    axes[3].set_ylabel('Aboveground Biomass (gDW)')
    axes[3].set_title('Total Aboveground Biomass by Species')
    axes[3].set_xlim(min_year, max_year)
    axes[3].grid(True, alpha=0.3)
    axes[3].legend(loc='upper left', fontsize=8)

    # Relative aboveground biomass stackplot
    axes[4].stackplot(years, *proportion_arrays,
                      labels=species_list,
                      colors=[species_colors[sp] for sp in species_list],
                      alpha=0.7)
    axes[4].set_xlabel('Year')
    axes[4].set_ylabel('Relative Proportion (%)')
    axes[4].set_title('Relative Aboveground Biomass by Species')
    axes[4].set_xlim(min_year, max_year)
    axes[4].set_ylim(0, 100)
    axes[4].grid(True, alpha=0.3)
    axes[4].legend(loc='upper left', fontsize=8)

    plt.subplots_adjust(hspace=0.4)
    plt.show()

    return all_data


def plot_single_cohort_results(csv_path, monthsToKeep=None):
    """Plot results of PnET Site Outputs for a single cohort (a single .csv file)
    to better understand the dynamic of cohorts."""
    
    # Read the CSV file
    df = pd.read_csv(csv_path)

    # Filter data by monthsToKeep if specified
    if monthsToKeep is not None:
        df = df.iloc[0:monthsToKeep]

    # Create figure with 4 subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Cohort Results: {csv_path.split("/")[-1]}', fontsize=14, fontweight='bold')

    # Subplot 1: Wood biomass
    axes[0, 0].plot(df['Time'], df['Wood(gDW)'], color='#8B4513', linewidth=2)
    axes[0, 0].set_xlabel('Time (year)')
    axes[0, 0].set_ylabel('Wood (gDW)')
    axes[0, 0].set_title('Wood Biomass Evolution')
    axes[0, 0].grid(True, alpha=0.3)

    # Subplot 2: LAI
    axes[0, 1].plot(df['Time'], df['LAI(m2)'], color='#228B22', linewidth=2)
    axes[0, 1].set_xlabel('Time (year)')
    axes[0, 1].set_ylabel('LAI (m²)')
    axes[0, 1].set_title('Leaf Area Index Evolution')
    axes[0, 1].grid(True, alpha=0.3)

    # Subplot 3: NSC
    axes[1, 0].plot(df['Time'], df['NSC(gC)'], color='#FF8C00', linewidth=2)
    axes[1, 0].set_xlabel('Time (year)')
    axes[1, 0].set_ylabel('NSC (gC)')
    axes[1, 0].set_title('Non-Structural Carbohydrates Evolution')
    axes[1, 0].grid(True, alpha=0.3)

    # Subplot 4: Environmental factors (0-1 range)
    axes[1, 1].plot(df['Time'], df['fWater(-)'], color='#1E90FF', linewidth=2, alpha=0.8, label='fWater')
    axes[1, 1].plot(df['Time'], df['fRad(-)'], color='#FFD700', linewidth=2, alpha=0.8, label='fRad')
    axes[1, 1].plot(df['Time'], df['fTemp_psn(-)'], color='#DC143C', linewidth=2, alpha=0.8, label='fTemp_psn')
    axes[1, 1].plot(df['Time'], df['fage(-)'], color='#9370DB', linewidth=2, alpha=0.8, label='fage')
    axes[1, 1].set_xlabel('Time (year)')
    axes[1, 1].set_ylabel('Factor value (-)')
    axes[1, 1].set_title('Environmental and Age Factors')
    axes[1, 1].set_ylim(-0.05, 1.05)
    axes[1, 1].legend(loc='best', framealpha=0.9)
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def create_species_csv(species_tuple, numberOfCells, ageRange, filename='output.csv'):
    """Function used to create the .csv files to initialize
    the landscapes for the little simulations used for the calibration."""
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')

        # Write header
        writer.writerow(['MapCode', 'SpeciesName', 'CohortAge', 'CohortBiomass'])

        # Loop through MapCode 1 to 1001
        for map_code in range(1, numberOfCells+2):
            # Write one row for each species
            for species in species_tuple:
                random_age = random.randint(ageRange[0], ageRange[1])
                writer.writerow([map_code, species, random_age, 0])

def min_max_top30_contiguous(series) -> tuple:
    """
    Returns (min, max) of a contiguous window of the top 30% values,
    anchored at the global maximum and expanded greedily left/right.
    Used to get the variation of the LAI around its maximum
    """
    values = series.reset_index(drop=True)
    n = len(values)
    n_top = max(1, int(n * 0.30))

    peak_idx = values.idxmax()

    left = peak_idx - 1
    right = peak_idx + 1
    window_indices = [peak_idx]

    while len(window_indices) < n_top:
        has_left = left >= 0
        has_right = right < n

        if has_left and has_right:
            # Expand toward the side with the larger neighbor
            if values[left] >= values[right]:
                window_indices.append(left)
                left -= 1
            else:
                window_indices.append(right)
                right += 1
        elif has_left:
            window_indices.append(left)
            left -= 1
        elif has_right:
            window_indices.append(right)
            right += 1
        else:
            break  # Series exhausted

    selected_values = values[window_indices]
    return selected_values.min(), selected_values.max()


import copy

def calibrationSimulationMonoculturemanawan(duration = 100,
                                            climate = "mild",
                                            soil = "SILO",
                                            speciesToSimulate = "ABIE.BAL",
                                            dictOfInitialCoreSpeciesParameters = json.load(open('./SpeciesParametersSets/Initial/initialCoreSpeciesParameters.json')),
                                            dictOfInitialPnETSpeciesParameters = json.load(open('./SpeciesParametersSets/Initial/initialPnETSpeciesParameters.json')),
                                            dictOfInitialPnETGenericParameters = json.load(open('./SpeciesParametersSets/Initial/InitialGenericParameters.json')),
                                            plotResults = False,
                                            plotAdjustedParameters = False,
                                            saveGrowthCurvePath = False):
    
    # We prepare the simulation files
    PnETGitHub_OneCellSim = parse_All_LANDIS_PnET_Files(r"./SimulationFiles/PnETGitHub_OneCellSim_v8")
    
    # - Species.txt : replace with the right initial core species parameters from the JSON file
    PnETGitHub_OneCellSim["species.txt"] = copy.deepcopy(dictOfInitialCoreSpeciesParameters)
    
    # - SpeciesParameters.txt : replace with the initial PnET species parameters from the JSOn file
    PnETGitHub_OneCellSim["SpeciesParameters.txt"] = copy.deepcopy(dictOfInitialPnETSpeciesParameters)
    
    # - PnETGenericParameters.txt : replace with initial generic parameters from the JSON file
    PnETGitHub_OneCellSim["PnETGenericParameters.txt"] = copy.deepcopy(dictOfInitialPnETGenericParameters)
    
    # Setting duration and timestep (minimal timestep for maximum spatial resolution, although it shouldn't change anything)
    PnETGitHub_OneCellSim["pnetsuccession.txt"]["Timestep"] = "1"
    PnETGitHub_OneCellSim["scenario.txt"]["Duration"] = str(duration)
    # Changing the soil if needed
    PnETGitHub_OneCellSim["EcoregionParameters.txt"]["EcoregionParameters"]["eco1"]["SoilType"] = soil
    # - pnetsuccession.txt : change startyear to 1900 and latitude to village of Manawan (47.2223)
    startYear = 1950
    PnETGitHub_OneCellSim["pnetsuccession.txt"]["StartYear"] = str(startYear)
    PnETGitHub_OneCellSim["pnetsuccession.txt"]["Latitude"] = "47.2223"
    # No dispersal, since we only simulate one pixel and the life of one cohort (monoculture and even-aged)
    PnETGitHub_OneCellSim["pnetsuccession.txt"]["SeedingAlgorithm"] = "NoDispersal"
    PnETGitHub_OneCellSim["PnETGenericParameters.txt"]["PreventEstablishment"] = "True"
    # Setting other parameters
    PnETGitHub_OneCellSim["scenario.txt"]["CellLength"] = "100"

    # Preparing the landscape
    # Removing Other species from the simulation
    speciesToRemove = []
    for species in PnETGitHub_OneCellSim["SpeciesParameters.txt"]["PnETSpeciesParameters"].keys():
        if species != speciesToSimulate and species != "LandisData":
            speciesToRemove.append(species)
    for species in speciesToRemove:
        if species in PnETGitHub_OneCellSim["species.txt"].keys():
            del PnETGitHub_OneCellSim["species.txt"][species]
        if species in PnETGitHub_OneCellSim["SpeciesParameters.txt"]["PnETSpeciesParameters"].keys():
            del PnETGitHub_OneCellSim["SpeciesParameters.txt"]["PnETSpeciesParameters"][species]
    # Inserting reading of the right climate file
    if climate == "mild":
        PnETGitHub_OneCellSim["pnetsuccession.txt"]["ClimateConfigFile"] = "ClimateConfigSimpleSims_MonthlyAveraged.txt"
    elif climate == "realHistorical":
        PnETGitHub_OneCellSim["pnetsuccession.txt"]["ClimateConfigFile"] = "ClimateConfigSimpleSims.txt"
    elif climate == "testFilesGithub":
        pass # The climate files from github are used by default if we don't input a climate config file
    else:
        raise ValueError("Climate value : " + str(climate) + " not recognized.")
    
    # -  PnEToutputsites_onecell.txt : replace site location
    PnETGitHub_OneCellSim["PnEToutputsites_onecell.txt"]["Site1"] = '1 1'

    # Writing the files in a temporary folder
    simulationPath = "/tmp/monocultureCalibrationPnET/"

    # We create the folder
    if not os.path.exists(simulationPath):
        os.mkdir(simulationPath)
    else:
        shutil.rmtree(simulationPath)
        os.mkdir(simulationPath)
    
    write_all_LANDIS_files(simulationPath,
                           PnETGitHub_OneCellSim,
                           True)

    # Copy the climate files
    if climate == "mild":
        shutil.copy("./SimulationFiles/ClimateConfigSimpleSims_MonthlyAveraged.txt", simulationPath)
        shutil.copy("./ReferencesAndData/Climate Data/dataFrameClimate_historicalMonthly_Ouranos_MonthlyAveraged.csv", simulationPath)
        shutil.copy("./ReferencesAndData/Climate Data/dataFrameClimate_SpinupMonthly_Ouranos_MonthlyAveraged.csv", simulationPath)
        os.remove(simulationPath + "/climate.txt")
        # I'm getting an issue when starting simulations at year 1900 with low longevity values. See https://github.com/LANDIS-II-Foundation/Library-Climate/issues/32
        # It seems to be related to the spinup code taking the wrong amount of year based on the longevity of the species
        # I'm removing years from the spinup file to avoid this
        spinupData = pd.read_csv(simulationPath + "dataFrameClimate_SpinupMonthly_Ouranos_MonthlyAveraged.csv")
        maxLongevity = dictOfInitialCoreSpeciesParameters[speciesToSimulate]["Longevity"]
        spinupData = spinupData[spinupData["Year"] > (spinupData['Year'].max() - int(maxLongevity))]
        spinupData.to_csv(simulationPath + "dataFrameClimate_SpinupMonthly_Ouranos_MonthlyAveraged.csv", index=False)
    elif climate == "realHistorical":
        shutil.copy("./SimulationFiles/ClimateConfigSimpleSims.txt", simulationPath)
        shutil.copy("./ReferencesAndData/Climate Data/dataFrameClimate_historicalMonthly_Ouranos.csv", simulationPath)
        shutil.copy("./ReferencesAndData/Climate Data/dataFrameClimate_SpinupMonthly_Ouranos.csv", simulationPath)
        os.remove(simulationPath + "/climate.txt")
        # I'm getting an issue when starting simulations at year 1900 with low longevity values. See https://github.com/LANDIS-II-Foundation/Library-Climate/issues/32
        # It seems to be related to the spinup code taking the wrong amount of year based on the longevity of the species
        # I'm removing years from the spinup file to avoid this
        spinupData = pd.read_csv(simulationPath + "dataFrameClimate_SpinupMonthly_Ouranos.csv")
        maxLongevity = dictOfInitialCoreSpeciesParameters[speciesToSimulate]["Longevity"]
        spinupData = spinupData[spinupData["Year"] > (spinupData['Year'].max() - int(maxLongevity))]
        spinupData.to_csv(simulationPath + "dataFrameClimate_SpinupMonthly_Ouranos.csv", index=False)
    elif climate == "testFilesGithub":
        pass # The climate files from github are used by default if we don't input a climate config file
    else:
        raise ValueError("Climate value : " + str(climate) + " not recognized.")
    # Removing climate.txt (old climate file from the test files)
    

    # Preparing rasters
    numberOfCells = 1
    ageRange = [1, 1]
    # Preparing the data we will put in the rasters
    data = np.ones((1, numberOfCells), dtype=np.uint8)
    # Transform used to settle the size of cells - not sure is very useful
    transform = Affine.translation(0, 0) * Affine.scale(1, 1)
    # Creating the ecoregion raster
    with rasterio.open(
    simulationPath + '/ecoregion.img',
    'w',
    driver='GTiff',
    height=1,
    width=numberOfCells,
    count=1,
    dtype=data.dtype,
    crs='EPSG:4326',
    transform=transform
    ) as dst:
        dst.write(data, 1)
    # Preparing the initial communities raster
    with rasterio.open(
    simulationPath + '/initial-communities.img',
    'w',
    driver='GTiff',
    height=1,
    width=numberOfCells,
    count=1,
    dtype=data.dtype,
    crs='EPSG:4326',
    transform=transform
    ) as dst:
        dst.write(data, 1)
    data = np.arange(1, numberOfCells+1, dtype="int32").reshape(1, numberOfCells)
    # Creating initial community .csv
    create_species_csv([speciesToSimulate], numberOfCells, ageRange, filename = simulationPath + "/initial-communities.csv")

    # We launch the simulation
    runLANDIS_Simulation(simulationPath,
                         "scenario.txt",
                        False)

    # We get the result file : only .csv file that should represent the cohort
    csv_file_cohort = pd.read_csv(glob.glob(f'{simulationPath}/Output/Site1/Cohort_*.csv')[0])
    # print(csv_file_cohort)
    # Add a measure of total biomass
    csv_file_cohort["SumFoliageWood_Site"] = csv_file_cohort["SiteFol(gDW)"] + csv_file_cohort["SiteWood(gDW)"]
    # We also get the foliage biomass from another output; this is because the foliage+wood biomass in the cohort file
    # is in gDW, or gram of dry weight. But we want g/m2 to compare to NFI data.
    csv_file_WoodFoliageBiomass = pd.read_csv(f'{simulationPath}output/WoodFoliageBiomass/WoodFoliageBiomass-AllYears.csv')
    # We also get the site csv that has the temperatures (for diagnostics)
    csv_file_site = pd.read_csv(f'{simulationPath}/Output/Site1/Site.csv')
    
    # We get the results we need
    variablesOutput = ["Biomass peak height", "Biomass peak time", "Biomass peak 95% time",
                       "Initation of decline", "Time of death",
                       "Maximum LAI", "LAI stability", "Average Fwater",
                      "Average July Temperature", "Biomass at 50% of biomass peak 95% time"]
    dictOfOutput = {}
    for variable in variablesOutput:
        if variable == "Biomass peak height":
            dictOfOutput[variable] = (csv_file_WoodFoliageBiomass[str(speciesToSimulate) + "_g/m2"].max())
        elif variable == "Biomass peak time":
            max_index = csv_file_cohort['SumFoliageWood_Site'].idxmax()
            # Here, we want the age of the cohort when it reaches the maximum
            # Since there is one row per month, and since the first row is the first month of life
            # of the cohort (even if there is a spinup), then the age of the cohorts (in years) is
            # simply row/12 (since 12 rows make a year of life.
            dictOfOutput[variable] = ((max_index+1)/12)
        elif variable == "Biomass peak 95% time":
            # Calculate 95% of the maximum value in the column
            threshold_value = csv_file_cohort['SumFoliageWood_Site'].max() * 0.95
            
            # Find the index of the first row where the column value is greater than or equal to the threshold
            # The .idxmax() method returns the index of the first occurrence of the maximum value.
            # To find the first occurrence of a value that meets a condition, we can create a boolean series.
            # However, idxmax() is for maximum values. A more direct approach for a threshold is to filter.
            
            # Filter the DataFrame to get rows that meet the condition
            rows_meeting_condition = csv_file_cohort[csv_file_cohort['SumFoliageWood_Site'] >= threshold_value]
            
            # Get the index of the first row from the filtered DataFrame
            if not rows_meeting_condition.empty:
                first_index = rows_meeting_condition.index[0]
                dictOfOutput[variable] = ((first_index+1)/12)
                # print(f"The first index where the column value is at least 95% of the maximum is: {first_index}")
            else:
                # print("No rows found that meet the condition.")
                dictOfOutput[variable] = "None"
        elif variable == "Initation of decline":
            # When fAge goes under 0.90 for the first time
            # Create a boolean mask for rows where 'my_column' is less than the threshold
            condition_met = csv_file_cohort['fage(-)'] < 0.9
            first_occurrence_index = csv_file_cohort.index[condition_met].min()
            # We want to get the age of the cohort where fAge goes under 0.90
            # Since one row per month, it's simply the number of row/index divided by 12.
            dictOfOutput[variable] = ((first_occurrence_index+1)/12)
        elif variable == "Time of death":
            # Cohort dies when NSCfrac is inferior to 0.01 at the end of a year (december)
            # We get the index of the row when this happens and divided it by 12 (since there
            # is one row per month), this gives us the age (in years) of death.
            mask = (csv_file_cohort['Month'] == 12) & (csv_file_cohort['NSCfrac(-)'] < 0.01)
            try:
                idx = csv_file_cohort[mask].index[0]
                dictOfOutput[variable] = ((idx+1)/12)
            except:
                # Code to handle any other error
                # print("Havent found the index where the cohort died. Might be an issue with the cohort.csv")
                dictOfOutput[variable] = "None"
        elif variable == "Maximum LAI":
            dictOfOutput[variable] = csv_file_cohort["LAI(m2)"].max()
        elif variable == "LAI stability":
            # dictOfOutput[variable] = min_max_top30_contiguous(csv_file_cohort["LAI(m2)"])

            # Step 1: Get max yearly LAI per year
            yearly_max_lai = csv_file_cohort.groupby("Year")["LAI(m2)"].max()
            
            # Step 2: Find the year where LAI reaches its overall maximum
            year_of_max_lai = yearly_max_lai.idxmax()
            
            # Step 3: Find the last year before fAge(-) drops below 0.6
            year_before_fage = csv_file_cohort[csv_file_cohort["fage(-)"] >= 0.7]["Year"].max()
            
            # Step 4: Filter and get the range (min, max) of yearly max LAI values in that window
            result = yearly_max_lai.loc[year_of_max_lai:year_before_fage]
            dictOfOutput[variable] = (result.min(), result.max())
            
        elif variable == "Average Fwater":
            dictOfOutput[variable] = csv_file_cohort["fWater(-)"].mean()
        elif variable == "Average July Temperature":
            dictOfOutput[variable] = csv_file_site[csv_file_site['Month'] == 7]['Tday(C)'].mean()
        elif variable == "Biomass at 50% of biomass peak 95% time":
            # Used to check if a cohort grows fast enough in its young years (see subphase 1.3)
            # We need to interpolate because the peak time is from the cohort file with a monthly timestep,
            # while the biomass peak heigh is from the csv file with the same timestep as PnET-Succession (e.g. 5 years)
            df = csv_file_WoodFoliageBiomass[["Time", str(speciesToSimulate) + "_g/m2"]].copy()
            df = df.sort_values("Time").reset_index(drop=True)
        
            # Rows below and above the target time
            below = df[df["Time"] <= 0.5*dictOfOutput["Biomass peak 95% time"]]
            above = df[df["Time"] >= 0.5*dictOfOutput["Biomass peak 95% time"]]
        
            # Exact match — no interpolation needed
            if not below.empty and below.iloc[-1]["Time"] == 0.5*dictOfOutput["Biomass peak 95% time"]:
                return below.iloc[-1][str(speciesToSimulate) + "_g/m2"]
            if not above.empty and above.iloc[0]["Time"] == 0.5*dictOfOutput["Biomass peak 95% time"]:
                return above.iloc[0][str(speciesToSimulate) + "_g/m2"]
        
            # Check that bracketing rows exist on both sides
            if below.empty or above.empty:
                # raise ValueError(
                #     f"dictOfOutput["Biomass peak 95% time"]={dictOfOutput["Biomass peak 95% time"]} is outside the range "
                #     f"of {"Time"} in csv_file_WoodFoliageBiomass "
                #     f"({df["Time"].min()} – {df["Time"].max()})."
                # )
                dictOfOutput[variable] = "Biomass peak 95% time is outside the range in csv_file_WoodFoliageBiomass"
        
            t0, b0 = below.iloc[-1]["Time"], below.iloc[-1][str(speciesToSimulate) + "_g/m2"]
            t1, b1 = above.iloc[0]["Time"],  above.iloc[0][str(speciesToSimulate) + "_g/m2"]
        
            # Linear interpolation
            interpolated_biomass = b0 + (b1 - b0) * (0.5*dictOfOutput["Biomass peak 95% time"] - t0) / (t1 - t0)
        
            dictOfOutput[variable] = interpolated_biomass
        else:
            raise ValueError("Value not recognized for output variable : " + str(variable))

    if plotResults: plot_all_cohort_results(str(simulationPath) + "/Output/Site1", {speciesToSimulate:"#5e81ac"})

    if plotAdjustedParameters:
        fRad = csv_file_cohort['fRad(-)']
        
        # --- Parameters ---
        MaxFracFol   = float(dictOfInitialPnETSpeciesParameters["PnETSpeciesParameters"][speciesToSimulate]["MaxFracFol"])
        FracFolShape = float(dictOfInitialPnETSpeciesParameters["PnETSpeciesParameters"][speciesToSimulate]["FracFolShape"])
        FracFol      = float(dictOfInitialPnETSpeciesParameters["PnETSpeciesParameters"][speciesToSimulate]["FracFol"])
        
        # --- Rolling average window ---
        rolling_window = 12  # <-- specify window size here
        
        # --- Computation ---
        AdjustedFracFol = FracFol + ((MaxFracFol - FracFol) * (fRad ** FracFolShape))
        RollingAvg      = AdjustedFracFol.rolling(window=rolling_window, center=True).mean()
        RollingAvgFRad  = fRad.rolling(window=rolling_window, center=True).mean()
        
        # --- Plot ---
        fig, ax = plt.subplots(figsize=(10, 4))
        ax2     = ax.twinx()  # secondary y-axis sharing the same x-axis
        
        # Push ax2 (fRad) behind ax (everything else)
        ax.set_zorder(ax2.get_zorder() + 1)
        ax.patch.set_visible(False)   # let ax2's content show through ax's transparent background
        
        # fRad (background, noisy) + its rolling average (background, solid green)
        l5, = ax2.plot(csv_file_cohort['Year'], fRad, color="lightgreen", linewidth=1,
                       alpha=0.6, zorder=1, label="fRad")
        l6, = ax2.plot(csv_file_cohort['Year'], RollingAvgFRad, color="green", linewidth=2,
                       zorder=2, label=f"fRad rolling avg (window={rolling_window})")
        
        # Foreground curves
        l1, = ax.plot(csv_file_cohort['Year'], AdjustedFracFol, color="darkorange", linewidth=1,
                      zorder=3, linestyle="--", alpha = 0.5, label="AdjFracFol (FracFolShape = " + str(dictOfInitialPnETSpeciesParameters["PnETSpeciesParameters"][speciesToSimulate]["FracFolShape"]) + ")")
        l2, = ax.plot(csv_file_cohort['Year'], RollingAvg, color="darkorange", linewidth=2,
                      zorder=4, label=f"AdjFracFol Rolling avg (window={rolling_window})")
        l3  = ax.axhline(FracFol,    color="blue",   linestyle="--", linewidth=1, zorder=4,
                          label=f"FracFol = {FracFol}")
        l4  = ax.axhline(MaxFracFol, color="tomato", linestyle="--", linewidth=1, zorder=4,
                          label=f"MaxFracFol = {MaxFracFol}")
        
        ax.set_xlabel("Timestep")
        ax.set_ylabel("AdjustedFracFol")
        ax2.set_ylabel("fRad (-)")
        ax.set_title("AdjustedFracFol over time")
        
        # Combined legend, drawn on ax so it sits above everything (including ax2's patch)
        lines  = [l1, l2, l3, l4, l5, l6]
        labels = [l.get_label() for l in lines]
        ax.legend(lines, labels, loc="best").set_zorder(5)
        
        plt.tight_layout()
        plt.show()



        # --- Parameters ---
        MaxFolN   = float(dictOfInitialPnETSpeciesParameters["PnETSpeciesParameters"][speciesToSimulate]["MaxFolN"])
        FolNShape = float(dictOfInitialPnETSpeciesParameters["PnETSpeciesParameters"][speciesToSimulate]["FolNShape"])
        FolN      = float(dictOfInitialPnETSpeciesParameters["PnETSpeciesParameters"][speciesToSimulate]["FolN"])
        
        # --- Computation ---
        AdjustedFolN = FolN + ((MaxFolN - FolN) * (fRad ** FolNShape))
        
        # --- Plot ---
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(csv_file_cohort['Year'], AdjustedFolN, color="steelblue", linewidth=1, label="AdjustedFolN")
        ax.axhline(FolN,    color="gray",   linestyle="--", linewidth=1, label=f"FolN = {FolN}")
        ax.axhline(MaxFolN, color="tomato", linestyle="--", linewidth=1, label=f"MaxFolN = {MaxFolN}")
        
        ax.set_xlabel("Timestep")
        ax.set_ylabel("AdjustedFolN")
        ax.set_title("AdjustedFolN over time")
        ax.legend()
        plt.tight_layout()
        plt.show()

    
    if saveGrowthCurvePath:
        output_path = Path(saveGrowthCurvePath)

        # Create all parent directories if they don't exist
        output_path.parent.mkdir(parents=True, exist_ok=True)
    
        csv_file_WoodFoliageBiomass.to_csv(output_path)
    
    return(dictOfOutput)
    # Delete the folder
    # shutil.rmtree(simulationPath)


#########################################################
# FUNCTION TO GET THE BOUNDS FOR EACH PARAMETERS WE WILL CALIBRATE
#########################################################

# Function to read the values taken from the litterature from the markdown table where they are

import re
import nbformat

def parse_parameters_bounds_table(notebook_path: str, markdown_cell_number: int) -> dict:
    """
    Reads a markdown cell by its markdown-cell index (1-based, skipping
    code/raw cells) from the given notebook, extracts the first markdown
    table found, and returns a nested dict:
        result[parameter_name][column_level] = [value1, value2, ...]

    Parameters
    ----------
    notebook_path : str
        Path to the .ipynb notebook file.
    markdown_cell_number : int
        1-based index counting only markdown cells.

    Returns
    -------
    dict
        Nested dictionary of parsed values, or {} on error.
    """

    # --- 1. Read the markdown cell ---
    source = read_markdown_cell(notebook_path, markdown_cell_number)

    if source is None:
        print(f"[ERROR] Could not find markdown cell number {markdown_cell_number} "
              f"in '{notebook_path}'.")
        return {}

    # --- 2. Extract the markdown table lines ---
    lines = source.splitlines()
    table_lines = []
    in_table = False

    for line in lines:
        stripped = line.strip()
        if re.match(r"^\|.*\|$", stripped):
            in_table = True
            table_lines.append(stripped)
        else:
            if in_table:
                # Stop at the first non-table line after the table started
                break

    if len(table_lines) < 3:
        print("[ERROR] No valid markdown table found in the cell "
              "(need at least a header row, a separator row, and one data row).")
        return {}

    # --- 3. Validate separator row ---
    separator = table_lines[1]
    if not re.match(r"^\|[\s\-:|]+\|$", separator):
        print(f"[ERROR] Row 1 does not look like a markdown table separator: "
              f"'{separator}'")
        return {}

    # --- 4. Helper: split a markdown row into cells ---
    def split_row(row: str) -> list:
        return [cell.strip() for cell in row.strip("|").split("|")]

    # --- 5. Parse header row ---
    header_cells = split_row(table_lines[0])

    if len(header_cells) < 3:
        print(f"[ERROR] Table header has fewer than 3 columns: {header_cells}")
        return {}

    # Column 0 = parameter name, column 1 = description (ignored), rest = levels
    levels = header_cells[2:]

    # --- 6. Helper: extract numeric values from a cell string ---
    def extract_values(cell_text: str) -> list:
        """
        Strips citation markers ([^N] or [N]) and splits on commas,
        returning a list of floats (or strings if not numeric).
        Empty cells return [].
        """
        cleaned = re.sub(r"\[\^?\d+\]", "", cell_text)
        parts = [p.strip() for p in cleaned.split(",")]
        values = [p for p in parts if p]
        result = []
        for v in values:
            try:
                result.append(float(v))
            except ValueError:
                result.append(v)
        return result

    # --- 7. Parse data rows ---
    result = {}

    for row_line in table_lines[2:]:  # skip header and separator
        row_cells = split_row(row_line)

        if len(row_cells) < 2:
            print(f"[WARNING] Skipping malformed row: '{row_line}'")
            continue

        param_name = row_cells[0]

        if not param_name:
            print(f"[WARNING] Skipping row with empty parameter name: '{row_line}'")
            continue

        # Pad row if it has fewer columns than the header
        while len(row_cells) < len(header_cells):
            row_cells.append("")

        result[param_name] = {}
        for i, level in enumerate(levels):
            col_index = i + 2  # offset: skip param name + description
            cell_text = row_cells[col_index] if col_index < len(row_cells) else ""
            result[param_name][level] = extract_values(cell_text)

    return result


# Functions to compute the bounds for each parameters depending on the list of values 
# If we don't have a lot of values, or for missing values, we compute bounds
# based on informations from other cells

import numpy as np
from typing import Union


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def detect_characteristic_type(level_keys: list) -> str:
    """Detect 'ordinal' (>2 levels) or 'binary' (exactly 2 levels)."""
    return "binary" if len(level_keys) == 2 else "ordinal"


def build_level_scores(level_keys: list) -> dict:
    """Map each level key to a numeric score from 1.0 to N."""
    return {key: float(i + 1) for i, key in enumerate(level_keys)}


def compute_pooled_std(level_data: dict) -> float:
    """Pooled sample std across all levels."""
    all_values = [v for vals in level_data.values() for v in vals]
    return float(np.std(all_values, ddof=1)) if len(all_values) >= 2 else 0.0


def fit_trend(level_data: dict, level_scores: dict, min_n: int = 3) -> Union[tuple, None]:
    """
    Fit a linear regression (level_score -> mean_value) using only
    levels with at least `min_n` observations.

    Returns (slope, intercept) or None if fewer than 2 qualifying levels.
    """
    xs, ys = [], []
    for level, values in level_data.items():
        if len(values) >= min_n:
            xs.append(level_scores[level])
            ys.append(float(np.mean(values)))

    if len(xs) < 2:
        return None

    slope, intercept = np.polyfit(np.array(xs), np.array(ys), 1)
    return float(slope), float(intercept)


# ---------------------------------------------------------------------------
# Core bound computation
# ---------------------------------------------------------------------------

def compute_bounds_for_level(
    values: list,
    level,
    char_type: str,
    level_data: dict,
    level_scores: dict,
    pooled_std: float,
    trend: Union[tuple, None],
    expansion_factor: float = 0.5,
) -> dict:
    """
    Compute (lower, upper) bounds for a single parameter-level combination.

    Rules:
        n >= 10  : empirical 5th–95th percentile
        3–9      : empirical range ± expansion_factor * pooled_std
        1–2      : extrapolated/borrowed mean ± pooled_std
        0        : extrapolated/borrowed mean ± pooled_std
    """
    n = len(values)
    result = {"n": n, "method": None, "lower": None, "upper": None}

    def _fallback_center() -> tuple[float, str]:
        if char_type == "ordinal" and trend is not None:
            slope, intercept = trend
            score = level_scores[level]
            center = slope * score + intercept
            return center, "extrapolated_trend"
        else:
            other_values = [
                v for k, v_list in level_data.items()
                if k != level for v in v_list
            ]
            if other_values:
                return float(np.mean(other_values)), "borrowed_mean"
            all_values = [v for vals in level_data.values() for v in vals]
            return (float(np.mean(all_values)) if all_values else float("nan")), "global_mean"

    if n >= 10:
        result["method"] = "empirical_percentile"
        result["lower"] = float(np.percentile(values, 5))
        result["upper"] = float(np.percentile(values, 95))

    elif 3 <= n <= 9:
        result["method"] = "empirical_expanded"
        result["lower"] = float(np.min(values)) - expansion_factor * pooled_std
        result["upper"] = float(np.max(values)) + expansion_factor * pooled_std

    elif 1 <= n <= 2:
        center, label = _fallback_center()
        result["method"] = f"{label}_pooled_std (n=1-2)"
        result["lower"] = center - pooled_std
        result["upper"] = center + pooled_std

    else:  # n == 0
        center, label = _fallback_center()
        result["method"] = f"{label}_pooled_std (n=0)"
        if np.isnan(center):
            result["lower"] = None
            result["upper"] = None
        else:
            result["lower"] = center - pooled_std
            result["upper"] = center + pooled_std

    return result


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def compute_all_bounds(
    data: dict,
    expansion_factor: float = 0.5,
    trend_min_n: int = 3,
    verbose: bool = False,
) -> dict:
    """
    Compute bounds for all parameters and their characteristic levels.

    Parameters
    ----------
    data : dict
        Nested dict: {param_name: {level_key: [values]}}
        Level keys can be any type (strings, ints, etc.); their order
        in the dict defines their ordinal score (first = 1, last = N).
    expansion_factor : float
        Multiplier for pooled std when expanding sparse empirical ranges.
    trend_min_n : int
        Minimum observations per level to include that level in the
        trend regression.
    verbose : bool
        If True, print a summary table of results after computation.

    Returns
    -------
    dict
        Nested dict: {param_name: {level_key: {n, method, lower, upper,
                                               characteristic_type,
                                               level_score, trend}}}
    """
    results = {}

    for param, level_data in data.items():
        level_keys = list(level_data.keys())
        char_type = detect_characteristic_type(level_keys)
        level_scores = build_level_scores(level_keys)
        pooled_std = compute_pooled_std(level_data)
        trend = (
            fit_trend(level_data, level_scores, min_n=trend_min_n)
            if char_type == "ordinal"
            else None
        )

        results[param] = {}
        for level, values in level_data.items():
            bounds = compute_bounds_for_level(
                values=values,
                level=level,
                char_type=char_type,
                level_data=level_data,
                level_scores=level_scores,
                pooled_std=pooled_std,
                trend=trend,
                expansion_factor=expansion_factor,
            )
            bounds["characteristic_type"] = char_type
            bounds["level_score"] = level_scores[level]
            if trend is not None:
                bounds["trend"] = {"slope": trend[0], "intercept": trend[1]}
            results[param][level] = bounds

    if verbose:
        for param, levels in results.items():
            print(f"\n=== {param} ===")
            for level, info in levels.items():
                trend_str = ""
                if "trend" in info:
                    t = info["trend"]
                    trend_str = f" | slope={t['slope']:.3f}, intercept={t['intercept']:.3f}"
                lo = f"{info['lower']:.3f}" if info["lower"] is not None else "None"
                hi = f"{info['upper']:.3f}" if info["upper"] is not None else "None"
                print(
                    f"  Level {str(level):>20} (score={info['level_score']:.0f}) | "
                    f"n={info['n']:>2} | "
                    f"method={info['method']:<40} | "
                    f"bounds=[{lo}, {hi}]"
                    f"{trend_str}"
                )

    return results

#################################################################
# TRYING TO RE-CREATE THE FUNCTIONS TO CREATE AND SELECT THE GAMS
#################################################################

# I've lost these functions because of a bug : createPercentilesBiomassDatasetsFromNFIData
# and getGamDiagnostics.

# I'm going to try to make them again, because they're needed.

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.mask import mask as rasterio_mask
from rasterio.features import geometry_mask
from shapely.ops import unary_union
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


def createPercentilesBiomassDatasetsFromNFIData(
    abundance_raster_path,
    age_raster_path,
    biomass_raster_path,
    speciesName,
    study_landscape_shp_path,
    canadian_ecozones_shp_path,
    age_window=5,
    abundance_window=5,
    percentile=100,
    mask_path=None,
    printPlot=False
):
    # ------------------------------------------------------------------
    # 1. Load shapefiles and reproject to raster CRS
    # ------------------------------------------------------------------
    with rasterio.open(abundance_raster_path) as src:
        raster_crs = src.crs
        raster_transform = src.transform
        raster_shape = (src.height, src.width)
        raster_nodata = src.nodata

    study_landscape = gpd.read_file(study_landscape_shp_path).to_crs(raster_crs)
    ecozones = gpd.read_file(canadian_ecozones_shp_path).to_crs(raster_crs)

    study_geom = unary_union(study_landscape.geometry)
    intersects_mask = ecozones.geometry.intersects(study_geom)
    study_ecozones = ecozones[intersects_mask]
    study_ecozones_union = unary_union(study_ecozones.geometry)

    surrounding_mask = ecozones.geometry.intersects(study_ecozones_union) & ~intersects_mask
    surrounding_ecozones = ecozones[surrounding_mask]
    surrounding_union = unary_union(
        pd.concat([study_ecozones, surrounding_ecozones]).geometry
    )
    canada_union = unary_union(ecozones.geometry)

    masks = {
        'study_landscape': study_geom,
        'ecozone': study_ecozones_union,
        'surrounding_ecozones': surrounding_union,
        'canada': canada_union,
    }

    # ------------------------------------------------------------------
    # 2. Helper: extract valid pixel triplets within a geometry
    #    Opens all 3 rasters once, clips to bounding box, applies
    #    geometry mask — avoids loading full rasters into RAM.
    # ------------------------------------------------------------------
    def extract_valid_pixels(geom):
        geom_geojson = [geom.__geo_interface__]

        with rasterio.open(abundance_raster_path) as src_a, \
             rasterio.open(age_raster_path) as src_ag, \
             rasterio.open(biomass_raster_path) as src_b:

            # Crop each raster to geometry bounding box
            abund_arr, abund_tf = rasterio_mask(src_a,  geom_geojson, crop=True, nodata=src_a.nodata)
            age_arr,   _        = rasterio_mask(src_ag, geom_geojson, crop=True, nodata=src_ag.nodata)
            bio_arr,   _        = rasterio_mask(src_b,  geom_geojson, crop=True, nodata=src_b.nodata)

            abund_nd = src_a.nodata
            age_nd   = src_ag.nodata
            bio_nd   = src_b.nodata

            # Build pixel-level geometry mask (True = inside geometry)
            geom_msk = geometry_mask(
                geom_geojson,
                transform=abund_tf,
                invert=True,
                out_shape=abund_arr.shape[1:]
            )

        abund = abund_arr[0].astype(np.float32)
        age   = age_arr[0].astype(np.float32)
        bio   = bio_arr[0].astype(np.float32)

        # Mask nodata
        valid = geom_msk.copy()
        if abund_nd is not None:
            valid &= (abund != abund_nd)
        if age_nd is not None:
            valid &= (age != age_nd)
        if bio_nd is not None:
            valid &= (bio != bio_nd)

        # Keep only positive/meaningful values
        valid &= (abund > 0) & (age > 0) & (bio >= 0)

        # Return only valid pixels — discard the rest immediately
        return age[valid], abund[valid], bio[valid]

    # ------------------------------------------------------------------
    # 3. Helper: build windowed percentile dataframe
    # ------------------------------------------------------------------
    def build_windowed_df(age_vals, abundance_vals, biomass_vals):
        print("Creating windowed dataframe...")

        # EDIT HERE : we change it so that it is in g/m2 instead of tons/hectares,
        # because LANDIS-II gives us g/m2.
        # To switch, we multiply by 100.
        species_biomass = (abundance_vals * biomass_vals / 100.0) * 100

        age_max   = age_vals.max()
        abund_max = abundance_vals.max()

        age_bins   = np.arange(0, age_max   + age_window,   age_window)
        abund_bins = np.arange(0, abund_max + abundance_window, abundance_window)

        # Digitize is faster than pd.cut for large arrays
        age_idx   = np.digitize(age_vals,   age_bins)   - 1
        abund_idx = np.digitize(abundance_vals, abund_bins) - 1

        # Clip indices to valid range
        age_idx   = np.clip(age_idx,   0, len(age_bins)   - 2)
        abund_idx = np.clip(abund_idx, 0, len(abund_bins) - 2)

        # Combine into a single integer key for groupby
        n_abund_bins = len(abund_bins) - 1
        combined_key = age_idx * n_abund_bins + abund_idx

        # Sort once for efficient grouping
        sort_order  = np.argsort(combined_key)
        sorted_keys = combined_key[sort_order]
        sorted_bio  = species_biomass[sort_order]

        unique_keys, first_idx, counts = np.unique(
            sorted_keys, return_index=True, return_counts=True
        )

        records = []
        for k, fi, cnt in zip(unique_keys, first_idx, counts):
            group_bio = sorted_bio[fi:fi + cnt]
            p_val     = np.percentile(group_bio, percentile)
            a_idx     = k // n_abund_bins
            ab_idx    = k  % n_abund_bins
            records.append({
                'age':               age_bins[a_idx]   + age_window   / 2,
                'abundance':         abund_bins[ab_idx] + abundance_window / 2,
                'biomass_percentile': p_val
            })

        if not records:
            return pd.DataFrame(columns=['age', 'abundance', 'biomass_percentile'])

        result_df = pd.DataFrame(records)

        # Monotonic abundance constraint
        kept_rows = []
        for age_b in sorted(result_df['age'].unique()):
            age_subset = result_df[result_df['age'] == age_b].sort_values('abundance')
            max_so_far = -np.inf
            for _, row in age_subset.iterrows():
                if row['biomass_percentile'] >= max_so_far:
                    max_so_far = row['biomass_percentile']
                    kept_rows.append(row)

        result_df = pd.DataFrame(kept_rows).reset_index(drop=True)
        print(f"Applied monotonic abundance constraint - kept {len(result_df)} entries")
        return result_df

    # ------------------------------------------------------------------
    # 4. Main loop
    # ------------------------------------------------------------------
    print("Creating initial dataframes for each mask...")
    result_dict = {}

    for mask_name, geom in masks.items():
        print(f"Dealing with mask : {mask_name}\n")
        age_vals, abundance_vals, biomass_vals = extract_valid_pixels(geom)
        result_dict[mask_name] = build_windowed_df(age_vals, abundance_vals, biomass_vals)

    print("\n===============")

    # ------------------------------------------------------------------
    # 5. Optional plot
    # ------------------------------------------------------------------
    if printPlot:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()
        for idx, (mask_name, df) in enumerate(result_dict.items()):
            ax = axes[idx]
            if df.empty:
                ax.set_title(f"{mask_name} (no data)")
                continue
            sc = ax.scatter(
                df['age'], df['biomass_percentile'],
                c=df['abundance'], cmap='RdYlGn',
                vmin=0, vmax=100, alpha=0.7,
                edgecolors='k', linewidths=0.3
            )
            plt.colorbar(sc, ax=ax, label='% Abundance')
            ax.set_xlabel('Forest Age (years)')
            ax.set_ylabel(f'Biomass (Percentile {percentile}) in g/m2')
            ax.set_title(f'{speciesName} — {mask_name}')
        plt.tight_layout()
        plt.show()

    return result_dict

import numpy as np
import pandas as pd
import json
import os
from pygam import LinearGAM, s, te
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import warnings
warnings.filterwarnings('ignore')


def getGamDiagnostics(
    dictOfPointsSubsets,
    speciesName,
    exportCsvOfSelectedCurve_Path,
    printPlot=False,
    dictionnaryOfSelectedCurves=None
):
    """
    Fits a GAM on each mask's dataframe from createPercentilesBiomassDatasetsFromNFIData,
    computes diagnostics for the 100% abundance prediction curve, selects the best
    mask, exports the predictions CSV and updates the JSON dictionary.

    Parameters
    ----------
    dictOfPointsSubsets             : output dict from createPercentilesBiomassDatasetsFromNFIData
    speciesName                     : species name string
    exportCsvOfSelectedCurve_Path   : path to export the predictions CSV
    printPlot                       : if True, plots GAM curves for all masks
    dictionnaryOfSelectedCurves     : path to the JSON file tracking selected masks per species
    """

    mask_order = ['study_landscape', 'ecozone', 'surrounding_ecozones', 'canada']
    abundance_levels = [100, 90, 80, 70, 60, 50, 40, 30, 20, 10]

    # ------------------------------------------------------------------
    # 1. Fit GAM and compute diagnostics for each mask
    # ------------------------------------------------------------------
    fitted_gams   = {}
    age_arrays    = {}
    diagnostics   = {}

    print("=" * 80)
    print("GAM 100% Abundance Curve Diagnostics")
    print("=" * 80)

    for mask_name in mask_order:
        df = dictOfPointsSubsets.get(mask_name)
        if df is None or df.empty:
            continue

        X = df[['age', 'abundance']].values
        y = df['biomass_percentile'].values

        # GAM: s(age, by=abundance) + s(abundance, by=age) + s(age) + s(abundance)
        # All splines constrained to be concave (constraint='concave')
        gam = LinearGAM(
            s(0, by=1, constraints='concave') +
            s(1, by=0, constraints='concave') +
            s(0, constraints='concave') +
            s(1, constraints='concave')
        ).fit(X, y)

        fitted_gams[mask_name] = gam

        # Age range for prediction (same resolution as example CSV)
        age_min  = df['age'].min()
        age_max  = df['age'].max()
        age_pred = np.linspace(age_min, age_max, 180)
        age_arrays[mask_name] = age_pred

        # 100% abundance prediction curve
        X_pred_100 = np.column_stack([age_pred, np.full_like(age_pred, 100.0)])
        pred_100   = gam.predict(X_pred_100)

        # --- Pseudo R² ---
        pseudo_r2 = gam.statistics_['pseudo_r2']['explained_deviance']

        # --- Peak age and biomass ---
        peak_idx     = np.argmax(pred_100)
        peak_age     = age_pred[peak_idx]
        peak_biomass = pred_100[peak_idx]

        # --- % negative biomass ---
        pct_negative = 100.0 * np.sum(pred_100 < 0) / len(pred_100)

        # --- Concavity: % decrease after peak ---
        post_peak    = pred_100[peak_idx:]
        if len(post_peak) > 1:
            pct_decrease = 100.0 * (post_peak[0] - post_peak[-1]) / post_peak[0] if post_peak[0] != 0 else 0.0
        else:
            pct_decrease = 0.0

        # --- Is truly concave: second derivative < 0 everywhere after age 0 ---
        is_concave = bool(np.all(np.diff(pred_100[:peak_idx + 1]) >= 0) and np.all(np.diff(pred_100[peak_idx:]) <= 0))

        # --- Max observed biomass in mask ---
        max_obs_biomass = df['biomass_percentile'].max()

        # --- % points above 40% abundance ---
        pct_above_40 = 100.0 * np.sum(df['abundance'] > 40) / len(df)

        # --- Validity ---
        is_valid = (pseudo_r2 > 0.85) and is_concave and (pct_negative < 20.0)

        diagnostics[mask_name] = {
            'Pseudo_R2':               round(pseudo_r2,       6),
            'Peak_Age':                round(peak_age,        6),
            'Peak_Biomass':            round(peak_biomass,    6),
            'Pct_Negative_Biomass':    round(pct_negative,    1),
            'Concavity_Pct_Decrease':  round(pct_decrease,    6),
            'Is_Truly_Concave':        is_concave,
            'Max_Observed_Biomass':    round(max_obs_biomass, 6),
            'Pct_Above_40_Abundance':  round(pct_above_40,    6),
            'Is_Valid':                is_valid,
        }

    diag_df = pd.DataFrame(diagnostics).T
    diag_df.index.name = 'Mask'

    # Reorder rows to match mask_order
    diag_df = diag_df.reindex([m for m in mask_order if m in diag_df.index])

    # Split display into column groups of 4 as in the example output
    all_cols   = list(diag_df.columns)
    col_groups = [all_cols[i:i+4] for i in range(0, len(all_cols), 4)]

    for gi, group in enumerate(col_groups):
        print(f"\nColumns {gi*4+1}-{min((gi+1)*4, len(all_cols))}:")
        print(diag_df[group].to_string())
        print("-" * 80)

    print("=" * 80)

    # ------------------------------------------------------------------
    # 6. Optional plot
    # ------------------------------------------------------------------
    if printPlot:
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()
        cmap = cm.get_cmap('RdYlGn')

        for idx, mask_name in enumerate(mask_order):
            if mask_name not in fitted_gams:
                continue
            ax       = axes[idx]
            gam_m    = fitted_gams[mask_name]
            age_p    = age_arrays[mask_name]
            df       = dictOfPointsSubsets[mask_name]

            # Scatter of data points coloured by abundance
            sc = ax.scatter(
                df['age'], df['biomass_percentile'],
                c=df['abundance'], cmap='RdYlGn',
                vmin=0, vmax=100, alpha=0.5,
                edgecolors='k', linewidths=0.2, s=20, label='Data'
            )
            plt.colorbar(sc, ax=ax, label='% Abundance')

            # GAM curves for each abundance level
            for abund in abundance_levels:
                X_p   = np.column_stack([age_p, np.full_like(age_p, float(abund))])
                p     = gam_m.predict(X_p)
                color = cmap(abund / 100.0)
                lw    = 2.5 if abund == 100 else 1.0
                ax.plot(age_p, p, color=color, linewidth=lw,
                        label=f'{abund}%' if abund == 100 else None)

            ax.axhline(0, color='grey', linestyle='--', linewidth=0.8)
            ax.set_xlabel('Forest Age (years)')
            ax.set_ylabel(f'Species Biomass (Percentile) in g/m2')
            valid_str = '✓ Valid' if diagnostics[mask_name]['Is_Valid'] else '✗ Invalid'
            ax.set_title(f'{speciesName} — {mask_name}\n'
                         f'R²={diagnostics[mask_name]["Pseudo_R2"]:.3f}  {valid_str}')

        plt.suptitle(f'GAM Diagnostics — {speciesName}', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()
    
    # ------------------------------------------------------------------
    # 2. Select best mask
    # ------------------------------------------------------------------
    valid_masks = [m for m in mask_order if m in diagnostics and diagnostics[m]['Is_Valid']]

    selected_mask = None
    selection_reason = None

    if valid_masks:
        # Among valid masks, prefer those whose peak age and peak biomass
        # are within 20% of the average across valid masks
        valid_peak_ages    = [diagnostics[m]['Peak_Age']    for m in valid_masks]
        valid_peak_biomass = [diagnostics[m]['Peak_Biomass'] for m in valid_masks]
        avg_peak_age       = np.mean(valid_peak_ages)
        avg_peak_biomass   = np.mean(valid_peak_biomass)

        for mask_name in mask_order:  # smallest first
            if mask_name not in valid_masks:
                continue
            d = diagnostics[mask_name]
            age_ok     = abs(d['Peak_Age']    - avg_peak_age)    <= 0.20 * avg_peak_age
            biomass_ok = abs(d['Peak_Biomass'] - avg_peak_biomass) <= 0.20 * avg_peak_biomass
            if age_ok and biomass_ok:
                selected_mask    = mask_name
                selection_reason = (
                    f"  - Curve is valid (R² > 0.85, truly concave, <20% negative values)\n"
                    f"  - Peak age within 20% of average ({avg_peak_age:.2f} years)\n"
                    f"  - Peak biomass within 20% of average ({avg_peak_biomass:.2f})"
                )
                break

        # Fallback: just pick the smallest valid mask
        if selected_mask is None:
            selected_mask    = valid_masks[0]
            selection_reason = "  - Smallest valid mask (peak age/biomass outside 20% window)"

    else:
        # No valid curve — let user choose
        print("\nNo valid curve found. Diagnostics summary:")
        print(diag_df.to_string())
        print("\nAvailable masks:", mask_order)
        user_choice = input("Please select a mask manually: ").strip()
        selected_mask    = user_choice if user_choice in diagnostics else mask_order[0]
        selection_reason = "  - Manually selected by user (no valid curve found)"

    # ------------------------------------------------------------------
    # 3. Print selection summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("BEST CURVE SELECTION")
    print("=" * 80)
    print(f"\nSelected Mask: {selected_mask}")
    print(f"\nReason for Selection:\n{selection_reason}")
    print(f"\nDiagnostics for Selected Curve:")
    print("-" * 80)
    for group in col_groups:
        print(diag_df.loc[[selected_mask], group].to_string())
        print()
    print("=" * 80)

    # ------------------------------------------------------------------
    # 4. Export predictions CSV for selected mask
    # ------------------------------------------------------------------
    gam_selected  = fitted_gams[selected_mask]
    age_pred_sel  = age_arrays[selected_mask]

    csv_data = {'Age': age_pred_sel}
    for abund in abundance_levels:
        X_pred = np.column_stack([age_pred_sel, np.full_like(age_pred_sel, float(abund))])
        preds  = gam_selected.predict(X_pred)
        csv_data[f'GAM_Prediction_{abund}%Abundance'] = preds

    pred_df = pd.DataFrame(csv_data)
    os.makedirs(os.path.dirname(exportCsvOfSelectedCurve_Path), exist_ok=True)
    pred_df.to_csv(exportCsvOfSelectedCurve_Path, index=False)

    # ------------------------------------------------------------------
    # 5. Update JSON dictionary of selected curves
    # ------------------------------------------------------------------
    if dictionnaryOfSelectedCurves is not None:
        if os.path.exists(dictionnaryOfSelectedCurves):
            with open(dictionnaryOfSelectedCurves, 'r') as f:
                existing = json.load(f)
        else:
            existing = {}
        existing[speciesName] = selected_mask
        with open(dictionnaryOfSelectedCurves, 'w') as f:
            json.dump(existing, f, indent=None)


    return {"Peak_age" : diag_df.at[selected_mask, 'Peak_Age'],
           "Peak_Biomass" : diag_df.at[selected_mask, 'Peak_Biomass']}


### FUNCTIONS FOR FIRST CALIBRATION STEP (MONOCULTURE IN IDEAL CONDITIONS)

import json
import copy
import shutil

def calibrate_phase1_1(
    species,
    functional_type,
    target_peak_biomass,
    LAI_min, LAI_target, LAI_max,
    dictOfInitialCoreSpeciesParameters,
    dictOfInitialPnETSpeciesParameters,
    dictOfInitialPnETGenericParameters,
    DictOfBounds
):

    ## HELPERS FUNCTIONS AND FIXED VALUES
    # ─── Literature reference values ──────────────────────────────────────────────
    AMAX_LITERATURE = {
        "evergreen": {"AmaxA": 5.3,  "AmaxB": 21.5},
        "deciduous": {"AmaxA": -46.0, "AmaxB": 71.9}
    }
    
    TOLERANCE = 0.01
    BOUND_FEASIBILITY_MARGIN = 0.20
    SIMULATION_DURATION = 300
    CLIMATE = "mild"
    SOIL = "SILO"
    
    
    def run_sim(species, core_params, pnet_params, generic_params, plotResultsSim = False):
        """Thin wrapper around the calibration simulation function."""
        return calibrationSimulationMonoculturemanawan(
            duration=SIMULATION_DURATION,
            climate=CLIMATE,
            soil=SOIL,
            speciesToSimulate=species,
            dictOfInitialCoreSpeciesParameters=core_params,
            dictOfInitialPnETSpeciesParameters=pnet_params,
            dictOfInitialPnETGenericParameters=generic_params,
            plotResults = plotResultsSim
        )
    
    
    def get_pnet_species_param(pnet_params, species, param_name):
        """
        Read a PnET species parameter as a float.
        Structure: pnet_params["PnETSpeciesParameters"][species][param_name]
        """
        return float(pnet_params["PnETSpeciesParameters"][species][param_name])
    
    
    def set_towood_toroot(pnet_params, species, towood_val, toroot_val):
        """
        Return a deep copy of pnet_params with TOWood and TORoot updated.
        Structure: pnet_params["PnETSpeciesParameters"][species][param_name]
        """
        params = copy.deepcopy(pnet_params)
        params["PnETSpeciesParameters"][species]["TOWood"] = towood_val
        params["PnETSpeciesParameters"][species]["TORoot"] = toroot_val
        return params
    
    def get_bounds(bounds, species, param_name):
        """Read lower/upper bounds as floats."""
        return (
            float(bounds[species][param_name]["lower"]),
            float(bounds[species][param_name]["upper"])
        )
    
    def diagnostic_checks(
        species,
        outputs,
        pnet_params,
        core_params,
        generic_params,
        bounds,
        functional_type,
        LAI_min, LAI_target, LAI_max,
        growing_too_much
    ):
        """
        Run diagnostic checks when TOWood/TORoot cannot reach the target.
        Prints actionable messages for the user.
        """
        issues_found = False
        direction_str = "too much" if growing_too_much else "not enough"
        print(f"\n[DIAGNOSTIC] Species is growing {direction_str} even at parameter limits.")
        print("=" * 65)
    
        # ── 1. LAI check ──────────────────────────────────────────────────────────
        max_lai = outputs["Maximum LAI"]
        if not (LAI_min <= max_lai <= LAI_max):
            issues_found = True
            print(
                f"[LAI] Maximum LAI = {max_lai:.3f} is outside the target range "
                f"[{LAI_min}, {LAI_max}] (target: {LAI_target}).\n"
                "  → Recommendation: Adjust FracFol, FrActWd, and/or SLWMax to obtain "
                "a more realistic LAI before continuing calibration."
            )
    
        # ── 2. FolN and FracBelowG within bounds ──────────────────────────────────
        for param_name in ["FolN", "FracBelowG"]:
            if param_name == "FolN":
                val = get_pnet_species_param(pnet_params, species, param_name)
            else:
                val = generic_params["FracBelowG"]
            lo  = bounds[species][param_name]["lower"]
            hi  = bounds[species][param_name]["upper"]
            if not (float(lo) <= float(val) <= float(hi)):
                issues_found = True
                print(
                    f"[{param_name}] Value = {val:.4f} is outside its empirical bounds "
                    f"[{lo}, {hi}].\n"
                    f"  → Recommendation: Reset {param_name} to a value within its bounds."
                )
    
        # ── 3. AmaxA and AmaxB within 10% of literature values ────────────────────
        lit = AMAX_LITERATURE[functional_type]
        for param_name, lit_val in lit.items():
            val       = get_pnet_species_param(pnet_params, species, param_name)
            threshold = abs(lit_val) * 0.10
            if abs(float(val) - float(lit_val)) > threshold:
                issues_found = True
                print(
                    f"[{param_name}] Value = {val:.4f} deviates more than 10% from the "
                    f"literature value for {functional_type} species ({lit_val}).\n"
                    f"  → Recommendation: Verify {param_name} and bring it closer to "
                    f"the literature value ({lit_val})."
                )
    
        # ── 4. Water stress check ──────────────────────────────────────────────────
        avg_fwater = outputs["Average Fwater"]
        if avg_fwater < 0.95:
            issues_found = True
            print(
                f"[Fwater] Average Fwater = {avg_fwater:.3f} is below 0.95, "
                "indicating significant water stress.\n"
                "  → Recommendation: Check precipitation inputs for this ecoregion. "
                "The biomass target may be unrealistic under current water availability."
            )
    
        # ── 5. PsnTOpt vs. Average July Temperature ────────────────────────────────
        psn_topt   = get_pnet_species_param(pnet_params, species, "PsnTOpt")
        avg_july_t = outputs["Average July Temperature"]
        if abs(float(psn_topt) - float(avg_july_t)) > 5.0:
            issues_found = True
            print(
                f"[PsnTOpt] PsnTOpt = {psn_topt:.1f} °C differs by more than 5 °C from "
                f"the simulated average July temperature ({avg_july_t:.1f} °C).\n"
                "  → Recommendation: The species may not be climatically adapted to this "
                "area. Consider whether the biomass target is realistic for this location."
                "  → If you're certain of the temperature parameters, you might lower your"
                "initial value of FolN for the species, as it might be too high, making the"
                "species too productive."
            )
    
        # ── 6. Catch-all ──────────────────────────────────────────────────────────
        if not issues_found:
            print(
                "[UNKNOWN] All diagnostic checks passed, yet the target cannot be reached.\n"
                "  → Recommendations:\n"
                "     • Verify that the target peak biomass is in the correct units (g/m²).\n"
                "     • Check that PsnTMin and PsnTMax are not set to unrealistic values.\n"
                "     • Consider whether the NFI-derived biomass target is appropriate "
                "for a monoculture simulation.\n"
                "     • Review other species parameters that influence carbon allocation.\n"
                "     • If this problem happens with most of your species, consider changing MaintResp."
                "     • If MaintResp have been changed already, then consider changing the photosynthesis\n"
                "       temperature parameters (PsnTMin/PsnTOpt) if your species performs too well/not well enough for the current climate."
                "     • If photosynthesis temperature parameters are good, consider reducing FolN for the species."
            )
    
        print("=" * 65)
    
    core_p    = copy.deepcopy(dictOfInitialCoreSpeciesParameters)
    pnet_p    = copy.deepcopy(dictOfInitialPnETSpeciesParameters)
    generic_p = copy.deepcopy(dictOfInitialPnETGenericParameters)
    bounds    = DictOfBounds
    # EDIT : Put longevity veeeeery far away so that fAge
    # doesn't influence the peak
    core_p[species]["Longevity"]="999"

    target_peak_biomass = float(target_peak_biomass)
    LAI_min             = float(LAI_min)
    LAI_target          = float(LAI_target)
    LAI_max             = float(LAI_max)

    towood_lo, towood_hi = get_bounds(bounds, species, "TOWood")
    toroot_lo, toroot_hi = get_bounds(bounds, species, "TORoot")
    # EDIT : If species is the red maple, we let ourselves use a much larger TOWood
    # The red maple tends to grow way too much, and is already constricted by
    # temperatures a lot; Gustafson suggets that it might do some self-thinning.
    if species == "ACER.RUB":
        print("WARNING : Red maple (ACER.RUB) detected.\n"
              "Will use a higher TOWood upper limit of 0.07\n"
              "to account for its higher growth, and the fact\n"
              "that it might be doing more self-thinning\n"
              "than other species.")
        towood_hi = 0.07

    # ── Baseline simulation ────────────────────────────────────────────────────
    print(f"\n[PHASE 1.1] Species: {species}")
    print(f"  Target peak biomass : {target_peak_biomass:.1f} g/m²")
    print(f"  LAI target range    : [{LAI_min}, {LAI_target}, {LAI_max}]")

    towood_cur = get_pnet_species_param(pnet_p, species, "TOWood")
    toroot_cur = get_pnet_species_param(pnet_p, species, "TORoot")

    print(f"\n[STEP 1] Baseline simulation  (TOWood={towood_cur:.6f}, TORoot={toroot_cur:.6f})")
    baseline_outputs = run_sim(species, core_p, pnet_p, generic_p)
    baseline_peak    = float(baseline_outputs["Biomass peak height"])
    print(f"  Baseline peak biomass : {baseline_peak:.1f} g/m²")
    print(f"  Target peak biomass   : {target_peak_biomass:.1f} g/m²")
    print(f"  Difference            : {baseline_peak - target_peak_biomass:.1f} g/m²")

    if abs(baseline_peak - target_peak_biomass) / target_peak_biomass < TOLERANCE:
        print("[RESULT] Baseline already within tolerance. No adjustment needed.")
        return {"TOWood": towood_cur, "TORoot": toroot_cur, "outputs": baseline_outputs}

    # ── Feasibility check at bound limits ─────────────────────────────────────
    # peak too LOW  → need to DECREASE TOWood/TORoot → check lower bounds
    # peak too HIGH → need to INCREASE TOWood/TORoot → check upper bounds
    if baseline_peak < target_peak_biomass:
        towood_limit = towood_lo
        toroot_limit = toroot_lo
        print(f"  Direction: baseline BELOW target → will DECREASE TOWood/TORoot")
    else:
        towood_limit = towood_hi
        toroot_limit = toroot_hi
        print(f"  Direction: baseline ABOVE target → will INCREASE TOWood/TORoot")

    print(f"\n[STEP 2] Feasibility check at limits "
          f"(TOWood={towood_limit:.6f}, TORoot={toroot_limit:.6f})")
    limit_pnet_p  = set_towood_toroot(pnet_p, species, towood_limit, toroot_limit)
    limit_outputs = run_sim(species, core_p, limit_pnet_p, generic_p)
    limit_peak    = float(limit_outputs["Biomass peak height"])
    print(f"  Peak at limits: {limit_peak:.1f} g/m²")

    if baseline_peak < target_peak_biomass:
        can_reach = limit_peak >= target_peak_biomass
    else:
        can_reach = limit_peak <= target_peak_biomass

    if not can_reach:
        growing_too_much = baseline_peak > target_peak_biomass
        diagnostic_checks(
            species, limit_outputs, limit_pnet_p, core_p, generic_p,
            bounds, functional_type,
            LAI_min, LAI_target, LAI_max,
            growing_too_much=growing_too_much
        )
        run_sim(species, core_p, limit_pnet_p, generic_p, True)
        return None

    # ── Incremental search ────────────────────────────────────────────────────
    # At every iteration:
    #   1. Look at current peak vs target → decide direction (no memory needed)
    #   2. If direction changed since last iteration → halve step
    #   3. Apply step in the decided direction
    # This is a pure bisection driven by the current state, not by history.
    print(f"\n[STEP 3] Incremental search toward target...")

    step      = max(towood_cur, toroot_cur) * 0.10
    min_step  = step * 1e-4
    cur_peak  = baseline_peak
    last_dir  = None   # tracks direction of previous iteration to detect changes

    best_params  = (towood_cur, toroot_cur)
    best_outputs = baseline_outputs

    iteration = 0
    max_iter  = 500

    while step >= min_step and iteration < max_iter:
        iteration += 1

        # Decide direction purely from current state
        # Above target → increase params (dir = +1)
        # Below target → decrease params (dir = -1)
        if cur_peak > target_peak_biomass:
            direction = +1.0
        else:
            direction = -1.0

        # Halve step whenever direction flips (we crossed the target)
        if last_dir is not None and direction != last_dir:
            step *= 0.5
            print(f"         → Direction changed. Halving step to {step:.6f}")

        last_dir = direction

        towood_new = max(towood_lo, min(towood_hi, towood_cur + direction * step))
        toroot_new = max(toroot_lo, min(toroot_hi, toroot_cur + direction * step))

        trial_pnet_p  = set_towood_toroot(pnet_p, species, towood_new, toroot_new)
        trial_outputs = run_sim(species, core_p, trial_pnet_p, generic_p)
        trial_peak    = float(trial_outputs["Biomass peak height"])

        rel_error = (trial_peak - target_peak_biomass) / target_peak_biomass

        print(f"  Iter {iteration:3d} | dir={direction:+.0f} | "
              f"TOWood={towood_new:.6f}  TORoot={toroot_new:.6f} | "
              f"Peak={trial_peak:.1f}  Error={rel_error*100:+.2f}%  Step={step:.6f}")

        if abs(rel_error) < TOLERANCE:
            print(f"\n[RESULT] Converged at iteration {iteration}.")
            print(f"  Final TOWood = {towood_new:.6f}")
            print(f"  Final TORoot = {toroot_new:.6f}")
            print(f"  Final peak biomass = {trial_peak:.1f} g/m²  "
                  f"(target: {target_peak_biomass:.1f} g/m²)")
            run_sim(species, core_p, trial_pnet_p, generic_p, True)
            return {"TOWood": towood_new, "TORoot": toroot_new, "outputs": trial_outputs}

        # Always accept the step and update current position
        towood_cur   = towood_new
        toroot_cur   = toroot_new
        cur_peak     = trial_peak
        best_params  = (towood_new, toroot_new)
        best_outputs = trial_outputs

    # ── Did not fully converge — return best found ────────────────────────────
    tw, tr    = best_params
    best_peak = float(best_outputs["Biomass peak height"])
    print(f"\n[WARNING] Did not fully converge within tolerance after {iteration} iterations.")
    print(f"  Best TOWood = {tw:.6f},  Best TORoot = {tr:.6f}")
    print(f"  Best peak biomass = {best_peak:.1f} g/m²  (target: {target_peak_biomass:.1f} g/m²)")
    run_sim(species, core_p, trial_pnet_p, generic_p, True)
    return {"TOWood": tw, "TORoot": tr, "outputs": best_outputs}


def calibrate_LAI_subphase_1_1_2(
    species,
    target_LAI_min,
    target_LAI_max,
    DictOfBounds,
    target_peak_biomass,
    initial_core_params_path='./SpeciesParametersSets/Initial/initialCoreSpeciesParameters.json',
    initial_pnet_species_params_path='./SpeciesParametersSets/Initial/initialPnETSpeciesParameters.json',
    initial_generic_params_path='./SpeciesParametersSets/Initial/InitialGenericParameters.json',
    duration=300,
    climate="mild",
    soil="SILO",
    n_bisection_iterations=20,
    fracfol_bypass_factor=0.20,
    verbose=True
):
    """
    Calibrates Maximum LAI and LAI stability for a given species by tuning
    FracFol (coarse), SLWmax (fine), and FrActWd (stability).
    Then recalibrates the biomass peak height via TOWood/TORoot (and FolN
    as a fallback if bounds are hit), iterating the LAI + biomass
    calibration up to 3 times if they pull each other out of bounds.

    Returns:
        dict: Updated parameter dictionaries and final simulation outputs.
    """

    # --- Load initial parameter dicts ---
    core_params = json.load(open(initial_core_params_path))
    pnet_species_params = json.load(open(initial_pnet_species_params_path))
    generic_params = json.load(open(initial_generic_params_path))
    # EDIT : Put longevity veeeeery far away so that fAge
    # doesn't influence the peak ?
    core_params[species]["Longevity"]="999"

    def run_sim(core_p, pnet_sp, gen_p, plotting = False):
        return calibrationSimulationMonoculturemanawan(
            duration=duration,
            climate=climate,
            soil=soil,
            speciesToSimulate=species,
            dictOfInitialCoreSpeciesParameters=copy.deepcopy(core_p),
            dictOfInitialPnETSpeciesParameters=copy.deepcopy(pnet_sp),
            dictOfInitialPnETGenericParameters=copy.deepcopy(gen_p),
            plotResults=plotting
        )

    def set_pnet_param(pnet_sp, param, value):
        pnet_sp_copy = copy.deepcopy(pnet_sp)
        pnet_sp_copy["PnETSpeciesParameters"][species][param] = value
        return pnet_sp_copy

    def set_core_param(core_p, param, value):
        core_p_copy = copy.deepcopy(core_p)
        core_p_copy[species][param] = value
        return core_p_copy

    def lai_in_target(max_lai):
        return round(target_LAI_min, 1) <= round(float(max_lai), 1) <= round(target_LAI_max, 1)

    def stability_in_target(lai_stability):
        stab_min = round(float(lai_stability[0]), 1)
        stab_max = round(float(lai_stability[1]), 1)
        return (stab_min >= round(target_LAI_min, 1)) and (stab_max <= round(target_LAI_max, 1))

    def set_towood_toroot(pnet_sp, towood_val, toroot_val):
        """Return a deep copy of pnet_sp with TOWood and TORoot updated."""
        p = copy.deepcopy(pnet_sp)
        p["PnETSpeciesParameters"][species]["TOWood"] = towood_val
        p["PnETSpeciesParameters"][species]["TORoot"] = toroot_val
        return p

    # --- TOWood/TORoot bounds ---
    towood_lo = float(DictOfBounds[species]["TOWood"]["lower"])
    towood_hi = float(DictOfBounds[species]["TOWood"]["upper"])
    toroot_lo = float(DictOfBounds[species]["TORoot"]["lower"])
    toroot_hi = float(DictOfBounds[species]["TORoot"]["upper"])
    if species == "ACER.RUB":
        if verbose:
            print("WARNING : Red maple (ACER.RUB) detected.\n"
                  "Will use a higher TOWood upper limit of 0.07\n"
                  "to account for its higher growth, and the fact\n"
                  "that it might be doing more self-thinning\n"
                  "than other species.")
        towood_hi = 0.07

    # --- FolN bounds ---
    foln_lo = float(DictOfBounds[species]["FolN"]["lower"])
    foln_hi = float(DictOfBounds[species]["FolN"]["upper"])

    target_peak_biomass = float(target_peak_biomass)
    PEAK_TOLERANCE = 0.01

    # =========================================================================
    # ITERATIVE LOOP: LAI calibration → biomass peak recalibration
    # Up to 3 iterations; stop early if both LAI and peak are in range.
    # =========================================================================
    MAX_OUTER_ITER = 3

    for outer_iter in range(1, MAX_OUTER_ITER + 1):
        if verbose:
            print(f"\n{'='*70}")
            print(f"  OUTER ITERATION {outer_iter}/{MAX_OUTER_ITER}")
            print(f"{'='*70}")

        # =====================================================================
        # STAGE 1: Coarse adjustment of Maximum LAI via FracFol (bisection)
        # If target is unreachable within empirical bounds, allow a 20% bypass.
        # If still unreachable after bypass, use the extended bound directly.
        # =====================================================================
        if verbose:
            print("=== Stage 1: Coarse LAI adjustment via FracFol ===")

        lo = float(DictOfBounds[species]["FracFol"]["lower"])
        hi = float(DictOfBounds[species]["FracFol"]["upper"])

        res_lo = run_sim(core_params, set_pnet_param(pnet_species_params, "FracFol", lo), generic_params)
        res_hi = run_sim(core_params, set_pnet_param(pnet_species_params, "FracFol", hi), generic_params)

        if verbose:
            print(f"  FracFol={lo:.4f} -> Max LAI={float(res_lo['Maximum LAI']):.3f}")
            print(f"  FracFol={hi:.4f} -> Max LAI={float(res_hi['Maximum LAI']):.3f}")

        best_fracfol = float(pnet_species_params["PnETSpeciesParameters"][species]["FracFol"])
        skip_bisection = False

        # --- Check lower bound: LAI too high even at minimum FracFol ---
        if float(res_lo["Maximum LAI"]) > target_LAI_max:
            lo_extended = lo * (1.0 - fracfol_bypass_factor)
            if verbose:
                print(f"  LAI too high at lower bound. Testing extended lower bound: "
                      f"FracFol={lo_extended:.5f}.")
            res_lo_ext = run_sim(core_params, set_pnet_param(pnet_species_params, "FracFol", lo_extended), generic_params)
            if verbose:
                print(f"  FracFol={lo_extended:.5f} -> Max LAI={float(res_lo_ext['Maximum LAI']):.3f}")
            if not lai_in_target(float(res_lo_ext["Maximum LAI"])):
                if verbose:
                    print(f"  Target still unreachable after bypass. Using extended lower bound directly.")
                best_fracfol = lo_extended
                skip_bisection = True
            else:
                lo = lo_extended
                if verbose:
                    print(f"  Target reachable within extended bounds. Proceeding with bisection.")

        # --- Check upper bound: LAI too low even at maximum FracFol ---
        elif float(res_hi["Maximum LAI"]) < target_LAI_min:
            hi_extended = hi * (1.0 + fracfol_bypass_factor)
            if verbose:
                print(f"  LAI too low at upper bound. Testing extended upper bound: "
                      f"FracFol={hi_extended:.5f}.")
            res_hi_ext = run_sim(core_params, set_pnet_param(pnet_species_params, "FracFol", hi_extended), generic_params)
            if verbose:
                print(f"  FracFol={hi_extended:.5f} -> Max LAI={float(res_hi_ext['Maximum LAI']):.3f}")
            if not lai_in_target(float(res_hi_ext["Maximum LAI"])):
                if verbose:
                    print(f"  Target still unreachable after bypass. Using extended upper bound directly.")
                best_fracfol = hi_extended
                skip_bisection = True
            else:
                hi = hi_extended
                if verbose:
                    print(f"  Target reachable within extended bounds. Proceeding with bisection.")

        # --- Bisection toward midpoint of target LAI range ---
        if not skip_bisection:
            # Changing target : we'll target the maximum.
            target_mid = target_LAI_max
            # target_mid = (target_LAI_min + target_LAI_max) / 2.0
            lo_b, hi_b = lo, hi

            for i in range(n_bisection_iterations):
                mid = (lo_b + hi_b) / 2.0
                res_mid = run_sim(core_params, set_pnet_param(pnet_species_params, "FracFol", mid), generic_params)
                max_lai = float(res_mid["Maximum LAI"])

                if verbose:
                    print(f"  Iter {i+1}: FracFol={mid:.5f}, Max LAI={max_lai:.4f}")

                if lai_in_target(max_lai):
                    best_fracfol = mid
                    if max_lai < target_mid:
                        lo_b = mid
                    else:
                        hi_b = mid
                elif max_lai < target_LAI_min:
                    lo_b = mid
                else:
                    hi_b = mid

                if (hi_b - lo_b) < 1e-6:
                    break

            best_fracfol = (lo_b + hi_b) / 2.0

        pnet_species_params = set_pnet_param(pnet_species_params, "FracFol", best_fracfol)
        res_stage1 = run_sim(core_params, pnet_species_params, generic_params)

        if verbose:
            print(f"  Stage 1 result: FracFol={best_fracfol:.5f}, Max LAI={float(res_stage1['Maximum LAI']):.4f}")

        # =====================================================================
        # STAGE 2: Fine-tuning via SLWmax (bisection, only if needed)
        # Note: SLWmax is inversely related to LAI — increasing SLWmax decreases LAI.
        # =====================================================================
        if verbose:
            print("=== Stage 2: Fine LAI tuning via SLWmax ===")

        current_max_lai = float(res_stage1["Maximum LAI"])
        best_slwmax = float(pnet_species_params["PnETSpeciesParameters"][species]["SLWmax"])

        if not lai_in_target(current_max_lai):
            lo_slw = float(DictOfBounds[species]["SLWmax"]["lower"])
            hi_slw = float(DictOfBounds[species]["SLWmax"]["upper"])

            res_lo_slw = run_sim(core_params, set_pnet_param(pnet_species_params, "SLWmax", lo_slw), generic_params)
            res_hi_slw = run_sim(core_params, set_pnet_param(pnet_species_params, "SLWmax", hi_slw), generic_params)

            if verbose:
                print(f"  SLWmax={lo_slw:.4f} -> Max LAI={float(res_lo_slw['Maximum LAI']):.3f}")
                print(f"  SLWmax={hi_slw:.4f} -> Max LAI={float(res_hi_slw['Maximum LAI']):.3f}")

            if float(res_lo_slw["Maximum LAI"]) < target_LAI_min:
                if verbose:
                    print(f"  WARNING: Even at minimum SLWmax ({lo_slw:.4f}), LAI is below target min. Using lower bound.")
                best_slwmax = lo_slw
            elif float(res_hi_slw["Maximum LAI"]) > target_LAI_max:
                if verbose:
                    print(f"  WARNING: Even at maximum SLWmax ({hi_slw:.4f}), LAI exceeds target max. Using upper bound.")
                best_slwmax = hi_slw
            else:
                # Targetting maximum instead
                # target_mid_slw = (target_LAI_min + target_LAI_max) / 2.0
                target_mid_slw = target_LAI_max
                lo_s, hi_s = lo_slw, hi_slw

                for i in range(n_bisection_iterations):
                    mid_s = (lo_s + hi_s) / 2.0
                    res_mid_s = run_sim(core_params, set_pnet_param(pnet_species_params, "SLWmax", mid_s), generic_params)
                    max_lai_s = float(res_mid_s["Maximum LAI"])

                    if verbose:
                        print(f"  Iter {i+1}: SLWmax={mid_s:.5f}, Max LAI={max_lai_s:.4f}")

                    if lai_in_target(max_lai_s):
                        best_slwmax = mid_s
                        # Higher SLWmax -> lower LAI
                        if max_lai_s > target_mid_slw:
                            lo_s = mid_s  # increase SLWmax to bring LAI down
                        else:
                            hi_s = mid_s  # decrease SLWmax to bring LAI up
                    elif max_lai_s < target_LAI_min:
                        hi_s = mid_s  # decrease SLWmax to raise LAI
                    else:
                        lo_s = mid_s  # increase SLWmax to lower LAI

                    if (hi_s - lo_s) < 1e-6:
                        break

                best_slwmax = (lo_s + hi_s) / 2.0

            pnet_species_params = set_pnet_param(pnet_species_params, "SLWmax", best_slwmax)
            res_stage2 = run_sim(core_params, pnet_species_params, generic_params)
        else:
            res_stage2 = res_stage1
            if verbose:
                print("  LAI already in target range after Stage 1. Skipping SLWmax adjustment.")

        if verbose:
            print(f"  Stage 2 result: SLWmax={best_slwmax:.5f}, Max LAI={float(res_stage2['Maximum LAI']):.4f}")

        # =====================================================================
        # STAGE 3: LAI stability correction via FrActWd
        # Start from the highest possible FrActWd value and decrease until we find
        # the largest value that brings LAI stability within target bounds.
        # =====================================================================
        if verbose:
            print("=== Stage 3: LAI stability correction via FrActWd ===")

        lai_stab = res_stage2["LAI stability"]
        stab_min_val = float(lai_stab[0])
        stab_max_val = float(lai_stab[1])
        best_fracactwd = float(pnet_species_params["PnETSpeciesParameters"][species]["FrActWd"])

        if verbose:
            print(f"  Initial LAI stability: min={stab_min_val:.4f}, max={stab_max_val:.4f}")

        if not stability_in_target(lai_stab):
            lo_fw = float(DictOfBounds[species]["FrActWd"]["lower"])
            hi_fw = float(DictOfBounds[species]["FrActWd"]["upper"])

            lo_f, hi_f = lo_fw, hi_fw

            # We want the largest FrActWd that keeps stability in target.
            # Use bisection starting from the upper bound, tracking the best
            # (highest) valid value found.
            best_valid_fracactwd = None

            # First, test the upper bound directly
            res_hi_fw = run_sim(core_params, set_pnet_param(pnet_species_params, "FrActWd", hi_fw), generic_params)
            stab_hi = res_hi_fw["LAI stability"]

            if verbose:
                print(f"  FrActWd={hi_fw:.4f} -> LAI stability: "
                      f"min={float(stab_hi[0]):.4f}, max={float(stab_hi[1]):.4f}")

            if stability_in_target(stab_hi):
                # Already valid at upper bound — use it directly
                best_valid_fracactwd = hi_fw
                if verbose:
                    print(f"  Stability in target at upper bound. Using FrActWd={hi_fw:.5f} directly.")
            else:
                # Bisect to find the largest valid value
                for i in range(n_bisection_iterations):
                    mid_f = (lo_f + hi_f) / 2.0
                    res_mid_f = run_sim(
                        core_params,
                        set_pnet_param(pnet_species_params, "FrActWd", mid_f),
                        generic_params
                    )
                    stab = res_mid_f["LAI stability"]
                    s_min = float(stab[0])
                    s_max = float(stab[1])

                    if verbose:
                        print(f"  Iter {i+1}: FrActWd={mid_f:.5f}, "
                              f"LAI stability: min={s_min:.4f}, max={s_max:.4f}")

                    if stability_in_target(stab):
                        # Valid — record it and try to go higher
                        best_valid_fracactwd = mid_f
                        lo_f = mid_f
                    else:
                        # Not valid — go lower
                        hi_f = mid_f

                    if (hi_f - lo_f) < 1e-6:
                        break

            if best_valid_fracactwd is not None:
                best_fracactwd = best_valid_fracactwd
                if verbose:
                    print(f"  Best valid FrActWd found: {best_fracactwd:.5f}")
            else:
                best_fracactwd = lo_f
                if verbose:
                    print(f"  No fully in-target candidate found. Using lowest tested value: "
                          f"FrActWd={best_fracactwd:.5f}")

            pnet_species_params = set_pnet_param(pnet_species_params, "FrActWd", best_fracactwd)
            res_stage3 = run_sim(core_params, pnet_species_params, generic_params)
        else:
            res_stage3 = res_stage2
            if verbose:
                print("  LAI stability already within target range. Skipping FrActWd adjustment.")

        final_stab = res_stage3["LAI stability"]
        if verbose:
            print(f"  Stage 3 result: FrActWd={best_fracactwd:.5f}, "
                  f"Max LAI={float(res_stage3['Maximum LAI']):.4f}, "
                  f"LAI stability: min={float(final_stab[0]):.4f}, max={float(final_stab[1]):.4f}")

        # =====================================================================
        # STAGE 4: Recalibrate biomass peak via TOWood/TORoot
        # (inspired by calibrate_phase1_1 incremental search)
        # If TOWood/TORoot hit their bounds, fall back to FolN adjustment.
        # =====================================================================
        if verbose:
            print("=== Stage 4: Biomass peak recalibration via TOWood/TORoot ===")

        towood_cur = float(pnet_species_params["PnETSpeciesParameters"][species]["TOWood"])
        toroot_cur = float(pnet_species_params["PnETSpeciesParameters"][species]["TORoot"])

        check_outputs = run_sim(core_params, pnet_species_params, generic_params)
        cur_peak = float(check_outputs["Biomass peak height"])

        if verbose:
            print(f"  Current: TOWood={towood_cur:.6f}, TORoot={toroot_cur:.6f}, Peak={cur_peak:.1f} g/m² (target: {target_peak_biomass:.1f})")

        if abs(cur_peak - target_peak_biomass) / target_peak_biomass < PEAK_TOLERANCE:
            if verbose:
                print("  Peak already within tolerance. No TOWood/TORoot adjustment needed.")
            res_stage4 = check_outputs
        else:
            step     = max(towood_cur, toroot_cur) * 0.10
            min_step = step * 1e-4
            last_dir = None
            best_towood, best_toroot = towood_cur, toroot_cur
            best_peak = cur_peak
            best_outputs_tw = check_outputs
            iteration = 0
            hit_bounds = False

            while step >= min_step and iteration < 500:
                iteration += 1
                direction = +1.0 if cur_peak > target_peak_biomass else -1.0
                if last_dir is not None and direction != last_dir:
                    step *= 0.5
                    if verbose:
                        print(f"         → Direction changed. Halving step to {step:.6f}")
                last_dir = direction

                towood_new = max(towood_lo, min(towood_hi, towood_cur + direction * step))
                toroot_new = max(toroot_lo, min(toroot_hi, toroot_cur + direction * step))

                # --- Detect if we've hit the bounds and are stuck ---
                if (towood_new == towood_cur and toroot_new == toroot_cur):
                    if verbose:
                        print(f"  Iter {iteration}: TOWood/TORoot at bounds "
                              f"(TOWood={towood_cur:.6f}, TORoot={toroot_cur:.6f}) — "
                              f"cannot move further. Switching to FolN adjustment.")
                    hit_bounds = True
                    break

                trial_pnet = set_towood_toroot(pnet_species_params, towood_new, toroot_new)
                trial_out  = run_sim(core_params, trial_pnet, generic_params)
                trial_peak = float(trial_out["Biomass peak height"])
                rel_error  = (trial_peak - target_peak_biomass) / target_peak_biomass

                if verbose:
                    print(f"  Iter {iteration:3d} | dir={direction:+.0f} | "
                          f"TOWood={towood_new:.6f}  TORoot={toroot_new:.6f} | "
                          f"Peak={trial_peak:.1f}  Error={rel_error*100:+.2f}%  Step={step:.6f}")

                if abs(rel_error) < PEAK_TOLERANCE:
                    if verbose:
                        print(f"  Converged at iteration {iteration}. TOWood={towood_new:.6f}, TORoot={toroot_new:.6f}")
                    best_towood, best_toroot = towood_new, toroot_new
                    best_peak = trial_peak
                    best_outputs_tw = trial_out
                    break

                towood_cur, toroot_cur = towood_new, toroot_new
                cur_peak = trial_peak
                best_towood, best_toroot = towood_new, toroot_new
                best_peak = trial_peak
                best_outputs_tw = trial_out

            pnet_species_params = set_towood_toroot(pnet_species_params, best_towood, best_toroot)
            res_stage4 = run_sim(core_params, pnet_species_params, generic_params)

            if verbose:
                print(f"  Stage 4 result: TOWood={best_towood:.6f}, TORoot={best_toroot:.6f}, "
                      f"Peak={float(res_stage4['Biomass peak height']):.1f} g/m²")

            # =================================================================
            # STAGE 5: FolN fallback (only if TOWood/TORoot hit bounds)
            # Lowering FolN → lowers peak; raising FolN → raises peak.
            # Uses the same incremental-search pattern as TOWood/TORoot.
            # =================================================================
            if hit_bounds:
                if verbose:
                    print("=== Stage 5: FolN fallback for peak calibration ===")

                foln_cur = float(pnet_species_params["PnETSpeciesParameters"][species]["FolN"])
                cur_peak = float(res_stage4["Biomass peak height"])

                if verbose:
                    print(f"  Current: FolN={foln_cur:.4f}, Peak={cur_peak:.1f} g/m² (target: {target_peak_biomass:.1f})")

                if abs(cur_peak - target_peak_biomass) / target_peak_biomass < PEAK_TOLERANCE:
                    if verbose:
                        print("  Peak already within tolerance after TOWood/TORoot bound fix. No FolN adjustment needed.")
                else:
                    step     = max(foln_cur * 0.10, 1e-4)
                    min_step = step * 1e-4
                    last_dir = None
                    best_foln = foln_cur
                    best_foln_peak = cur_peak
                    iteration = 0

                    while step >= min_step and iteration < 500:
                        iteration += 1
                        # Lower FolN → lower peak; higher FolN → higher peak
                        direction = -1.0 if cur_peak > target_peak_biomass else +1.0
                        if last_dir is not None and direction != last_dir:
                            step *= 0.5
                            if verbose:
                                print(f"         → Direction changed. Halving step to {step:.6f}")
                        last_dir = direction

                        foln_new = max(foln_lo, min(foln_hi, foln_cur + direction * step))

                        if foln_new == foln_cur:
                            if verbose:
                                print(f"  FolN at bounds (FolN={foln_cur:.4f}) — cannot move further. Stopping FolN adjustment.")
                            break

                        trial_pnet = set_pnet_param(pnet_species_params, "FolN", foln_new)
                        trial_out  = run_sim(core_params, trial_pnet, generic_params)
                        trial_peak = float(trial_out["Biomass peak height"])
                        rel_error  = (trial_peak - target_peak_biomass) / target_peak_biomass

                        if verbose:
                            print(f"  Iter {iteration:3d} | dir={direction:+.0f} | "
                                  f"FolN={foln_new:.4f} | "
                                  f"Peak={trial_peak:.1f}  Error={rel_error*100:+.2f}%  Step={step:.6f}")

                        if abs(rel_error) < PEAK_TOLERANCE:
                            if verbose:
                                print(f"  FolN converged at iteration {iteration}. FolN={foln_new:.4f}")
                            best_foln = foln_new
                            best_foln_peak = trial_peak
                            break

                        foln_cur = foln_new
                        cur_peak = trial_peak
                        best_foln = foln_new
                        best_foln_peak = trial_peak

                    pnet_species_params = set_pnet_param(pnet_species_params, "FolN", best_foln)
                    res_stage4 = run_sim(core_params, pnet_species_params, generic_params)

                    if verbose:
                        print(f"  Stage 5 result: FolN={best_foln:.4f}, "
                              f"Peak={float(res_stage4['Biomass peak height']):.1f} g/m²")

        # =====================================================================
        # Check convergence: are both LAI and peak in range?
        # If so, break out of the outer loop early.
        # =====================================================================
        final_lai_ok = lai_in_target(res_stage4["Maximum LAI"]) and stability_in_target(res_stage4["LAI stability"])
        final_peak_ok = abs(float(res_stage4["Biomass peak height"]) - target_peak_biomass) / target_peak_biomass < PEAK_TOLERANCE

        if final_lai_ok and final_peak_ok:
            if verbose:
                print(f"\n  Converged after outer iteration {outer_iter}. Both LAI and peak are in range.")
            break
        elif not final_lai_ok:
            if verbose:
                print(f"\n  LAI out of bounds after biomass recalibration (Max LAI={float(res_stage4['Maximum LAI']):.4f}). "
                      f"Will re-calibrate LAI{'.' if outer_iter < MAX_OUTER_ITER else ' — but no iterations remain.'}")
        else:
            if verbose:
                print(f"\n  Peak still out of tolerance after iteration {outer_iter}.")

    else:
        # Loop exhausted without converging
        print(f"\n[WARNING] Failed to converge after {MAX_OUTER_ITER} outer iterations: "
              f"could not simultaneously satisfy LAI target [{target_LAI_min}, {target_LAI_max}] "
              f"and peak biomass target {target_peak_biomass:.1f} g/m². "
              f"Final Max LAI={float(res_stage4['Maximum LAI']):.4f}, "
              f"Final Peak={float(res_stage4['Biomass peak height']):.1f} g/m².")

    # =========================================================================
    # Final summary
    # =========================================================================
    final_stab = res_stage4["LAI stability"]
    final_params = {
        "FracFol": best_fracfol,
        "SLWmax": best_slwmax,
        "FrActWd": best_fracactwd,
        "TOWood": float(pnet_species_params["PnETSpeciesParameters"][species]["TOWood"]),
        "TORoot": float(pnet_species_params["PnETSpeciesParameters"][species]["TORoot"]),
        "FolN": float(pnet_species_params["PnETSpeciesParameters"][species]["FolN"])
    }

    if verbose:
        print("\n=== Calibration Summary ===")
        print(f"  Species            : {species}")
        print(f"  Target LAI range   : [{target_LAI_min}, {target_LAI_max}]")
        print(f"  Final Max LAI      : {float(res_stage4['Maximum LAI']):.4f}")
        print(f"  Final LAI stability: min={float(final_stab[0]):.4f}, max={float(final_stab[1]):.4f}")
        print(f"  Target peak biomass: {target_peak_biomass:.1f} g/m²")
        print(f"  Final peak biomass : {float(res_stage4['Biomass peak height']):.1f} g/m²")
        print(f"  FracFol            : {best_fracfol:.5f}")
        print(f"  SLWmax             : {best_slwmax:.5f}")
        print(f"  FrActWd            : {best_fracactwd:.5f}")
        print(f"  TOWood             : {final_params['TOWood']:.6f}")
        print(f"  TORoot             : {final_params['TORoot']:.6f}")
        print(f"  FolN               : {final_params['FolN']:.4f}")
        in_range = lai_in_target(res_stage4["Maximum LAI"]) and stability_in_target(res_stage4["LAI stability"])
        peak_ok  = abs(float(res_stage4["Biomass peak height"]) - target_peak_biomass) / target_peak_biomass < PEAK_TOLERANCE
        print(f"  LAI calibration successful : {in_range}")
        print(f"  Peak calibration successful: {peak_ok}")
        run_sim(core_params, pnet_species_params, generic_params, True)
        # print(pnet_species_params)

    return {
        "final_params": final_params,
        "final_simulation_outputs": res_stage4,
        "updated_core_params": core_params,
        "updated_pnet_species_params": pnet_species_params,
        "updated_generic_params": generic_params
    }




import json
import copy
import shutil

def calibrate_subphase_1_1_3(
    species: str,
    target_peak_biomass: float,
    DictOfBounds: dict,
    path_core: str = './SpeciesParametersSets/Initial/initialCoreSpeciesParameters.json',
    path_pnet: str = './SpeciesParametersSets/Initial/initialPnETSpeciesParameters.json',
    path_generic: str = './SpeciesParametersSets/Initial/InitialGenericParameters.json',
    duration: int = 300,
    climate: str = "mild",
    soil: str = "SILO",
    tolerance: float = 0.01,
    max_iter: int = 50
):
    # --- Load parameter dicts ---
    core_params    = json.load(open(path_core))
    pnet_params    = json.load(open(path_pnet))
    generic_params = json.load(open(path_generic))
    # EDIT : Put longevity veeeeery far away so that fAge
    # doesn't influence the peak
    core_params[species]["Longevity"]="999"

    # --- Retrieve FolN bounds ---
    foln_lower = float(DictOfBounds[species]["FolN"]["lower"])
    foln_upper = float(DictOfBounds[species]["FolN"]["upper"])

    target_peak_biomass = float(target_peak_biomass)

    print(f"[1.1.3] Species          : {species}")
    print(f"[1.1.3] Target biomass   : {target_peak_biomass:.2f} g/m²")
    print(f"[1.1.3] FolN bounds      : [{foln_lower}, {foln_upper}]")

    # --- Helper: run simulation for a given FolN value ---
    def run_sim(foln_value, printPlots = False):
        foln_value = float(foln_value)
        pnet_trial = copy.deepcopy(pnet_params)
        pnet_trial["PnETSpeciesParameters"][species]["FolN"] = foln_value
        result = calibrationSimulationMonoculturemanawan(
            duration=duration,
            climate=climate,
            soil=soil,
            speciesToSimulate=species,
            dictOfInitialCoreSpeciesParameters=copy.deepcopy(core_params),
            dictOfInitialPnETSpeciesParameters=pnet_trial,
            dictOfInitialPnETGenericParameters=copy.deepcopy(generic_params),
            plotResults = printPlots
        )
        return float(result["Biomass peak height"])

    # --- Baseline check ---
    current_foln = float(pnet_params["PnETSpeciesParameters"][species]["FolN"])
    baseline_peak = run_sim(current_foln)
    rel_error = abs(baseline_peak - target_peak_biomass) / target_peak_biomass

    print(f"[1.1.3] Baseline FolN    : {current_foln:.4f}")
    print(f"[1.1.3] Baseline peak    : {baseline_peak:.2f} g/m²  "
          f"(relative error: {rel_error * 100:.2f}%)")

    if rel_error <= tolerance:
        print(f"[1.1.3] Already within tolerance. No adjustment needed.")
        return current_foln, baseline_peak

    # --- Evaluate biomass at both bounds ---
    peak_at_lower = run_sim(foln_lower)
    peak_at_upper = run_sim(foln_upper)

    print(f"[1.1.3] Biomass at FolN lower bound ({foln_lower:.4f}): {peak_at_lower:.2f} g/m²")
    print(f"[1.1.3] Biomass at FolN upper bound ({foln_upper:.4f}): {peak_at_upper:.2f} g/m²")

    # Check that the target lies within the achievable range
    foln_min_peak = min(peak_at_lower, peak_at_upper)
    foln_max_peak = max(peak_at_lower, peak_at_upper)

    if not (foln_min_peak <= target_peak_biomass <= foln_max_peak):
        if abs(peak_at_lower - target_peak_biomass) < abs(peak_at_upper - target_peak_biomass):
            best_foln = foln_lower
            best_peak = peak_at_lower
        else:
            best_foln = foln_upper
            best_peak = peak_at_upper

        rel_error_best = abs(best_peak - target_peak_biomass) / target_peak_biomass
        print(f"\n[WARNING] Target biomass ({target_peak_biomass:.2f} g/m²) is outside the range "
              f"achievable by FolN [{foln_min_peak:.2f}, {foln_max_peak:.2f}] g/m².")
        print(f"[WARNING] Best achievable peak: {best_peak:.2f} g/m² at FolN={best_foln:.4f} "
              f"(relative error: {rel_error_best * 100:.2f}%).")
        print("[WARNING] Please return to subphase 1.1.1 to make major adjustments "
              "to the biomass peak before retrying subphase 1.1.3.")
        return best_foln, best_peak

    # --- Bisection ---
    if peak_at_lower <= target_peak_biomass:
        low, high = foln_lower, foln_upper
    else:
        low, high = foln_upper, foln_lower

    print(f"\n[1.1.3] Starting bisection...")

    best_foln = float((low + high) / 2.0)
    best_peak = None

    for i in range(max_iter):
        mid      = float((low + high) / 2.0)
        mid_peak = run_sim(mid)
        rel_error = abs(mid_peak - target_peak_biomass) / target_peak_biomass

        print(f"  Iter {i+1:02d} | FolN={mid:.4f} | "
              f"Biomass peak={mid_peak:.2f} g/m² | "
              f"Relative error={rel_error * 100:.2f}%")

        best_foln = mid
        best_peak = mid_peak

        if rel_error <= tolerance:
            print(f"\n[1.1.3] Converged at iteration {i+1}.")
            break

        if mid_peak < target_peak_biomass:
            low = mid
        else:
            high = mid

    else:
        rel_error_final = abs(float(best_peak) - target_peak_biomass) / target_peak_biomass
        print(f"\n[WARNING] Bisection reached max iterations ({max_iter}) without converging "
              f"within tolerance (final relative error: {rel_error_final * 100:.2f}%).")
        print("[WARNING] Please return to subphase 1.1.1 to make major adjustments "
              "to the biomass peak before retrying subphase 1.1.3.")

    best_peak = float(best_peak)
    best_foln = float(best_foln)

    run_sim(best_foln, printPlots = True)

    print(f"\n[1.1.3] Calibrated FolN  : {best_foln:.4f}")
    print(f"[1.1.3] Achieved peak    : {best_peak:.2f} g/m²")
    print(f"[1.1.3] Relative error   : {abs(best_peak - target_peak_biomass) / target_peak_biomass * 100:.2f}%")

    return best_foln


import json
import copy
import numpy as np
import shutil
import math

def calibrate_subphase_1_2(
    species: str,
    target_peak_time: float,
    target_peak_biomass: float,
    target_LAI: float,
    DictOfBounds: dict,
    path_core: str = './SpeciesParametersSets/Calibrated_SubSubPhase1.1.3/initialCoreSpeciesParameters.json',
    path_pnet: str = './SpeciesParametersSets/Calibrated_SubSubPhase1.1.3/initialPnETSpeciesParameters.json',
    path_generic: str = './SpeciesParametersSets/Calibrated_SubSubPhase1.1.3/InitialGenericParameters.json',
    duration: int = 300,
    climate: str = "mild",
    soil: str = "SILO",
    disableLongevity = True,
    peak_time_tolerance = 7.0):

    # ─────────────────────────────────────────────
    # ARBITRARY INPUTS FOUND BY TRIAL AND ERROR
    # ─────────────────────────────────────────────

    # peak_time_tolerance    = 7.0   # ± years
    peak_biomass_rel_tol   = 0.05   # ± 5%

    MaxFracFol_step  = 0.02
    FrActWd_step     = 0.000005
    FracFol_step     = 0.005
    FolN_step        = 0.1       # step for FolN in Phase C
    turnover_pct_step = 0.05   # 5% increment per iteration for TOWood and TORoot

    # FracFolShape coupling constants - Arbitrary, found via trial and error.
    # Will need to change these if PnET-Succession code changes
    MAXFRACFOL_THRESHOLD  = 0.05
    MAXFRACFOL_CEILING    = 0.141
    FRACFOLSHAPE_LOW      = 8.0
    FRACFOLSHAPE_HIGH     = 16.0

    # ─────────────────────────────────────────────
    # LOAD INITIAL PARAMETER DICTIONARIES
    # ─────────────────────────────────────────────
    initial_core_params = json.load(open(path_core))
    initial_pnet_species_params = json.load(open(path_pnet))
    initial_generic_params = json.load(open(path_generic))

    # We edit the longevity because at this step, we still want it to be 
    # unlimited to avoid confusion with the effect of age.
    if disableLongevity:
        initial_core_params[species]["Longevity"]="999"

    # ─────────────────────────────────────────────
    # BOUNDS
    # ─────────────────────────────────────────────
    lb_MaxFracFol = float(DictOfBounds[species]["MaxFracFol"]["lower"])
    ub_MaxFracFol = float(DictOfBounds[species]["MaxFracFol"]["upper"])
    ceiling_MaxFracFol = float(ub_MaxFracFol * 1.5)

    lb_FrActWd = float(DictOfBounds[species]["FrActWd"]["lower"])
    ub_FrActWd = float(DictOfBounds[species]["FrActWd"]["upper"])

    lb_FracFol = float(DictOfBounds[species]["FracFol"]["lower"])
    ub_FracFol = float(DictOfBounds[species]["FracFol"]["upper"])

    lb_TOWood = float(DictOfBounds[species]["TOWood"]["lower"])
    ub_TOWood = float(DictOfBounds[species]["TOWood"]["upper"])
    lb_TORoot  = float(DictOfBounds[species]["TORoot"]["lower"])
    ub_TORoot  = float(DictOfBounds[species]["TORoot"]["upper"])

    lb_FolN = float(DictOfBounds[species]["FolN"]["lower"])
    ub_FolN = float(DictOfBounds[species]["FolN"]["upper"])

    # ─────────────────────────────────────────────
    # CURRENT PARAMETER STATE (mutable working values)
    # ─────────────────────────────────────────────
    current_MaxFracFol = float(
        initial_pnet_species_params["PnETSpeciesParameters"][species]["MaxFracFol"])
    current_FrActWd = float(
        initial_pnet_species_params["PnETSpeciesParameters"][species]["FrActWd"])
    current_FracFol = float(
        initial_pnet_species_params["PnETSpeciesParameters"][species]["FracFol"])
    current_TOWood = float(
        initial_pnet_species_params["PnETSpeciesParameters"][species]["TOWood"])
    current_TORoot = float(
        initial_pnet_species_params["PnETSpeciesParameters"][species]["TORoot"])
    current_FolN = float(
        initial_pnet_species_params["PnETSpeciesParameters"][species]["FolN"])

    # Initial value of FracFoLShape
    def coupled_FracFolShape(MaxFracFol):
        """Return FracFolShape with a fast-rise/slow-approach curve coupled to MaxFracFol."""
        MaxFracFol = float(MaxFracFol)
        if MaxFracFol <= MAXFRACFOL_THRESHOLD:
            return float(FRACFOLSHAPE_LOW)
        else:
            # Normalise position in [0, 1] across the active range
            t = (MaxFracFol - MAXFRACFOL_THRESHOLD) / (MAXFRACFOL_CEILING - MAXFRACFOL_THRESHOLD)
            t = min(t, 1.0)  # clamp before applying curve
    
            # Reverse exponential: f(t) = (1 - e^(-k*t)) / (1 - e^(-k)), maps [0,1] -> [0,1]
            # Higher k = faster early rise, slower approach to ceiling
            k = 5.0
            curved_t = (1.0 - math.exp(-k * t)) / (1.0 - math.exp(-k))
    
            value = FRACFOLSHAPE_LOW + curved_t * (FRACFOLSHAPE_HIGH - FRACFOLSHAPE_LOW)
            return float(min(max(value, FRACFOLSHAPE_LOW), FRACFOLSHAPE_HIGH))

    if "FracFolShape" in initial_pnet_species_params["PnETSpeciesParameters"][species].keys():
        current_FracFolShape = float(initial_pnet_species_params["PnETSpeciesParameters"][species]["FracFolShape"])
    else:
        current_FracFolShape = coupled_FracFolShape(current_MaxFracFol)

    # ─────────────────────────────────────────────
    # BEST PARAMETER SET TRACKER
    # (closest peak age while having right peak height and right LAI)
    # ─────────────────────────────────────────────
    best_params = None          # will hold a dict of the best parameter set
    best_peak_age_diff = None   # |peak_time - target| for that set

    def update_best_params(sim_peak_time, sim_max_LAI, sim_peak_biomass):
        """Register the current parameter set if peak height and LAI are matched
        and the peak age is closer to target than any previously recorded set."""
        nonlocal best_params, best_peak_age_diff
        lai_ok = abs(sim_max_LAI - original_max_LAI_phB) / original_max_LAI_phB <= 0.05
        biomass_ok = (abs(sim_peak_biomass - original_peak_biomass_phB)
                      / original_peak_biomass_phB <= 0.05)
        if lai_ok and biomass_ok:
            age_diff = abs(sim_peak_time - target_peak_time)
            if (best_peak_age_diff is None) or (age_diff < best_peak_age_diff):
                best_peak_age_diff = age_diff
                best_params = {
                    "MaxFracFol": current_MaxFracFol,
                    "FracFolShape": current_FracFolShape,
                    "FrActWd": current_FrActWd,
                    "FracFol": current_FracFol,
                    "TOWood": current_TOWood,
                    "TORoot": current_TORoot,
                    "FolN": current_FolN,
                }
                print(f"    ★ New best parameter set registered "
                      f"(peak age diff = {age_diff:.1f} yr)")

    # ─────────────────────────────────────────────
    # HELPERS FUNCTIONS
    # ─────────────────────────────────────────────

    def build_param_dicts(MaxFracFol, FrActWd, FracFol, TOWood, TORoot,
                          FolN, fracFolShape=current_FracFolShape):
        core   = copy.deepcopy(initial_core_params)
        pnet_s = copy.deepcopy(initial_pnet_species_params)
        gen    = copy.deepcopy(initial_generic_params)

        sp = pnet_s["PnETSpeciesParameters"][species]
        sp["MaxFracFol"]   = float(MaxFracFol)
        sp["FracFolShape"] = float(fracFolShape)
        sp["FrActWd"]      = float(FrActWd)
        sp["FracFol"]      = float(FracFol)
        sp["TOWood"]       = float(TOWood)
        sp["TORoot"]        = float(TORoot)
        sp["FolN"]          = float(FolN)

        return core, pnet_s, gen

    def run_sim(MaxFracFol, FrActWd, FracFol, TOWood, TORoot, FolN,
                makePlots=False, printParams=False, fracFolShape=current_FracFolShape):
        core, pnet_s, gen = build_param_dicts(
            MaxFracFol, FrActWd, FracFol, TOWood, TORoot, FolN, fracFolShape)
        result = calibrationSimulationMonoculturemanawan(
            duration=duration,
            climate=climate,
            soil=soil,
            speciesToSimulate=species,
            dictOfInitialCoreSpeciesParameters=core,
            dictOfInitialPnETSpeciesParameters=pnet_s,
            dictOfInitialPnETGenericParameters=gen,
            plotResults=makePlots,
            
        )
        if printParams:
            print("Core species parameters : \n")
            print(core)
            print("PnET species parameters : \n")
            print(pnet_s)
            print("PnET generic parameters : \n")
            print(gen)
        # print("\033[42m\nPnET species parameters : \033[0m")
        # print("\033[42m" + str(pnet_s) + "\033[0m")
        # print("\n")
        # print("\033[42mSim results : "+ str(result) + "\033[0m\n")
        return (float(result["Biomass peak 95% time"]),
                float(result["Maximum LAI"]),
                float(result["Biomass peak height"]))

    # ─────────────────────────────────────────────
    # HELPER FUNCTION: Recalibrate LAI and Biomass Peak Height
    # ─────────────────────────────────────────────
    def recalibrate_LAI_and_biomass(
            current_MaxFracFol, current_FrActWd, current_FracFol,
            current_TOWood, current_TORoot, current_FolN,
            original_max_LAI, original_peak_biomass,
            LAI_rel_tol=0.05, biomass_rel_tol=0.05, CalibratePeakHeight=False):
        """
        Adjusts FracFol to restore Maximum LAI to original_max_LAI,
        then adjusts TOWood and TORoot together to restore peak biomass
        to original_peak_biomass. Uses overshoot detection with step halving
        for both adjustments.
        Returns updated (current_FracFol, current_TOWood, current_TORoot).
        """

        # ── Step 1: Recalibrate Biomass Peak Height via TOWood and TORoot ──
        if True:
            print("    [Helper] Recalibrating Biomass Peak Height via TOWood and TORoot...")
            _, _, sim_biomass = run_sim(
                current_MaxFracFol, current_FrActWd, current_FracFol,
                current_TOWood, current_TORoot, current_FolN,
                False, False, current_FracFolShape)

            biomass_matched = abs(sim_biomass - original_peak_biomass) / original_peak_biomass <= biomass_rel_tol

            if biomass_matched:
                print(f"    [Helper] Biomass already within tolerance ({sim_biomass:.2f}). "
                      f"Skipping TOWood/TORoot adjustment.")
            else:
                direction_turnover = 1 if sim_biomass > original_peak_biomass else -1
                turnover_step = turnover_pct_step
                prev_biomass = sim_biomass
                bio_iter = 0

                while not biomass_matched:
                    pct_change = 1.0 + direction_turnover * turnover_step
                    candidate_TOWood = float(current_TOWood * pct_change)
                    candidate_TORoot = float(current_TORoot * pct_change)

                    if candidate_TOWood > ub_TOWood or candidate_TOWood < lb_TOWood:
                        print(f"    [Helper] ⚠ WARNING: TOWood={candidate_TOWood:.6f} outside empirical "
                              f"bounds [{lb_TOWood:.6f}, {ub_TOWood:.6f}].")
                    if candidate_TORoot > ub_TORoot or candidate_TORoot < lb_TORoot:
                        print(f"    [Helper] ⚠ WARNING: TORoot={candidate_TORoot:.6f} outside empirical "
                              f"bounds [{lb_TORoot:.6f}, {ub_TORoot:.6f}].")

                    _, sim_LAI, sim_biomass = run_sim(
                        current_MaxFracFol, current_FrActWd, current_FracFol,
                        candidate_TOWood, candidate_TORoot, current_FolN,
                        False, False, current_FracFolShape)

                    bio_iter += 1
                    print(f"    [Helper-BIO {bio_iter:>3d}] TOWood={candidate_TOWood:.6f}  "
                          f"TORoot={candidate_TORoot:.6f} (step={turnover_step:.4f}) | "
                          f"PeakBiomass={sim_biomass:.2f} (target={original_peak_biomass:.2f}) | "
                          f"MaxLAI={sim_LAI:.4f}")

                    # Overshoot detection
                    overshot = (
                        (direction_turnover == 1  and
                         sim_biomass < original_peak_biomass * (1.0 - biomass_rel_tol)) or
                        (direction_turnover == -1 and
                         sim_biomass > original_peak_biomass * (1.0 + biomass_rel_tol))
                    )
                    if overshot:
                        turnover_step = float(turnover_step / 2.0)
                        direction_turnover = -direction_turnover
                        print(f"    [Helper] Overshoot detected (Biomass: {prev_biomass:.2f} → "
                              f"{sim_biomass:.2f}). Reversing direction, halving step to "
                              f"{turnover_step:.6f}.")
                        continue

                    prev_biomass = sim_biomass
                    current_TOWood = candidate_TOWood
                    current_TORoot = candidate_TORoot
                    biomass_matched = (
                        abs(sim_biomass - original_peak_biomass) / original_peak_biomass <= biomass_rel_tol)

                print(f"    [Helper] ✓ Biomass recalibrated. TOWood={current_TOWood:.6f}  "
                      f"TORoot={current_TORoot:.6f} | PeakBiomass={sim_biomass:.2f}")

        # ── Step 2: Recalibrate Maximum LAI via FracFol ──
        print("    [Helper] Recalibrating Maximum LAI via FracFol...")
        _, sim_LAI, _ = run_sim(
            current_MaxFracFol, current_FrActWd, current_FracFol,
            current_TOWood, current_TORoot, current_FolN,
            False, False, current_FracFolShape)

        LAI_matched = abs(sim_LAI - original_max_LAI) / original_max_LAI <= LAI_rel_tol

        if LAI_matched:
            print(f"    [Helper] LAI already within tolerance ({sim_LAI:.4f}). Skipping FracFol adjustment.")
        else:
            fracfol_step = FracFol_step
            direction_fracfol = 1 if sim_LAI < original_max_LAI else -1
            prev_LAI = sim_LAI
            lai_iter = 0

            while not LAI_matched:
                candidate_FracFol = float(current_FracFol + direction_fracfol * fracfol_step)
                if candidate_FracFol < 0:
                    raise ValueError("The algorithm has reached negative values for FracFol because it had "
                                     "too much trouble reducing the LAI. FracFol cannot be negative. "
                                     "Please check the assumptions or parameters for this species.")

                if candidate_FracFol > ub_FracFol:
                    print(f"    [Helper] ⚠ WARNING: FracFol={candidate_FracFol:.6f} exceeds upper "
                          f"empirical bound ({ub_FracFol:.6f}).")
                elif candidate_FracFol < lb_FracFol:
                    print(f"    [Helper] ⚠ WARNING: FracFol={candidate_FracFol:.6f} is below lower "
                          f"empirical bound ({lb_FracFol:.6f}).")

                _, sim_LAI, _ = run_sim(
                    current_MaxFracFol, current_FrActWd, candidate_FracFol,
                    current_TOWood, current_TORoot, current_FolN,
                    False, False, current_FracFolShape)

                lai_iter += 1
                print(f"    [Helper-LAI {lai_iter:>3d}] FracFol={candidate_FracFol:.6f} "
                      f"(step={fracfol_step:.6f}) | MaxLAI={sim_LAI:.4f} "
                      f"(target={original_max_LAI:.4f})")

                # Overshoot detection
                overshot = (
                    (direction_fracfol == 1  and sim_LAI > original_max_LAI * (1.0 + LAI_rel_tol)) or
                    (direction_fracfol == -1 and sim_LAI < original_max_LAI * (1.0 - LAI_rel_tol))
                )
                if overshot:
                    fracfol_step = float(fracfol_step / 2.0)
                    direction_fracfol = -direction_fracfol
                    print(f"    [Helper] Overshoot detected (LAI: {prev_LAI:.4f} → {sim_LAI:.4f}). "
                          f"Reversing direction, halving step to {fracfol_step:.6f}.")
                    continue

                prev_LAI = sim_LAI
                current_FracFol = candidate_FracFol
                LAI_matched = abs(sim_LAI - original_max_LAI) / original_max_LAI <= LAI_rel_tol

                if lai_iter == 10:
                    print("LAI recalibration seems to fail. Resetting FracFol to lower values and retrying.")
                    current_FracFol = lb_FracFol
                    fracfol_step = FracFol_step

                if lai_iter == 25:
                    print("LAI recalibration seems to fail. Printing sim and parameters :")
                    _, sim_LAI, _ = run_sim(
                        current_MaxFracFol, current_FrActWd, candidate_FracFol,
                        current_TOWood, current_TORoot, current_FolN,
                        False, False, current_FracFolShape)
                    raise ValueError("LAI recalibration is failing. Seems like the algorithm has dug itself into a hole.")

            print(f"    [Helper] ✓ LAI recalibrated. FracFol={current_FracFol:.6f} | "
                  f"MaxLAI={sim_LAI:.4f}")

        return current_FracFol, current_TOWood, current_TORoot
    
    # ─────────────────────────────────────────────
    # BASELINE RUN
    # ─────────────────────────────────────────────
    print("=" * 70)
    print(f"Subphase 1.2 — Species: {species}")
    print("=" * 70)
    print("\n[Baseline] Running baseline simulation...")
    
    baseline_peak_time, baseline_max_LAI, baseline_peak_biomass = run_sim(
        current_MaxFracFol, current_FrActWd, current_FracFol,
        current_TOWood, current_TORoot, current_FolN, False, False)
    
    original_max_LAI = baseline_max_LAI
    
    print(f"  Baseline peak time    : {baseline_peak_time:.1f} yr")
    print(f"  Baseline max LAI      : {baseline_max_LAI:.4f}")
    print(f"  Baseline peak biomass : {baseline_peak_biomass:.2f} g/m²")
    
    peak_time_matched = abs(baseline_peak_time - target_peak_time) <= peak_time_tolerance
    too_slow = baseline_peak_time > target_peak_time + peak_time_tolerance
    too_fast = baseline_peak_time < target_peak_time - peak_time_tolerance

    # ─────────────────────────────────────────────
    # RECORD ORIGINAL TARGETS FOR HELPER
    # ─────────────────────────────────────────────
    original_max_LAI_phB = baseline_max_LAI
    original_peak_biomass_phB = baseline_peak_biomass

    LAI_trigger_low  = 2.2
    LAI_trigger_high = 6.0
    
    # ─────────────────────────────────────────────
    # PHASE A — ADJUST MaxFracFol (with coupled FracFolShape)
    # ─────────────────────────────────────────────
    print("\n--- Phase A: Adjusting MaxFracFol (FracFolShape coupled) ---")
    
    # closestPeakTime = -999
    # finalMaxFracFol  = current_MaxFracFol
    # finalFracFolShape = current_FracFolShape
    
    if peak_time_matched:
        print("  Peak timing already within tolerance. Skipping Phase A.")
    else:
        direction  = 1 if too_slow else -1
        step       = MaxFracFol_step          # adaptive step size
        prev_error = baseline_peak_time - target_peak_time   # signed error
        iteration  = 0
    
        while not peak_time_matched:
            candidate_MaxFracFol  = float(current_MaxFracFol + direction * step)
            candidate_FracFolShape = float(coupled_FracFolShape(candidate_MaxFracFol))
    
            # ── Bounds check (use the current direction to pick the right bound) ──
            if direction > 0 and candidate_MaxFracFol > MAXFRACFOL_CEILING:
                print(f"  MaxFracFol reached ceiling ({MAXFRACFOL_CEILING:.4f}). Stopping Phase A.")
                break
            if direction < 0 and candidate_MaxFracFol < lb_MaxFracFol:
                print(f"  MaxFracFol reached lower bound ({lb_MaxFracFol:.6f}). Stopping Phase A.")
                break
    
            sim_peak_time, sim_max_LAI, _ = run_sim(
                candidate_MaxFracFol, current_FrActWd, current_FracFol,
                current_TOWood, current_TORoot, current_FolN,
                False, False, candidate_FracFolShape)
    
            iteration += 1
            print(
                f"  [A-{iteration:>3d}] MaxFracFol={candidate_MaxFracFol:.4f}  "
                f"FracFolShape={candidate_FracFolShape:.4f} | "
                f"PeakTime={sim_peak_time:.1f} yr | MaxLAI={sim_max_LAI:.4f}"
            )
    
            current_error = sim_peak_time - target_peak_time
    
            # ── Bisection: if we crossed the target, flip direction & halve step ──
            if prev_error * current_error < 0:      # sign change → overshot
                direction = -direction
                step      = step / 2.0
    
            prev_error           = current_error
            current_MaxFracFol   = float(candidate_MaxFracFol)
            current_FracFolShape = float(candidate_FracFolShape)
            peak_time_matched    = abs(current_error) <= peak_time_tolerance
    
            if peak_time_matched:
                print(f"  ✓ Peak timing matched in Phase A.")
                print(f"    MaxFracFol={current_MaxFracFol:.6f}  "
                      f"FracFolShape={coupled_FracFolShape(current_MaxFracFol):.6f}")
                # Now that the peak time has matched, we attempt to recalirate LAI and peak.
                # We then re-check if the peak time is OK before we go.
                print("Phase A Finished. Recalibrating LAI and peak before going to phase B.")
                current_FracFol, current_TOWood, current_TORoot = recalibrate_LAI_and_biomass(
                    current_MaxFracFol, current_FrActWd, current_FracFol,
                    current_TOWood, current_TORoot, current_FolN,
                    original_max_LAI_phB, original_peak_biomass_phB)
                print("Re-checking peak time one last time before going into phase B")
                sim_peak_time, sim_max_LAI, _ = run_sim(
                candidate_MaxFracFol, current_FrActWd, current_FracFol,
                current_TOWood, current_TORoot, current_FolN,
                False, False, candidate_FracFolShape)
                # Same code as above
                current_error = sim_peak_time - target_peak_time
                # ── Bisection: if we crossed the target, flip direction & halve step ──
                if prev_error * current_error < 0:      # sign change → overshot
                    direction = -direction
                    step      = step / 2.0
        
                prev_error           = current_error
                current_MaxFracFol   = float(candidate_MaxFracFol)
                peak_time_matched    = abs(current_error) <= peak_time_tolerance # If true, we escape the loop and go to phase B. If False, we go back into phase A.
                if peak_time_matched:
                    print("Peak time is OK after recalibration of LAI and peak. Going to phase B.")
                    current_MaxFracFol = candidate_MaxFracFol
                    current_FracFolShape = candidate_FracFolShape
                else:
                    print("Peak time has changed too much after recalibration of LAI and peak. Going back to phase A.")
                
    
    #         if (closestPeakTime == -999) or (abs(current_error) < closestPeakTime):
    #             closestPeakTime   = abs(current_error)
    #             finalMaxFracFol   = candidate_MaxFracFol
    #             finalFracFolShape = float(coupled_FracFolShape(candidate_MaxFracFol))
    
    # if closestPeakTime > -999:
    #     current_MaxFracFol    = finalMaxFracFol
    #     current_FracFolShape  = finalFracFolShape
    
    print(f"Final values of MaxFracFol and FracFolShape to be used going forward :"
          f"    MaxFracFol={current_MaxFracFol:.6f}  "
          f"FracFolShape={current_FracFolShape:.6f}")

    # ─────────────────────────────────────────────
    # PHASE B — ADJUST FrActWd
    # ─────────────────────────────────────────────
    print("\n--- Phase B: Adjusting FrActWd ---")
    print(f"  Reference LAI for recalibration    : {original_max_LAI_phB:.4f}")
    print(f"  Reference biomass for recalibration: {original_peak_biomass_phB:.2f} g/m²")
    print(f"  LAI trigger range                  : [{LAI_trigger_low:.1f}, {LAI_trigger_high:.1f}]")

    # Flag: did FrActWd hit its upper bound, triggering Phase C?
    frActWd_hit_ub = False
    frActWd_hit_lb = False
    # Flag: has full convergence been achieved?
    fully_converged = False

    if peak_time_matched:
        print("  Peak timing already matched. Skipping Phase B.")
        fully_converged = True
    else:
        sim_peak_time, sim_max_LAI, sim_peak_biomass = run_sim(
            current_MaxFracFol, current_FrActWd, current_FracFol,
            current_TOWood, current_TORoot, current_FolN,
            False, False, current_FracFolShape)

        too_slow = sim_peak_time > target_peak_time + peak_time_tolerance
        direction = 1 if too_slow else -1
        current_step = FrActWd_step
        prev_peak_time = sim_peak_time
        iteration = 0

        # Outer loop: keep going until peak timing, LAI and biomass are all matched
        all_matched = (
            abs(sim_peak_time - target_peak_time) <= peak_time_tolerance and
            abs(sim_max_LAI - original_max_LAI_phB) / original_max_LAI_phB <= 0.05 and
            abs(sim_peak_biomass - original_peak_biomass_phB) / original_peak_biomass_phB <= 0.05
        )

        iterationsWithoutRecalibration = 0

        while not all_matched:

            # ── Adjust FrActWd until peak timing is matched ──
            peak_time_matched = abs(sim_peak_time - target_peak_time) <= peak_time_tolerance

            while not peak_time_matched:
                candidate_FrActWd = float(current_FrActWd + direction * current_step)

                # ── NEW: Stop Phase B if FrActWd exceeds upper bound ──
                if candidate_FrActWd > ub_FrActWd:
                    print(f"  ⚠ FrActWd={candidate_FrActWd:.8f} exceeds upper bound "
                          f"({ub_FrActWd:.6f}). Stopping Phase B, will proceed to Phase C.")
                    frActWd_hit_ub = True
                    # Clamp FrActWd to its upper bound before exiting
                    current_FrActWd = float(ub_FrActWd)
                    break

                if candidate_FrActWd < lb_FrActWd:
                    print(f"  ⚠ WARNING: FrActWd={candidate_FrActWd:.6f} is below lower empirical "
                          f"bound ({lb_FrActWd:.6f}). Stopping Phase B, will proceed to Phase C.")
                    frActWd_hit_lb = True
                    # Clamp FrActWd to its upper bound before exiting
                    current_FrActWd = float(lb_FrActWd)
                    break

                sim_peak_time, sim_max_LAI, sim_peak_biomass = run_sim(
                    current_MaxFracFol, candidate_FrActWd, current_FracFol,
                    current_TOWood, current_TORoot, current_FolN,
                    False, False, current_FracFolShape)

                iteration += 1
                iterationsWithoutRecalibration += 1
                print(
                    f"  [B-{iteration:>3d}] FrActWd={candidate_FrActWd:.8f} "
                    f"(step={current_step:.6f}) | "
                    f"PeakTime={sim_peak_time:.1f} yr | "
                    f"MaxLAI={sim_max_LAI:.4f} | "
                    f"PeakBiomass={sim_peak_biomass:.2f} g/m²"
                )

                # Overshoot detection for peak timing
                overshot = (
                    (direction == 1  and
                     sim_peak_time < target_peak_time - peak_time_tolerance) or
                    (direction == -1 and
                     sim_peak_time > target_peak_time + peak_time_tolerance)
                )
                if overshot:
                    current_step = float(current_step / 2.0)
                    direction = -direction
                    print(f"  Overshoot detected (PeakTime: {prev_peak_time:.1f} → "
                          f"{sim_peak_time:.1f} yr). Reversing direction, halving step to "
                          f"{current_step:.6f}.")
                    continue

                prev_peak_time = sim_peak_time
                current_FrActWd = candidate_FrActWd
                peak_time_matched = abs(sim_peak_time - target_peak_time) <= peak_time_tolerance

                # Trigger helper if iterationsWithoutRecalibration == 2
                if iterationsWithoutRecalibration >= 4:
                    print("  Relacalibrating LAI after 4 iterations of changing FrActWd...")
                    current_FracFol, current_TOWood, current_TORoot = recalibrate_LAI_and_biomass(
                        current_MaxFracFol, current_FrActWd, current_FracFol,
                        current_TOWood, current_TORoot, current_FolN,
                        original_max_LAI_phB, original_peak_biomass_phB)
                    # Re-evaluate after helper
                    sim_peak_time, sim_max_LAI, sim_peak_biomass = run_sim(
                        current_MaxFracFol, current_FrActWd, current_FracFol,
                        current_TOWood, current_TORoot, current_FolN,
                        False, False, current_FracFolShape)
                    peak_time_matched = abs(sim_peak_time - target_peak_time) <= peak_time_tolerance
                    iterationsWithoutRecalibration = 0

            # If FrActWd hit its upper bound, break out of the outer loop too
            if frActWd_hit_ub or frActWd_hit_lb:
                break

            if peak_time_matched:
                print(f"  ✓ Peak timing matched. FrActWd={current_FrActWd:.6f}. "
                      f"Launching final helper recalibration...")

            # ── Final recalibration of LAI and biomass once timing is matched ──
            current_FracFol, current_TOWood, current_TORoot = recalibrate_LAI_and_biomass(
                current_MaxFracFol, current_FrActWd, current_FracFol,
                current_TOWood, current_TORoot, current_FolN,
                original_max_LAI_phB, original_peak_biomass_phB,
                CalibratePeakHeight=True)

            # ── Final check: are all three targets met? ──
            sim_peak_time, sim_max_LAI, sim_peak_biomass = run_sim(
                current_MaxFracFol, current_FrActWd, current_FracFol,
                current_TOWood, current_TORoot, current_FolN,
                False, False, current_FracFolShape)

            peak_time_ok    = abs(sim_peak_time - target_peak_time) <= peak_time_tolerance
            lai_ok          = abs(sim_max_LAI - original_max_LAI_phB) / original_max_LAI_phB <= 0.05
            biomass_ok      = (abs(sim_peak_biomass - original_peak_biomass_phB)
                               / original_peak_biomass_phB <= 0.05)

            print(f"\n  [B — End-of-cycle check]")
            print(f"    PeakTime={sim_peak_time:.1f} yr  "
                  f"(target={target_peak_time:.1f} ± {peak_time_tolerance:.1f})  "
                  f"{'✓' if peak_time_ok else '✗'}")
            print(f"    MaxLAI={sim_max_LAI:.4f}  "
                  f"(target={original_max_LAI_phB:.4f} ± 5%)  "
                  f"{'✓' if lai_ok else '✗'}")
            print(f"    PeakBiomass={sim_peak_biomass:.2f}  "
                  f"(target={original_peak_biomass_phB:.2f} ± 5%)  "
                  f"{'✓' if biomass_ok else '✗'}")

            # ── NEW: Register best parameter set at end-of-cycle ──
            update_best_params(sim_peak_time, sim_max_LAI, sim_peak_biomass)

            all_matched = peak_time_ok and lai_ok and biomass_ok

            if not all_matched:
                too_slow = sim_peak_time > target_peak_time + peak_time_tolerance
                direction = 1 if too_slow else -1
                prev_peak_time = sim_peak_time
                print("  One or more targets not yet met. Continuing FrActWd adjustment...\n")

        if not frActWd_hit_ub and not frActWd_hit_lb:
            print(f"\n  ✓ Phase B complete.")
            print(f"    FrActWd={current_FrActWd:.6f} | FracFol={current_FracFol:.6f} | "
                  f"TOWood={current_TOWood:.6f} | TORoot={current_TORoot:.6f}")
            fully_converged = all_matched

    # ─────────────────────────────────────────────
    # PHASE C — ADJUST FolN (optional, entered when FrActWd hit ub)
    # ─────────────────────────────────────────────
    folN_hit_ub = False

    if (frActWd_hit_ub or frActWd_hit_lb) and not fully_converged:
        print("\n--- Phase C: Adjusting FolN (FrActWd reached upper bound) ---")
        print(f"  FolN bounds: [{lb_FolN:.6f}, {ub_FolN:.6f}]")
        print(f"  Current FolN: {current_FolN:.6f}")
        print(f"  Reference LAI for recalibration    : {original_max_LAI_phB:.4f}")
        print(f"  Reference biomass for recalibration: {original_peak_biomass_phB:.2f} g/m²")

        # Recalibrate LAI and biomass at the start of Phase C
        # (since we clamped FrActWd to ub, values may have shifted)
        current_FracFol, current_TOWood, current_TORoot = recalibrate_LAI_and_biomass(
            current_MaxFracFol, current_FrActWd, current_FracFol,
            current_TOWood, current_TORoot, current_FolN,
            original_max_LAI_phB, original_peak_biomass_phB)

        sim_peak_time, sim_max_LAI, sim_peak_biomass = run_sim(
            current_MaxFracFol, current_FrActWd, current_FracFol,
            current_TOWood, current_TORoot, current_FolN,
            False, False, current_FracFolShape)

        too_slow = sim_peak_time > target_peak_time + peak_time_tolerance
        direction = 1 if too_slow else -1
        current_step_c = FolN_step
        prev_peak_time = sim_peak_time
        iteration_c = 0

        all_matched_c = (
            abs(sim_peak_time - target_peak_time) <= peak_time_tolerance and
            abs(sim_max_LAI - original_max_LAI_phB) / original_max_LAI_phB <= 0.05 and
            abs(sim_peak_biomass - original_peak_biomass_phB) / original_peak_biomass_phB <= 0.05
        )

        iterationsWithoutRecalibration_c = 0

        while not all_matched_c:

            # ── Adjust FolN until peak timing is matched ──
            peak_time_matched = abs(sim_peak_time - target_peak_time) <= peak_time_tolerance

            while not peak_time_matched:
                candidate_FolN = float(current_FolN + direction * current_step_c)

                # ── Stop Phase C if FolN exceeds upper bound ──
                if candidate_FolN > ub_FolN:
                    print(f"  ⚠ FolN={candidate_FolN:.6f} exceeds upper bound "
                          f"({ub_FolN:.6f}). Stopping Phase C.")
                    folN_hit_ub = True
                    current_FolN = float(ub_FolN)
                    break

                if candidate_FolN < lb_FolN:
                    print(f"  ⚠ WARNING: FolN={candidate_FolN:.6f} is below lower empirical "
                          f"bound ({lb_FolN:.6f}). Proceeding anyway.")

                sim_peak_time, sim_max_LAI, sim_peak_biomass = run_sim(
                    current_MaxFracFol, current_FrActWd, current_FracFol,
                    current_TOWood, current_TORoot, candidate_FolN,
                    False, False, current_FracFolShape)

                iteration_c += 1
                iterationsWithoutRecalibration_c += 1
                print(
                    f"  [C-{iteration_c:>3d}] FolN={candidate_FolN:.6f} "
                    f"(step={current_step_c:.6f}) | "
                    f"PeakTime={sim_peak_time:.1f} yr | "
                    f"MaxLAI={sim_max_LAI:.4f} | "
                    f"PeakBiomass={sim_peak_biomass:.2f} g/m²"
                )

                # Overshoot detection for peak timing
                overshot = (
                    (direction == 1  and
                     sim_peak_time < target_peak_time - peak_time_tolerance) or
                    (direction == -1 and
                     sim_peak_time > target_peak_time + peak_time_tolerance)
                )
                if overshot:
                    current_step_c = float(current_step_c / 2.0)
                    direction = -direction
                    print(f"  Overshoot detected (PeakTime: {prev_peak_time:.1f} → "
                          f"{sim_peak_time:.1f} yr). Reversing direction, halving step to "
                          f"{current_step_c:.6f}.")
                    continue

                prev_peak_time = sim_peak_time
                current_FolN = candidate_FolN
                peak_time_matched = abs(sim_peak_time - target_peak_time) <= peak_time_tolerance

                # Trigger helper recalibration every 2 iterations (same as Phase B)
                if iterationsWithoutRecalibration_c == 2:
                    print(f"  Launching helper recalibration after {iterationsWithoutRecalibration_c} FolN steps...")
                    current_FracFol, current_TOWood, current_TORoot = recalibrate_LAI_and_biomass(
                        current_MaxFracFol, current_FrActWd, current_FracFol,
                        current_TOWood, current_TORoot, current_FolN,
                        original_max_LAI_phB, original_peak_biomass_phB)
                    sim_peak_time, sim_max_LAI, sim_peak_biomass = run_sim(
                        current_MaxFracFol, current_FrActWd, current_FracFol,
                        current_TOWood, current_TORoot, current_FolN,
                        False, False, current_FracFolShape)
                    peak_time_matched = abs(sim_peak_time - target_peak_time) <= peak_time_tolerance
                    iterationsWithoutRecalibration_c = 0

            # If FolN hit its upper bound, break out of the outer loop too
            if folN_hit_ub:
                break

            if peak_time_matched:
                print(f"  ✓ Peak timing matched. FolN={current_FolN:.6f}. "
                      f"Launching final helper recalibration...")

            # ── Final recalibration of LAI and biomass once timing is matched ──
            current_FracFol, current_TOWood, current_TORoot = recalibrate_LAI_and_biomass(
                current_MaxFracFol, current_FrActWd, current_FracFol,
                current_TOWood, current_TORoot, current_FolN,
                original_max_LAI_phB, original_peak_biomass_phB,
                CalibratePeakHeight=True)

            # ── Final check: are all three targets met? ──
            sim_peak_time, sim_max_LAI, sim_peak_biomass = run_sim(
                current_MaxFracFol, current_FrActWd, current_FracFol,
                current_TOWood, current_TORoot, current_FolN,
                False, False, current_FracFolShape)

            peak_time_ok    = abs(sim_peak_time - target_peak_time) <= peak_time_tolerance
            lai_ok          = abs(sim_max_LAI - original_max_LAI_phB) / original_max_LAI_phB <= 0.05
            biomass_ok      = (abs(sim_peak_biomass - original_peak_biomass_phB)
                               / original_peak_biomass_phB <= 0.05)

            print(f"\n  [C — End-of-cycle check]")
            print(f"    PeakTime={sim_peak_time:.1f} yr  "
                  f"(target={target_peak_time:.1f} ± {peak_time_tolerance:.1f})  "
                  f"{'✓' if peak_time_ok else '✗'}")
            print(f"    MaxLAI={sim_max_LAI:.4f}  "
                  f"(target={original_max_LAI_phB:.4f} ± 5%)  "
                  f"{'✓' if lai_ok else '✗'}")
            print(f"    PeakBiomass={sim_peak_biomass:.2f}  "
                  f"(target={original_peak_biomass_phB:.2f} ± 5%)  "
                  f"{'✓' if biomass_ok else '✗'}")

            # ── Register best parameter set at end-of-cycle ──
            update_best_params(sim_peak_time, sim_max_LAI, sim_peak_biomass)

            all_matched_c = peak_time_ok and lai_ok and biomass_ok

            if not all_matched_c:
                too_slow = sim_peak_time > target_peak_time + peak_time_tolerance
                direction = 1 if too_slow else -1
                prev_peak_time = sim_peak_time
                print("  One or more targets not yet met. Continuing FolN adjustment...\n")

        if not folN_hit_ub:
            print(f"\n  ✓ Phase C complete.")
            print(f"    FolN={current_FolN:.6f} | FrActWd={current_FrActWd:.6f} | "
                  f"FracFol={current_FracFol:.6f} | TOWood={current_TOWood:.6f} | "
                  f"TORoot={current_TORoot:.6f}")
            fully_converged = all_matched_c
        else:
            # FolN reached its upper bound without full convergence
            fully_converged = False

    # Final recalibration of LAI and peak
    print("  Calibrating LAI and biomass peak one last time...\n")
    current_FracFol, current_TOWood, current_TORoot = recalibrate_LAI_and_biomass(
        current_MaxFracFol, current_FrActWd, current_FracFol,
        current_TOWood, current_TORoot, current_FolN,
        original_max_LAI_phB, original_peak_biomass_phB,
        CalibratePeakHeight=True)

    # ─────────────────────────────────────────────
    # DETERMINE FINAL PARAMETERS
    # ─────────────────────────────────────────────
    # If not fully converged, fall back to the best recorded parameter set
    if not fully_converged and best_params is not None:
        print("\n  ⚠ WARNING: Peak timing was NOT achieved.")
        print(f"    Falling back to the best parameter set recorded "
              f"(peak age diff = {best_peak_age_diff:.1f} yr, "
              f"with matched peak height and LAI).")
        current_MaxFracFol  = best_params["MaxFracFol"]
        current_FracFolShape = best_params["FracFolShape"]
        current_FrActWd      = best_params["FrActWd"]
        current_FracFol      = best_params["FracFol"]
        current_TOWood       = best_params["TOWood"]
        current_TORoot       = best_params["TORoot"]
        current_FolN         = best_params["FolN"]
    elif not fully_converged and best_params is None:
        print("\n  ⚠ WARNING: Peak timing was NOT achieved and no valid "
              "fallback parameter set was recorded (no end-of-cycle had both "
              "peak height and LAI matched). Returning current parameters.")

    # ─────────────────────────────────────────────
    # FINAL REPORT
    # ─────────────────────────────────────────────

    print("\n" + "=" * 70)
    # Final sim run to show a plot
    sim_peak_time, sim_max_LAI, sim_peak_biomass = run_sim(
        current_MaxFracFol, current_FrActWd, current_FracFol,
        current_TOWood, current_TORoot, current_FolN,
        True, False, current_FracFolShape)

    print("CALIBRATION RESULTS — Subphase 1.2")
    print("=" * 70)
    print(f"  MaxFracFol        : {current_MaxFracFol:.6f}")
    print(f"  FracFolShape      : {current_FracFolShape:.6f}")
    print(f"  FrActWd           : {current_FrActWd:.6f}")
    print(f"  FolN              : {current_FolN:.6f}"
          + (" ⚠ WARNING: outside empirical bounds"
             if current_FolN > ub_FolN or current_FolN < lb_FolN else ""))
    print(f"  FracFol           : {current_FracFol:.6f}"
          + (" ⚠ WARNING: outside empirical bounds"
             if current_FracFol > ub_FracFol or current_FracFol < lb_FracFol else ""))
    print(f"  TOWood            : {current_TOWood:.6f}"
          + (" ⚠ WARNING: outside empirical bounds"
             if current_TOWood > ub_TOWood or current_TOWood < lb_TOWood else ""))
    print(f"  TORoot            : {current_TORoot:.6f}"
          + (" ⚠ WARNING: outside empirical bounds"
             if current_TORoot > ub_TORoot or current_TORoot < lb_TORoot else ""))
    print()
    final_peak_time_ok = abs(sim_peak_time - target_peak_time) <= peak_time_tolerance
    print(f"  Final peak time   : {sim_peak_time:.1f} yr  "
          f"(target: {target_peak_time:.1f} ± {peak_time_tolerance:.1f} yr)  "
          f"{'✓ Still within tolerance' if final_peak_time_ok else '✗ OUT OF TOLERANCE — review needed'}")
    peak_biomass_matched = (
             abs(sim_peak_biomass - target_peak_biomass) / target_peak_biomass <= peak_biomass_rel_tol)
    print(f"  Final peak biomass: {sim_peak_biomass:.2f} g/m²  "
          f"(target: {target_peak_biomass:.2f} ± {peak_biomass_rel_tol*100:.0f}%)  "
          f"{'✓ OK' if peak_biomass_matched else '✗ Not matched'}")
    print()

    if not fully_converged:
        print("  ✗ Peak timing was NOT fully matched. The returned parameters are the "
              "best set found (closest peak age with correct peak height and LAI). "
              "Consider revising targets or adjusting additional parameters.")
    else:
        print("  ✓ Subphase 1.2 calibration complete.")
    print("=" * 70)

    return {"MaxFracFol": current_MaxFracFol,
            "FracFolShape": current_FracFolShape,
            "FrActWd": current_FrActWd,
            "FracFol": current_FracFol,
            "TOWood": current_TOWood,
            "TORoot": current_TORoot,
            "FolN": current_FolN}



import copy
import json

def calibrate_subphase_1_3(
    species,
    target_peak_time,
    target_peak_biomass,
    peak_time_tolerance,
    target_LAI,
    dictOfInitialCoreSpeciesParameters,
    dictOfInitialPnETSpeciesParameters,
    dictOfInitialPnETGenericParameters,
    DictOfBounds,
    duration=300,
    climate="mild",
    soil="SILO"
):
    # ------------------------------------------------------------------ #
    # Helper: deep-copy parameter dicts to avoid mutating originals
    # ------------------------------------------------------------------ #
    def get_param_copies():
        return (
            copy.deepcopy(dictOfInitialCoreSpeciesParameters),
            copy.deepcopy(dictOfInitialPnETSpeciesParameters),
            copy.deepcopy(dictOfInitialPnETGenericParameters)
        )

    # ------------------------------------------------------------------ #
    # Helper: run simulation and return outputs as floats
    # ------------------------------------------------------------------ #
    def run_sim(core_params, pnet_species_params, generic_params, plottingResults = False):
        # print(core_params)
        # print(pnet_species_params)
        # print(generic_params)
        results = calibrationSimulationMonoculturemanawan(
            duration=duration,
            climate=climate,
            soil=soil,
            speciesToSimulate=species,
            dictOfInitialCoreSpeciesParameters=core_params,
            dictOfInitialPnETSpeciesParameters=pnet_species_params,
            dictOfInitialPnETGenericParameters=generic_params,
            plotResults = plottingResults
        )
        return results
        # Ensure all returned values are floats
        # return {k: float(v) for k, v in results.items()}

    # ------------------------------------------------------------------ #
    # Helper: set a PnET species parameter
    # ------------------------------------------------------------------ #
    def set_pnet_species_param(pnet_params, param_name, value):
        pnet_params["PnETSpeciesParameters"][species][param_name] = float(value)

    # ------------------------------------------------------------------ #
    # Helper: get a PnET species parameter
    # ------------------------------------------------------------------ #
    def get_pnet_species_param(pnet_params, param_name):
        if param_name in pnet_params["PnETSpeciesParameters"][species].keys():
            return float(pnet_params["PnETSpeciesParameters"][species][param_name])
        else:
            return("Parameter is not in dictionnary")

    # ------------------------------------------------------------------ #
    # Helper: check bounds
    # ------------------------------------------------------------------ #
    def check_bounds(param_name, value):
        lower = float(DictOfBounds[species][param_name]["lower"])
        upper = float(DictOfBounds[species][param_name]["upper"])
        if value < lower or value > upper:
            return False, lower, upper
        return True, lower, upper

    # ------------------------------------------------------------------ #
    # Helper: Change FracFolShape depending on MaxFracFol
    # ------------------------------------------------------------------ #
    
    # FracFolShape coupling constants
    MAXFRACFOL_THRESHOLD  = 0.05
    MAXFRACFOL_CEILING    = 0.141
    FRACFOLSHAPE_LOW      = 8.0
    FRACFOLSHAPE_HIGH     = 16.0
    
    def coupled_FracFolShape(MaxFracFol):
        """Return FracFolShape with a fast-rise/slow-approach curve coupled to MaxFracFol."""
        MaxFracFol = float(MaxFracFol)
        if MaxFracFol <= MAXFRACFOL_THRESHOLD:
            return float(FRACFOLSHAPE_LOW)
        else:
            # Normalise position in [0, 1] across the active range
            t = (MaxFracFol - MAXFRACFOL_THRESHOLD) / (MAXFRACFOL_CEILING - MAXFRACFOL_THRESHOLD)
            t = min(t, 1.0)  # clamp before applying curve
    
            # Reverse exponential: f(t) = (1 - e^(-k*t)) / (1 - e^(-k)), maps [0,1] -> [0,1]
            # Higher k = faster early rise, slower approach to ceiling
            k = 5.0
            curved_t = (1.0 - math.exp(-k * t)) / (1.0 - math.exp(-k))
    
            value = FRACFOLSHAPE_LOW + curved_t * (FRACFOLSHAPE_HIGH - FRACFOLSHAPE_LOW)
            return float(min(max(value, FRACFOLSHAPE_LOW), FRACFOLSHAPE_HIGH))

    # ------------------------------------------------------------------ #
    # Helper: Change FracFolN depending on MaxFolN
    # ------------------------------------------------------------------ #
    
    # FracFolShape coupling constants
    MAXFOLN_THRESHOLD  = 1
    MAXFOLN_CEILING    = 15
    FOLNSHAPE_LOW      = 6.0
    FOLNSHAPE_HIGH     = 50.0
    
    def coupled_FolNShape(MaxFolN):
        """Return FracFolN with a fast-rise/slow-approach curve coupled to MaxFolN."""
        MaxFolN = float(MaxFolN)
        if MaxFolN <= MAXFOLN_THRESHOLD:
            return float(FOLNSHAPE_LOW)
        else:
            # Normalise position in [0, 1] across the active range
            t = (MaxFolN - MAXFOLN_THRESHOLD) / (MAXFOLN_CEILING - MAXFOLN_THRESHOLD)
            t = min(t, 1.0)  # clamp before applying curve
    
            # Reverse exponential: f(t) = (1 - e^(-k*t)) / (1 - e^(-k)), maps [0,1] -> [0,1]
            # Higher k = faster early rise, slower approach to ceiling
            k = 5.0
            curved_t = (1.0 - math.exp(-k * t)) / (1.0 - math.exp(-k))
    
            value = FOLNSHAPE_LOW + curved_t * (FOLNSHAPE_HIGH - FOLNSHAPE_LOW)
            return float(min(max(value, FOLNSHAPE_LOW), FOLNSHAPE_HIGH))
    
    # ------------------------------------------------------------------ #
    # STEP 1 — Baseline simulation
    # ------------------------------------------------------------------ #
    print(f"\n{'='*60}")
    print(f"Subphase 1.3 — Accelerating early growth for: {species}")
    print(f"{'='*60}")
    print("\n[Step 1] Running baseline simulation...")

    core_p, pnet_sp, gen_p = get_param_copies()

    # We put the longevity very far to avoid confusions with the effect of age
    for speciesInDict in core_p.keys():
        if speciesInDict != "LandisData":
            core_p[speciesInDict]["Longevity"]="999"

    results = run_sim(core_p, pnet_sp, gen_p)

    biomass_at_50pct = results["Biomass at 50% of biomass peak 95% time"]
    peak_biomass     = results["Biomass peak height"]
    peak_time        = results["Biomass peak 95% time"]
    threshold        = 0.3 * float(target_peak_biomass)

    print(f"  Biomass at 50% of peak time : {biomass_at_50pct:.4f} g/m²")
    print(f"  Peak biomass                : {peak_biomass:.4f} g/m²")
    print(f"  Peak time                   : {peak_time:.1f} years")
    print(f"  30% threshold (target)      : {threshold:.4f} g/m²")

    # ------------------------------------------------------------------ #
    # STEP 2 — Check if calibration is needed
    # ------------------------------------------------------------------ #
    if biomass_at_50pct >= threshold:
        print("\n[PASS] Species already reaches ≥30% of peak biomass at 50% of peak time.")
        print("No calibration needed for Subphase 1.3. Returning current parameter values.")
    
        current_params = {
            "MaxFracFol"  : float(get_pnet_species_param(pnet_sp, "MaxFracFol")),
            "FracFolShape": float(get_pnet_species_param(pnet_sp, "FracFolShape")),
            "FolN"        : float(get_pnet_species_param(pnet_sp, "FolN")),
            "FracFol"        : float(get_pnet_species_param(pnet_sp, "FracFol")),
            "TOWood"      : float(get_pnet_species_param(pnet_sp, "TOWood")),
            "TORoot"      : float(get_pnet_species_param(pnet_sp, "TORoot")),
        }

        if get_pnet_species_param(pnet_sp, "MaxFolN") != "Parameter is not in dictionnary":
            current_params["MaxFolN"] = float(get_pnet_species_param(pnet_sp, "MaxFolN"))
        else:
            current_params["MaxFolN"] = "Parameter is not in dictionnary"
            
        if get_pnet_species_param(pnet_sp, "FolNShape") != "Parameter is not in dictionnary":
            current_params["FolNShape"] = float(get_pnet_species_param(pnet_sp, "FolNShape"))
        else:
            current_params["FolNShape"] = "Parameter is not in dictionnary"
            
        return current_params

    # Track which parameters were changed
    changed_params = {}

    # ------------------------------------------------------------------ #
    # STAGE A — Increase MaxFracFol + FracFolShape
    # ------------------------------------------------------------------ #
    print("\n--- Stage A: Adjusting MaxFracFol + FracFolShape ---")

    MaxFracFol_current = get_pnet_species_param(pnet_sp, "MaxFracFol")
    MaxFracFol_max     = 0.141

    while biomass_at_50pct < threshold and MaxFracFol_current < MaxFracFol_max:
        MaxFracFol_current = min(round(MaxFracFol_current + 0.01, 6), MaxFracFol_max)
        FracFolShape_current = float(coupled_FracFolShape(MaxFracFol_current))

        set_pnet_species_param(pnet_sp, "MaxFracFol",   MaxFracFol_current)
        set_pnet_species_param(pnet_sp, "FracFolShape", FracFolShape_current)

        results = run_sim(core_p, pnet_sp, gen_p)
        biomass_at_50pct = results["Biomass at 50% of biomass peak 95% time"]
        peak_biomass     = results["Biomass peak height"]
        peak_time        = results["Biomass peak 95% time"]
        threshold        = 0.30 * float(target_peak_biomass)

        print(f"  MaxFracFol={MaxFracFol_current:.4f}, FracFolShape={FracFolShape_current:.4f} "
              f"→ Biomass@50%={biomass_at_50pct:.4f} (threshold={threshold:.4f})")

    changed_params["MaxFracFol"]   = MaxFracFol_current
    changed_params["FracFolShape"] = float(get_pnet_species_param(pnet_sp, "FracFolShape"))

    # ------------------------------------------------------------------ #
    # STAGE B — Increase MaxFolN + FolNShape
    # ------------------------------------------------------------------ #
    if biomass_at_50pct < threshold:
        print("\n--- Stage B: Adjusting MaxFolN + FolNShape ---")

        MaxFolN_current = get_pnet_species_param(pnet_sp, "FolN")
        MaxFolN_max     = 15.0

        while biomass_at_50pct < threshold and MaxFolN_current < MaxFolN_max:
            MaxFolN_current = min(round(MaxFolN_current + 1.0, 6), MaxFolN_max)
            FolNShape_current = float(coupled_FolNShape(MaxFolN_current))

            set_pnet_species_param(pnet_sp, "MaxFolN",   MaxFolN_current)
            set_pnet_species_param(pnet_sp, "FolNShape", FolNShape_current)

            results = run_sim(core_p, pnet_sp, gen_p)
            biomass_at_50pct = results["Biomass at 50% of biomass peak 95% time"]
            peak_biomass     = results["Biomass peak height"]
            peak_time        = results["Biomass peak 95% time"]
            # Threshold stays anchored to the original target peak biomass
            threshold = 0.25 * float(target_peak_biomass)

            print(f"  MaxFolN={MaxFolN_current:.1f}, FolNShape={FolNShape_current:.4f} "
                  f"→ Biomass@50%={biomass_at_50pct:.4f} (threshold={threshold:.4f})")

        changed_params["MaxFolN"]   = MaxFolN_current
        changed_params["FolNShape"] = float(get_pnet_species_param(pnet_sp, "FolNShape"))

        # --- Check if Stage B succeeded ---
        if biomass_at_50pct < threshold:
            print("\n[ERROR] Species still does not reach 30% of peak biomass at 50% of peak time "
                  "after maximizing MaxFracFol and MaxFolN.")
            print("  → Please re-check species assumptions, climate files, soil parameters, "
                  "and initial parameter values before proceeding.")
            return {}

    # ------------------------------------------------------------------ #
    # STAGE C — Re-adjust peak timing using FolN
    # ------------------------------------------------------------------ #
    print("\n--- Stage C: Re-adjusting peak timing using FolN ---")

    FolN_current = get_pnet_species_param(pnet_sp, "FolN")
    FolN_min     = 0.5

    results = run_sim(core_p, pnet_sp, gen_p)
    peak_time = results["Biomass peak 95% time"]

    print(f"  Current peak time: {peak_time:.1f} years | "
          f"Target: {float(target_peak_time):.1f} ± {float(peak_time_tolerance):.1f} years")

    while abs(peak_time - float(target_peak_time)) > float(peak_time_tolerance) and FolN_current > FolN_min:
        FolN_current = max(round(FolN_current - 0.1, 6), FolN_min)
        set_pnet_species_param(pnet_sp, "FolN", FolN_current)

        results = run_sim(core_p, pnet_sp, gen_p)
        peak_time    = results["Biomass peak 95% time"]
        peak_biomass = results["Biomass peak height"]

        print(f"  FolN={FolN_current:.2f} → Peak time={peak_time:.1f} years "
              f"(target={float(target_peak_time):.1f} ± {float(peak_time_tolerance):.1f})")

    changed_params["FolN"] = FolN_current

    if abs(peak_time - float(target_peak_time)) > float(peak_time_tolerance):
        print("\n[ERROR] Peak time could not be brought back to the target after reducing FolN to 0.5.")
        print("  → Please re-check assumptions, climate files, and initial parameter values.")
        return {}

    print(f"  [OK] Peak time successfully adjusted to {peak_time:.1f} years.")

    # ------------------------------------------------------------------ #
    # STAGE D — Re-adjust LAI using FracFol
    # ------------------------------------------------------------------ #
    print("\n--- Stage D: Adjusting Maximum LAI using FracFol ---")
    
    results = run_sim(core_p, pnet_sp, gen_p)
    current_lai = results["Maximum LAI"]
    FracFol_current = get_pnet_species_param(pnet_sp, "FracFol")
    
    _, FracFol_lower, FracFol_upper = check_bounds("FracFol", FracFol_current)
    
    print(f"  Current Maximum LAI : {current_lai:.4f} | Target: {float(target_LAI):.4f}")
    print(f"  Current FracFol     : {FracFol_current:.4f} | Bounds: [{FracFol_lower:.4f}, {FracFol_upper:.4f}]")
    
    while not abs(current_lai - float(target_LAI)) < 0.05:
        if current_lai < float(target_LAI):
            new_FracFol = round(FracFol_current + 0.001, 6)
            if new_FracFol > FracFol_upper:
                print(f"  [WARNING] FracFol would exceed upper bound ({FracFol_upper:.4f}). Stopping LAI adjustment.")
                break
        else:
            new_FracFol = round(FracFol_current - 0.001, 6)
            if new_FracFol < FracFol_lower:
                print(f"  [WARNING] FracFol would go below lower bound ({FracFol_lower:.4f}). Stopping LAI adjustment.")
                break
    
        FracFol_current = new_FracFol
        set_pnet_species_param(pnet_sp, "FracFol", FracFol_current)
    
        results = run_sim(core_p, pnet_sp, gen_p)
        current_lai = results["Maximum LAI"]
    
        print(f"  FracFol={FracFol_current:.4f} → Maximum LAI={current_lai:.4f} (target={float(target_LAI):.4f})")
    
    changed_params["FracFol"] = FracFol_current
    print(f"  [OK] LAI adjustment complete. Final Maximum LAI={current_lai:.4f}, FracFol={FracFol_current:.4f}")

    
    # ------------------------------------------------------------------ #
    # STAGE D — Re-adjust peak height using TOWood + TORoot
    # ------------------------------------------------------------------ #
    print("\n--- Stage D: Adjusting peak height using TOWood + TORoot ---")
    
    results = run_sim(core_p, pnet_sp, gen_p)
    peak_biomass = results["Biomass peak height"]
    
    TOWood_current = get_pnet_species_param(pnet_sp, "TOWood")
    TORoot_current = get_pnet_species_param(pnet_sp, "TORoot")
    
    _, TOWood_lower, TOWood_upper = check_bounds("TOWood", TOWood_current)
    _, TORoot_lower, TORoot_upper = check_bounds("TORoot", TORoot_current)
    
    print(f"  Current peak biomass: {peak_biomass:.4f} g/m² | Target: {float(target_peak_biomass):.4f} g/m²")
    print(f"  TOWood bounds: [{TOWood_lower:.4f}, {TOWood_upper:.4f}] | TORoot bounds: [{TORoot_lower:.4f}, {TORoot_upper:.4f}]")
    
    step_size = 0.005
    min_step  = 0.0001
    prev_direction = None  # tracks the last adjustment direction to detect overshoot

    # Tolerance : 500g/m2
    while abs(peak_biomass - float(target_peak_biomass)) > 500.0 and step_size >= min_step:
        direction = -1.0 if peak_biomass < float(target_peak_biomass) else 1.0
    
        # Overshoot detected — halve the step size and reverse
        if prev_direction is not None and direction != prev_direction:
            step_size = round(step_size / 2.0, 8)
            print(f"  [Overshoot detected] Reducing step size to {step_size:.6f}")
    
        new_TOWood = round(TOWood_current + direction * step_size, 8)
        new_TORoot = round(TORoot_current + direction * step_size, 8)
    
    
        TOWood_current = new_TOWood
        TORoot_current = new_TORoot
    
        set_pnet_species_param(pnet_sp, "TOWood", TOWood_current)
        set_pnet_species_param(pnet_sp, "TORoot", TORoot_current)
    
        results = run_sim(core_p, pnet_sp, gen_p)
        peak_biomass = results["Biomass peak height"]
        peak_time    = results["Biomass peak 95% time"]
    
        print(f"  step={step_size:.6f}, TOWood={TOWood_current:.6f}, TORoot={TORoot_current:.6f} "
              f"→ Peak biomass={peak_biomass:.4f} g/m² (target={float(target_peak_biomass):.4f})")
    
        prev_direction = direction
    
    changed_params["TOWood"] = TOWood_current
    changed_params["TORoot"] = TORoot_current
    print(f"  [OK] Peak height adjustment complete. Final peak biomass={peak_biomass:.4f} g/m²")



    # ------------------------------------------------------------------ #
    # FINAL SUMMARY
    # ------------------------------------------------------------------ #
    print(f"\n{'='*60}")
    print(f"CALIBRATION SUMMARY — Subphase 1.3 — {species}")
    print(f"{'='*60}")
    print(f"  Final peak biomass      : {peak_biomass:.4f} g/m²  (target: {float(target_peak_biomass):.4f})")
    print(f"  Final peak time         : {peak_time:.1f} years    (target: {float(target_peak_time):.1f} ± {float(peak_time_tolerance):.1f})")
    print(f"  Biomass at 50% peak time: {biomass_at_50pct:.4f} g/m²  (threshold: {threshold:.4f})")

    print(f"\n  Parameters changed:")
    for param, val in changed_params.items():
        if param in DictOfBounds[species].keys():
            in_bounds, lower, upper = check_bounds(param, val)
            status = "OK" if in_bounds else f"OUT OF BOUNDS [{lower:.6f}, {upper:.6f}]"
            print(f"    {param:20s} = {float(val):.6f}  [{status}]")

    print(f"\n  Parameters outside of bounds:")
    any_oob = False
    for param, val in changed_params.items():
        if param in DictOfBounds[species].keys():
            in_bounds, lower, upper = check_bounds(param, val)
            if not in_bounds:
                any_oob = True
                print(f"    *** {param}: value={float(val):.6f}, bounds=[{lower:.6f}, {upper:.6f}]")
    if not any_oob:
        print("    None — all changed parameters are within empirical bounds.")

    print(f"{'='*60}\n")

    results = run_sim(core_p, pnet_sp, gen_p, True)
    
    return changed_params



import json
import math
import copy

def calibrate_senescence(
    species,
    typical_mortality,
    maximum_mortality,
    target_peak_biomass,
    DictOfBounds,
    dictOfInitialCoreSpeciesParameters,
    dictOfInitialPnETSpeciesParameters,
    dictOfInitialPnETGenericParameters,
    tolerance=3,
    max_iterations=50,
    simulation_duration=500,
    climate="mild",
    soil="SILO"
):
    # --- Ensure targets are floats ---
    typical_mortality  = float(typical_mortality)
    maximum_mortality  = float(maximum_mortality)
    target_peak_biomass = float(target_peak_biomass)
    tolerance          = float(tolerance)

    # --- Load parameter dictionaries ---
    core_params    = copy.deepcopy(dictOfInitialCoreSpeciesParameters)
    pnet_params    = copy.deepcopy(dictOfInitialPnETSpeciesParameters)
    generic_params = copy.deepcopy(dictOfInitialPnETGenericParameters)

    # --- Retrieve bounds for PsnAgeRed, TOWood, TORoot ---
    psn_age_red_lower = float(DictOfBounds[species]["PsnAgeRed"]["lower"])
    psn_age_red_upper = float(DictOfBounds[species]["PsnAgeRed"]["upper"])
    TOWood_lower      = float(DictOfBounds[species]["TOWood"]["lower"])
    TOWood_upper      = float(DictOfBounds[species]["TOWood"]["upper"])
    TORoot_lower      = float(DictOfBounds[species]["TORoot"]["lower"])
    TORoot_upper      = float(DictOfBounds[species]["TORoot"]["upper"])

    # -------------------------------------------------------------------------
    # Helper functions
    # -------------------------------------------------------------------------
    def compute_PsnAgeRed(Longevity, typ_mort):
        ratio = typ_mort / Longevity
        if ratio <= 0.0 or ratio >= 1.0:
            raise ValueError(
                f"Cannot compute PsnAgeRed: typical_mortality/Longevity = {ratio:.4f} "
                f"must be strictly between 0 and 1."
            )
        return (-1 / math.log10(typical_mortality / Longevity))

    def run_sim(core_p, pnet_p, gen_p, plot=False):
        return calibrationSimulationMonoculturemanawan(
            duration=simulation_duration,
            climate=climate,
            soil=soil,
            speciesToSimulate=species,
            dictOfInitialCoreSpeciesParameters=copy.deepcopy(core_p),
            dictOfInitialPnETSpeciesParameters=copy.deepcopy(pnet_p),
            dictOfInitialPnETGenericParameters=copy.deepcopy(gen_p),
            plotResults=plot
        )

    # -------------------------------------------------------------------------
    # STEP 1 – Analytical initialization
    # -------------------------------------------------------------------------
    if maximum_mortality <= typical_mortality:
        raise ValueError("maximum_mortality must be strictly greater than typical_mortality.")

    ratio_mort        = maximum_mortality / typical_mortality
    initial_PsnAgeRed = math.log(4.0) / math.log(ratio_mort)
    initial_Longevity = typical_mortality / (0.1 ** (1.0 / initial_PsnAgeRed))

    # Clamp only PsnAgeRed to bounds
    initial_PsnAgeRed = max(psn_age_red_lower, min(psn_age_red_upper, initial_PsnAgeRed))

    print(f"\n{'='*60}")
    print(f"Subphase 1.4 – Senescence calibration for {species}")
    print(f"{'='*60}")
    print(f"  Target typical mortality (initiation of decline) : {typical_mortality:.1f} years")
    print(f"  Target maximum mortality (time of death)         : {maximum_mortality:.1f} years")
    print(f"  Target peak biomass                              : {target_peak_biomass:.2f} g/m²")
    print(f"  Tolerance                                        : ±{tolerance:.0f} years")
    print(f"\n  [Init] PsnAgeRed = {initial_PsnAgeRed:.4f}")
    print(f"  [Init] Longevity  = {initial_Longevity:.2f} years")
    print(f"  [Init] Analytical initiation of decline = "
          f"{initial_Longevity * (0.1 ** (1.0 / initial_PsnAgeRed)):.2f} years "
          f"(target: {typical_mortality:.1f})")

    # -------------------------------------------------------------------------
    # STEP 2 – Iterative correction loop for senescence
    # -------------------------------------------------------------------------
    current_PsnAgeRed = initial_PsnAgeRed
    current_Longevity = initial_Longevity

    converged             = False
    time_of_death         = None
    initiation_of_decline = None

    for iteration in range(1, max_iterations + 1):

        pnet_params["PnETSpeciesParameters"][species]["PsnAgeRed"] = float(current_PsnAgeRed)
        if "PsnAgeRed" in generic_params.keys(): del generic_params["PsnAgeRed"] # Since we are changing PsnAgeRed by species, we remove it as a generic parameters.
        core_params[species]["Longevity"] = str(int(round(current_Longevity)))

        print(f"\n  [Iter {iteration}] PsnAgeRed = {current_PsnAgeRed:.4f} | "
              f"Longevity = {current_Longevity:.2f}")

        results = run_sim(core_params, pnet_params, generic_params)

        time_of_death         = float(results["Time of death"])
        initiation_of_decline = float(results["Initation of decline"])

        print(f"           → Simulated time of death         : {time_of_death:.1f} years "
              f"(target: {maximum_mortality:.1f})")
        print(f"           → Simulated initiation of decline : {initiation_of_decline:.1f} years "
              f"(target: {typical_mortality:.1f})")

        death_error   = time_of_death - maximum_mortality
        decline_error = initiation_of_decline - typical_mortality

        death_ok   = abs(death_error)   <= tolerance
        decline_ok = abs(decline_error) <= tolerance

        if death_ok and decline_ok:
            print(f"\n  ✓ Senescence convergence reached at iteration {iteration}!")
            converged = True
            break

        # Adjust Longevity (no bounds clamping)
        longevity_adjustment = abs(death_error) / 2.0
        if death_error < 0:
            new_Longevity = current_Longevity + longevity_adjustment
        else:
            new_Longevity = current_Longevity - longevity_adjustment

        # Recompute PsnAgeRed from updated Longevity, clamp to bounds
        try:
            new_PsnAgeRed = compute_PsnAgeRed(new_Longevity, typical_mortality)
        except ValueError as e:
            print(f"  ✗ Could not recompute PsnAgeRed: {e}")
            print("    Stopping iteration.")
            break

        # new_PsnAgeRed = max(psn_age_red_lower, min(psn_age_red_upper, new_PsnAgeRed))

        print(f"           → Longevity adjustment: "
              f"{'+' if death_error < 0 else '-'}{longevity_adjustment:.2f} "
              f"→ new Longevity = {new_Longevity:.2f}")
        print(f"           → Recomputed PsnAgeRed = {new_PsnAgeRed:.4f}")

        current_Longevity = new_Longevity
        current_PsnAgeRed = new_PsnAgeRed

    if not converged:
        print(f"\n  ⚠ Maximum iterations ({max_iterations}) reached without full convergence.")
        print(f"    Last time of death         : {time_of_death:.1f} (target: {maximum_mortality:.1f})")
        print(f"    Last initiation of decline : {initiation_of_decline:.1f} (target: {typical_mortality:.1f})")

    # -------------------------------------------------------------------------
    # STEP 2.5 – Re-adjust peak biomass using TOWood and TORoot
    # -------------------------------------------------------------------------
    print(f"\n--- Step 2.5: Adjusting peak biomass using TOWood + TORoot ---")

    results      = run_sim(core_params, pnet_params, generic_params)
    peak_biomass = float(results["Biomass peak height"])

    TOWood_current = float(pnet_params["PnETSpeciesParameters"][species]["TOWood"])
    TORoot_current = float(pnet_params["PnETSpeciesParameters"][species]["TORoot"])

    print(f"  Current peak biomass : {peak_biomass:.4f} g/m² | Target: {target_peak_biomass:.4f} g/m²")
    print(f"  TOWood bounds        : [{TOWood_lower:.4f}, {TOWood_upper:.4f}]")
    print(f"  TORoot bounds        : [{TORoot_lower:.4f}, {TORoot_upper:.4f}]")

    step_size      = 0.005
    min_step       = 0.0001
    prev_direction = None

    # Tolerance: 500 g/m²
    while abs(peak_biomass - target_peak_biomass) > 500.0 and step_size >= min_step:
        direction = -1.0 if peak_biomass < target_peak_biomass else 1.0

        # Overshoot detected — halve the step size
        if prev_direction is not None and direction != prev_direction:
            step_size = round(step_size / 2.0, 8)
            print(f"  [Overshoot detected] Reducing step size to {step_size:.6f}")

        new_TOWood = round(TOWood_current + direction * step_size, 8)
        new_TORoot = round(TORoot_current + direction * step_size, 8)

        # Clamp to bounds
        # new_TOWood = max(0, min(TOWood_upper, new_TOWood))
        # new_TORoot = max(0, min(TORoot_upper, new_TORoot))

        TOWood_current = new_TOWood
        TORoot_current = new_TORoot

        pnet_params["PnETSpeciesParameters"][species]["TOWood"] = float(TOWood_current)
        pnet_params["PnETSpeciesParameters"][species]["TORoot"] = float(TORoot_current)

        results      = run_sim(core_params, pnet_params, generic_params)
        peak_biomass = float(results["Biomass peak height"])

        print(f"  step={step_size:.6f}, TOWood={TOWood_current:.6f}, TORoot={TORoot_current:.6f} "
              f"→ Peak biomass={peak_biomass:.4f} g/m² (target={target_peak_biomass:.4f})")

        prev_direction = direction

    print(f"  [OK] Peak biomass adjustment complete. Final peak biomass = {peak_biomass:.4f} g/m²")

    # -------------------------------------------------------------------------
    # STEP 3 – Final summary and plot
    # -------------------------------------------------------------------------
    final_PsnAgeRed = float(current_PsnAgeRed)
    final_Longevity = float(current_Longevity)

    analytical_decline_final = final_Longevity * (0.1 ** (1.0 / final_PsnAgeRed))
    analytical_death_fAge06  = final_Longevity * (0.4 ** (1.0 / final_PsnAgeRed))

    print(f"\n{'='*60}")
    print(f"  FINAL CALIBRATED PARAMETERS for {species}")
    print(f"{'='*60}")
    print(f"  PsnAgeRed : {final_PsnAgeRed:.4f}")
    print(f"  Longevity : {final_Longevity:.2f} years  "
          f"(stored as integer: {int(round(final_Longevity))})")
    print(f"  TOWood    : {TOWood_current:.6f}")
    print(f"  TORoot    : {TORoot_current:.6f}")
    print(f"\n  Analytical initiation of decline (fAge=0.90) : "
          f"{analytical_decline_final:.2f} years  (target: {typical_mortality:.1f})")
    print(f"  Analytical fAge=0.60 threshold               : "
          f"{analytical_death_fAge06:.2f} years  (target: {maximum_mortality:.1f})")
    print(f"\n  Simulated initiation of decline : {initiation_of_decline:.1f} years")
    print(f"  Simulated time of death         : {time_of_death:.1f} years")
    print(f"  Final peak biomass              : {peak_biomass:.4f} g/m²  "
          f"(target: {target_peak_biomass:.4f} g/m²)")

    # Final simulation with plot
    print(f"\n  Running final simulation with plot...")
    run_sim(core_params, pnet_params, generic_params, plot=True)

    return {
        "PsnAgeRed": final_PsnAgeRed,
        "Longevity": int(round(final_Longevity)),
        "TOWood": float(TOWood_current),
        "TORoot": float(TORoot_current),
        "simulated_initiation_of_decline": float(initiation_of_decline),
        "simulated_time_of_death": float(time_of_death),
        "final_peak_biomass": float(peak_biomass),
        "converged": converged
    }



import pandas as pd
import matplotlib.pyplot as plt

def plot_growth_curves(
    nfi_gam_path: str,
    pnet_growth_path: str,
    species: str,
    pnet_growth_path_initial: str | None = None
) -> None:
    """
    Plot and compare tree species growth curves from NFI_GAM observations
    and PnET_Growth model predictions.

    Args:
        nfi_gam_path: Path to the NFI_GAM .csv file.
        pnet_growth_path: Path to the calibrated PnET_Growth .csv file.
        species: Name of the tree species to display in the plot.
        pnet_growth_path_initial: Optional path to the uncalibrated PnET_Growth .csv file.
    """
    nfi_df = pd.read_csv(nfi_gam_path)
    pnet_df = pd.read_csv(pnet_growth_path)

    nfi_df = nfi_df.sort_values("Age")
    pnet_df = pnet_df.sort_values("Time")

    # Load optional initial (uncalibrated) PnET data
    pnet_initial_df = None
    if pnet_growth_path_initial is not None:
        pnet_initial_df = pd.read_csv(pnet_growth_path_initial).sort_values("Time")

    # Compute y-axis limits based on global maximum across all loaded curves
    all_maxima = [
        nfi_df["GAM_Prediction_100%Abundance"].max(),
        pnet_df["AllSpp_g/m2"].max()
    ]
    if pnet_initial_df is not None:
        all_maxima.append(pnet_initial_df["AllSpp_g/m2"].max())

    global_max = max(all_maxima)
    y_min = -0.10 * global_max
    y_max =  1.10 * global_max

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(
        nfi_df["Age"],
        nfi_df["GAM_Prediction_100%Abundance"],
        color="#2196F3",
        linewidth=2,
        label=f"{species} — NFI GAM (Observations)"
    )
    ax.plot(
        pnet_df["Time"],
        pnet_df["AllSpp_g/m2"],
        color="#F44336",
        linewidth=2,
        label=f"{species} — PnET Growth (Calibrated)"
    )
    if pnet_initial_df is not None:
        ax.plot(
            pnet_initial_df["Time"],
            pnet_initial_df["AllSpp_g/m2"],
            color="#FF9800",
            linewidth=2,
            linestyle="--",
            label=f"{species} — PnET Growth (Uncalibrated)"
        )

    ax.set_ylim(y_min, y_max)
    ax.set_xlabel("Age (years)", fontsize=12)
    ax.set_ylabel("Biomass (g/m²)", fontsize=12)
    ax.set_title(f"{species} Growth Curves: Observations vs. Model", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.show()



def plot_multispecies_growth_curves(
    nfi_gam_paths: list[str],
    pnet_growth_paths: list[str],
    species_names: list[str]
) -> None:
    """
    Plot NFI GAM and PnET Growth curves for multiple species on two separate
    plots sharing the same x and y axis ranges.

    Args:
        nfi_gam_paths: List of paths to NFI_GAM .csv files, one per species.
        pnet_growth_paths: List of paths to PnET_Growth .csv files, one per species.
        species_names: List of species names, one per species.
    """
    if not (len(nfi_gam_paths) == len(pnet_growth_paths) == len(species_names)):
        raise ValueError("All three lists must have the same number of items.")

    # Load all dataframes
    nfi_dfs  = [pd.read_csv(p).sort_values("Age")  for p in nfi_gam_paths]
    pnet_dfs = [pd.read_csv(p).sort_values("Time") for p in pnet_growth_paths]

    # Compute shared axis limits across all species and both sources
    global_x_max = max(
        max(df["Age"].max()  for df in nfi_dfs),
        max(df["Time"].max() for df in pnet_dfs)
    )
    global_y_max = max(
        max(df["GAM_Prediction_100%Abundance"].max() for df in nfi_dfs),
        max(df["AllSpp_g/m2"].max()                 for df in pnet_dfs)
    )
    x_min, x_max = 0, global_x_max
    y_min, y_max = -0.10 * global_y_max, 1.10 * global_y_max

    # Assign one color per species
    colors = cm.tab10(np.linspace(0, 1, len(species_names)))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))

    for nfi_df, pnet_df, species, color in zip(nfi_dfs, pnet_dfs, species_names, colors):
        ax1.plot(
            nfi_df["Age"],
            nfi_df["GAM_Prediction_100%Abundance"],
            color=color,
            linewidth=2,
            label=species
        )
        ax2.plot(
            pnet_df["Time"],
            pnet_df["AllSpp_g/m2"],
            color=color,
            linewidth=2,
            label=species
        )

    for ax, title in zip(
        (ax1, ax2),
        ("NFI GAM — Observations", "PnET Growth — Model")
    ):
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_xlabel("Age (years)", fontsize=12)
        ax.set_ylabel("Biomass (g/m²)", fontsize=12)
        ax.set_title(title, fontsize=14)
        ax.legend(fontsize=10)
        ax.grid(True, linestyle="--", alpha=0.5)

    fig.suptitle("Multi-Species Growth Curves: Observations vs. Model", fontsize=15, fontweight="bold")
    plt.tight_layout()
    plt.show()



## FUNCTIONS FOR DERIVING PARAMETERS THROUGH REGRESSION WITH PUBLISHED STUDIES
# Drought and waterlogging

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pygam import LinearGAM, s

def predict_H_parameters(
    drought_tolerance: float,
    waterlogging_tolerance: float,
    plot: bool = False,
    gam_summary: bool = False,
    csv_path: str = "./SpeciesParametersSets/PreviousStudies/GustafsonParameters_UserGuidev5.1_WithDroughtAndWtTolerance.csv"
) -> tuple[float, float, float, float]:
    """
    Predicts H1, H2, H3 and H4 parameter values using GAMs.
    - H1, H2 are predicted from Waterlogging Tolerance
    - H3, H4 are predicted from Drought Tolerance

    Parameters
    ----------
    drought_tolerance : float
        Drought tolerance score to predict H3 and H4 for (typically 1–5).
    waterlogging_tolerance : float
        Waterlogging tolerance score to predict H1 and H2 for (typically 1–5).
    plot : bool, optional
        If True, plots all 4 GAM fits and marks the predicted values. Default False.
    gam_summary : bool, optional
        If True, prints GAM summary and performance metrics for all 4 models. Default False.
    csv_path : str, optional
        Path to the input CSV file.

    Returns
    -------
    tuple[float, float, float, float]
        Predicted (H1, H2, H3, H4) values.
    """

    def _clean_data(df: pd.DataFrame, predictor_col: str, response_cols: list[str]) -> tuple[np.ndarray, list[np.ndarray]]:
        """Filter, clean and return X and y arrays for a given predictor and response columns."""
        subset = df.copy()
        subset = subset[subset[predictor_col].astype(str).str.strip().str.lower() != "not found"]
        subset = subset.dropna(subset=[predictor_col] + response_cols)
        subset[predictor_col] = pd.to_numeric(subset[predictor_col], errors="coerce")
        for col in response_cols:
            subset[col] = pd.to_numeric(subset[col], errors="coerce")
        subset = subset.dropna(subset=[predictor_col] + response_cols)
        X = subset[predictor_col].values.reshape(-1, 1)
        ys = [subset[col].values for col in response_cols]
        return X, ys

    def _print_summary(label: str, predictor_label: str, gam: LinearGAM, X: np.ndarray, y: np.ndarray):
        """Print pygam native summary + performance metrics."""
        y_pred = gam.predict(X)
        residuals = y - y_pred
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2 = 1 - ss_res / ss_tot
        rmse = np.sqrt(np.mean(residuals ** 2))
        mae = np.mean(np.abs(residuals))
        n = len(y)

        print("=" * 55)
        print(f"  GAM Summary — {label} = s({predictor_label})")
        print("=" * 55)
        gam.summary()
        print(f"\n  📊 Performance Metrics ({n} observations)")
        print(f"  {'R²':<30} {r2:.4f}")
        print(f"  {'RMSE':<30} {rmse:.4f}")
        print(f"  {'MAE':<30} {mae:.4f}")
        print(f"  {'Residual Std Dev':<30} {np.std(residuals):.4f}")
        print(f"  {'Residual Min':<30} {residuals.min():.4f}")
        print(f"  {'Residual Max':<30} {residuals.max():.4f}")
        print("=" * 55)
        print()

    def _plot_gam(ax, X: np.ndarray, y: np.ndarray, gam: LinearGAM,
                  param_label: str, predictor_label: str,
                  pred_x: float, pred_y: float):
        """Plot a single GAM fit with data points, CI, and predicted value."""
        X_range = np.linspace(1, 5, 200).reshape(-1, 1)
        curve = gam.predict(X_range)
        ci = gam.prediction_intervals(X_range, width=0.95)

        ax.scatter(X.flatten(), y, alpha=0.6, edgecolors="steelblue",
                   facecolors="lightblue", zorder=3, label="Observed data")
        ax.plot(X_range.flatten(), curve, color="steelblue",
                linewidth=2, label="GAM fit")
        ax.fill_between(X_range.flatten(), ci[:, 0], ci[:, 1],
                        alpha=0.2, color="steelblue", label="95% CI")
        ax.axvline(pred_x, color="tomato", linestyle="--", linewidth=1.2, alpha=0.7)
        ax.scatter([pred_x], [pred_y], color="tomato", zorder=5, s=80,
                   label=f"Predicted {param_label} = {pred_y:.4f}")
        ax.set_xlabel(predictor_label, fontsize=11)
        ax.set_ylabel(param_label, fontsize=11)
        ax.set_title(f"{param_label} = s({predictor_label})", fontsize=12)
        ax.set_xlim(1, 5)
        ax.legend(fontsize=9)
        ax.grid(True, linestyle="--", alpha=0.4)

    # --- Load data ---
    df = pd.read_csv(csv_path)

    # --- Clean data for each predictor ---
    X_drought, (y_H3, y_H4) = _clean_data(df, "Drought Tolerance", ["H3", "H4"])
    X_wlog,    (y_H1, y_H2) = _clean_data(df, "Waterlogging Tolerance", ["H1", "H2"])

    # --- Fit GAMs ---
    gam_H1 = LinearGAM(s(0)).gridsearch(X_wlog,   y_H1, progress=False)
    gam_H2 = LinearGAM(s(0)).gridsearch(X_wlog,   y_H2, progress=False)
    gam_H3 = LinearGAM(s(0)).gridsearch(X_drought, y_H3, progress=False)
    gam_H4 = LinearGAM(s(0)).gridsearch(X_drought, y_H4, progress=False)

    # --- GAM Summaries ---
    if gam_summary:
        _print_summary("H1", "Waterlogging Tolerance", gam_H1, X_wlog,    y_H1)
        _print_summary("H2", "Waterlogging Tolerance", gam_H2, X_wlog,    y_H2)
        _print_summary("H3", "Drought Tolerance",      gam_H3, X_drought, y_H3)
        _print_summary("H4", "Drought Tolerance",      gam_H4, X_drought, y_H4)

    # --- Predict ---
    pred_H1 = float(gam_H1.predict([[waterlogging_tolerance]])[0])
    pred_H2 = float(gam_H2.predict([[waterlogging_tolerance]])[0])
    pred_H3 = float(gam_H3.predict([[drought_tolerance]])[0])
    pred_H4 = float(gam_H4.predict([[drought_tolerance]])[0])

    # --- Plot ---
    if plot:
        fig, axes = plt.subplots(1, 4, figsize=(22, 5))
        fig.suptitle("GAM Regression: H1–H4 vs Tolerance Scores", fontsize=14)

        _plot_gam(axes[0], X_wlog,    y_H1, gam_H1, "H1", "Waterlogging Tolerance", waterlogging_tolerance, pred_H1)
        _plot_gam(axes[1], X_wlog,    y_H2, gam_H2, "H2", "Waterlogging Tolerance", waterlogging_tolerance, pred_H2)
        _plot_gam(axes[2], X_drought, y_H3, gam_H3, "H3", "Drought Tolerance",      drought_tolerance,      pred_H3)
        _plot_gam(axes[3], X_drought, y_H4, gam_H4, "H4", "Drought Tolerance",      drought_tolerance,      pred_H4)

        plt.tight_layout()
        plt.show()

    return {"H1":round(pred_H1, 2), "H2":round(pred_H2, 2), "H3":round(pred_H3, 2), "H4":round(pred_H4, 2)}




# TEMPERATURE PARAMETERS

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pygam import LinearGAM, s

def predict_temperature_parameters(species_map, return_df=True, plotAndSummaries = True):
    # ─────────────────────────────────────────────
    # Hardcoded CSV paths
    # ─────────────────────────────────────────────
    CSV1_PATH = "./SpeciesParametersSets/PreviousStudies/GustafsonParameters_UserGuidev5.1_WithDroughtAndWtTolerance.csv"   # ← your first CSV
    CSV2_PATH = "./ReferencesAndData/Others/DendonckerEtAl2026_Average_niche_all_spp.csv"       # ← your second CSV
    
    # ─────────────────────────────────────────────
    # 1. Load & merge data
    # ─────────────────────────────────────────────
    def load_and_merge() -> pd.DataFrame:
        df1 = pd.read_csv(CSV1_PATH)
        df2 = pd.read_csv(CSV2_PATH)
        df1 = df1.rename(columns={'Species': 'species'})
        df = df1.merge(df2, on="species", how="left", suffixes=("", "_df2"))
        return df
    
    
    # ─────────────────────────────────────────────
    # 2. Helpers: fit GAM, plot, summarise
    # ─────────────────────────────────────────────
    def fit_plot_gam(df: pd.DataFrame, formula_str: str, y_col: str,
                     x_cols: list[str]) -> LinearGAM:
        sub = df[[y_col] + x_cols].dropna()
        y = sub[y_col].values
        X = sub[x_cols].values
    
        n = len(x_cols)
        if n == 1:
            gam = LinearGAM(s(0))
        elif n == 2:
            gam = LinearGAM(s(0) + s(1))
        else:
            gam = LinearGAM(s(0) + s(1) + s(2))
    
        gam.fit(X, y)
    
        print(f"\n{'='*60}")
        print(f"Model: {formula_str}")
        print(f"{'='*60}")
        if plotAndSummaries:
            gam.summary()
        
            if n == 1:
                _plot_1d(gam, sub, y_col, x_cols[0], formula_str)
            else:
                _plot_partial(gam, sub, y_col, x_cols, formula_str)
    
        return gam
    
    
    def _plot_1d(gam: LinearGAM, sub: pd.DataFrame, y_col: str,
                 x_col: str, title: str) -> None:
        fig, ax = plt.subplots(figsize=(7, 4))
        XX = gam.generate_X_grid(term=0, n=200)
        preds = gam.predict(XX)
        ci = gam.prediction_intervals(XX, width=0.95)
    
        ax.scatter(sub[x_col], sub[y_col], color="steelblue", alpha=0.7,
                   edgecolors="white", s=60, label="Observed", zorder=3)
        ax.plot(XX[:, 0], preds, color="tomato", lw=2, label="GAM prediction")
        ax.fill_between(XX[:, 0], ci[:, 0], ci[:, 1],
                        color="tomato", alpha=0.15, label="95% CI")
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        ax.set_title(title)
        ax.legend()
        plt.tight_layout()
        plt.show()
    
    
    def _plot_partial(gam: LinearGAM, sub: pd.DataFrame, y_col: str,
                      x_cols: list[str], title: str) -> None:
        n = len(x_cols)
        fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
        if n == 1:
            axes = [axes]
    
        y_centered = sub[y_col] - sub[y_col].mean()
    
        for i, (ax, x_col) in enumerate(zip(axes, x_cols)):
            XX = gam.generate_X_grid(term=i, n=200)
            pdep, confi = gam.partial_dependence(term=i, X=XX, width=0.95)
    
            ax.scatter(sub[x_col], y_centered, color="steelblue", alpha=0.6,
                       s=50, edgecolors="white", label="Observed (centred)", zorder=3)
            ax.plot(XX[:, i], pdep, color="tomato", lw=2, label="Partial effect")
            ax.fill_between(XX[:, i], confi[:, 0], confi[:, 1],
                            color="tomato", alpha=0.15, label="95% CI")
            ax.set_xlabel(x_col)
            ax.set_ylabel(f"Partial effect on {y_col}" if i == 0 else "")
            ax.legend(fontsize=8)
    
        fig.suptitle(title)
        plt.tight_layout()
        plt.show()
    
    
    # ─────────────────────────────────────────────
    # 3. Train all GAMs, return fitted model registry
    # ─────────────────────────────────────────────
    def train_gams(df: pd.DataFrame) -> dict:
        """
        Fits all GAMs and returns a registry:
            { formula_str: (fitted_gam, x_cols) }
        """
        models = [
            # PsnTMin
            ("PsnTMin = s(MAT.q05)",
             "PsnTMin", ["MAT.q05"]),
    
            # PsnTOpt
            ("PsnTOpt = s(MAT.q95) + s(MTWM.q95)",
             "PsnTOpt", ["MAT.q95", "MTWM.q95"]),
    
            # PsnTMax
            ("PsnTMax = s(MTWM.q95)",
             "PsnTMax", ["MTWM.q95"]),
    
            # LeafOnMinT — four candidate models
            ("LeafOnMinT = s(MAT.q05)",
             "LeafOnMinT", ["MAT.q05"])
        ]
    
        registry = {}
        for formula_str, y_col, x_cols in models:
            gam = fit_plot_gam(df, formula_str, y_col, x_cols)
            registry[formula_str] = (gam, x_cols)
    
        return registry
    
    
    # ─────────────────────────────────────────────
    # 4. Main pipeline
    # ─────────────────────────────────────────────
    def run_pipeline(
        species_code_map: dict[str, str],
        return_df: bool = False,
        plotAndSummaries: bool = False
    ) -> tuple[dict, pd.DataFrame | None]:
        """
        Parameters
        ----------
        species_code_map : dict
            Keys   = species codes (e.g. "ABAL")
            Values = full latin names (e.g. "Abies alba")
    
        return_df : bool
            If True, also return the merged dataframe with prediction columns.
    
        Returns
        -------
        predictions : dict
            { species_code: { "PsnTMin": float, "PsnTOpt": float,
                              "PsnTMax": float, "LeafOnMinT": float } }
        df : pd.DataFrame or None
            Merged dataframe with prediction columns (only if return_df=True).
        """
        # ── Load, merge, train ────────────────────
        df = load_and_merge()
        registry = train_gams(df)
    
        # ── Load df2 separately for species lookup ─
        # (predictions for new species use their climate variables from df2)
        df2 = pd.read_csv(CSV2_PATH).set_index("species")
    
        # ── Selected models for final predictions ──
        # Change these keys to swap the chosen model for each variable
        selected = {
            "PsnTMin":    ("PsnTMin = s(MAT.q05)",                ["MAT.q05"]),
            "PsnTOpt":    ("PsnTOpt = s(MAT.q95) + s(MTWM.q95)", ["MAT.q95", "MTWM.q95"]),
            "PsnTMax":    ("PsnTMax = s(MTWM.q95)",               ["MTWM.q95"]),
            "LeafOnMinT": ("LeafOnMinT = s(MAT.q05)",             ["MAT.q05"])
        }
    
        # ── Predict for each species code ──────────
        predictions = {}
    
        for code, latin_name in species_code_map.items():
            # Look up the species row in df2
            if latin_name not in df2.index:
                print(f"Warning: '{latin_name}' (code: '{code}') not found in "
                      f"{CSV2_PATH}. Returning NaN for all variables.")
                predictions[code] = {var: np.nan for var in selected}
                continue
    
            row = df2.loc[latin_name]
            species_preds = {}
    
            for var, (formula_str, x_cols) in selected.items():
                gam, _ = registry[formula_str]
    
                # Extract predictor values for this species
                try:
                    x_vals = np.array([[row[col] for col in x_cols]], dtype=float)
                except KeyError as e:
                    print(f"Warning: predictor {e} not found in {CSV2_PATH} "
                          f"for species '{latin_name}'. Setting {var} to NaN.")
                    species_preds[var] = np.nan
                    continue
    
                if np.isnan(x_vals).any():
                    print(f"Warning: NaN predictor value(s) for '{latin_name}' "
                          f"when predicting {var}. Setting to NaN.")
                    species_preds[var] = np.nan
                else:
                    species_preds[var] = float(gam.predict(x_vals)[0])
    
            predictions[code] = species_preds
    
        # ── Optionally add prediction columns to df ─
        if return_df:
            for var, (formula_str, x_cols) in selected.items():
                gam, _ = registry[formula_str]
                X_all = df[x_cols].values
                mask = ~np.isnan(X_all).any(axis=1)
                df[f"{var}_pred"] = np.nan
                df.loc[mask, f"{var}_pred"] = gam.predict(X_all[mask])
            return predictions, df
    
        return predictions, None

    if return_df:
        predictions, df = run_pipeline(species_map, return_df=True)
        return(predictions, df)
    else:
        predictions, df = run_pipeline(species_map, return_df=False)
        return(predictions)


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pygam import LinearGAM, s


def predict_temperature_parameters_single_species(
    species_code: str,
    species_latin_name: str,
    MAT_q05: float,
    MAT_q95: float,
    MTWM_q95: float,
    return_df: bool = True,
    plotAndSummaries: bool = True,
):
    CSV1_PATH = "./SpeciesParametersSets/PreviousStudies/GustafsonParameters_UserGuidev5.1_WithDroughtAndWtTolerance.csv"
    CSV2_PATH = "./ReferencesAndData/Others/DendonckerEtAl2026_Average_niche_all_spp.csv"

    # ── Load & merge (training data) ──────────────────────────────────────────
    df1 = pd.read_csv(CSV1_PATH)
    df2 = pd.read_csv(CSV2_PATH)
    # df1 = df1.rename(columns={"Species": "species"})
    df = df1.merge(df2, on="Species", how="left", suffixes=("", "_df2"))

    # ── Helper: fit GAM + optional plot ──────────────────────────────────────
    def fit_plot_gam(formula_str: str, y_col: str, x_cols: list[str],
                     monotonic: bool = True) -> LinearGAM:
        sub = df[[y_col] + x_cols].dropna()
        y = sub[y_col].values
        X = sub[x_cols].values
        n = len(x_cols)

        if monotonic:
            if n == 1:
                gam_terms = s(0, constraints="monotonic_inc")
            elif n == 2:
                gam_terms = s(0, constraints="monotonic_inc") + s(1, constraints="monotonic_inc")
            else:
                gam_terms = (s(0, constraints="monotonic_inc") +
                             s(1, constraints="monotonic_inc") +
                             s(2, constraints="monotonic_inc"))
        else:
            if n == 1:
                gam_terms = s(0)
            elif n == 2:
                gam_terms = s(0) + s(1)
            else:
                gam_terms = s(0) + s(1) + s(2)

        gam = LinearGAM(gam_terms)
        gam.fit(X, y)

        if plotAndSummaries:
            print(f"\n{'='*60}\nModel: {formula_str}\n{'='*60}")
            gam.summary()
            if n == 1:
                fig, ax = plt.subplots(figsize=(7, 4))
                XX = gam.generate_X_grid(term=0, n=200)
                preds = gam.predict(XX)
                ci = gam.prediction_intervals(XX, width=0.95)
                ax.scatter(sub[x_cols[0]], sub[y_col], color="steelblue", alpha=0.7,
                           edgecolors="white", s=60, label="Observed", zorder=3)
                ax.plot(XX[:, 0], preds, color="tomato", lw=2, label="GAM prediction")
                ax.fill_between(XX[:, 0], ci[:, 0], ci[:, 1], color="tomato",
                                alpha=0.15, label="95% CI")
                ax.set_xlabel(x_cols[0]); ax.set_ylabel(y_col); ax.set_title(formula_str)
                ax.legend(); plt.tight_layout(); plt.show()
            else:
                fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
                y_centered = sub[y_col] - sub[y_col].mean()
                for i, (ax, x_col) in enumerate(zip(axes, x_cols)):
                    XX = gam.generate_X_grid(term=i, n=200)
                    pdep, confi = gam.partial_dependence(term=i, X=XX, width=0.95)
                    ax.scatter(sub[x_col], y_centered, color="steelblue", alpha=0.6,
                               s=50, edgecolors="white", label="Observed (centred)", zorder=3)
                    ax.plot(XX[:, i], pdep, color="tomato", lw=2, label="Partial effect")
                    ax.fill_between(XX[:, i], confi[:, 0], confi[:, 1], color="tomato",
                                    alpha=0.15, label="95% CI")
                    ax.set_xlabel(x_col)
                    ax.set_ylabel(f"Partial effect on {y_col}" if i == 0 else "")
                    ax.legend(fontsize=8)
                fig.suptitle(formula_str); plt.tight_layout(); plt.show()

        return gam

    # ── Fit all GAMs ──────────────────────────────────────────────────────────
    # monotonic=True for all single-predictor models, False for PsnTOpt (2 predictors, free)
    models = {
        "PsnTMin":    ("PsnTMin = s(MAT.q05, constraints='monotonic_inc')",                ["MAT.q05"],              True),
        "PsnTOpt":    ("PsnTOpt = s(MAT.q95) + s(MTWM.q95)",                              ["MAT.q95", "MTWM.q95"],  False),
        "PsnTMax":    ("PsnTMax = s(MTWM.q95, constraints='monotonic_inc')",               ["MTWM.q95"],             True),
        "LeafOnMinT": ("LeafOnMinT = s(MAT.q05, constraints='monotonic_inc')",             ["MAT.q05"],              True),
    }
    fitted = {var: fit_plot_gam(formula_str, var, x_cols, monotonic)
              for var, (formula_str, x_cols, monotonic) in models.items()}

    # ── Compute calibration bounds per predictor ──────────────────────────────
    temp_inputs = {"MAT.q05": MAT_q05, "MAT.q95": MAT_q95, "MTWM.q95": MTWM_q95}

    calib_bounds = {col: (df[col].dropna().min(), df[col].dropna().max())
                    for col in temp_inputs}

    # ── Check for extrapolation and warn ─────────────────────────────────────
    for var, (_, x_cols, _) in models.items():
        for col in x_cols:
            val = temp_inputs[col]
            lo, hi = calib_bounds[col]
            if val < lo:
                print(
                    f"⚠️  EXTRAPOLATION WARNING | Species: {species_latin_name} ({species_code}) | "
                    f"Predictor '{col}' = {val:.2f} is BELOW the calibration range "
                    f"[{lo:.2f}, {hi:.2f}] → predicted parameter '{var}' may be unreliable."
                )
            elif val > hi:
                print(
                    f"⚠️  EXTRAPOLATION WARNING | Species: {species_latin_name} ({species_code}) | "
                    f"Predictor '{col}' = {val:.2f} is ABOVE the calibration range "
                    f"[{lo:.2f}, {hi:.2f}] → predicted parameter '{var}' may be unreliable."
                )

    # ── Predict for the species using provided temperature values ─────────────
    species_preds = {}
    for var, (_, x_cols, _) in models.items():
        x_vals = np.array([[temp_inputs[col] for col in x_cols]], dtype=float)
        if np.isnan(x_vals).any():
            print(f"Warning: NaN temperature value(s) for '{species_latin_name}' "
                  f"when predicting {var}. Setting to NaN.")
            species_preds[var] = np.nan
        else:
            species_preds[var] = round(float(fitted[var].predict(x_vals)[0]), 2)

    predictions = {species_code: species_preds}

    # ── Optionally annotate df with predictions ───────────────────────────────
    if return_df:
        for var, (_, x_cols, _) in models.items():
            X_all = df[x_cols].values
            mask = ~np.isnan(X_all).any(axis=1)
            df[f"{var}_pred"] = np.nan
            df.loc[mask, f"{var}_pred"] = fitted[var].predict(X_all[mask])
        return predictions, df

    return predictions, None




# GET PARAMETER VALUES FOR A NEW SPECIES ACCORDING TO THE CALIBRATION TIPS
# OF ERIC GUSTAFSON

import numpy as np

def get_parameters_gustafson_rules(shade_tolerance: float, tree_type: str, wood_type: str,
                                   fullLatinSpeciesName : str, IMAX : int,
                                   diazDatasetPath : str = "./ReferencesAndData/Others/DiazEtAlDatasetTraits.csv") -> dict[str, str]:
    """
    This function uses the rules indicated by Eric Gustafson in the user guide of PnET-Succession v5.1.
    There is two parameters from which Gustafson has no rule to get them : TOFol and FrActWd.
    For these, I used the averaged values for deciduous/evergreen and hardwood/softwoods in previous publications of PnET-Succession parameters
    this gives the following :
    - TOFol : 1 for deciduous trees, 0.3 for evergreens
    - FrActWd : 0.000035 for hardwoods, 0.000029 for softwoods

    Args:
        shade_tolerance: float from 1 (very intolerant) to 5 (very tolerant)
        tree_type: 'deciduous' or 'evergreen'
        wood_type: 'hardwood' or 'softwood'

    Returns:
        Dictionary of parameter names to string values.
    """
    assert 1 <= shade_tolerance <= 5, "shade_tolerance must be between 1 and 5"
    tree_type = tree_type.lower()
    wood_type = wood_type.lower()
    assert tree_type in ("deciduous", "evergreen"), "tree_type must be 'deciduous' or 'evergreen'"
    assert wood_type in ("hardwood", "softwood"), "wood_type must be 'hardwood' or 'softwood'"

    # Shade tolerance scores corresponding to table columns (Tolerant=5 ... Intolerant=1)
    scores = np.array([5, 4, 3, 2, 1], dtype=float)

    def linear_predict(values: list[float], x: float) -> float:
        """Fit a line through the 5 table points and predict at x."""
        coeffs = np.polyfit(scores, values, 1)
        return float(np.polyval(coeffs, x))

    st = shade_tolerance

    # --- Linear models from table of the calibration tips of Eric Gustafson ---
    HalfSat = linear_predict([150, 181, 212.5, 244, 275], st)

    # I'm changing things here; I'm making the difference in FolN being one of gymnosperm or not
    # rather than deciduous or evergreen, as it seems from the literature than the biggest differences
    # in productivity is between angiosperms/gymnosperms
    if wood_type == "hardwood":
        FolN       = linear_predict([2.2, 2.4, 2.6, 2.8, 2.9], st)
        # SLWmax     = linear_predict([70, 75, 80, 85, 90], st) # Now estimated with data from Diaz et al., see below
    else:  # softwood
        FolN       = linear_predict([1.1, 1.3, 1.5, 1.7, 1.9], st)
        # SLWmax     = linear_predict([150, 175, 200, 225, 250], st) # Now estimated with data from Diaz et al., see below

    # For FracFol, however, the difference will obviously be between deciduous and evergreen.
    if tree_type == "deciduous":
        FracFol    = linear_predict([0.014, 0.014, 0.015, 0.017, 0.018], st)
    else:  # evergreen
        FracFol    = linear_predict([0.05, 0.055, 0.06, 0.065, 0.07], st)
    
    # FracBelowG  = linear_predict([0.37, 0.35, 0.33, 0.31, 0.29], st)
    EstRad      = linear_predict([0.976, 0.954, 0.928, 0.900, 0.870], st)
    CFracBiomass = linear_predict([0.5, 0.475, 0.45, 0.425, 0.4], st)

    # --- Fixed rules ---
    if tree_type == "softwood":
        AmaxA  = 5.3
        AmaxB  = 21.5
        k      = 0.5
        SLWDel = 0.0
        TOFol  = 0.3
    else:  # deciduous
        AmaxA  = -46.0
        AmaxB  = 71.9
        k      = 0.58
        SLWDel = 0.2
        TOFol  = 1.0

    KWdLit    = 0.075 if wood_type == "hardwood" else 0.125
    FrActWd  = 0.000035 if wood_type == "hardwood" else 0.000029

    TOWood       = 0.03
    TORoot       = 0.03
    FracFolShape = 6
    MaxFracFol   = FracFol
    EstMoist     = 1

    # Determining SLWmax from Diaz et al.

    # We retrieve LMA (= SLW) from Diaz et al. using the full latin species name; we will consider it as the average for the species
    diazDataSet = pd.read_csv(diazDatasetPath)
    if fullLatinSpeciesName in list(diazDataSet["Species name standardized against TPL"]):
        SLWaverage = diazDataSet.loc[diazDataSet['Species name standardized against TPL'] == fullLatinSpeciesName, 'LMA (g/m2)'].values[0]
        # We use the function to get SLWMax from IMAX, SLWDel and SLWAverage.
        SLWmax = float(SLWaverage) / (1 - (float(SLWDel) * (int(IMAX) - 1) / 2))
    else: # If the species is not found in Diaz et al., we return a warning and use previous rules to determine SLWmax
        print("WARNING : Species not found in the data of Diaz et al. for retrieving its SLW value.\n"
              "This is most probably a problem with the latin name of the species you provided.\n"
              "Please look at the Diaz et al. dataset in /Others to understand why your species is not found.\n")
        if wood_type == "hardwood":
            SLWmax     = linear_predict([70, 75, 80, 85, 90], st) # From Gustafson's rules
        else:  # softwood
            SLWmax     = linear_predict([150, 175, 200, 225, 250], st) # From Gustafson's rules


    # --- Assemble dictionary (all values as strings) ---
    params = {
        "HalfSat":      str(round(HalfSat, 6)),
        "FolN":         str(round(FolN, 6)),
        "SLWmax":       str(round(SLWmax, 6)),
        "FracFol":      str(round(FracFol, 6)),
        # "FracBelowG":   str(round(FracBelowG, 6)), # Not this one, as we use it as a generic parameter
        "EstRad":       str(round(EstRad, 6)),
        "CFracBiomass": str(round(CFracBiomass, 6)),
        "AmaxA":        str(AmaxA),
        "AmaxB":        str(AmaxB),
        "k":            str(k),
        "TOWood":       str(TOWood),
        "TORoot":       str(TORoot),
        "KWdLit":        str(KWdLit),
        "SLWDel":       str(SLWDel),
        "FracFolShape": str(FracFolShape),
        "MaxFracFol":   str(round(MaxFracFol, 6)),
        "EstMoist":     str(EstMoist),
        "TOFol":        str(TOFol),
        "FrActWd":      str(FrActWd),
    }

    return params



#######################################################
# FUNCTIONS FOR INVESTIGATING COMPETITION OUTCOMES
#######################################################

import copy

def competitionSimulationMonoculturemanawan(duration = 200,
                                            climate = "mild",
                                            soil = "SILO",
                                            speciesToSimulate = ["ABIE.BAL", "ACER.RUB"],
                                            dictOfInitialCoreSpeciesParameters_path = './SpeciesParametersSets/Calibrated_SubSubPhase1.4/initialCoreSpeciesParameters.json',
                                            dictOfInitialPnETSpeciesParameters_path = './SpeciesParametersSets/Calibrated_SubSubPhase1.4/initialPnETSpeciesParameters.json',
                                            dictOfInitialPnETGenericParameters_path = './SpeciesParametersSets/Calibrated_SubSubPhase1.4/InitialGenericParameters.json',
                                            equalizeWater = False, # Can be false to keep the differences in water parameters between the two species
                                            equalizeTemperature = False, # Can be false to keep the differences in water parameters between the two species,
                                            equalizeRest = False, # Can be false to keep the differences in the rest of the parameters
                                            numberOfCells = 100,
                                            timestep = 10,
                                            plotResults = False,
                                            dispersal = True,
                                            simulationPath = "/tmp/competitionCalibrationPnET/"):


    # Dicts are loaded here because without it, there was a strange interaction in the run_all_pairs() function
    # where the edited dict where kept in the loop. By loading them here, we avoid this.
    dictOfInitialCoreSpeciesParameters = json.load(open(dictOfInitialCoreSpeciesParameters_path))
    dictOfInitialPnETSpeciesParameters = json.load(open(dictOfInitialPnETSpeciesParameters_path))
    dictOfInitialPnETGenericParameters = json.load(open(dictOfInitialPnETGenericParameters_path))

    # print("\n")
    # print(dictOfInitialCoreSpeciesParameters)
    # print("\n")
    
    # We prepare the simulation files
    PnETGitHub_OneCellSim = parse_All_LANDIS_PnET_Files(r"./SimulationFiles/PnETGitHub_OneCellSim_v8")
    
    # - Species.txt : replace with the right initial core species parameters from the JSON file
    PnETGitHub_OneCellSim["species.txt"] = dictOfInitialCoreSpeciesParameters
    
    # - SpeciesParameters.txt : replace with the initial PnET species parameters from the JSOn file
    PnETGitHub_OneCellSim["SpeciesParameters.txt"] = dictOfInitialPnETSpeciesParameters
    
    # - PnETGenericParameters.txt : replace with initial generic parameters from the JSON file
    PnETGitHub_OneCellSim["PnETGenericParameters.txt"] = dictOfInitialPnETGenericParameters
    
    # Setting duration and timestep
    # WARNING : Timestep can change things here by increasing the amount
    # of opportunities of implantation of younger cohorts
    PnETGitHub_OneCellSim["pnetsuccession.txt"]["Timestep"] = str(timestep)
    PnETGitHub_OneCellSim["scenario.txt"]["Duration"] = str(duration)
    # Changing the soil if needed
    PnETGitHub_OneCellSim["EcoregionParameters.txt"]["EcoregionParameters"]["eco1"]["SoilType"] = soil
    # - pnetsuccession.txt : change startyear to 1900 and latitude to village of Manawan (47.2223)
    startYear = 1900
    PnETGitHub_OneCellSim["pnetsuccession.txt"]["StartYear"] = str(startYear)
    PnETGitHub_OneCellSim["pnetsuccession.txt"]["Latitude"] = "47.2223"
    if dispersal:
        # Full dispersal (but without influence of neighbouring cells) so that competition takes into account regeneration
        PnETGitHub_OneCellSim["pnetsuccession.txt"]["SeedingAlgorithm"] = "UniversalDispersal"
        PnETGitHub_OneCellSim["PnETGenericParameters.txt"]["PreventEstablishment"] = "False"
    else:
        # No dispersal
        PnETGitHub_OneCellSim["pnetsuccession.txt"]["SeedingAlgorithm"] = "NoDispersal"
        PnETGitHub_OneCellSim["PnETGenericParameters.txt"]["PreventEstablishment"] = "True"
    # Setting other parameters
    PnETGitHub_OneCellSim["scenario.txt"]["CellLength"] = "100"

    # Preparing the landscape
    # Removing Other species from the simulation
    speciesToRemove = []
    for species in PnETGitHub_OneCellSim["SpeciesParameters.txt"]["PnETSpeciesParameters"].keys():
        if species not in speciesToSimulate and species != "LandisData":
            speciesToRemove.append(species)
    for species in speciesToRemove:
        if species in PnETGitHub_OneCellSim["species.txt"].keys():
            del PnETGitHub_OneCellSim["species.txt"][species]
        if species in PnETGitHub_OneCellSim["SpeciesParameters.txt"]["PnETSpeciesParameters"].keys():
            del PnETGitHub_OneCellSim["SpeciesParameters.txt"]["PnETSpeciesParameters"][species]
    # Inserting reading of the right climate file
    if climate == "mild":
        PnETGitHub_OneCellSim["pnetsuccession.txt"]["ClimateConfigFile"] = "ClimateConfigSimpleSims_MonthlyAveraged.txt"
    elif climate == "realHistorical":
        PnETGitHub_OneCellSim["pnetsuccession.txt"]["ClimateConfigFile"] = "ClimateConfigSimpleSims.txt"
    elif climate == "testFilesGithub":
        pass # The climate files from github are used by default if we don't input a climate config file
    else:
        raise ValueError("Climate value : " + str(climate) + " not recognized.")
    
    # -  PnEToutputsites_onecell.txt : replace site location
    PnETGitHub_OneCellSim["PnEToutputsites_onecell.txt"]["Site1"] = '1 1'

    # EQUILIZATION
    # We put parameters at intermediate value between the two species
    parametersToEqualize = []
    waterParameters = ["H1", "H2", "H3", "H4"]
    temperatureParameters = ["LeafOnMinT", "PsnTMin", "PsnTOpt", "PsnTMax"]
    coreParameters = ['Longevity', 'Sexual Maturity', 'Seed Dispersal Distance - Effective', 'Seed Dispersal Distance - Maximum', 'Vegetative Reproduction Probability', 'Sprout Age - Min', 'Sprout Age - Max', 'Post Fire Regen']
    coreParametersIntegers = ["Longevity", 'Sexual Maturity', 'Seed Dispersal Distance - Effective', 'Seed Dispersal Distance - Maximum', 'Sprout Age - Min', 'Sprout Age - Max']
    if equalizeWater:
        for parameter in waterParameters:
            parametersToEqualize.append(parameter)
    if equalizeTemperature:
        for parameter in temperatureParameters:
            parametersToEqualize.append(parameter)
    if equalizeRest:
        for parameter in dictOfInitialPnETSpeciesParameters["PnETSpeciesParameters"][speciesToSimulate[0]]:
            if parameter not in waterParameters and parameter not in temperatureParameters:
                parametersToEqualize.append(parameter)

    for parameter in parametersToEqualize:
        halfValue = (float(dictOfInitialPnETSpeciesParameters["PnETSpeciesParameters"][speciesToSimulate[0]][parameter]) + float(dictOfInitialPnETSpeciesParameters["PnETSpeciesParameters"][speciesToSimulate[1]][parameter])) / 2
        dictOfInitialPnETSpeciesParameters["PnETSpeciesParameters"][speciesToSimulate[0]][parameter] = str(halfValue)
        dictOfInitialPnETSpeciesParameters["PnETSpeciesParameters"][speciesToSimulate[1]][parameter] = str(halfValue)

    # If we equalize "The rest", we also have to equalize longevity, etc.
    if equalizeRest:
        for parameter in coreParameters:
            if parameter != 'Post Fire Regen': #Post Fire regen is ignored - no fires.
                halfValue = (float(dictOfInitialCoreSpeciesParameters[speciesToSimulate[0]][parameter]) + float(dictOfInitialCoreSpeciesParameters[speciesToSimulate[1]][parameter])) / 2
                if parameter in coreParametersIntegers:
                    halfValue = int(halfValue)
                dictOfInitialCoreSpeciesParameters[speciesToSimulate[0]][parameter] = str(halfValue)
                dictOfInitialCoreSpeciesParameters[speciesToSimulate[1]][parameter] = str(halfValue)
                

    # Writing the files in a temporary folder
    # We create the folder
    if not os.path.exists(simulationPath):
        os.mkdir(simulationPath)
    else:
        shutil.rmtree(simulationPath)
        os.mkdir(simulationPath)

    simulationPath = simulationPath + "/"
    
    write_all_LANDIS_files(simulationPath,
                           PnETGitHub_OneCellSim,
                           True)


    # print(dictOfInitialCoreSpeciesParameters)
    # print("\n")
    # print(dictOfInitialPnETSpeciesParameters["PnETSpeciesParameters"][speciesToSimulate[0]].keys())
    # print("\n")
    # print(dictOfInitialPnETSpeciesParameters["PnETSpeciesParameters"][speciesToSimulate[1]].keys())
    # Copy the climate files
    if climate == "mild":
        shutil.copy("./SimulationFiles/ClimateConfigSimpleSims_MonthlyAveraged.txt", simulationPath)
        shutil.copy("./ReferencesAndData/Climate Data/dataFrameClimate_historicalMonthly_Ouranos_MonthlyAveraged.csv", simulationPath)
        shutil.copy("./ReferencesAndData/Climate Data/dataFrameClimate_SpinupMonthly_Ouranos_MonthlyAveraged.csv", simulationPath)
        os.remove(simulationPath + "/climate.txt")
        # I'm getting an issue when starting simulations at year 1900 with low longevity values. See https://github.com/LANDIS-II-Foundation/Library-Climate/issues/32
        # It seems to be related to the spinup code taking the wrong amount of year based on the longevity of the species
        # I'm removing years from the spinup file to avoid this
        spinupData = pd.read_csv(simulationPath + "dataFrameClimate_SpinupMonthly_Ouranos_MonthlyAveraged.csv")
        maxLongevity = 0
        for species in speciesToSimulate:
            maxLongevity = max(maxLongevity, int(float(dictOfInitialCoreSpeciesParameters[species]["Longevity"])))
        spinupData = spinupData[spinupData["Year"] > (spinupData['Year'].max() - int(maxLongevity))]
        spinupData.to_csv(simulationPath + "dataFrameClimate_SpinupMonthly_Ouranos_MonthlyAveraged.csv", index=False)
    elif climate == "realHistorical":
        shutil.copy("./SimulationFiles/ClimateConfigSimpleSims.txt", simulationPath)
        shutil.copy("./ReferencesAndData/Climate Data/dataFrameClimate_historicalMonthly_Ouranos.csv", simulationPath)
        shutil.copy("./ReferencesAndData/Climate Data/dataFrameClimate_SpinupMonthly_Ouranos.csv", simulationPath)
        os.remove(simulationPath + "/climate.txt")
        # I'm getting an issue when starting simulations at year 1900 with low longevity values. See https://github.com/LANDIS-II-Foundation/Library-Climate/issues/32
        # It seems to be related to the spinup code taking the wrong amount of year based on the longevity of the species
        # I'm removing years from the spinup file to avoid this
        spinupData = pd.read_csv(simulationPath + "dataFrameClimate_SpinupMonthly_Ouranos.csv")
        maxLongevity = 0
        for species in speciesToSimulate:
            maxLongevity = max(maxLongevity, int(float(dictOfInitialCoreSpeciesParameters[species]["Longevity"])))
        spinupData = spinupData[spinupData["Year"] > (spinupData['Year'].max() - int(maxLongevity))]
        spinupData.to_csv(simulationPath + "dataFrameClimate_SpinupMonthly_Ouranos.csv", index=False)
    elif climate == "testFilesGithub":
        pass # The climate files from github are used by default if we don't input a climate config file
    else:
        raise ValueError("Climate value : " + str(climate) + " not recognized.")
    # Removing climate.txt (old climate file from the test files)

    # Preparing rasters
    # All cohorts start at 1 year olf
    ageRange = [1, 1]
    # Preparing the data we will put in the rasters
    data = np.ones((1, numberOfCells), dtype=np.uint8)
    # Transform used to settle the size of cells - not sure is very useful
    transform = Affine.translation(0, 0) * Affine.scale(1, 1)
    # Creating the ecoregion raster
    with rasterio.open(
    simulationPath + '/ecoregion.img',
    'w',
    driver='GTiff',
    height=1,
    width=numberOfCells,
    count=1,
    dtype=data.dtype,
    crs='EPSG:4326',
    transform=transform
    ) as dst:
        dst.write(data, 1)
    # Preparing the initial communities raster
    data = np.arange(1, numberOfCells+1, dtype="int32").reshape(1, numberOfCells)
    with rasterio.open(
    simulationPath + '/initial-communities.img',
    'w',
    driver='GTiff',
    height=1,
    width=numberOfCells,
    count=1,
    dtype=data.dtype,
    crs='EPSG:4326',
    transform=transform
    ) as dst:
        dst.write(data, 1)
    # Creating initial community .csv
    create_species_csv(speciesToSimulate, numberOfCells, ageRange, filename = simulationPath + "/initial-communities.csv")

    # We launch the simulation
    runLANDIS_Simulation(simulationPath,
                         "scenario.txt",
                        False,
                        printRunning = False)

    

    if plotResults:
        plot_all_cohort_results(str(simulationPath) + "/Output/Site1", {speciesToSimulate[0]:"#5e81ac", speciesToSimulate[1]:"#bf616a"})

    
    return(compute_Y(
    species1 = speciesToSimulate[0],
    species2 = speciesToSimulate[1],
    base_dir = simulationPath + "/output/WoodFoliageBiomass",
    timestep_short = 40,
    timestep_long = 200,
    long_term_mode = "snapshot"))
    # Delete the folder
    # shutil.rmtree(simulationPath)


import numpy as np
import rasterio
from pathlib import Path
from dataclasses import dataclass, field


@dataclass
class CompetitionResult:
    """Holds Y indices, win fractions, and growth curves for one species pair."""
    Y_short: float          # mean relative competitive index at short-term timestep
    Y_long: float           # mean relative competitive index at long-term timestep
    p_short: float          # fraction of cells where sp1 wins short-term
    p_long: float           # fraction of cells where sp1 wins long-term
    n_cells: int            # number of valid (non-masked) cells used

    # Growth curves: shape (n_cells, n_timesteps)
    biomass_sp1: np.ndarray = field(repr=False)  # biomass of species 1 across time
    biomass_sp2: np.ndarray = field(repr=False)  # biomass of species 2 across time
    timesteps: np.ndarray   = field(repr=False)  # 1D array of timestep values


def _get_available_timesteps(species_name: str, base_dir: Path) -> list[int]:
    """
    Scan the species folder and return a sorted list of available timesteps,
    inferred from filenames matching 'WoodFoliageBiomass-{timestep}.img'.
    """
    folder = base_dir / species_name
    timesteps = sorted(
        int(f.stem.split("-")[-1])
        for f in folder.glob("WoodFoliageBiomass-*.img")
    )
    if not timesteps:
        raise FileNotFoundError(f"No WoodFoliageBiomass-*.img files found in {folder}")
    return timesteps


def _load_biomass(species_name: str, timestep: int, base_dir: Path) -> np.ndarray:
    """
    Load a flat 1D array of valid (non-NoData) biomass values for a given
    species and timestep.

    Returns
    -------
    np.ndarray
        1D array of valid biomass values across all cells.
    """
    path = base_dir / species_name / f"WoodFoliageBiomass-{timestep}.img"

    with rasterio.open(path) as src:
        data = src.read(1).astype(np.float64)
        nodata = src.nodata

    if nodata is not None:
        valid_mask = data != nodata
    else:
        valid_mask = np.isfinite(data)

    return data[valid_mask]


def _load_all_timesteps(
    species_name: str,
    timesteps: list[int],
    base_dir: Path,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load biomass for all timesteps for a species.

    Returns
    -------
    biomass : np.ndarray, shape (n_cells, n_timesteps)
        Biomass values per cell per timestep.
    valid_cells : np.ndarray of bool, shape (n_cells_total,)
        Mask of cells that are valid across ALL timesteps (intersection).
    """
    # Load each timestep as a 2D raster, keep the full grid for mask intersection
    arrays = []
    masks = []

    for ts in timesteps:
        path = base_dir / species_name / f"WoodFoliageBiomass-{ts}.img"
        with rasterio.open(path) as src:
            data = src.read(1).astype(np.float64).ravel()  # flatten to 1D
            nodata = src.nodata

        if nodata is not None:
            mask = data != nodata
        else:
            mask = np.isfinite(data)

        arrays.append(data)
        masks.append(mask)

    # Intersect masks: a cell is valid only if it's valid at every timestep
    combined_mask = np.ones(len(arrays[0]), dtype=bool)
    for m in masks:
        combined_mask &= m

    # Stack into (n_cells, n_timesteps), keeping only consistently valid cells
    biomass = np.column_stack([a[combined_mask] for a in arrays])  # (n_cells, n_timesteps)

    return biomass, combined_mask


def compute_Y(
    species1: str,
    species2: str,
    base_dir: str | Path = "/output/WoodFoliageBiomass",
    timestep_short: int = 40,
    timestep_long: int = 200,
    long_term_mode: str = "snapshot",  # "snapshot" or "gain"
) -> CompetitionResult:
    """
    Compute the relative competitive index Y for a pair of species,
    and return full biomass growth curves across all available timesteps.

    Y = mean over cells of (B1 - B2) / (B1 + B2), in [-1, 1].
    Y > 0 means species1 dominates, Y < 0 means species2 dominates.

    Parameters
    ----------
    species1, species2 : str
        Species folder names.
    base_dir : str or Path
        Root directory containing per-species subfolders.
    timestep_short : int
        Timestep used for short-term comparison (default: 40).
    timestep_long : int
        Timestep used for long-term comparison (default: 200).
    long_term_mode : str
        "snapshot" : compare raw biomass at timestep_long.
        "gain"     : compare biomass gained between timestep_short
                     and timestep_long.

    Returns
    -------
    CompetitionResult
        Includes Y indices, win fractions, and full growth curves.
    """
    base_dir = Path(base_dir)

    # --- Discover timesteps (use union of both species, warn on mismatch) ---
    ts1 = _get_available_timesteps(species1, base_dir)
    ts2 = _get_available_timesteps(species2, base_dir)

    if ts1 != ts2:
        common = sorted(set(ts1) & set(ts2))
        print(
            f"Warning: timestep mismatch between '{species1}' and '{species2}'. "
            f"Using {len(common)} common timesteps."
        )
    else:
        common = ts1

    # Validate that the key timesteps are present
    for ts, label in [(timestep_short, "timestep_short"), (timestep_long, "timestep_long")]:
        if ts not in common:
            raise ValueError(f"{label}={ts} not found among available timesteps: {common}")

    # --- Load full growth curves (n_cells, n_timesteps) ---
    biomass_sp1, mask1 = _load_all_timesteps(species1, common, base_dir)
    biomass_sp2, mask2 = _load_all_timesteps(species2, common, base_dir)

    # Align cells: keep only cells valid for both species across all timesteps
    if biomass_sp1.shape[0] != biomass_sp2.shape[0]:
        raise ValueError(
            f"Cell count mismatch after masking: "
            f"{biomass_sp1.shape[0]} (sp1) vs {biomass_sp2.shape[0]} (sp2). "
            "Check that both species rasters share the same grid."
        )

    n_cells = biomass_sp1.shape[0]
    ts_array = np.array(common)

    # --- Extract slices at key timesteps ---
    idx_short = common.index(timestep_short)
    idx_long  = common.index(timestep_long)

    b1_short = biomass_sp1[:, idx_short]
    b2_short = biomass_sp2[:, idx_short]
    b1_long  = biomass_sp1[:, idx_long]
    b2_long  = biomass_sp2[:, idx_long]

    # --- Helper for ties --
    def _win_fraction_with_tiebreak(a: np.ndarray, b: np.ndarray) -> float:
        """
        Compute fraction of cells where a 'wins' over b,
        with ties broken randomly (50/50) instead of always favouring a.
        """
        wins  = a > b                          # strict win for species 1
        ties  = a == b                         # exact ties
        # Each tie is independently assigned to species 1 with p=0.5
        tie_wins = ties & (np.random.random(ties.shape) < 0.5)
        return float(np.mean(wins | tie_wins))
    
    # --- Short-term Y ---
    denom_short = b1_short + b2_short
    valid_short = denom_short > 0
    rci_short = np.where(valid_short, (b1_short - b2_short) / denom_short, np.nan)

    Y_short = float(np.nanmean(rci_short))
    p_short = _win_fraction_with_tiebreak(b1_short[valid_short], b2_short[valid_short])

    # --- Long-term Y ---
    if long_term_mode == "gain":
        b1_eff = b1_long - b1_short
        b2_eff = b2_long - b2_short
    elif long_term_mode == "snapshot":
        b1_eff = b1_long
        b2_eff = b2_long
    else:
        raise ValueError(f"long_term_mode must be 'snapshot' or 'gain', got '{long_term_mode}'")

    denom_long = b1_eff + b2_eff
    valid_long = denom_long > 0
    rci_long = np.where(valid_long, (b1_eff - b2_eff) / denom_long, np.nan)

    Y_long = float(np.nanmean(rci_long))
    p_long  = _win_fraction_with_tiebreak(b1_eff[valid_long],   b2_eff[valid_long])
    
    return CompetitionResult(
        Y_short=Y_short,
        Y_long=Y_long,
        p_short=p_short,
        p_long=p_long,
        n_cells=n_cells,
        biomass_sp1=biomass_sp1,   # shape: (n_cells, n_timesteps)
        biomass_sp2=biomass_sp2,   # shape: (n_cells, n_timesteps)
        timesteps=ts_array,        # shape: (n_timesteps,)
    )


from itertools import combinations
from dataclasses import dataclass, field
import numpy as np


# Represents one row in the full factorial design (one of the 8 configurations)
FACTORIAL_CONFIGS = [
    # (equalizeWater, equalizeTemperature, equalizeRest)  — all 2^3 combinations
    (False, False, False),  # full competition         (+1, +1, +1)
    (True,  False, False),  # water equalized          (-1, +1, +1)
    (False, True,  False),  # temperature equalized    (+1, -1, +1)
    (False, False, True),   # rest equalized           (+1, +1, -1)
    (True,  True,  False),  # water + temp equalized   (-1, -1, +1)
    (True,  False, True),   # water + rest equalized   (-1, +1, -1)
    (False, True,  True),   # temp + rest equalized    (+1, -1, -1)
    (True,  True,  True),   # all equalized            (-1, -1, -1)
]

# Corresponding ±1 codes for the factorial model (W, T, R)
FACTORIAL_CODES = np.array([
    [+1, +1, +1],
    [-1, +1, +1],
    [+1, -1, +1],
    [+1, +1, -1],
    [-1, -1, +1],
    [-1, +1, -1],
    [+1, -1, -1],
    [-1, -1, -1],
], dtype=float)


@dataclass
class PairSimulationResults:
    """All 8 factorial CompetitionResults for a single species pair."""
    species1: str
    species2: str
    # Keyed by (equalizeWater, equalizeTemperature, equalizeRest)
    results: dict = field(default_factory=dict)   # config_tuple -> CompetitionResult


import numpy as np
from itertools import combinations
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field


# ── Worker (must be top-level for pickling) ────────────────────────────────────

def _run_single_config(
    sp1: str,
    sp2: str,
    dictOfInitialCoreSpeciesParameters_path: str,
    dictOfInitialPnETSpeciesParameters_path: str,
    dictOfInitialPnETGenericParameters_path: str,
    climate: str,
    soil: str,
    config: tuple,
    worker_output_dir: str,
    timestep_short: int,
    timestep_long: int,
    long_term_mode: str,
    *_,  # absorbs the now-unused extra positional arg passed by run_all_pairs
) -> tuple[tuple, "CompetitionResult"]:
    """
    Top-level worker function executed in a separate process.

    Delegates entirely to competitionSimulationMonoculturemanawan, which
    already calls compute_Y internally and returns a CompetitionResult.
    We do NOT call compute_Y again here — the simulator handles the correct
    output path (simulationPath + "/output/WoodFoliageBiomass") itself.
    """
    eq_water, eq_temp, eq_rest = config

    result = competitionSimulationMonoculturemanawan(
        speciesToSimulate=[sp1, sp2],
        dictOfInitialCoreSpeciesParameters_path=dictOfInitialCoreSpeciesParameters_path,
        dictOfInitialPnETSpeciesParameters_path=dictOfInitialPnETSpeciesParameters_path,
        dictOfInitialPnETGenericParameters_path=dictOfInitialPnETGenericParameters_path,
        climate = climate,
        soil = soil,
        equalizeWater=eq_water,
        equalizeTemperature=eq_temp,
        equalizeRest=eq_rest,
        simulationPath=worker_output_dir,
    )

    return config, result



# ── Main automation function ───────────────────────────────────────────────────

def run_all_pairs(
    species_list: list[str],
    base_dir: str = "/output/WoodFoliageBiomass",
    timestep_short: int = 40,
    timestep_long: int = 200,
    long_term_mode: str = "snapshot",
    n_workers: int = 1,
    temp_dir: str = "/tmp/manawan_parallel",
    dictOfInitialCoreSpeciesParameters_path: str = './SpeciesParametersSets/Calibrated_SubSubPhase1.4/initialCoreSpeciesParameters.json',
    dictOfInitialPnETSpeciesParameters_path: str = './SpeciesParametersSets/Calibrated_SubSubPhase1.4/initialPnETSpeciesParameters.json',
    dictOfInitialPnETGenericParameters_path: str = './SpeciesParametersSets/Calibrated_SubSubPhase1.4/InitialGenericParameters.json',
    climate: str = "mild",
    soil: str = "SILO"
) -> list[PairSimulationResults]:
    """
    For every unique (unordered) pair of species, run all 8 factorial
    configurations of competitionSimulationMonoculturemanawan and collect
    CompetitionResult objects.

    The all-equalized config (equalizeWater=True, equalizeTemperature=True,
    equalizeRest=True) is never simulated: both species are identical in that
    configuration so a perfect-tie CompetitionResult is injected directly.

    Parallel execution
    ------------------
    When n_workers > 1, up to `n_workers` simulation configs are run
    simultaneously in separate processes. Each worker writes its raster
    outputs to an isolated subdirectory under `temp_dir` to avoid file
    collisions, then compute_Y reads from that same subdirectory.
    The final structure of the returned list is identical regardless of
    n_workers.

    Parameters
    ----------
    species_list : list[str]
        All species names to consider.
    base_dir : str
        Root output directory used for sequential runs (n_workers=1).
    timestep_short, timestep_long : int
        Passed through to compute_Y.
    long_term_mode : str
        Passed through to compute_Y.
    n_workers : int
        Number of parallel simulation processes. 1 = sequential (default).
        A sensible value is the number of available CPU cores minus one.
    temp_dir : str
        Root temporary directory under which per-worker isolated output
        folders are created when n_workers > 1. Created automatically if
        it does not exist.

    Returns
    -------
    list[PairSimulationResults]
        One entry per unique species pair, each holding 8 CompetitionResults.
        Order matches the iteration order of combinations(species_list, 2).
    """
    PERFECT_TIE = CompetitionResult(
        Y_short=0.0,
        Y_long=0.0,
        p_short=0.5,
        p_long=0.5,
        n_cells=0,
        biomass_sp1=np.array([]),
        biomass_sp2=np.array([]),
        timesteps=np.array([]),
    )

    ALL_EQUALIZED = (True, True, True)

    # Pre-build the ordered list of pairs so we can preserve output order
    pairs = list(combinations(species_list, 2))

    # Initialise result containers (one per pair, in order)
    pair_results_map: dict[tuple, PairSimulationResults] = {
        (sp1, sp2): PairSimulationResults(species1=sp1, species2=sp2)
        for sp1, sp2 in pairs
    }

    # Inject the guaranteed tie for the all-equalized config upfront
    for sp1, sp2 in pairs:
        pair_results_map[(sp1, sp2)].results[ALL_EQUALIZED] = PERFECT_TIE

    # Build the list of jobs to actually run (all configs except ALL_EQUALIZED)
    jobs = [
        (sp1, sp2, config)
        for sp1, sp2 in pairs
        for config in FACTORIAL_CONFIGS
        if config != ALL_EQUALIZED
    ]

    # ── Sequential path ────────────────────────────────────────────────────────
    if n_workers == 1:
        for sp1, sp2, config in jobs:
            eq_water, eq_temp, eq_rest = config
            print(
                f"[{sp1} vs {sp2}] "
                f"W={eq_water} T={eq_temp} R={eq_rest}"
            )
            competitionSimulationMonoculturemanawan(
                speciesToSimulate=[sp1, sp2],
                dictOfInitialCoreSpeciesParameters_path=dictOfInitialCoreSpeciesParameters_path,
                dictOfInitialPnETSpeciesParameters_path=dictOfInitialPnETSpeciesParameters_path,
                dictOfInitialPnETGenericParameters_path=dictOfInitialPnETGenericParameters_path,
                climate = climate,
                soil = soil,
                equalizeWater=eq_water,
                equalizeTemperature=eq_temp,
                equalizeRest=eq_rest,
            )
            result = compute_Y(
                species1=sp1,
                species2=sp2,
                base_dir=base_dir,
                timestep_short=timestep_short,
                timestep_long=timestep_long,
                long_term_mode=long_term_mode,
            )
            pair_results_map[(sp1, sp2)].results[config] = result

    # ── Parallel path ──────────────────────────────────────────────────────────
    else:
        temp_root = Path(temp_dir)
        temp_root.mkdir(parents=True, exist_ok=True)

        # Map each job to a unique isolated output directory so workers never
        # write to the same path simultaneously.
        def _worker_dir(sp1: str, sp2: str, config: tuple) -> str:
            tag = f"{sp1}__{sp2}__W{int(config[0])}T{int(config[1])}GS{int(config[2])}"
            d = temp_root / tag
            d.mkdir(parents=True, exist_ok=True)
            return str(d)

        futures = {}
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            for sp1, sp2, config in jobs:
                worker_out = _worker_dir(sp1, sp2, config)
                future = executor.submit(
                    _run_single_config,
                    sp1, sp2,
                    dictOfInitialCoreSpeciesParameters_path, dictOfInitialPnETSpeciesParameters_path, dictOfInitialPnETGenericParameters_path,
                    climate, soil,
                    config,
                    worker_out,       # passed as outputDir to the simulator
                    timestep_short,
                    timestep_long,
                    long_term_mode,
                    # no second worker_out needed anymore
                )
                futures[future] = (sp1, sp2)

            completed = 0
            total = len(futures)
            for future in as_completed(futures):
                sp1, sp2 = futures[future]
                completed += 1
                try:
                    config, result = future.result()
                    pair_results_map[(sp1, sp2)].results[config] = result
                    print(
                        f"[{completed}/{total}] Done: {sp1} vs {sp2} "
                        f"W={config[0]} T={config[1]} GS={config[2]}"
                    )
                except Exception as exc:
                    print(
                        f"[{completed}/{total}] FAILED: {sp1} vs {sp2} "
                        f"config={futures[future]} → {exc}"
                    )
                    raise  # re-raise so the caller knows something went wrong

    # Return in the original pair order
    return [pair_results_map[(sp1, sp2)] for sp1, sp2 in pairs]


@dataclass
class EffectDecomposition:
    """
    Factorial effect decomposition for one species pair, one time window.

    Attributes
    ----------
    Y_full : float
        Competitive index under full (unmodified) competition.
    beta : dict
        All 7 factorial contrasts (main effects + interactions), keyed by
        label strings: 'W', 'T', 'R', 'WT', 'WR', 'TR', 'WTR'.
    e_W, e_T, e_R : float
        Main-effect shares in [0, 1], summing to 1.
        Proportion of the main-effect variance explained by each factor.
    iota : float
        Non-additivity index in [0, 1].
        Share of total variance carried by interaction terms.
        High iota means W/T/R effects are entangled; interpret e_W/T/R with caution.
    var_total : float
        Total variance of Y across the 8 configurations (denominator of iota).
    """
    species1: str
    species2: str
    window: str          # "short" or "long"
    Y_full: float
    beta: dict
    e_W: float
    e_T: float
    e_R: float
    iota: float
    var_total: float


def compute_effects(
    pair_results: PairSimulationResults,
) -> tuple[EffectDecomposition, EffectDecomposition]:
    """
    Apply the 2^3 factorial contrast decomposition to extract the contribution
    of water (W), temperature (T), and 'rest' (R) parameters to competition
    outcome, for both the short-term and long-term time windows.

    The model is:
        Y = β0 + βW·xW + βT·xT + βR·xR
              + βWT·xW·xT + βWR·xW·xR + βTR·xT·xR + βWTR·xW·xT·xR

    where xW, xT, xR ∈ {-1, +1}:
        +1 = real (species-specific) parameter values
        -1 = equalized (mean of both species) parameter values

    Because the design matrix is orthogonal, each coefficient is simply:
        βj = (1/8) * Σ_i [ φj(i) * Y(i) ]
    where φj(i) is the product of the ±1 codes for effect j at run i.

    The variance of Y across the 8 runs partitions exactly as:
        Var(Y) = βW² + βT² + βR² + βWT² + βWR² + βTR² + βWTR²

    Main-effect shares (sum to 1 over W, T, R):
        e_W = βW² / (βW² + βT² + βR²)

    Non-additivity index (share of variance in interactions):
        ι = (βWT² + βWR² + βTR² + βWTR²) / Var(Y)

    Parameters
    ----------
    pair_results : PairSimulationResults
        Output of run_all_pairs for a single species pair.

    Returns
    -------
    (decomp_short, decomp_long) : tuple of EffectDecomposition
    """
    # Collect Y values in the same row order as FACTORIAL_CONFIGS / FACTORIAL_CODES
    Y_short_vec = np.array([
        pair_results.results[cfg].Y_short for cfg in FACTORIAL_CONFIGS
    ])
    Y_long_vec = np.array([
        pair_results.results[cfg].Y_long for cfg in FACTORIAL_CONFIGS
    ])

    results = []
    for Y_vec, window_label in [(Y_short_vec, "short"), (Y_long_vec, "long")]:

        # ── Step 1: build the full contrast matrix (8×8) ──────────────────────
        # Columns: intercept, W, T, R, WT, WR, TR, WTR
        X = np.column_stack([
            np.ones(8),                                          # β0 (intercept)
            FACTORIAL_CODES[:, 0],                               # βW
            FACTORIAL_CODES[:, 1],                               # βT
            FACTORIAL_CODES[:, 2],                               # βR
            FACTORIAL_CODES[:, 0] * FACTORIAL_CODES[:, 1],      # βWT
            FACTORIAL_CODES[:, 0] * FACTORIAL_CODES[:, 2],      # βWR
            FACTORIAL_CODES[:, 1] * FACTORIAL_CODES[:, 2],      # βTR
            FACTORIAL_CODES[:, 0] * FACTORIAL_CODES[:, 1] * FACTORIAL_CODES[:, 2],  # βWTR
        ])

        # ── Step 2: compute all 8 coefficients via orthogonal contrasts ───────
        # Because X is orthogonal with column norms = sqrt(8),
        # β = (X'X)^{-1} X' Y = (1/8) X' Y
        beta_vec = X.T @ Y_vec / 8.0

        beta_labels = ["intercept", "W", "T", "R", "WT", "WR", "TR", "WTR"]
        beta_dict = dict(zip(beta_labels, beta_vec))

        # ── Step 3: variance partition ────────────────────────────────────────
        # Var(Y) across the 8 design points = sum of squared non-intercept betas
        # (exact identity for a saturated 2^k design)
        var_total = float(np.sum(beta_vec[1:] ** 2))  # exclude intercept

        beta_W   = float(beta_vec[1])
        beta_T   = float(beta_vec[2])
        beta_R   = float(beta_vec[3])
        beta_WT  = float(beta_vec[4])
        beta_WR  = float(beta_vec[5])
        beta_TR  = float(beta_vec[6])
        beta_WTR = float(beta_vec[7])

        # ── Step 4: main-effect shares (normalized to sum to 1 over W, T, R) ──
        main_var = beta_W**2 + beta_T**2 + beta_R**2

        if main_var > 0:
            e_W = beta_W**2 / main_var
            e_T = beta_T**2 / main_var
            e_R = beta_R**2 / main_var
        else:
            # No main effects at all — competition is purely stochastic or
            # driven entirely by interactions; set shares to equal thirds
            e_W = e_T = e_R = 1.0 / 3.0

        # ── Step 5: non-additivity index ──────────────────────────────────────
        interaction_var = beta_WT**2 + beta_WR**2 + beta_TR**2 + beta_WTR**2
        iota = float(interaction_var / var_total) if var_total > 0 else 0.0

        # ── Step 6: Y under full (unmodified) competition ─────────────────────
        Y_full = float(pair_results.results[(False, False, False)].Y_short
                       if window_label == "short"
                       else pair_results.results[(False, False, False)].Y_long)

        results.append(EffectDecomposition(
            species1=pair_results.species1,
            species2=pair_results.species2,
            window=window_label,
            Y_full=Y_full,
            beta=beta_dict,
            e_W=e_W,
            e_T=e_T,
            e_R=e_R,
            iota=iota,
            var_total=var_total,
        ))

    return results[0], results[1]  # (short, long)

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from matplotlib.cm import ScalarMappable
import numpy as np

def _to_ternary(e_W, e_T, e_R):
    """
    Convert (e_W, e_T, e_R) barycentric coordinates (summing to 1)
    to 2D Cartesian coordinates for a standard equilateral triangle.

    Triangle corners:
        Water       → bottom-left  (0, 0)
        Temperature → bottom-right (1, 0)
        Rest        → top-center   (0.5, sqrt(3)/2)
    """
    x = e_T + e_R * 0.5
    y = e_R * (3 ** 0.5 / 2)
    return x, y



def _assign_species_colors(species_list: list[str]) -> dict[str, str]:
    """Assign a unique, visually distinct color to each species."""
    palette = plt.cm.tab20.colors  # 20 distinct colors
    return {sp: palette[i % len(palette)] for i, sp in enumerate(species_list)}


import warnings
import matplotlib.colors as mcolors

def make_species_colors(
    species_list: list[str],
    use_glasbey: bool = True,
) -> dict[str, str]:
    """
    Assign a unique color to each species from a curated palette.

    If `use_glasbey` is True, uses the Glasbey palette from the `colorcet`
    package, which is algorithmically designed to maximize perceptual distance
    between colors — ideal for large categorical sets (supports up to 256).

    If `use_glasbey` is False (default), uses the hardcoded curated palette,
    falling back to Matplotlib's tab20/tab20b qualitative colors with a warning
    if the species list exceeds the palette length.

    Parameters
    ----------
    species_list : list[str]
    use_glasbey : bool, optional
        If True, use the Glasbey palette from `colorcet`. Default is False.

    Returns
    -------
    dict mapping species name -> hex color string
    """
    PALETTE = [
        "#0F4C81", "#A23E48", "#4C956C", "#E0A458", "#6B6D76",
        "#E69F00", "#56B4E9", "#009E73", "#F0E442",
        "#0072B2", "#D55E00", "#CC79A7", "#000000", "#4477AA",
        "#66CCEE", "#228833", "#CCBB44", "#EE6677", "#AA3377",
        "#BBBBBB", "#1F77B4", "#D62728", "#2CA02C", "#9467BD",
        "#FF7F0E", "#17BECF",
    ]
    MAX_GLASBEY = 256

    if use_glasbey:
        try:
            import colorcet as cc
        except ImportError as e:
            raise ImportError(
                "The `colorcet` package is required when use_glasbey=True. "
                "Install it with: pip install colorcet"
            ) from e

        if len(species_list) > MAX_GLASBEY:
            raise ValueError(
                f"Species list ({len(species_list)}) exceeds the maximum "
                f"Glasbey palette size ({MAX_GLASBEY})."
            )

        return {sp: cc.glasbey[i] for i, sp in enumerate(species_list)}

    # --- Hardcoded palette path ---
    if len(species_list) > len(PALETTE):
        warnings.warn(
            f"Species list ({len(species_list)}) exceeds curated palette "
            f"({len(PALETTE)} colors). Falling back to Matplotlib tab20 colors.",
            UserWarning,
        )
        fallback = [mcolors.to_hex(c) for c in plt.cm.tab20.colors]
        fallback += [mcolors.to_hex(c) for c in plt.cm.tab20b.colors]
        return {sp: fallback[i % len(fallback)] for i, sp in enumerate(species_list)}

    return {sp: PALETTE[i] for i, sp in enumerate(species_list)}


def plot_win_matrices(
    all_pairs: list[PairSimulationResults],
    species_list: list[str],
    species_colors: dict[str, str],
    figsize_per_panel: float = 4.0,
) -> plt.Figure:
    """
    Two side-by-side full matrices showing short-term and long-term
    win fraction (p_short / p_long) of the row species ("Challenger") vs the
    column species ("Opponent"), under full competition.

    Color scale: dark blue = 1.0 (challenger always wins),
                 dark red  = 0.0 (opponent always wins),
                 white     = 0.5 (tie).

    Tick labels are colored with each species' color from species_colors.

    Parameters
    ----------
    all_pairs : list[PairSimulationResults]
    species_list : list[str]
        Ordered list of species (determines axis order).
    species_colors : dict[str, str]
        Output of make_species_colors.
    figsize_per_panel : float
    """
    n = len(species_list)
    sp_idx = {sp: i for i, sp in enumerate(species_list)}

    mat_short = np.full((n, n), np.nan)
    mat_long  = np.full((n, n), np.nan)

    full_config = (False, False, False)

    for pair in all_pairs:
        i = sp_idx[pair.species1]
        j = sp_idx[pair.species2]
        p_s = pair.results[full_config].p_short
        p_l = pair.results[full_config].p_long

        # Fill both triangles: [i,j] = win fraction of species1 vs species2
        #                       [j,i] = mirror (win fraction of species2 vs species1)
        mat_short[i, j] = p_s
        mat_short[j, i] = 1.0 - p_s
        mat_long[i, j]  = p_l
        mat_long[j, i]  = 1.0 - p_l

    cmap = plt.cm.RdBu  # red (opponent wins) → white (tie) → blue (challenger wins)

    fig, axes = plt.subplots(
        1, 2,
        figsize=(figsize_per_panel * 2 + 4.0, figsize_per_panel),
        constrained_layout=True,
    )
    fig.get_layout_engine().set(w_pad=0.4, wspace=0.15)

    titles = ["Short-term win fraction (t=40)", "Long-term win fraction (t=200)"]

    for ax, mat, title in zip(axes, [mat_short, mat_long], titles):
        ax.imshow(mat, cmap=cmap, vmin=0, vmax=1, aspect="equal")

        # Annotate non-diagonal cells
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                if not np.isnan(mat[i, j]):
                    ax.text(
                        j, i, f"{mat[i, j]:.2f}",
                        ha="center", va="center", fontsize=8,
                        color="white" if abs(mat[i, j] - 0.5) > 0.3 else "black",
                    )

        # Grey out only the diagonal
        for i in range(n):
            ax.add_patch(
                mpatches.Rectangle(
                    (i - 0.5, i - 0.5), 1, 1,
                    color="lightgrey", zorder=2,
                )
            )

        # Axis ticks
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))

        # Color tick labels by species color
        ax.set_xticklabels(species_list, rotation=45, ha="right", fontsize=9)
        ax.set_yticklabels(species_list, fontsize=9)

        for tick, sp in zip(ax.get_xticklabels(), species_list):
            tick.set_color(species_colors[sp])
        for tick, sp in zip(ax.get_yticklabels(), species_list):
            tick.set_color(species_colors[sp])

        ax.set_xlabel("Opponent →", fontsize=10, fontweight="bold")
        ax.set_ylabel("← Challenger", fontsize=10, fontweight="bold")
        ax.set_title(title, fontsize=11, fontweight="bold")

    # Single shared colorbar
    sm = ScalarMappable(cmap=cmap, norm=mcolors.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, shrink=0.7, pad=0.02)
    cbar.set_label("Win fraction", fontsize=9)
    cbar.set_ticks([0, 0.25, 0.5, 0.75, 1.0])
    cbar.set_ticklabels(
        ["0\n(opponent wins)", "0.25", "0.5\n(tie)", "0.75", "1.0\n(challenger wins)"]
    )

    return fig


def plot_ternary_effects(
    decompositions_short: list[EffectDecomposition],
    decompositions_long:  list[EffectDecomposition],
    species_colors: dict[str, str],
    n_cols: int = 3,
    subplot_size: float = 3.5,
) -> plt.Figure:
    """
    One small ternary subplot per SPECIES (not per pair), arranged in a grid.

    Each subplot shows all competition outcomes involving that species.
    For each pair the species appears in:
        ○  open circle  = short-term effect balance
        ●  filled circle = long-term effect balance
        →  arrow connecting short to long, colored with the OPPONENT's species color

    The short- and long-term dots are colored on a blue/red scale:
        blue  = the focal species won (Y_full > 0 if focal == species1, < 0 if focal == species2)
        red   = the focal species lost
        grey  = tie (|Y_full| < 0.05)

    Opacity of the dots encodes trust: faded = high non-additivity (ι).

    Parameters
    ----------
    decompositions_short, decompositions_long : list[EffectDecomposition]
        Outputs of compute_effects for all pairs, in the same order.
    species_colors : dict[str, str]
        Output of make_species_colors.
    n_cols : int
        Number of subplots per row.
    subplot_size : float
        Size of each square subplot in inches.
    """

    # ── Collect all species that appear in at least one decomposition ─────────
    all_species = sorted({
        sp
        for d in decompositions_short
        for sp in (d.species1, d.species2)
    })
    n_species = len(all_species)
    n_rows = int(np.ceil(n_species / n_cols))

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(subplot_size * n_cols, subplot_size * n_rows + 0.6),
        squeeze=False,
    )

    fig.subplots_adjust(hspace=0.5)

    # Pre-build a lookup: (species1, species2) -> (d_short, d_long)
    pair_lookup = {
        (d.species1, d.species2): (d_s, d_l)
        for d_s, d_l in zip(decompositions_short, decompositions_long)
        for d in [d_s]   # key by species1/species2 as stored
    }

    def _draw_triangle(ax):
        """Draw the ternary triangle, corner labels, and grid lines."""
        corners = np.array([[0, 0], [1, 0], [0.5, 3**0.5 / 2], [0, 0]])
        ax.plot(corners[:, 0], corners[:, 1], "k-", lw=1.2, zorder=1)
        off = 0.04
        ax.text(0 - off,            0 - off,            "W", ha="center", fontsize=8, fontweight="bold")
        ax.text(1 + off,            0 - off,            "T", ha="center", fontsize=8, fontweight="bold")
        ax.text(0.5, 3**0.5/2 + off, "G/S",               ha="center", fontsize=8, fontweight="bold")
        for level in [0.25, 0.5, 0.75]:
            for fixed_axis in range(3):
                pts = []
                for t in np.linspace(0, 1 - level, 40):
                    coords = [0.0, 0.0, 0.0]
                    coords[fixed_axis] = level
                    remaining = [k for k in range(3) if k != fixed_axis]
                    coords[remaining[0]] = t
                    coords[remaining[1]] = 1 - level - t
                    pts.append(_to_ternary(coords[0], coords[1], coords[2]))
                pts = np.array(pts)
                ax.plot(pts[:, 0], pts[:, 1], color="lightgrey", lw=0.5, zorder=0)

    for sp_idx, focal_species in enumerate(all_species):
        row, col = divmod(sp_idx, n_cols)
        ax = axes[row, col]
        ax.set_aspect("equal")
        ax.axis("off")
        _draw_triangle(ax)

        # ── Find all pairs involving this focal species ────────────────────────
        for (d_short, d_long) in zip(decompositions_short, decompositions_long):
            if focal_species not in (d_short.species1, d_short.species2):
                continue

            # Determine opponent and whether focal species won
            focal_is_sp1 = (focal_species == d_short.species1)
            opponent = d_short.species2 if focal_is_sp1 else d_short.species1

            # Y_full > 0 means species1 wins; adjust sign for focal perspective
            y_focal_short = d_short.Y_full if focal_is_sp1 else -d_short.Y_full
            y_focal_long  = d_long.Y_full  if focal_is_sp1 else -d_long.Y_full

            # Dot color: blue = focal wins, red = focal loses, grey = tie
            def _outcome_color(y_val):
                if abs(y_val) < 0.05:
                    return "grey"
                return "#2166ac" if y_val > 0 else "#d6604d"

            color_short = _outcome_color(y_focal_short)
            color_long  = _outcome_color(y_focal_long)

            # Arrow/line color = opponent's species color
            opponent_color = species_colors.get(opponent, "grey")

            alpha_s = max(0.3, 1.0 - d_short.iota)
            alpha_l = max(0.3, 1.0 - d_long.iota)
            alpha_line = (alpha_s + alpha_l) / 2

            xs, ys = _to_ternary(d_short.e_W, d_short.e_T, d_short.e_R)
            xl, yl = _to_ternary(d_long.e_W,  d_long.e_T,  d_long.e_R)

            # Connecting arrow colored by opponent
            ax.annotate(
                "", xy=(xl, yl), xytext=(xs, ys),
                arrowprops=dict(
                    arrowstyle="-|>",
                    color=opponent_color,
                    lw=1.2,
                    alpha=alpha_line,
                ),
                zorder=2,
            )

            # Short-term: open circle
            ax.scatter(xs, ys, s=55,
                       facecolors="none", edgecolors=color_short,
                       linewidths=1.5, alpha=alpha_s, zorder=3)
            # Long-term: filled circle
            ax.scatter(xl, yl, s=55,
                       facecolors=color_long, edgecolors=color_long,
                       alpha=alpha_l, zorder=3)

        # ── Subplot title: focal species in its own color ──────────────────────
        focal_color = species_colors.get(focal_species, "black")
        title_y = 3**0.38 / 2 + 0.22
        ax.text(
            0.5, title_y, focal_species,
            ha="center", va="bottom", fontsize=10, fontweight="bold",
            color=focal_color, transform=ax.transData,
        )

    # Hide unused subplots
    for sp_idx in range(n_species, n_rows * n_cols):
        row, col = divmod(sp_idx, n_cols)
        axes[row, col].set_visible(False)

    # ── Shared legend ──────────────────────────────────────────────────────────
    legend_elements = [
        plt.scatter([], [], s=55, facecolors="none", edgecolors="grey",
                    linewidths=1.5, label="Short-term"),
        plt.scatter([], [], s=55, facecolors="grey", edgecolors="grey",
                    label="Long-term"),
        plt.scatter([], [], s=55, facecolors="#2166ac", edgecolors="#2166ac",
                    label="Focal species wins"),
        plt.scatter([], [], s=55, facecolors="#d6604d", edgecolors="#d6604d",
                    label="Focal species loses"),
        plt.Line2D([0], [0], color="grey", lw=1.5,
                   label="Arrow color = opponent species"),
    ]
    fig.legend(
        handles=legend_elements,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.0),
        ncol=3, fontsize=8, frameon=False,
    )

    fig.suptitle(
        "Effect decomposition per species  —  W: Water · T: Temperature · G/S: Growth and Shade",
        fontsize=11, fontweight="bold",
    )
    plt.tight_layout(rect=[0, 0.06, 1, 0.95], h_pad=4.0)

    return fig


def plot_growth_curves_competition(
    all_pairs: list[PairSimulationResults],
    species_colors: dict[str, str],
    alpha: float = 0.15,
    figsize_per_row: tuple = (14, 2.0),  # ← height halved from 4.0 to 2.0; adjust here
) -> plt.Figure:
    """
    Grid of growth curve plots, one row per species pair, three panels per row:
        Left panel   : water-equalized simulation   (equalizeWater=True,  
equalizeTemperature=False, equalizeRest=False)
        Middle panel : temperature-equalized sim     (equalizeWater=False, 
equalizeTemperature=True,  equalizeRest=False)
        Right panel  : full competition (no equalization)

    Each panel superimposes the growth curves of both species across all cells.
    Biomass is converted from g/m² to metric ton/ha (× 0.01).
    Individual cell curves are semi-transparent; a solid median is drawn on top.
    Within each row, all three panels share the same x- and y-axis limits.

    Parameters
    ----------
    all_pairs : list[PairSimulationResults]
    species_colors : dict[str, str]
        Output of make_species_colors.
    alpha : float
        Transparency of individual cell curves.
    figsize_per_row : tuple
        (width, height) per figure row. Change the second value to adjust row height.
    """
    CONFIGS = {
        "Water equalized":       (True,  False, False),
        "Temperature equalized": (False, True,  False),
        "Full competition":      (False, False, False),
    }

    G_M2_TO_T_HA = 0.01  # conversion factor

    n_pairs = len(all_pairs)

    fig, axes = plt.subplots(
        n_pairs, 3,
        figsize=(figsize_per_row[0], figsize_per_row[1] * n_pairs),
        squeeze=False,
    )

    for pair_idx, pair in enumerate(all_pairs):
        c1 = species_colors.get(pair.species1, "#333333")
        c2 = species_colors.get(pair.species2, "#333333")

        # Compute shared y-axis limits across all 3 configs for this row
        y_max_all, y_min_all = -np.inf, np.inf
        for config in CONFIGS.values():
            result = pair.results[config]
            b1 = result.biomass_sp1 * G_M2_TO_T_HA
            b2 = result.biomass_sp2 * G_M2_TO_T_HA
            if b1.size > 0:
                y_max_all = max(y_max_all, b1.max(), b2.max())
                y_min_all = min(y_min_all, b1.min(), b2.min())
        y_max_all *= 1.05

        for col_idx, (panel_title, config) in enumerate(CONFIGS.items()):
            ax = axes[pair_idx, col_idx]

            result    = pair.results[config]
            timesteps = result.timesteps
            b1        = result.biomass_sp1 * G_M2_TO_T_HA
            b2        = result.biomass_sp2 * G_M2_TO_T_HA

            # Individual cell curves (transparent)
            for cell_curve in b1:
                ax.plot(timesteps, cell_curve, color=c1, alpha=alpha, lw=0.8)
            for cell_curve in b2:
                ax.plot(timesteps, cell_curve, color=c2, alpha=alpha, lw=0.8)

            # Median curves
            ax.plot(timesteps, np.median(b1, axis=0), color=c1, lw=2.0,
                    alpha=0.9, label=pair.species1)
            ax.plot(timesteps, np.median(b2, axis=0), color=c2, lw=2.0,
                    alpha=0.9, label=pair.species2)

            ax.set_ylim(y_min_all, y_max_all)
            ax.set_xlim(timesteps[0], timesteps[-1])
            ax.spines[["top", "right"]].set_visible(False)

            # Panel title (config name) on EVERY row  ← changed
            ax.set_title(panel_title, fontsize=9, fontweight="bold")

            # y-axis label and ticks only on leftmost panel
            if col_idx == 0:
                ax.set_ylabel("Biomass (t/ha)", fontsize=8)
            else:
                ax.set_yticklabels([])

            # x-axis label only on bottom row
            if pair_idx == n_pairs - 1:
                ax.set_xlabel("Year", fontsize=8)

            # Legend on EVERY panel  ← changed
            ax.legend(fontsize=7.5, frameon=False, loc="upper left")

        # Row label (pair name) on the left margin
        axes[pair_idx, 0].annotate(
            f"{pair.species1} vs {pair.species2}",
            xy=(-0.22, 0.5), xycoords="axes fraction",
            fontsize=8, style="italic", color="grey",
            ha="center", va="center", rotation=90,
        )

    fig.suptitle(
        "Growth curves across all cells",
        fontsize=12, fontweight="bold", y=1.01,
    )
    plt.tight_layout()

    return fig

#######################################################
# FUNCTIONS TO CALIBRATE MAXPEST
#######################################################

import random

def create_species_csv_maxpestcalibration(listOfSpecies, numberOfCells, ageRange, filename='output.csv'):
    """Function used to create the .csv files to initialize
    the landscapes for the little simulations used for the calibration.
    This function is adapted to make the larger landscape to calibrate MaxPest."""
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')

        # Write header
        writer.writerow(['MapCode', 'SpeciesName', 'CohortAge', 'CohortBiomass'])

        # Loop through MapCode 1 to 1001
        for map_code in range(1, (numberOfCells*numberOfCells)+2):
            # Pick a random number of species
            # Pick a random number of elements between 1 and the length of the list
            k = random.randint(1, len(listOfSpecies))
            # Select k unique elements
            random_species = random.sample(listOfSpecies, k)
            
            # Write one row for each species
            for species in random_species:
                random_age = random.randint(ageRange[0], ageRange[1])
                writer.writerow([map_code, species, random_age, 0])


import copy

def calibrationSimulationLandscapeMaxPestManawan(duration = 500,
                                                climate = "realHistorical",
                                                soilsProportions = {"SILO":0.5, "SAND":0.25, "CLAY":0.25},
                                                timestep = 10,
                                                numberOfCells = 100,
                                                maxPest = 1,
                                                dictOfCoreSpeciesParameters = json.load(open('./SpeciesParametersSets/Initial/initialCoreSpeciesParameters.json')),
                                                dictOfPnETSpeciesParameters = json.load(open('./SpeciesParametersSets/Initial/initialPnETSpeciesParameters.json')),
                                                dictOfPnETGenericParameters = json.load(open('./SpeciesParametersSets/Initial/InitialGenericParameters.json')),
                                                plotResults = False,
                                                parallel = 0):    
    # We prepare the simulation files
    PnETGitHub_OneCellSim = parse_All_LANDIS_PnET_Files(r"./SimulationFiles/PnETGitHub_OneCellSim_v8")
    
    # - Species.txt : replace with the right initial core species parameters from the JSON file
    PnETGitHub_OneCellSim["species.txt"] = copy.deepcopy(dictOfCoreSpeciesParameters)
    
    # - SpeciesParameters.txt : replace with the initial PnET species parameters from the JSOn file
    PnETGitHub_OneCellSim["SpeciesParameters.txt"] = copy.deepcopy(dictOfPnETSpeciesParameters)
    
    # - PnETGenericParameters.txt : replace with initial generic parameters from the JSON file
    PnETGitHub_OneCellSim["PnETGenericParameters.txt"] = copy.deepcopy(dictOfPnETGenericParameters)

    listOfSpeciesToSimulate = list(PnETGitHub_OneCellSim["SpeciesParameters.txt"]["PnETSpeciesParameters"].keys())
    
    # Setting duration and timestep (minimal timestep for maximum spatial resolution, although it shouldn't change anything)
    PnETGitHub_OneCellSim["pnetsuccession.txt"]["Timestep"] = str(timestep)
    PnETGitHub_OneCellSim["scenario.txt"]["Duration"] = str(duration)
    # Adding all of the soil types we will need
    # In the default file, we have two SAND ecoregions.
    # First, we modify the second
    # Then, we add others
    PnETGitHub_OneCellSim["EcoregionParameters.txt"]["EcoregionParameters"]["eco1"]["SoilType"] = "SILO"
    soilTypesAlreadyThere = ["SAND", "SILO"]
    soilsTypesNumberDict = {"SILO":1, "SAND":2}
    ecoNumber = 2
    for soilType in soilsProportions.keys():
        if soilType not in soilTypesAlreadyThere:
            PnETGitHub_OneCellSim["EcoregionParameters.txt"]["EcoregionParameters"]["eco" + str(ecoNumber)] = copy.deepcopy(PnETGitHub_OneCellSim["EcoregionParameters.txt"]["EcoregionParameters"]["eco1"])
            PnETGitHub_OneCellSim["EcoregionParameters.txt"]["EcoregionParameters"]["eco" + str(ecoNumber)]["SoilType"] = soilType
            soilsTypesNumberDict[soilType] = ecoNumber + 1
            ecoNumber += 1
    # Finally, we need to add these in the ecoregion file too
    PnETGitHub_OneCellSim["ecoregion.txt"]["eco0"]["active"] = "yes"
    PnETGitHub_OneCellSim["ecoregion.txt"]["eco0"]["Map Code"] = "1"
    PnETGitHub_OneCellSim["ecoregion.txt"]["eco1"]["Map Code"] = "2"
    for ecoregion in PnETGitHub_OneCellSim["EcoregionParameters.txt"]["EcoregionParameters"].keys():
        if ecoregion not in PnETGitHub_OneCellSim["ecoregion.txt"].keys():
            PnETGitHub_OneCellSim["ecoregion.txt"][ecoregion] = copy.deepcopy(PnETGitHub_OneCellSim["ecoregion.txt"]["eco0"])
            PnETGitHub_OneCellSim["ecoregion.txt"][ecoregion]["Map Code"] = str(soilsTypesNumberDict[PnETGitHub_OneCellSim["EcoregionParameters.txt"]["EcoregionParameters"][ecoregion]["SoilType"]])
        PnETGitHub_OneCellSim["ecoregion.txt"][ecoregion]["Description"] = f'\"{PnETGitHub_OneCellSim["EcoregionParameters.txt"]["EcoregionParameters"][ecoregion]["SoilType"]} soils\"'
    
    # - pnetsuccession.txt : change startyear to 1900 and latitude to village of Manawan (47.2223)
    startYear = 1950
    PnETGitHub_OneCellSim["pnetsuccession.txt"]["StartYear"] = str(startYear)
    PnETGitHub_OneCellSim["pnetsuccession.txt"]["Latitude"] = "47.2223"
    # Dispersal activated
    PnETGitHub_OneCellSim["pnetsuccession.txt"]["SeedingAlgorithm"] = "WardSeedDispersal"
    PnETGitHub_OneCellSim["PnETGenericParameters.txt"]["PreventEstablishment"] = "False"
    # Setting other parameters
    PnETGitHub_OneCellSim["scenario.txt"]["CellLength"] = "100"
    # Changing MaxPest
    PnETGitHub_OneCellSim["PnETGenericParameters.txt"]["MaxPest"] = str(maxPest)

    # Activating parallelisation for faster sim
    if parallel > 0:
        PnETGitHub_OneCellSim["PnETGenericParameters.txt"]["Parallel"] = str(parallel)

    # Preparing the landscape
    # Inserting reading of the right climate file
    if climate == "mild":
        PnETGitHub_OneCellSim["pnetsuccession.txt"]["ClimateConfigFile"] = "ClimateConfigSimpleSims_MonthlyAveraged.txt"
    elif climate == "realHistorical":
        PnETGitHub_OneCellSim["pnetsuccession.txt"]["ClimateConfigFile"] = "ClimateConfigSimpleSims.txt"
    elif climate == "testFilesGithub":
        pass # The climate files from github are used by default if we don't input a climate config file
    else:
        raise ValueError("Climate value : " + str(climate) + " not recognized.")

    # Writing the files in a temporary folder
    simulationPath = "/tmp/landscapeCalibrationMaxPest/"

    # Removing PnET Output Site
    del PnETGitHub_OneCellSim["pnetsuccession.txt"]["PNEToutputsites"]
    del PnETGitHub_OneCellSim["PnEToutputsites_onecell.txt"]

    # We create the folder
    if not os.path.exists(simulationPath):
        os.mkdir(simulationPath)
    else:
        shutil.rmtree(simulationPath)
        os.mkdir(simulationPath)
    
    write_all_LANDIS_files(simulationPath,
                           PnETGitHub_OneCellSim,
                           True)

    # Copy the climate files
    if climate == "mild":
        shutil.copy("./SimulationFiles/ClimateConfigSimpleSims_MonthlyAveraged.txt", simulationPath)
        shutil.copy("./ReferencesAndData/Climate Data/dataFrameClimate_historicalMonthly_Ouranos_MonthlyAveraged.csv", simulationPath)
        shutil.copy("./ReferencesAndData/Climate Data/dataFrameClimate_SpinupMonthly_Ouranos_MonthlyAveraged.csv", simulationPath)
        os.remove(simulationPath + "/climate.txt")
        # I'm getting an issue when starting simulations at year 1900 with low longevity values. See https://github.com/LANDIS-II-Foundation/Library-Climate/issues/32
        # It seems to be related to the spinup code taking the wrong amount of year based on the longevity of the species
        # I'm removing years from the spinup file to avoid this
        spinupData = pd.read_csv(simulationPath + "dataFrameClimate_SpinupMonthly_Ouranos_MonthlyAveraged.csv")
        maxLongevity = 0
        for species in listOfSpeciesToSimulate:
            maxLongevity = max(float(dictOfCoreSpeciesParameters[species]["Longevity"]), maxLongevity)
        spinupData = spinupData[spinupData["Year"] > (spinupData['Year'].max() - int(maxLongevity))]
        spinupData.to_csv(simulationPath + "dataFrameClimate_SpinupMonthly_Ouranos_MonthlyAveraged.csv", index=False)
        # We edit the climate files to add the missing ecoregions to it
        dataFrameClimate_historicalMonthly_Ouranos_MonthlyAveraged = pd.read_csv(simulationPath + "/dataFrameClimate_historicalMonthly_Ouranos_MonthlyAveraged.csv")
        dataFrameClimate_SpinupMonthly_Ouranos_MonthlyAveraged = pd.read_csv(simulationPath + "/dataFrameClimate_SpinupMonthly_Ouranos_MonthlyAveraged.csv")
        for ecoregion in PnETGitHub_OneCellSim["EcoregionParameters.txt"]["EcoregionParameters"].keys():
            if ecoregion != "eco1":
                dataFrameClimate_historicalMonthly_Ouranos_MonthlyAveraged[ecoregion] = dataFrameClimate_historicalMonthly_Ouranos_MonthlyAveraged['eco1']
                dataFrameClimate_SpinupMonthly_Ouranos_MonthlyAveraged[ecoregion] = dataFrameClimate_SpinupMonthly_Ouranos_MonthlyAveraged['eco1']
        dataFrameClimate_historicalMonthly_Ouranos_MonthlyAveraged.to_csv(simulationPath + "/dataFrameClimate_historicalMonthly_Ouranos_MonthlyAveraged.csv", index=False)
        dataFrameClimate_SpinupMonthly_Ouranos_MonthlyAveraged.to_csv(simulationPath + "/dataFrameClimate_SpinupMonthly_Ouranos_MonthlyAveraged.csv", index=False)
        
    elif climate == "realHistorical":
        shutil.copy("./SimulationFiles/ClimateConfigSimpleSims.txt", simulationPath)
        shutil.copy("./ReferencesAndData/Climate Data/dataFrameClimate_historicalMonthly_Ouranos.csv", simulationPath)
        shutil.copy("./ReferencesAndData/Climate Data/dataFrameClimate_SpinupMonthly_Ouranos.csv", simulationPath)
        os.remove(simulationPath + "/climate.txt")
        # I'm getting an issue when starting simulations at year 1900 with low longevity values. See https://github.com/LANDIS-II-Foundation/Library-Climate/issues/32
        # It seems to be related to the spinup code taking the wrong amount of year based on the longevity of the species
        # I'm removing years from the spinup file to avoid this
        spinupData = pd.read_csv(simulationPath + "dataFrameClimate_SpinupMonthly_Ouranos.csv")
        maxLongevity = 0
        for species in listOfSpeciesToSimulate:
            maxLongevity = max(float(dictOfCoreSpeciesParameters[species]["Longevity"]), maxLongevity)
        spinupData = spinupData[spinupData["Year"] > (spinupData['Year'].max() - int(maxLongevity))]
        spinupData.to_csv(simulationPath + "dataFrameClimate_SpinupMonthly_Ouranos.csv", index=False)
        # We edit the climate files to add the missing ecoregions to it
        dataFrameClimate_historicalMonthly_Ouranos = pd.read_csv(simulationPath + "/dataFrameClimate_historicalMonthly_Ouranos.csv")
        dataFrameClimate_SpinupMonthly_Ouranos = pd.read_csv(simulationPath + "/dataFrameClimate_SpinupMonthly_Ouranos.csv")
        for ecoregion in PnETGitHub_OneCellSim["EcoregionParameters.txt"]["EcoregionParameters"].keys():
            if ecoregion != "eco1":
                dataFrameClimate_historicalMonthly_Ouranos[ecoregion] = dataFrameClimate_historicalMonthly_Ouranos['eco1']
                dataFrameClimate_SpinupMonthly_Ouranos[ecoregion] = dataFrameClimate_SpinupMonthly_Ouranos['eco1']
        dataFrameClimate_historicalMonthly_Ouranos.to_csv(simulationPath + "/dataFrameClimate_historicalMonthly_Ouranos.csv", index=False)
        dataFrameClimate_SpinupMonthly_Ouranos.to_csv(simulationPath + "/dataFrameClimate_SpinupMonthly_Ouranos.csv", index=False)

        
    elif climate == "testFilesGithub":
        pass # The climate files from github are used by default if we don't input a climate config file
    else:
        raise ValueError("Climate value : " + str(climate) + " not recognized.")
    # Removing climate.txt (old climate file from the test files)
    

    # Preparing rasters
    # numberOfCells = 500
    ageRange = [1, 100]
    # Preparing the data we will put in the rasters
    data = np.ones((1, numberOfCells), dtype=np.uint8)
    # Transform used to settle the size of cells - not sure is very useful
    transform = Affine.translation(0, 0) * Affine.scale(1, 1)
    # Creating the ecoregion raster
    # First, we need to translate the soil proportions into ecoregion codes
    finalSoilsProportionsDict = {}
    for soilType in soilsProportions.keys():
        if soilType not in soilsTypesNumberDict.keys():
            raise ValueError(f"Error : Soil type {soilType} not found in soilsTypesNumberDict.")
        finalSoilsProportionsDict[soilsTypesNumberDict[soilType]] = soilsProportions[soilType]
    # Then, we generate the map
    data_ecoregions = generate_soil_map((numberOfCells,numberOfCells),
                                        finalSoilsProportionsDict,
                                        20, # Mean patch size; can be adjusted
                                        soil_names = soilsTypesNumberDict, plot=True, seed=22)
    
    with rasterio.open(
    simulationPath + '/ecoregion.img',
    'w',
    driver='GTiff',
    height=numberOfCells,
    width=numberOfCells,
    count=1,
    dtype=data.dtype,
    crs='EPSG:4326',
    transform=transform
    ) as dst:
        dst.write(data_ecoregions, 1)
    # Preparing the initial communities raster
    data = np.arange(1, numberOfCells**2 + 1, dtype="int32").reshape(numberOfCells, numberOfCells)
    with rasterio.open(
    simulationPath + '/initial-communities.img',
    'w',
    driver='GTiff',
    height=numberOfCells,
    width=numberOfCells,
    count=1,
    dtype=data.dtype,
    crs='EPSG:4326',
    transform=transform
    ) as dst:
        dst.write(data, 1)
    # Creating initial community .csv
    create_species_csv_maxpestcalibration(listOfSpeciesToSimulate, numberOfCells, ageRange, filename = simulationPath + "/initial-communities.csv")

    # We launch the simulation
    runLANDIS_Simulation(simulationPath,
                         "scenario.txt",
                        False)

    cohort_balance_csv = pd.read_csv(f'{simulationPath}/output/CohortBalance.csv')

    cohortPerCellVector = [x / (numberOfCells*numberOfCells) for x in cohort_balance_csv["#Cohorts"].tolist()]

    if plotResults:
        # We plot the evolution in the average number of cohorts per cell
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(cohort_balance_csv["Time"], cohort_balance_csv["#Cohorts"]/(numberOfCells*numberOfCells), color="steelblue", linewidth=1, label="Cohorts per cell (average)")
        
        ax.set_xlabel("Timestep")
        ax.set_ylabel("Cohorts per cell (average)")
        ax.set_title(f"Cohorts per cell (average) through time in a landscape of dimensions {numberOfCells} * {numberOfCells} cells")
        plt.tight_layout()
        plt.show()

        # We plot a stack plot of the biomass per species
        speciesBiomass = pd.read_csv(f'{simulationPath}/output/WoodFoliageBiomass/WoodFoliageBiomass-AllYears.csv')
        speciesBiomass = speciesBiomass.drop(columns=['AllSpp_g/m2'])
        speciesBiomass = speciesBiomass.set_index("Time").copy()
        percentage = True # Switch if you want absolute values
        if percentage:
            speciesBiomass = speciesBiomass.div(speciesBiomass.sum(axis=1), axis=0) * 100

        fig, ax = plt.subplots(figsize=(10, 6))
    
        ax.stackplot(speciesBiomass.index, speciesBiomass.T, labels=speciesBiomass.columns)
        ax.legend(loc="upper left")
        ax.set_xlabel("Time")
        ax.set_ylabel("Biomass (%)" if percentage else "Biomass")
        ax.set_title("Species biomass through time")
        if percentage:
            ax.set_ylim(0, 100)


    return(cohortPerCellVector)

    # # We get the result file : only .csv file that should represent the cohort
    # csv_file_cohort = pd.read_csv(glob.glob(f'{simulationPath}/Output/Site1/Cohort_*.csv')[0])
    # # print(csv_file_cohort)
    # # Add a measure of total biomass
    # csv_file_cohort["SumFoliageWood_Site"] = csv_file_cohort["SiteFol(gDW)"] + csv_file_cohort["SiteWood(gDW)"]
    # # We also get the foliage biomass from another output; this is because the foliage+wood biomass in the cohort file
    # # is in gDW, or gram of dry weight. But we want g/m2 to compare to NFI data.
    # csv_file_WoodFoliageBiomass = pd.read_csv(f'{simulationPath}output/WoodFoliageBiomass/WoodFoliageBiomass-AllYears.csv')
    # # We also get the site csv that has the temperatures (for diagnostics)
    # csv_file_site = pd.read_csv(f'{simulationPath}/Output/Site1/Site.csv')
    
    # # We get the results we need
    # variablesOutput = ["Biomass peak height", "Biomass peak time", "Biomass peak 95% time",
    #                    "Initation of decline", "Time of death",
    #                    "Maximum LAI", "LAI stability", "Average Fwater",
    #                   "Average July Temperature", "Biomass at 50% of biomass peak 95% time"]
    # dictOfOutput = {}
    # for variable in variablesOutput:
    #     if variable == "Biomass peak height":
    #         dictOfOutput[variable] = (csv_file_WoodFoliageBiomass[str(speciesToSimulate) + "_g/m2"].max())
    #     elif variable == "Biomass peak time":
    #         max_index = csv_file_cohort['SumFoliageWood_Site'].idxmax()
    #         # Here, we want the age of the cohort when it reaches the maximum
    #         # Since there is one row per month, and since the first row is the first month of life
    #         # of the cohort (even if there is a spinup), then the age of the cohorts (in years) is
    #         # simply row/12 (since 12 rows make a year of life.
    #         dictOfOutput[variable] = ((max_index+1)/12)
    #     elif variable == "Biomass peak 95% time":
    #         # Calculate 95% of the maximum value in the column
    #         threshold_value = csv_file_cohort['SumFoliageWood_Site'].max() * 0.95
            
    #         # Find the index of the first row where the column value is greater than or equal to the threshold
    #         # The .idxmax() method returns the index of the first occurrence of the maximum value.
    #         # To find the first occurrence of a value that meets a condition, we can create a boolean series.
    #         # However, idxmax() is for maximum values. A more direct approach for a threshold is to filter.
            
    #         # Filter the DataFrame to get rows that meet the condition
    #         rows_meeting_condition = csv_file_cohort[csv_file_cohort['SumFoliageWood_Site'] >= threshold_value]
            
    #         # Get the index of the first row from the filtered DataFrame
    #         if not rows_meeting_condition.empty:
    #             first_index = rows_meeting_condition.index[0]
    #             dictOfOutput[variable] = ((first_index+1)/12)
    #             # print(f"The first index where the column value is at least 95% of the maximum is: {first_index}")
    #         else:
    #             # print("No rows found that meet the condition.")
    #             dictOfOutput[variable] = "None"
    #     elif variable == "Initation of decline":
    #         # When fAge goes under 0.90 for the first time
    #         # Create a boolean mask for rows where 'my_column' is less than the threshold
    #         condition_met = csv_file_cohort['fage(-)'] < 0.9
    #         first_occurrence_index = csv_file_cohort.index[condition_met].min()
    #         # We want to get the age of the cohort where fAge goes under 0.90
    #         # Since one row per month, it's simply the number of row/index divided by 12.
    #         dictOfOutput[variable] = ((first_occurrence_index+1)/12)
    #     elif variable == "Time of death":
    #         # Cohort dies when NSCfrac is inferior to 0.01 at the end of a year (december)
    #         # We get the index of the row when this happens and divided it by 12 (since there
    #         # is one row per month), this gives us the age (in years) of death.
    #         mask = (csv_file_cohort['Month'] == 12) & (csv_file_cohort['NSCfrac(-)'] < 0.01)
    #         try:
    #             idx = csv_file_cohort[mask].index[0]
    #             dictOfOutput[variable] = ((idx+1)/12)
    #         except:
    #             # Code to handle any other error
    #             # print("Havent found the index where the cohort died. Might be an issue with the cohort.csv")
    #             dictOfOutput[variable] = "None"
    #     elif variable == "Maximum LAI":
    #         dictOfOutput[variable] = csv_file_cohort["LAI(m2)"].max()
    #     elif variable == "LAI stability":
    #         # dictOfOutput[variable] = min_max_top30_contiguous(csv_file_cohort["LAI(m2)"])

    #         # Step 1: Get max yearly LAI per year
    #         yearly_max_lai = csv_file_cohort.groupby("Year")["LAI(m2)"].max()
            
    #         # Step 2: Find the year where LAI reaches its overall maximum
    #         year_of_max_lai = yearly_max_lai.idxmax()
            
    #         # Step 3: Find the last year before fAge(-) drops below 0.6
    #         year_before_fage = csv_file_cohort[csv_file_cohort["fage(-)"] >= 0.7]["Year"].max()
            
    #         # Step 4: Filter and get the range (min, max) of yearly max LAI values in that window
    #         result = yearly_max_lai.loc[year_of_max_lai:year_before_fage]
    #         dictOfOutput[variable] = (result.min(), result.max())
            
    #     elif variable == "Average Fwater":
    #         dictOfOutput[variable] = csv_file_cohort["fWater(-)"].mean()
    #     elif variable == "Average July Temperature":
    #         dictOfOutput[variable] = csv_file_site[csv_file_site['Month'] == 7]['Tday(C)'].mean()
    #     elif variable == "Biomass at 50% of biomass peak 95% time":
    #         # Used to check if a cohort grows fast enough in its young years (see subphase 1.3)
    #         # We need to interpolate because the peak time is from the cohort file with a monthly timestep,
    #         # while the biomass peak heigh is from the csv file with the same timestep as PnET-Succession (e.g. 5 years)
    #         df = csv_file_WoodFoliageBiomass[["Time", str(speciesToSimulate) + "_g/m2"]].copy()
    #         df = df.sort_values("Time").reset_index(drop=True)
        
    #         # Rows below and above the target time
    #         below = df[df["Time"] <= 0.5*dictOfOutput["Biomass peak 95% time"]]
    #         above = df[df["Time"] >= 0.5*dictOfOutput["Biomass peak 95% time"]]
        
    #         # Exact match — no interpolation needed
    #         if not below.empty and below.iloc[-1]["Time"] == 0.5*dictOfOutput["Biomass peak 95% time"]:
    #             return below.iloc[-1][str(speciesToSimulate) + "_g/m2"]
    #         if not above.empty and above.iloc[0]["Time"] == 0.5*dictOfOutput["Biomass peak 95% time"]:
    #             return above.iloc[0][str(speciesToSimulate) + "_g/m2"]
        
    #         # Check that bracketing rows exist on both sides
    #         if below.empty or above.empty:
    #             # raise ValueError(
    #             #     f"dictOfOutput["Biomass peak 95% time"]={dictOfOutput["Biomass peak 95% time"]} is outside the range "
    #             #     f"of {"Time"} in csv_file_WoodFoliageBiomass "
    #             #     f"({df["Time"].min()} – {df["Time"].max()})."
    #             # )
    #             dictOfOutput[variable] = "Biomass peak 95% time is outside the range in csv_file_WoodFoliageBiomass"
        
    #         t0, b0 = below.iloc[-1]["Time"], below.iloc[-1][str(speciesToSimulate) + "_g/m2"]
    #         t1, b1 = above.iloc[0]["Time"],  above.iloc[0][str(speciesToSimulate) + "_g/m2"]
        
    #         # Linear interpolation
    #         interpolated_biomass = b0 + (b1 - b0) * (0.5*dictOfOutput["Biomass peak 95% time"] - t0) / (t1 - t0)
        
    #         dictOfOutput[variable] = interpolated_biomass
    #     else:
    #         raise ValueError("Value not recognized for output variable : " + str(variable))

    # if plotResults: plot_all_cohort_results(str(simulationPath) + "/Output/Site1", {speciesToSimulate:"#5e81ac"})

    # if plotAdjustedParameters:
    #     fRad = csv_file_cohort['fRad(-)']
        
    #     # --- Parameters ---
    #     MaxFracFol   = float(dictOfInitialPnETSpeciesParameters["PnETSpeciesParameters"][speciesToSimulate]["MaxFracFol"])
    #     FracFolShape = float(dictOfInitialPnETSpeciesParameters["PnETSpeciesParameters"][speciesToSimulate]["FracFolShape"])
    #     FracFol      = float(dictOfInitialPnETSpeciesParameters["PnETSpeciesParameters"][speciesToSimulate]["FracFol"])
        
    #     # --- Rolling average window ---
    #     rolling_window = 12  # <-- specify window size here
        
    #     # --- Computation ---
    #     AdjustedFracFol = FracFol + ((MaxFracFol - FracFol) * (fRad ** FracFolShape))
    #     RollingAvg      = AdjustedFracFol.rolling(window=rolling_window, center=True).mean()
    #     RollingAvgFRad  = fRad.rolling(window=rolling_window, center=True).mean()
        
    #     # --- Plot ---
    #     fig, ax = plt.subplots(figsize=(10, 4))
    #     ax2     = ax.twinx()  # secondary y-axis sharing the same x-axis
        
    #     # Push ax2 (fRad) behind ax (everything else)
    #     ax.set_zorder(ax2.get_zorder() + 1)
    #     ax.patch.set_visible(False)   # let ax2's content show through ax's transparent background
        
    #     # fRad (background, noisy) + its rolling average (background, solid green)
    #     l5, = ax2.plot(csv_file_cohort['Year'], fRad, color="lightgreen", linewidth=1,
    #                    alpha=0.6, zorder=1, label="fRad")
    #     l6, = ax2.plot(csv_file_cohort['Year'], RollingAvgFRad, color="green", linewidth=2,
    #                    zorder=2, label=f"fRad rolling avg (window={rolling_window})")
        
    #     # Foreground curves
    #     l1, = ax.plot(csv_file_cohort['Year'], AdjustedFracFol, color="darkorange", linewidth=1,
    #                   zorder=3, linestyle="--", alpha = 0.5, label="AdjFracFol (FracFolShape = " + str(dictOfInitialPnETSpeciesParameters["PnETSpeciesParameters"][speciesToSimulate]["FracFolShape"]) + ")")
    #     l2, = ax.plot(csv_file_cohort['Year'], RollingAvg, color="darkorange", linewidth=2,
    #                   zorder=4, label=f"AdjFracFol Rolling avg (window={rolling_window})")
    #     l3  = ax.axhline(FracFol,    color="blue",   linestyle="--", linewidth=1, zorder=4,
    #                       label=f"FracFol = {FracFol}")
    #     l4  = ax.axhline(MaxFracFol, color="tomato", linestyle="--", linewidth=1, zorder=4,
    #                       label=f"MaxFracFol = {MaxFracFol}")
        
    #     ax.set_xlabel("Timestep")
    #     ax.set_ylabel("AdjustedFracFol")
    #     ax2.set_ylabel("fRad (-)")
    #     ax.set_title("AdjustedFracFol over time")
        
    #     # Combined legend, drawn on ax so it sits above everything (including ax2's patch)
    #     lines  = [l1, l2, l3, l4, l5, l6]
    #     labels = [l.get_label() for l in lines]
    #     ax.legend(lines, labels, loc="best").set_zorder(5)
        
    #     plt.tight_layout()
    #     plt.show()



    #     # --- Parameters ---
    #     MaxFolN   = float(dictOfInitialPnETSpeciesParameters["PnETSpeciesParameters"][speciesToSimulate]["MaxFolN"])
    #     FolNShape = float(dictOfInitialPnETSpeciesParameters["PnETSpeciesParameters"][speciesToSimulate]["FolNShape"])
    #     FolN      = float(dictOfInitialPnETSpeciesParameters["PnETSpeciesParameters"][speciesToSimulate]["FolN"])
        
    #     # --- Computation ---
    #     AdjustedFolN = FolN + ((MaxFolN - FolN) * (fRad ** FolNShape))
        
    #     # --- Plot ---
    #     fig, ax = plt.subplots(figsize=(10, 4))
    #     ax.plot(csv_file_cohort['Year'], AdjustedFolN, color="steelblue", linewidth=1, label="AdjustedFolN")
    #     ax.axhline(FolN,    color="gray",   linestyle="--", linewidth=1, label=f"FolN = {FolN}")
    #     ax.axhline(MaxFolN, color="tomato", linestyle="--", linewidth=1, label=f"MaxFolN = {MaxFolN}")
        
    #     ax.set_xlabel("Timestep")
    #     ax.set_ylabel("AdjustedFolN")
    #     ax.set_title("AdjustedFolN over time")
    #     ax.legend()
    #     plt.tight_layout()
    #     plt.show()

    
    # if saveGrowthCurvePath:
    #     output_path = Path(saveGrowthCurvePath)

    #     # Create all parent directories if they don't exist
    #     output_path.parent.mkdir(parents=True, exist_ok=True)
    
    #     csv_file_WoodFoliageBiomass.to_csv(output_path)
    
    # return(dictOfOutput)
    # Delete the folder
    # shutil.rmtree(simulationPath)


#######
# Functions to make soil maps
#######

import random
from collections import defaultdict
import numpy as np


class _IndexedSet:
    """Set with O(1) add / discard / uniform random pick."""
    __slots__ = ("items", "index")

    def __init__(self):
        self.items = []
        self.index = {}

    def add(self, item):
        if item not in self.index:
            self.index[item] = len(self.items)
            self.items.append(item)

    def discard(self, item):
        idx = self.index.pop(item, None)
        if idx is None:
            return
        last = self.items.pop()
        if idx < len(self.items):
            self.items[idx] = last
            self.index[last] = idx

    def random(self, rng):
        return self.items[rng.randrange(len(self.items))]

    def __len__(self):
        return len(self.items)

    def __bool__(self):
        return bool(self.items)


def generate_soil_map(size, proportions, mean_patch_size, soil_names=None,
                       plot=False, seed=None):
    """
    Generate a 2-D numpy array representing a randomized soil map made of
    organic patches, matching the requested proportions per soil code.

    Parameters
    ----------
    size : tuple(int, int)
        (rows, cols) of the map.
    proportions : dict[int, float]
        {soil_code: proportion}. Must sum to 1.
    mean_patch_size : float
        Target average number of cells per patch (controls patch size).
    soil_names : dict[str, int], optional
        {soil_name: soil_code}, used to label the plot legend.
    plot : bool, optional
        If True, display the map with matplotlib.
    seed : int, optional
        Seed for reproducibility.

    Returns
    -------
    np.ndarray (rows, cols) of int, dtype matching soil codes.
    """
    rows, cols = size
    n_cells = rows * cols
    if n_cells <= 0:
        raise ValueError("size must contain two positive integers.")
    if mean_patch_size <= 0:
        raise ValueError("mean_patch_size must be > 0.")

    codes = list(proportions.keys())
    props = np.array([proportions[c] for c in codes], dtype=float)
    if not np.isclose(props.sum(), 1.0):
        raise ValueError("Proportions must sum to 1.")

    rng = random.Random(seed)

    # --- Exact cell counts via largest-remainder method ---
    raw = props * n_cells
    counts = np.floor(raw).astype(int)
    remainder = n_cells - counts.sum()
    order = np.argsort(-(raw - counts))
    for i in range(remainder):
        counts[order[i]] += 1
    target_counts = {c: int(n) for c, n in zip(codes, counts)}
    remaining = dict(target_counts)

    # --- Seeds per code (controls patch size) ---
    n_seeds = {
        c: max(1, min(target_counts[c], round(target_counts[c] / mean_patch_size)))
        for c in codes
    }

    grid = np.full((rows, cols), -1, dtype=int)

    unassigned = _IndexedSet()
    for r in range(rows):
        for c in range(cols):
            unassigned.add((r, c))

    def neighbours(cell):
        r, c = cell
        if r > 0: yield (r - 1, c)
        if r < rows - 1: yield (r + 1, c)
        if c > 0: yield (r, c - 1)
        if c < cols - 1: yield (r, c + 1)

    frontier = defaultdict(_IndexedSet)

    def place_seed(code):
        if remaining[code] <= 0 or not unassigned:
            return False
        cell = unassigned.random(rng)
        grid[cell] = code
        unassigned.discard(cell)
        remaining[code] -= 1
        frontier[code].add(cell)
        return True

    # initial seeds
    for c in codes:
        for _ in range(n_seeds[c]):
            if not place_seed(c):
                break

    # --- Random growth loop ---
    active_codes = [c for c in codes if remaining[c] > 0]
    while active_codes and unassigned:
        code = rng.choice(active_codes)

        if not frontier[code]:
            place_seed(code)
            active_codes = [c for c in codes if remaining[c] > 0]
            continue

        fcell = frontier[code].random(rng)
        candidates = [n for n in neighbours(fcell) if grid[n] == -1]

        if not candidates:
            frontier[code].discard(fcell)
            continue

        new_cell = rng.choice(candidates)
        grid[new_cell] = code
        unassigned.discard(new_cell)
        remaining[code] -= 1
        frontier[code].add(new_cell)

        if not any(grid[n] == -1 for n in neighbours(fcell)):
            frontier[code].discard(fcell)

        active_codes = [c for c in codes if remaining[c] > 0]

    if plot:
        _plot_soil_map(grid, codes, soil_names)

    return grid


def _plot_soil_map(grid, codes, soil_names):
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap, BoundaryNorm
    from matplotlib.patches import Patch

    # code -> name lookup (fallback to str(code) if not provided/missing)
    code_to_name = {}
    if soil_names:
        code_to_name = {code: name for name, code in soil_names.items()}

    sorted_codes = sorted(codes)
    cmap = plt.get_cmap("tab20", len(sorted_codes))
    boundaries = [c - 0.5 for c in sorted_codes] + [sorted_codes[-1] + 0.5]
    norm = BoundaryNorm(boundaries, cmap.N)

    plt.figure(figsize=(6, 6))
    plt.imshow(grid, cmap=cmap, norm=norm)
    plt.title("Randomized Soil Map")

    legend_handles = [
        Patch(color=cmap(i), label=code_to_name.get(code, str(code)))
        for i, code in enumerate(sorted_codes)
    ]
    plt.legend(handles=legend_handles, bbox_to_anchor=(1.02, 1), loc="upper left",
               title="Soil type", borderaxespad=0.)
    plt.tight_layout()
    plt.show()