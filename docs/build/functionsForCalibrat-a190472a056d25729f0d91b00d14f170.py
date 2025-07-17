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
from datetime import datetime
import geopandas as gpd
import cftime
import requests
from tqdm import tqdm
import matplotlib.dates as mdates
from matplotlib.colors import to_rgba, LinearSegmentedColormap
from matplotlib.gridspec import GridSpec
from rasterio.windows import Window
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
    # To speed up : If we're in same year/month, we avoid re-reading the data
    year_previous = "-99"
    month_previous = "-99"
    value_previous = 0
    for index, row in df.iterrows():
        year = int(row['year'])
        month = int(row['month'])

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
    plot_figsize=(20, 8)
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

        print("✓ Rasters loaded successfully")
        return prevalence, age, biomass

    def process_data(prevalence, age, biomass):
        """Process the data and create masks"""
        print("Processing data and applying masks...")

        # Mask pixels with no species presence
        mask = (prevalence > 0) & (~np.isnan(prevalence)) & (~np.isnan(age)) & (~np.isnan(biomass))

        # Calculate species biomass (prevalence as proportion * total biomass)
        species_biomass = (prevalence / 100.0) * biomass

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
    prevalence, age, biomass = read_rasters()

    # Process data
    age_flat, species_biomass_flat, prevalence_flat = process_data(prevalence, age, biomass)

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
    
        # Apply additional mask if provided (keep pixels with value 0, exclude pixels with value > 1)
        if mask is not None:
            print("Mask has been detected. Masking...")
            values, counts = np.unique(mask, return_counts=True)
            print("Values:", values)
            print("Counts:", counts)
            valid_mask = (~np.isnan(abundance)) & (~np.isnan(age)) & (~np.isnan(biomass)) & (abundance > 0) & (mask < 1)
            values, counts = np.unique((mask < 1), return_counts=True)
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
    
        # Fit the model with weights
        gam.fit(X, y, weights=weights)
    
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