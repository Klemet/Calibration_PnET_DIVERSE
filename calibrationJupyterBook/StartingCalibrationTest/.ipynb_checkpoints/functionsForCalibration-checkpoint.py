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
    keys = [
        "Longevity",
        "Sexual Maturity",
        "Shade Tolerance",
        "Fire Tolerance",
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
        if filename.endswith('.txt'):
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
                elif ["LandisData", "Initial Communities"] in content_separated or ["LandisData", '"Initial Communities"'] in content_separated or ["LandisData", "Initial", "Communities"] in content_separated or ['LandisData', '"Initial', 'Communities"'] in content_separated:
                    print("Found : Initial Communities file : " + str(filename))
                    results[filename] = parse_LANDIS_SimpleParameterFile(file_path)

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
        file.write(">> Active  Code  Name   Description\n")
        file.write(">> ------  ----  -----  -----------\n")

        # Define fixed widths for each column
        widths = {
            'active': 8,
            'Map code': 4,
            'name': 8,
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
                print("The LANDIS-II simulation has finished properly !")
                break
        
            # Prints the last line
            if printSim:
                print(output.splitlines()[-1])


    
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


def plot_TimeSeries_CSV_PnETSitesOutputs(df, referenceDict = {}, columnToPlotSelector = [], trueTime = False, realBiomass = True, cellLength = 30):
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
        if realBiomass and (column == "Wood(gDW)" or column == "Root(gDW)" or column == "Fol(gDW)"):
            columnData = df[column] * (cellLength*cellLength)
            columnData = columnData / 1000000
            columnName = column[:-5] + " (Mg or Metric tons)"
        else:
            columnData = df[column]
        
        
        if trueTime:
            plt.plot(df['Time'], columnData, label = "PnET Succession")
        else: #We edit the time to remove the years (e.g. 2000, 2001, etc.) and just use 0 as starting year. Makes things easier for the reference curve.
            timeNormalized = df['Time'] - min(df['Time'])
            plt.plot(timeNormalized, columnData, label = "PnET Succession")
            
        # If reference curve exists for the variable, we display it on the curve
        if column in referenceDict:
            plt.plot(referenceDict[column]["Time"], referenceDict[column]["Values"], color='#ebcb8b', label = "Reference")
        
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
                                        numberOfTrees,
                                        siteIndex,
                                        variant = "FVSon",
                                        timestep = 10,
                                        numberOfTimesteps = 12,
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
    'Site_Index': str(siteIndex)
    }

    columns = ', '.join(data_to_insert.keys())
    values = ', '.join(['%s'] * len(data_to_insert))
    sql = "INSERT INTO FVS_StandInit (" + str(columns) + ") VALUES" + str(tuple(data_to_insert.values()))

    # print(sql)

    cursor.execute(sql)
    # print(f"{cursor.rowcount} record inserted in database.")

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
    result = subprocess.run(['FVSon', '--keywordfile=SingleStandSim_Keywords.key'], cwd=folderForFiles, capture_output=True, text=True)
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
    target_string_lastLineBeforeValues = "YEAR    Total    Merch     Live     Dead     Dead      DDW    Floor  Shb/Hrb   Carbon   Carbon  from Fire"
    target_dashesLine = "--------------------------------------------------------------------------------------------------------------"

    with open(folderForFiles + "/SingleStandSim_Keywords.out", 'r') as file:
        lines = file.readlines()

    # Initialize variables to track whether we've found the target string and to collect report lines
    found_target_carbonReport = False
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
    
    # Delete the folder created for the inputs and outputs if specified
    if clearFiles:
        print("Clearing files")
        shutil.rmtree(folderForFiles)
        print(f"The directory '{folderForFiles}' has been deleted.")
    
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