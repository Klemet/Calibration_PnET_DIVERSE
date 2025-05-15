#!/usr/bin/env python3
import os
import sys
import glob
import re

# The goal of this script is to automatically insert all relevant references to the dll
# used by LANDIS-II (especially the extensions) to the Console.csproj file before recompiling
# the core at the end. The location of the folder with the dll and of Console.csproj are given
# as arguments to the script.
# This script was done with a LLM. It looks at existing dll references in Console.csproj (to avoid
# duplicates); then look at all of the dll files, select which one to insert (focusing on the dll of the
# extensions, avoiding the Landis.Console.dll and the raster libraries to avoid errors during the recompiling
# of the core; also adding the library log4net which seem to be needed), and then insert the references in
# Console.csproj.

def find_existing_references(file_content):
    """
    Find all existing references in the file content that might correspond to DLLs.
    Returns a set of base names (without extension) that are already referenced.
    """
    # Pattern to match any Include="something" attribute
    pattern = r'Include="([^"]+)"'
    matches = re.findall(pattern, file_content)

    # Extract base names without extensions or version info
    base_names = set()
    for match in matches:
        # Split on comma to handle version info
        base_name = match.split(',')[0]
        # Remove path if present
        base_name = os.path.basename(base_name)
        # Remove extension if present
        base_name = os.path.splitext(base_name)[0]
        base_names.add(base_name)

    return base_names

def generate_dll_references(folder_path, existing_references):
    """
    Generate XML-like string for Landis or log4net DLL files in the specified folder,
    excluding those that match existing references.
    """
    # Find all .dll files in the folder
    dll_pattern = os.path.join(folder_path, "*.dll")
    all_dll_files = glob.glob(dll_pattern)

    if not all_dll_files:
        print(f"No DLL files found in {folder_path}")
        return None

    # Filter to only include DLLs that start with "Landis" or "log4net"
    # Also filter to avoid main libraries ?
    dll_files = []
    # These .dll need to avoid being referenced to avoid creating errors once
    # LANDIS-II is running. Not sure why.
    filterToAvoid = ["Landis.Console",
                     "Landis.Raster",
                     "Landis.Landscapes",
                     "Landis.Core",
                     "Landis.SpatialModeling",
                     "SiteHarvest",
                     "Landis.Utilities",
                     "Landis.Extensions.Dataset",
                     "Landis.Library.Parameters"]
    for dll_path in all_dll_files:
        filename = os.path.basename(dll_path)
        if filename.startswith("Landis") or filename.startswith("log4net"):
            if not any([x in filename for x in filterToAvoid]):
                dll_files.append(dll_path)

    if not dll_files:
        print(f"No Landis or log4net DLL files found in {folder_path}")
        return None

    # Extract filenames and check against existing references
    new_dlls = []
    for dll_path in dll_files:
        dll_filename = os.path.basename(dll_path)
        # Get base name without extension
        base_name = os.path.splitext(dll_filename)[0]

        # Only add if the base name is not already referenced
        if base_name not in existing_references:
            new_dlls.append(dll_filename)

    if not new_dlls:
        print("All relevant DLLs are already referenced in the file. No changes needed.")
        return None

    # Start building the output string
    output = "<ItemGroup>\n"

    # Add a reference for each new DLL file
    for dll_filename in new_dlls:
        reference = f"    <Reference Include=\"{dll_filename}\">\n" \
                   f"        <HintPath>../../build/extensions/{dll_filename}</HintPath>\n" \
                   f"    </Reference>\n"
        output += reference

    # Close the ItemGroup tag and add the Project closing tag
    output += "</ItemGroup></Project>"

    print(f"Adding {len(new_dlls)} new DLL references: {', '.join(new_dlls)}")
    return output

def insert_at_project_end(file_path, content):
    """
    Insert content at the last occurrence of </Project> in the file.
    """
    try:
        # Read the entire file
        with open(file_path, 'r') as file:
            file_content = file.read()

        # Find the last occurrence of </Project>
        project_end_pos = file_content.rfind("</Project>")

        if project_end_pos == -1:
            print("Error: No </Project> tag found in the file")
            return False

        # Replace the last </Project> with our content
        new_content = file_content[:project_end_pos] + content

        # Write the modified content back to the file
        with open(file_path, 'w') as file:
            file.write(new_content)

        print(f"Successfully inserted content into {file_path}")
        return True

    except Exception as e:
        print(f"Error modifying file: {e}")
        return False

def main():
    # Check if correct number of arguments is provided
    if len(sys.argv) != 3:
        print("Usage: python script.py /path/to/folder/ /path/to/text_file.txt")
        return

    folder_path = sys.argv[1]
    output_file = sys.argv[2]

    # Validate folder path
    if not os.path.isdir(folder_path):
        print(f"Error: {folder_path} is not a valid directory")
        return

    # Validate output file exists
    if not os.path.isfile(output_file):
        print(f"Error: {output_file} is not a valid file")
        return

    # Read the file content to find existing references
    try:
        with open(output_file, 'r') as file:
            file_content = file.read()

        existing_references = find_existing_references(file_content)
        print(f"Found {len(existing_references)} existing references")

        # Generate the DLL references, excluding existing ones
        dll_references = generate_dll_references(folder_path, existing_references)

        if dll_references:
            # Insert at the last </Project> tag
            insert_at_project_end(output_file, dll_references)
        else:
            print("No changes made to the file")

    except Exception as e:
        print(f"Error processing file: {e}")

if __name__ == "__main__":
    main()
