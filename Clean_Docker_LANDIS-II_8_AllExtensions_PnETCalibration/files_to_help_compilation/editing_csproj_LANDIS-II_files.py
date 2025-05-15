import sys
import re

def remove_previous_debug_path_and_others(file_path):
    """
    This function removes the already existing lines in the style of
    <PropertyGroup Condition="'$(Configuration)|$(Platform)'=='Debug|AnyCPU'"> that
    gives the path were the output should be put. This is because we're going to add our own,
    and such lines will mess up where the dll are put after output.
    """
    
    fileEditLines = list()
    with open(file_path, 'r') as file:
        ignoreUntilClosingPropertyGroup = False
        ignoreUntilClosingTarget = False
        for line in file:
            # Check if </HintPath> is in the line
            if '<PropertyGroup Condition="\'$(Configuration)|$(Platform)\'==\'Debug|AnyCPU\'">' in line or '<PropertyGroup Condition="\'$(Configuration)|$(Platform)\'==\'Release|AnyCPU\'">' in line:
                ignoreUntilClosingPropertyGroup = True
                # print("PROPERTY GROUP FOUND !")
            elif '<Target Name="PostBuild" AfterTargets="PostBuildEvent">' in line:
                ignoreUntilClosingTarget = True
            else:
                # print("Property group not found.")
                if not ignoreUntilClosingPropertyGroup and not ignoreUntilClosingTarget:
                    fileEditLines.append(line)
                else:
                    if "</PropertyGroup>" in line:
                        ignoreUntilClosingPropertyGroup = False
                    elif "</Target>" in line:
                        ignoreUntilClosingTarget = False
        
    with open(file_path, 'w') as file:
       for modified_line in fileEditLines:
           file.write(modified_line)

def replace_in_file(file_path):
    """
    This functions adds the lines necessary to the csproj file to to output the dll to the right place.
    Takes into account if the edit is already present.
    """
    
    # Define the strings to search for and replace
    search_string = "  </PropertyGroup>"
    replace_string_withoutAppendTarget = (
        "  </PropertyGroup>\n"
        "  <PropertyGroup Condition=\"'$(Configuration)|$(Platform)'=='Release|AnyCPU'\">\n"
        "    <OutputPath>..\\..\\build\\extensions</OutputPath>\n"
        "  </PropertyGroup>"
    )
    replace_string_withAppendTarget = (
    "    <AppendTargetFrameworkToOutputPath>false</AppendTargetFrameworkToOutputPath>"
    "  </PropertyGroup>\n"
    "  <PropertyGroup Condition=\"'$(Configuration)|$(Platform)'=='Release|AnyCPU'\">\n"
    "    <OutputPath>..\\..\\build\\extensions</OutputPath>\n"
    "  </PropertyGroup>"
    )
    check_string = "<AppendTargetFrameworkToOutputPath>false</AppendTargetFrameworkToOutputPath>"

    # Read the file content
    with open(file_path, 'r') as file:
        content = file.read()

    # Check if the check_string is not present in the content
    if check_string not in content and replace_string_withAppendTarget not in content:
        # Replace the first occurrence of search_string with replace_string
        updated_content = content.replace(search_string, replace_string_withAppendTarget, 1)

        # Write the updated content back to the file
        with open(file_path, 'w') as file:
            file.write(updated_content)
        print("AppendTargetFrameworkToOutputPath not found. Replacing with AppendTargetFrameworkToOutputPath added.")
    else:
        if replace_string_withoutAppendTarget not in content:
            updated_content = content.replace(search_string, replace_string_withoutAppendTarget, 1)
            with open(file_path, 'w') as file:
                file.write(updated_content)
            print("AppendTargetFrameworkToOutputPath found. Replacing without AppendTargetFrameworkToOutputPath added.")


def replace_hint_paths(file_path):
    """
    This function replaces the paths in the csproj file
    with ones that correctly point out to where the support libraries are found,
    according to the tutorial to compile LANDIS-II for Linux on the repositories of 
    the LANDIS-II foundation
    """
    
    fileEditLines = list()
    with open(file_path, 'r') as file:
        for line in file:
            # Check if </HintPath> is in the line
            if '</HintPath>' in line and '</HintPath>' in line:
                betweenHintsTags = line.strip()[10:][0:-11]
                substrings = re.split(r'\.\.|/|\\', betweenHintsTags.strip())
                # Filter out empty strings from the result
                dllFile = [s for s in substrings if "dll" in s][0]
                # Print or process the substrings as needed
                lineToCompose = ("    <HintPath>..\\..\\build\\extensions\\" +
                                   str(dllFile) +
                                   "</HintPath>\n")
                # print(lineToCompose)
                fileEditLines.append(lineToCompose)
            else:
                fileEditLines.append(line)
        
    with open(file_path, 'w') as file:
       for modified_line in fileEditLines:
           file.write(modified_line)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py <file_path>")
    else:
        remove_previous_debug_path_and_others(sys.argv[1])
        replace_in_file(sys.argv[1])
        replace_hint_paths(sys.argv[1])