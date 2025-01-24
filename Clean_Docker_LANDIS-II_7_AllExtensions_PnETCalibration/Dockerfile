FROM ubuntu:22.04

################################################################
# PREPARATIONS
################################################################

# Set the DEBIAN_FRONTEND environment variable to noninteractive
# Avoid prompts when installing packages
ENV DEBIAN_FRONTEND=noninteractive

# PREPARING PACKAGES AND UPGRADING, AND INSTALLING DEPENDENCIES OF LANDIS-II
# apt-get clean and rm -rf /var/lib/apt/lists/* are used to clear the packages folder caches
# to avoid putting them in the docker layerin system, freeing space
RUN apt update && apt upgrade \
&& apt-get update && apt-get -y upgrade \
&& apt-get install -y wget vim pip nano git python3 python-is-python3 \
&& apt-get install -y libjpeg62 \
&& apt-get install -y libpng16-16 \
&& apt-get install -y gdal-bin \
&& apt-get install -y libgdal-dev \
# Installing libssl1 sometimes needed for Dotnet
&& wget http://archive.ubuntu.com/ubuntu/pool/main/o/openssl/libssl1.1_1.1.0g-2ubuntu4_amd64.deb \
&& dpkg -i libssl1.1_1.1.0g-2ubuntu4_amd64.deb \
&& apt-get clean && rm -rf /var/lib/apt/lists/*
ENV C_INCLUDE_PATH=/usr/include/gdal
ENV CPLUS_INCLUDE_PATH=/usr/include/gdal

# INSTALLING DOTNET SDK AND RUNTIME 2.2
RUN mkdir /bin/.dotnet/ && cd /bin/.dotnet/ \
&& wget https://dot.net/v1/dotnet-install.sh -O dotnet-install.sh \
&& chmod +x ./dotnet-install.sh \
&& ./dotnet-install.sh --channel 2.2 -InstallDir /bin/.dotnet/  \
&& ./dotnet-install.sh --channel 2.2 --runtime aspnetcore -InstallDir /bin/.dotnet/ \
&& apt-get clean && rm -rf /var/lib/apt/lists/*
ENV DOTNET_ROOT=/bin/.dotnet
ENV PATH=$PATH:$DOTNET_ROOT:$DOTNET_ROOT/tools

# PREPARING VARIABLES FOR DOWNLOADS
# WARNING : With a bad internet connection, git clone can sometimes fail to.
# These commands should help git on bad internet connections
RUN git config --global http.version HTTP/1.1 \
&& git config --global http.postBuffer 524288000 \
&& git config --global http.lowSpeedLimit 0 \
&& git config --global http.lowSpeedTime 999999

# PREPARING VARIABLE FOR MORE EASY COMMANDS 
ENV LANDIS_EXTENSIONS_TOOL="/bin/LANDIS_Linux/build/Release/Landis.Extensions.dll"
ENV LANDIS_CONSOLE="/bin/LANDIS_Linux/build/Release/Landis.Console.dll"
ENV LANDIS_FOLDER="/bin/LANDIS_Linux"

################################################################
# COMPILING LANDIS-II
################################################################

###### COMPILING THE CORE 

RUN mkdir /bin/LANDIS_Linux \
&& cd /bin/LANDIS_Linux && git clone https://github.com/LANDIS-II-Foundation/Core-Model-v7-LINUX.git \
&& cd /bin/LANDIS_Linux/Core-Model-v7-LINUX/Tool-Console/src && dotnet build -c Release \
&& mkdir /bin/LANDIS_Linux/build && mv /bin/LANDIS_Linux/Core-Model-v7-LINUX/build/* /bin/LANDIS_Linux/build \
&& rm -r /bin/LANDIS_Linux/Core-Model-v7-LINUX
# Downloading support libraries and installing them
RUN cd /bin/LANDIS_Linux && git clone https://github.com/LANDIS-II-Foundation/Support-Library-Dlls-v7.git \
&& mv /bin/LANDIS_Linux/Support-Library-Dlls-v7/* /bin/LANDIS_Linux/build/extensions \
&& rm -r /bin/LANDIS_Linux/Support-Library-Dlls-v7

# Transfering the python script that helps the compilation of the extensions
COPY "./files_to_help_compilation/editing_csproj_LANDIS-II_files.py" /bin/LANDIS_Linux

# Transfering the sh script that allows us to only download one folder from one commit from github
COPY "./files_to_help_compilation/downloadSpecificGitCommitAndFolder.sh" /bin/LANDIS_Linux


###### COMPILING AND REGISTERING EXTENSIONS 
# INFO: The files necessary for the compilation are downloaded through the script downloadSpecificGitCommitAndFolder.sh
# It is used to avoid downloading the entire repository as it's not needed.
#
# It takes three arguments :
# The URL of the repo
# The commit hash for the commit from which we want to download files
# The folder we want to download files from (excludes all of the others)
#
# The folder we want is almost always /src which contains the source code.
#
# We also always end up deleting the files downloaded for the build to avoid saving them in docker layering system (saves space).
#
# For the .txt file necessary to register the extension in the extension.xml file, I've decided to download them through
# their unique Github URL (corresponding to the commit). This is again to avoid downloading a lot of files for nothing.

#### SUCCESSION EXTENSIONS

# Biomass Succession
RUN cd /bin/LANDIS_Linux \
# Commit 4bdbe475c3def1f6f2082ecb4682402a87ba4775 is apparently most recent commit compatible with core v7.
&& ./downloadSpecificGitCommitAndFolder.sh https://github.com/LANDIS-II-Foundation/Extension-Biomass-Succession.git 4bdbe475c3def1f6f2082ecb4682402a87ba4775 /src \ 
&& cd /bin/LANDIS_Linux/Extension-Biomass-Succession/src \
&& python /bin/LANDIS_Linux/editing_csproj_LANDIS-II_files.py ./biomass-succession.csproj \
&& dotnet build -c Release \
&& wget https://raw.githubusercontent.com/LANDIS-II-Foundation/Extension-Biomass-Succession/4bdbe475c3def1f6f2082ecb4682402a87ba4775/deploy/installer/Biomass%20Succession%206.1.txt \
&& dotnet $LANDIS_EXTENSIONS_TOOL add "Biomass Succession 6.1.txt" \
&& cd /bin/LANDIS_Linux/ && rm -r Extension-Biomass-Succession 

# # NECN Succession
RUN cd /bin/LANDIS_Linux \
# Commit 7203fc14a757d05673389ce42b4e7cd59e433d69 is apparently most recent commit compatible with core v7.
&& ./downloadSpecificGitCommitAndFolder.sh https://github.com/LANDIS-II-Foundation/Extension-NECN-Succession.git 7203fc14a757d05673389ce42b4e7cd59e433d69 /src \ 
&& cd /bin/LANDIS_Linux/Extension-NECN-Succession/src \
&& python /bin/LANDIS_Linux/editing_csproj_LANDIS-II_files.py ./NECN-succession.csproj \
# The devellopers of NECN forgot to add a line referencing the location of one of the support libraries.
# I add it here manually at the right place in the file.
&& sed -i '63i    <HintPath>..\\..\\build\\extensions\\Landis.Library.Succession-v8.dll</HintPath>' NECN-succession.csproj \
&& dotnet build -c Release \
&& wget https://raw.githubusercontent.com/LANDIS-II-Foundation/Extension-NECN-Succession/7203fc14a757d05673389ce42b4e7cd59e433d69/deploy/installer/NECN_Succession%207.txt \
&& dotnet $LANDIS_EXTENSIONS_TOOL add "NECN_Succession 7.txt" \
&& cd /bin/LANDIS_Linux/ && rm -r Extension-NECN-Succession

# # PnET Succession
RUN cd /bin/LANDIS_Linux \
# Commit 472353bb1059b6e1ac80f4c3750d1bca7d7673cd is apparently most recent commit compatible with core v7.
&& ./downloadSpecificGitCommitAndFolder.sh https://github.com/LANDIS-II-Foundation/Extension-PnET-Succession.git 472353bb1059b6e1ac80f4c3750d1bca7d7673cd /src \ 
&& cd /bin/LANDIS_Linux/Extension-PnET-Succession/src \
&& python /bin/LANDIS_Linux/editing_csproj_LANDIS-II_files.py ./PnET-Succession.csproj \
# We need a library that is not in the support libraries to compile PnET. We compile it and put it in the extensions folder
# I download the last commit at the date of this script
&& cd /bin/LANDIS_Linux \
&& ./downloadSpecificGitCommitAndFolder.sh https://github.com/LANDIS-II-Foundation/Library-PnET-Cohort.git 1f1f0e846e3e501f510d13b0d1918ba9bad3e3f5 /src \ 
&& cd /bin/LANDIS_Linux/Library-PnET-Cohort/src \
&& python /bin/LANDIS_Linux/editing_csproj_LANDIS-II_files.py ./Library-PnET-Cohort.csproj \
&& dotnet build -c Release \
# Back to PnET proper
# A definition seems to be missing in Plugin.cs, even on newer versions; I add it here. Strange ?
&& cd /bin/LANDIS_Linux/Extension-PnET-Succession/src && sed -i '57i        public int ThreadCount;' PlugIn.cs \
# We need to remove the .sln file; it's not useful to build, and is full of bad/missing references to other projects.
&& rm /bin/LANDIS_Linux/Extension-PnET-Succession/src/PnET-Succession.sln \ 
&& dotnet build -c Release \
&& cd /bin/LANDIS_Linux/ && rm -r Extension-PnET-Succession \
# To finish, we have to transfer some "Default files" of PnET in the extension folder
&& ./downloadSpecificGitCommitAndFolder.sh https://github.com/LANDIS-II-Foundation/Extension-PnET-Succession.git 472353bb1059b6e1ac80f4c3750d1bca7d7673cd /deploy \ 
&& cp -r /bin/LANDIS_Linux/Extension-PnET-Succession/deploy/Defaults /bin/LANDIS_Linux/build/extensions \
&& wget https://raw.githubusercontent.com/LANDIS-II-Foundation/Extension-PnET-Succession/472353bb1059b6e1ac80f4c3750d1bca7d7673cd/deploy/installer/PnET-Succession.txt \
&& dotnet $LANDIS_EXTENSIONS_TOOL add "PnET-Succession.txt" \
&& cd /bin/LANDIS_Linux/ && rm -r Extension-PnET-Succession

# # Forest Carbon Succession (ForCS)
# # This one is a head scratcher; commit with v3.0 is missing the csproj file to compile; commit with v3.1 (the latest compatible with core v7) is missing some key files. So i'm obliged to "mix" the two commits.
RUN cd /bin/LANDIS_Linux \
# Commit 9d5cf85564484a8ee2389055b35dc0dd02dcc10a for 3.0
&& ./downloadSpecificGitCommitAndFolder.sh https://github.com/LANDIS-II-Foundation/Extension-ForCS-Succession.git 9d5cf85564484a8ee2389055b35dc0dd02dcc10a /src \ 
&& mkdir /tmp/ForCSCode \
&& cp -r /bin/LANDIS_Linux/Extension-ForCS-Succession/src/* /tmp/ForCSCode \
&& rm -r /bin/LANDIS_Linux/Extension-ForCS-Succession \
# Commit for 3.1
&& ./downloadSpecificGitCommitAndFolder.sh https://github.com/LANDIS-II-Foundation/Extension-ForCS-Succession.git 5fda9973fd6c2b58c39cc396479606dceaea295b /src \ 
&& cp -r -n /tmp/ForCSCode/* /bin/LANDIS_Linux/Extension-ForCS-Succession/src/ \
&& cd /bin/LANDIS_Linux/Extension-ForCS-Succession/src \
&& python /bin/LANDIS_Linux/editing_csproj_LANDIS-II_files.py ./CForCS.csproj \
# # Before we build : we need to move several libraries needed. They are normally provided through NuGet packages; but here,
# # that's not the case. We'll just put them in the extension folder.
&& mv -n /bin/LANDIS_Linux/Extension-ForCS-Succession/src/bin/Debug/netcoreapp2.0/* /bin/LANDIS_Linux/build/extensions \
&& cd /bin/LANDIS_Linux/Extension-ForCS-Succession/src && dotnet build -c Release \
&& wget https://raw.githubusercontent.com/LANDIS-II-Foundation/Extension-ForCS-Succession/5fda9973fd6c2b58c39cc396479606dceaea295b/deploy/installer/ForCS%203.1.txt \
&& dotnet $LANDIS_EXTENSIONS_TOOL add "ForCS 3.1.txt" \
&& cd /bin/LANDIS_Linux/ && rm -r Extension-ForCS-Succession

# # Age Only succession
RUN cd /bin/LANDIS_Linux \
# Commit bcf48e41b385ff0fec9e994c6ba907ffc20d6e83 is latest commit at time of writing. Repo is archived because of v8.
&& ./downloadSpecificGitCommitAndFolder.sh https://github.com/LANDIS-II-Foundation/Extension-Age-Only-Succession-Archived.git bcf48e41b385ff0fec9e994c6ba907ffc20d6e83 /src \ 
&& cd /bin/LANDIS_Linux/Extension-Age-Only-Succession-Archived/src \
&& python /bin/LANDIS_Linux/editing_csproj_LANDIS-II_files.py ./age-only-successsion.csproj \
# # Again here, we have to add a line to precise where a support library is, as it's missing from the file.
&& cd /bin/LANDIS_Linux/Extension-Age-Only-Succession-Archived/src && sed -i '43i    <HintPath>..\\..\\build\\extensions\\Landis.Library.Succession-v8.dll</HintPath>' age-only-successsion.csproj \
&& cd /bin/LANDIS_Linux/Extension-Age-Only-Succession-Archived/src && dotnet build -c Release \
&& wget https://raw.githubusercontent.com/LANDIS-II-Foundation/Extension-Age-Only-Succession-Archived/bcf48e41b385ff0fec9e994c6ba907ffc20d6e83/deploy/installer/AgeOnly%20Succession%205.2.txt \
&& dotnet $LANDIS_EXTENSIONS_TOOL add "AgeOnly Succession 5.2.txt" \
&& cd /bin/LANDIS_Linux/ && rm -r Extension-Age-Only-Succession-Archived

# DGS Succession extension : seems to experimental. Will not be included in this image.


#### DISTURBANCE EXTENSIONS

## FIRE

# Base Fire
RUN cd /bin/LANDIS_Linux \
# Commit c54071a1b6e23951e9bb8970441a599318008408 is apparently most recent commit compatible with core v7.
&& ./downloadSpecificGitCommitAndFolder.sh https://github.com/LANDIS-II-Foundation/Extension-Base-Fire.git c54071a1b6e23951e9bb8970441a599318008408 /src \ 
&& cd /bin/LANDIS_Linux/Extension-Base-Fire/src \
&& python /bin/LANDIS_Linux/editing_csproj_LANDIS-II_files.py ./base-fire.csproj \
&& dotnet build -c Release \
&& wget https://raw.githubusercontent.com/LANDIS-II-Foundation/Extension-Base-Fire/c54071a1b6e23951e9bb8970441a599318008408/deploy/installer/Base%20Fire%204.0.txt \
&& dotnet $LANDIS_EXTENSIONS_TOOL add "Base Fire 4.0.txt" \
&& cd /bin/LANDIS_Linux/ && rm -r Extension-Base-Fire

# BFOLDS Fire
# Source code provided directly by Marc Ouelette from the Ontario Forest Research Institute
# As the source code is not yet available in Open-source format due to copyrights reasons,
# I include it for now as a dll already compiled.
# Commands necessary to compile the .dll
# COPY "./BFOLDS_v2.3" /bin/LANDIS_Linux/BFOLDSFire2.3
# RUN cd /bin/LANDIS_Linux/BFOLDSFire2.3/src \
# && python /bin/LANDIS_Linux/editing_csproj_LANDIS-II_files.py ./BFOLDS-FireModel.csproj \
# && cp /bin/LANDIS_Linux/build/Release/Landis.Console.dll /bin/LANDIS_Linux/build/extensions \
# && cd /bin/LANDIS_Linux/BFOLDSFire2.3/src && dotnet build -c Release BFOLDS-FireModel.csproj
# Commands to just copy the .dll
COPY "./files_to_help_compilation/BFOLDS_v2.3_nocode" /bin/LANDIS_Linux/BFOLDS_v2.3_nocode
RUN cp /bin/LANDIS_Linux/BFOLDS_v2.3_nocode/Landis.Extension.BFOLDSFire.dll /bin/LANDIS_Linux/build/extensions \
&& dotnet $LANDIS_EXTENSIONS_TOOL add $LANDIS_FOLDER/BFOLDS_v2.3_nocode/BFOLDSFire_v2.3.txt \
&& rm -r /bin/LANDIS_Linux/BFOLDS_v2.3_nocode/

# Climate-social Fire
RUN cd /bin/LANDIS_Linux \
# Commit 78b550b5ce61aa83d4a6553055ebabe06d863512 is apparently most recent commit compatible with core v7.
&& ./downloadSpecificGitCommitAndFolder.sh https://github.com/LANDIS-II-Foundation/Extension-Social-Climate-Fire.git 78b550b5ce61aa83d4a6553055ebabe06d863512 /src \ 
&& cd /bin/LANDIS_Linux/Extension-Social-Climate-Fire/src \
&& python /bin/LANDIS_Linux/editing_csproj_LANDIS-II_files.py ./scrapple.csproj \
# We remove the .sln file, which is not useful to us and blocks the build.
&& rm /bin/LANDIS_Linux/Extension-Social-Climate-Fire/src/scrapple.sln \
&& dotnet build -c Release \
&& wget https://raw.githubusercontent.com/LANDIS-II-Foundation/Extension-Social-Climate-Fire/78b550b5ce61aa83d4a6553055ebabe06d863512/deploy/installer/Scrapple%203.txt \
&& dotnet $LANDIS_EXTENSIONS_TOOL add "Scrapple 3.txt" \
&& cd /bin/LANDIS_Linux/ && rm -r Extension-Social-Climate-Fire

# Dynamic Fuel system
RUN cd /bin/LANDIS_Linux \
# Commit 7d24f401ed3104494421284f709f9e9ee63789c6 latest commit as of writting, compatible with v7.
&& ./downloadSpecificGitCommitAndFolder.sh https://github.com/LANDIS-II-Foundation/Extension-Dynamic-Fuel-System.git 7d24f401ed3104494421284f709f9e9ee63789c6 /src \ 
&& cd /bin/LANDIS_Linux/Extension-Dynamic-Fuel-System/src \
&& python /bin/LANDIS_Linux/editing_csproj_LANDIS-II_files.py ./dynamic-fuel-system.csproj \
&& dotnet build -c Release \
&& wget https://raw.githubusercontent.com/LANDIS-II-Foundation/Extension-Dynamic-Fuel-System/7d24f401ed3104494421284f709f9e9ee63789c6/deploy/installer/Dynamic%20Fuel%20System%203.0.txt \
&& dotnet $LANDIS_EXTENSIONS_TOOL add "Dynamic Fuel System 3.0.txt" \
&& cd /bin/LANDIS_Linux/ && rm -r Extension-Dynamic-Fuel-System

# Dynamic Biomass Fuels
RUN cd /bin/LANDIS_Linux \
# Commit 14b65b9d356dcf7bdf5bd7f5003c00be25535958 latest commit compatible with v7.
&& ./downloadSpecificGitCommitAndFolder.sh https://github.com/LANDIS-II-Foundation/Extension-Dynamic-Biomass-Fuels.git 14b65b9d356dcf7bdf5bd7f5003c00be25535958 /src \ 
&& cd /bin/LANDIS_Linux/Extension-Dynamic-Biomass-Fuels/src \
&& python /bin/LANDIS_Linux/editing_csproj_LANDIS-II_files.py ./dynamic-biomass-fuels.csproj \
&& dotnet build -c Release \
&& wget https://raw.githubusercontent.com/LANDIS-II-Foundation/Extension-Dynamic-Biomass-Fuels/14b65b9d356dcf7bdf5bd7f5003c00be25535958/deploy/installer/Dynamic%20Biomass%20Fuels%203.0.txt \
&& dotnet $LANDIS_EXTENSIONS_TOOL add "Dynamic Biomass Fuels 3.0.txt" \
&& cd /bin/LANDIS_Linux/ && rm -r Extension-Dynamic-Biomass-Fuels

# Dynamic Fire System
RUN cd /bin/LANDIS_Linux \
# Commit d75578eebca9d06185a6412466edb95cd45afeb1 latest commit compatible with v7.
&& ./downloadSpecificGitCommitAndFolder.sh https://github.com/LANDIS-II-Foundation/Extension-Dynamic-Fire-System.git d75578eebca9d06185a6412466edb95cd45afeb1 /src \ 
&& cd /bin/LANDIS_Linux/Extension-Dynamic-Fire-System/src \
&& python /bin/LANDIS_Linux/editing_csproj_LANDIS-II_files.py ./dynamic-fire.csproj \
&& dotnet build -c Release \
&& wget https://raw.githubusercontent.com/LANDIS-II-Foundation/Extension-Dynamic-Fire-System/d75578eebca9d06185a6412466edb95cd45afeb1/deploy/installer/Dynamic%20Fire%20Component%203.0.txt \
&& dotnet $LANDIS_EXTENSIONS_TOOL add "Dynamic Fire Component 3.0.txt" \
&& cd /bin/LANDIS_Linux/ && rm -r Extension-Dynamic-Fire-System


## HARVEST

# Base Harvest
RUN cd /bin/LANDIS_Linux \
# Commit cdb4861eb8389cbe165d2be1d435d01e62e2a8f2 latest commit as of today, and compatible with v7.
&& ./downloadSpecificGitCommitAndFolder.sh https://github.com/LANDIS-II-Foundation/Extension-Base-Harvest.git cdb4861eb8389cbe165d2be1d435d01e62e2a8f2 /src \ 
&& cd /bin/LANDIS_Linux/Extension-Base-Harvest/src \
&& python /bin/LANDIS_Linux/editing_csproj_LANDIS-II_files.py ./base-harvest.csproj \
&& dotnet build -c Release \
&& wget https://raw.githubusercontent.com/LANDIS-II-Foundation/Extension-Base-Harvest/cdb4861eb8389cbe165d2be1d435d01e62e2a8f2/deploy/installer/Base%20Harvest%205.2.txt \
&& dotnet $LANDIS_EXTENSIONS_TOOL add "Base Harvest 5.2.txt" \
&& cd /bin/LANDIS_Linux/ && rm -r Extension-Base-Harvest

# Biomass Harvest
# WARNING HERE
# The last commit compatible with v7 is c588c94590dcd1ec5e39790a46f7cbe9a7aa41d6
# However, I take one commit before it; this is because c588c94590dcd1ec5e39790a46f7cbe9a7aa41d6 uses the 
# Landis.Library.Biomass-v2.1 instead of the 2.0. Problem is, the latest version of Biomass Succession
# that works with core v7 uses Landis.Library.Biomass-v2.0. It has never been updated to use 2.1.
# There doesn't seem to be much differences in the end. this less recent commit is even indicated as a more
# recent version (Biomass harvest 4.6) than the newer commit (4.5). Very strange. 
# If the more recent commit is used (c588c94590dcd1ec5e39790a46f7cbe9a7aa41d6), it will throw a mysterious error when used
# with biomass succession during a variable initialisation (Data type of site variable "Succession.WoodyDebris" is Landis.Library.Biomass.Pool and not Landis.Library.Biomass.Pool); it's because both extensions are using the same library, but
# without the same version.
# We also don't take commit c4771b5294806b51ebcd498c3fd46eb765e5d7c5 (which is also more recent than the one used here)
# because it's a small commit that removes a file which make the build impossible. I don't understand why the devs did this.
RUN cd /bin/LANDIS_Linux \
&& ./downloadSpecificGitCommitAndFolder.sh https://github.com/LANDIS-II-Foundation/Extension-Biomass-Harvest.git 883e93340b75df2759f528a5c17fe4846433c6aa /src \ 
&& cd /bin/LANDIS_Linux/Extension-Biomass-Harvest/src \
&& python /bin/LANDIS_Linux/editing_csproj_LANDIS-II_files.py ./biomass-harvest-ext.csproj \
&& dotnet build -c Release \
&& wget https://raw.githubusercontent.com/LANDIS-II-Foundation/Extension-Biomass-Harvest/883e93340b75df2759f528a5c17fe4846433c6aa/deploy/installer/Biomass%20Harvest%204.x.txt \
&& dotnet $LANDIS_EXTENSIONS_TOOL add "Biomass Harvest 4.x.txt" \
&& cd /bin/LANDIS_Linux/ && rm -r Extension-Biomass-Harvest

# # SOSIEL Harvest
# First, we need to compile two SOSIEL libraries used by the harvest extension
RUN cd /bin/LANDIS_Linux \
# Commit bcbb57528ce2a7c57a1ebc6ab68f91f91a30eb9e latest commit compatible with v7.
&& ./downloadSpecificGitCommitAndFolder.sh https://github.com/LANDIS-II-Foundation/Library-Algorithm-SOSIEL.git bcbb57528ce2a7c57a1ebc6ab68f91f91a30eb9e /src \ 
&& cd /bin/LANDIS_Linux/Library-Algorithm-SOSIEL/src \
&& python /bin/LANDIS_Linux/editing_csproj_LANDIS-II_files.py ./SOSIEL.csproj \
&& dotnet build -c Release \
&& cd /bin/LANDIS_Linux/ && rm -r Library-Algorithm-SOSIEL \
# Commit 53cd7b26ca39e1b28f0f6dcc71888c1ba3b2cf01 latest commit compatible with v7.
&& ./downloadSpecificGitCommitAndFolder.sh https://github.com/LANDIS-II-Foundation/Library-SOSIEL-VBGP.git 53cd7b26ca39e1b28f0f6dcc71888c1ba3b2cf01 /src \ 
&& cd /bin/LANDIS_Linux/Library-SOSIEL-VBGP/src \
&& python /bin/LANDIS_Linux/editing_csproj_LANDIS-II_files.py ./SOSIEL.VBGP.csproj \
# We remove the .sln file, which is not useful to us and blocks the build.
&& rm /bin/LANDIS_Linux/Library-SOSIEL-VBGP/src/SOSIEL.VBGP.sln \
&& dotnet build -c Release \
&& cd /bin/LANDIS_Linux/ && rm -r Library-SOSIEL-VBGP \
# Back to compiling SOSIEL proper; commit 04b52ad9b28837b3ca550faff73c3635fdb04338 is latest for v7
&& ./downloadSpecificGitCommitAndFolder.sh https://github.com/LANDIS-II-Foundation/Extension-SOSIEL-Harvest.git 04b52ad9b28837b3ca550faff73c3635fdb04338 /src \ 
&& cd /bin/LANDIS_Linux/Extension-SOSIEL-Harvest/src \
&& python /bin/LANDIS_Linux/editing_csproj_LANDIS-II_files.py ./sosiel-harvest.csproj \
# There is a reference issue during the build; we have to update the version of a library that is provided.
# EDIT : This was solved when making a change to the version of biomass harvest (see above). LANDIS-II compilation works in mysterious ways.
# && cd /bin/LANDIS_Linux/Extension-SOSIEL-Harvest/src && sed -i 's/BiomassCohorts-v3/BiomassCohorts-v4/g' sosiel-harvest.csproj
&& dotnet build -c Release \
&& wget https://raw.githubusercontent.com/LANDIS-II-Foundation/Extension-SOSIEL-Harvest/04b52ad9b28837b3ca550faff73c3635fdb04338/deploy/installer/SHE%201.2.txt \
&& dotnet $LANDIS_EXTENSIONS_TOOL add "SHE 1.2.txt" \
&& cd /bin/LANDIS_Linux/ && rm -r Extension-SOSIEL-Harvest

# FRS Module
RUN cd /bin/LANDIS_Linux \
# Commit dbc52588fcc47a1dcbaf5f72c71f20c626e1707e is latest for v7
# Not using the download script here as the repo is organized differently (source code is in the root)
&& git clone https://github.com/Klemet/LANDIS-II-Forest-Roads-Simulation-extension.git \
&& cd /bin/LANDIS_Linux/LANDIS-II-Forest-Roads-Simulation-extension && git checkout dbc52588fcc47a1dcbaf5f72c71f20c626e1707e \
&& rm /bin/LANDIS_Linux/LANDIS-II-Forest-Roads-Simulation-extension/Forest-Roads-Extension.sln
# Sadly, the csproj file here is completly wrong for compiling on Linux. 
# I got to replace it entirely by something that is more in line with the other extensions.
COPY "files_to_help_compilation/Forest-Roads-Extension.csproj" /bin/LANDIS_Linux/LANDIS-II-Forest-Roads-Simulation-extension
RUN cd /bin/LANDIS_Linux/LANDIS-II-Forest-Roads-Simulation-extension && dotnet build -c Release \
&& wget https://raw.githubusercontent.com/Klemet/LANDIS-II-Forest-Roads-Simulation-extension/dbc52588fcc47a1dcbaf5f72c71f20c626e1707e/Deploy/Installation%20Files/plug-ins-installer-files/Forest%20Roads%20Simulation%201.3.1.txt \
&& dotnet $LANDIS_EXTENSIONS_TOOL add "Forest Roads Simulation 1.3.1.txt" \
&& cd /bin/LANDIS_Linux/ && rm -r LANDIS-II-Forest-Roads-Simulation-extension

# Magic harvest
RUN cd /bin/LANDIS_Linux \
# Commit 1e4f9254ff4b96a3d96e1208f79e6c94b9d1576c is latest for v7
# Not using the download script here as the repo is organized differently (source code is in the root)
&& git clone https://github.com/Klemet/LANDIS-II-Magic-Harvest.git \
&& cd /bin/LANDIS_Linux/LANDIS-II-Magic-Harvest/ && git checkout 1e4f9254ff4b96a3d96e1208f79e6c94b9d1576c \
&& rm /bin/LANDIS_Linux/LANDIS-II-Magic-Harvest/"Magic Harvest.sln"
# Sadly, the csproj file here is completly wrong for compiling on Linux. 
# I got to replace it entirely by something that is more in line with the other extensions.
COPY ["./files_to_help_compilation/Magic Harvest.csproj", "/bin/LANDIS_Linux/LANDIS-II-Magic-Harvest"]  
RUN cd /bin/LANDIS_Linux/LANDIS-II-Magic-Harvest && dotnet build -c Release \
&& wget https://raw.githubusercontent.com/Klemet/LANDIS-II-Magic-Harvest/1e4f9254ff4b96a3d96e1208f79e6c94b9d1576c/Deploy/Installation%20Files/plug-ins-installer-files/Magic%20harvest%20v1.3.txt \
&& dotnet $LANDIS_EXTENSIONS_TOOL add "Magic harvest v1.3.txt" \
&& cd /bin/LANDIS_Linux/ && rm -r LANDIS-II-Magic-Harvest

## WIND

# Base Wind
RUN cd /bin/LANDIS_Linux \
# Commit 7b49d9589b7ccf6735aeba643d10fe0f4e9ad627 latest commit compatible with v7.
&& ./downloadSpecificGitCommitAndFolder.sh https://github.com/LANDIS-II-Foundation/Extension-Base-Wind.git 7b49d9589b7ccf6735aeba643d10fe0f4e9ad627 /src \ 
&& cd /bin/LANDIS_Linux/Extension-Base-Wind/src \
&& python /bin/LANDIS_Linux/editing_csproj_LANDIS-II_files.py ./base-wind.csproj \
# We need to edit a typo in the csproj.
&& sed -i 's/Landis.Library.Landis.Library/Landis.Library/g' base-wind.csproj \
&& dotnet build -c Release \
&& wget https://raw.githubusercontent.com/LANDIS-II-Foundation/Extension-Base-Wind/7b49d9589b7ccf6735aeba643d10fe0f4e9ad627/deploy/installer/Base%20Wind%203.0.txt \
&& dotnet $LANDIS_EXTENSIONS_TOOL add "Base Wind 3.0.txt" \
&& cd /bin/LANDIS_Linux/ && rm -r Extension-Base-Wind

# Linear Wind
RUN cd /bin/LANDIS_Linux \
# Commit f0bdd46223e87609e12c9d9d940cb95e475f3927 latest commit compatible with v7.
&& ./downloadSpecificGitCommitAndFolder.sh https://github.com/LANDIS-II-Foundation/Extension-LinearWind.git f0bdd46223e87609e12c9d9d940cb95e475f3927 /src \ 
&& cd /bin/LANDIS_Linux/Extension-LinearWind/src \
&& python /bin/LANDIS_Linux/editing_csproj_LANDIS-II_files.py ./linear-wind.csproj \
&& dotnet build -c Release \
&& wget https://raw.githubusercontent.com/LANDIS-II-Foundation/Extension-LinearWind/f0bdd46223e87609e12c9d9d940cb95e475f3927/deploy/installer/Linear%20Wind%202.0.txt \
&& dotnet $LANDIS_EXTENSIONS_TOOL add "Linear Wind 2.0.txt" \
&& cd /bin/LANDIS_Linux/ && rm -r Extension-LinearWind

# Hurricane
RUN cd /bin/LANDIS_Linux \
# Commit 8aeed8eccd4dbc88109e2dc2926cd5a3b8400c14 latest commit compatible with v7.
&& ./downloadSpecificGitCommitAndFolder.sh https://github.com/LANDIS-II-Foundation/Extension-Biomass-Hurricane.git 8aeed8eccd4dbc88109e2dc2926cd5a3b8400c14 /src \ 
&& cd /bin/LANDIS_Linux/Extension-Biomass-Hurricane/src \
&& python /bin/LANDIS_Linux/editing_csproj_LANDIS-II_files.py ./biomass-hurricane.csproj \
# We need to edit a typo in the csproj.
&& sed -i 's/Landis.Library.Landis.Library/Landis.Library/g' biomass-hurricane.csproj \
&& dotnet build -c Release \
&& wget https://raw.githubusercontent.com/LANDIS-II-Foundation/Extension-Biomass-Hurricane/8aeed8eccd4dbc88109e2dc2926cd5a3b8400c14/deploy/Biomass%20Hurricane%202.txt \
&& dotnet $LANDIS_EXTENSIONS_TOOL add "Biomass Hurricane 2.txt" \
&& cd /bin/LANDIS_Linux/ && rm -r Extension-Biomass-Hurricane


## DISEASES, INSECTS AND OTHERS

# Base BDA (now named Climate biological Disturbance agent)
RUN cd /bin/LANDIS_Linux \
# Commit 1216055f55474a83038542bc80e70518303c32a4 latest commit compatible with v7.
&& ./downloadSpecificGitCommitAndFolder.sh https://github.com/LANDIS-II-Foundation/Extension-Base-BDA.git 1216055f55474a83038542bc80e70518303c32a4 /src \ 
&& cd /bin/LANDIS_Linux/Extension-Base-BDA/src \
&& python /bin/LANDIS_Linux/editing_csproj_LANDIS-II_files.py ./Base-BDA-Climate.csproj \
&& dotnet build -c Release \
&& wget https://raw.githubusercontent.com/LANDIS-II-Foundation/Extension-Base-BDA/1216055f55474a83038542bc80e70518303c32a4/deploy/installer/Base%20BDA%204.1.txt \
&& dotnet $LANDIS_EXTENSIONS_TOOL add "Base BDA 4.1.txt" \
&& cd /bin/LANDIS_Linux/ && rm -r Extension-Base-BDA

# Root rot
RUN cd /bin/LANDIS_Linux \
# Commit b2f5d591a6d9cf86e0a546f5477dedcb9345fe3f latest commit as of today and compatible with v7.
&& ./downloadSpecificGitCommitAndFolder.sh https://github.com/LANDIS-II-Foundation/Extension-Root-Rot.git b2f5d591a6d9cf86e0a546f5477dedcb9345fe3f /src \ 
&& cd /bin/LANDIS_Linux/Extension-Root-Rot/src \
&& python /bin/LANDIS_Linux/editing_csproj_LANDIS-II_files.py ./root-rot.csproj \
# We remove the .sln
&& rm /bin/LANDIS_Linux/Extension-Root-Rot/src/root-rot.sln \
&& dotnet build -c Release \
&& wget https://raw.githubusercontent.com/LANDIS-II-Foundation/Extension-Root-Rot/b2f5d591a6d9cf86e0a546f5477dedcb9345fe3f/Deploy/installer/Root%20Rot%201.1.txt \
&& dotnet $LANDIS_EXTENSIONS_TOOL add "Root Rot 1.1.txt" \
&& cd /bin/LANDIS_Linux/ && rm -r Extension-Root-Rot

# Biomass Browse
RUN cd /bin/LANDIS_Linux \
# Commit 1755224fb4fd04b85c3b24e2677376681492b7c4 latest commit as of today and compatible with v7.
&& ./downloadSpecificGitCommitAndFolder.sh https://github.com/LANDIS-II-Foundation/Extension-Biomass-Browse.git 1755224fb4fd04b85c3b24e2677376681492b7c4 /src \ 
&& cd /bin/LANDIS_Linux/Extension-Biomass-Browse/src \
&& python /bin/LANDIS_Linux/editing_csproj_LANDIS-II_files.py ./biomass-browse.csproj \
# There are two csproj file; it seems to be the same extension, but under different names. I use biomass browse,
# which seems to be more recent.
&& dotnet build -c Release biomass-browse.csproj \
&& wget https://raw.githubusercontent.com/LANDIS-II-Foundation/Extension-Biomass-Browse/1755224fb4fd04b85c3b24e2677376681492b7c4/deploy/installer/Biomass%20Browse%202.txt \
&& dotnet $LANDIS_EXTENSIONS_TOOL add "Biomass Browse 2.txt" \
&& cd /bin/LANDIS_Linux/ && rm -r Extension-Biomass-Browse

# Land-use plus
RUN cd /bin/LANDIS_Linux \
# Commit f5085b01da380dec94db47caa0b66adcc02c5308 latest commit compatible with v7.
&& ./downloadSpecificGitCommitAndFolder.sh https://github.com/LANDIS-II-Foundation/Extension-Land-Use-Plus.git f5085b01da380dec94db47caa0b66adcc02c5308 /src \ 
&& cd /bin/LANDIS_Linux/Extension-Land-Use-Plus/src \
&& python /bin/LANDIS_Linux/editing_csproj_LANDIS-II_files.py ./land-use.csproj \
# We remove the .sln
&& rm /bin/LANDIS_Linux/Extension-Land-Use-Plus/src/land-use.sln \
&& dotnet build -c Release \
&& wget https://raw.githubusercontent.com/LANDIS-II-Foundation/Extension-Land-Use-Plus/f5085b01da380dec94db47caa0b66adcc02c5308/deploy/installer/Land%20Use%203.1.txt \
&& dotnet $LANDIS_EXTENSIONS_TOOL add "Land Use 3.1.txt" \
&& cd /bin/LANDIS_Linux/ && rm -r Extension-Land-Use-Plus

### OUTPUT EXTENSIONS

# Max species Age
RUN cd /bin/LANDIS_Linux \
# Commit d28232fc9025d359e6fc2ccefee64866e75a84ef latest commit compatible with v7.
&& ./downloadSpecificGitCommitAndFolder.sh https://github.com/LANDIS-II-Foundation/Extension-Output-Max-Species-Age.git d28232fc9025d359e6fc2ccefee64866e75a84ef /src \ 
&& cd /bin/LANDIS_Linux/Extension-Output-Max-Species-Age/src \
&& python /bin/LANDIS_Linux/editing_csproj_LANDIS-II_files.py ./max-species-age.csproj \
&& dotnet build -c Release \
&& wget https://raw.githubusercontent.com/LANDIS-II-Foundation/Extension-Output-Max-Species-Age/d28232fc9025d359e6fc2ccefee64866e75a84ef/deploy/installer/Output%20MaxSpeciesAge%203.0.txt \
&& dotnet $LANDIS_EXTENSIONS_TOOL add "Output MaxSpeciesAge 3.0.txt" \
&& cd /bin/LANDIS_Linux/ && rm -r Extension-Output-Max-Species-Age

# Biomass-by-age
RUN cd /bin/LANDIS_Linux \
# Commit 8f96e3b1c6598fd8b5ecfe32a9506de3488da85e latest commit compatible with v7.
&& ./downloadSpecificGitCommitAndFolder.sh https://github.com/LANDIS-II-Foundation/Extension-Output-Biomass-By-Age.git 8f96e3b1c6598fd8b5ecfe32a9506de3488da85e /src \ 
&& cd /bin/LANDIS_Linux/Extension-Output-Biomass-By-Age/src \
&& python /bin/LANDIS_Linux/editing_csproj_LANDIS-II_files.py ./output-biomass-by-age.csproj \
&& dotnet build -c Release \
&& wget https://raw.githubusercontent.com/LANDIS-II-Foundation/Extension-Output-Biomass-By-Age/8f96e3b1c6598fd8b5ecfe32a9506de3488da85e/deploy/installer/Output%20Biomass-by-Age%203.0.txt \
&& dotnet $LANDIS_EXTENSIONS_TOOL add "Output Biomass-by-Age 3.0.txt" \
&& cd /bin/LANDIS_Linux/ && rm -r Extension-Output-Biomass-By-Age

# Biomass Community
RUN cd /bin/LANDIS_Linux \
# Commit fc7db62ba68d9b315e5bd207de169c7515747446 latest commit compatible with v7.
&& ./downloadSpecificGitCommitAndFolder.sh https://github.com/LANDIS-II-Foundation/Extension-Output-Biomass-Community.git fc7db62ba68d9b315e5bd207de169c7515747446 /src \ 
&& cd /bin/LANDIS_Linux/Extension-Output-Biomass-Community/src \
&& python /bin/LANDIS_Linux/editing_csproj_LANDIS-II_files.py ./output-biomass-community.csproj \
&& dotnet build -c Release \
&& wget https://raw.githubusercontent.com/LANDIS-II-Foundation/Extension-Output-Biomass-Community/fc7db62ba68d9b315e5bd207de169c7515747446/deploy/installer/Output%20Biomass%20Community%202.0.txt \
&& dotnet $LANDIS_EXTENSIONS_TOOL add "Output Biomass Community 2.0.txt" \
&& cd /bin/LANDIS_Linux/ && rm -r Extension-Output-Biomass-Community

# Biomass Output
RUN cd /bin/LANDIS_Linux \
# Commit 26a4178de05008b62b570226c8e5c8ab27ac7ac3 latest commit compatible with v7.
&& ./downloadSpecificGitCommitAndFolder.sh https://github.com/LANDIS-II-Foundation/Extension-Output-Biomass.git 26a4178de05008b62b570226c8e5c8ab27ac7ac3 /src \ 
&& cd /bin/LANDIS_Linux/Extension-Output-Biomass/src \
&& python /bin/LANDIS_Linux/editing_csproj_LANDIS-II_files.py ./output-biomass.csproj \
&& dotnet build -c Release \
&& wget https://raw.githubusercontent.com/LANDIS-II-Foundation/Extension-Output-Biomass/26a4178de05008b62b570226c8e5c8ab27ac7ac3/deploy/installer/Output%20Biomass%203.0.txt \
# Problem with Output Biomass : The .txt file to register the extensio doesn't point to the right name for the assembly
# We change it
&& sed -i '6s/.*/Assembly     Landis.Extension.Output.Biomass-v3/' "Output Biomass 3.0.txt" \
&& dotnet $LANDIS_EXTENSIONS_TOOL add "Output Biomass 3.0.txt" \
&& cd /bin/LANDIS_Linux/ && rm -r Extension-Output-Biomass

# Biomass reclassification
RUN cd /bin/LANDIS_Linux \
# Commit 571536db574f643a27dd3bbddd7b76f52826198c latest commit compatible with v7.
&& ./downloadSpecificGitCommitAndFolder.sh https://github.com/LANDIS-II-Foundation/Extension-Output-Biomass-Reclass.git 571536db574f643a27dd3bbddd7b76f52826198c /src \ 
&& cd /bin/LANDIS_Linux/Extension-Output-Biomass-Reclass/src \
&& python /bin/LANDIS_Linux/editing_csproj_LANDIS-II_files.py ./output-biomass.csproj \
&& dotnet build -c Release \
&& wget https://raw.githubusercontent.com/LANDIS-II-Foundation/Extension-Output-Biomass-Reclass/571536db574f643a27dd3bbddd7b76f52826198c/deploy/installer/Output%20Biomass%20Reclass%203.0.txt \
&& dotnet $LANDIS_EXTENSIONS_TOOL add "Output Biomass Reclass 3.0.txt" \
&& cd /bin/LANDIS_Linux/ && rm -r Extension-Output-Biomass-Reclass

# Cohort statistics outputs
RUN cd /bin/LANDIS_Linux \
# Commit c64ad5b1e9a4916de7f608fa43d436ef6ec2e4e6 latest commit compatible with v7.
&& ./downloadSpecificGitCommitAndFolder.sh https://github.com/LANDIS-II-Foundation/Extension-Output-Cohort-Statistics.git c64ad5b1e9a4916de7f608fa43d436ef6ec2e4e6 /src \ 
&& cd /bin/LANDIS_Linux/Extension-Output-Cohort-Statistics/src \
&& python /bin/LANDIS_Linux/editing_csproj_LANDIS-II_files.py ./output-cohort-stats.csproj \
&& dotnet build -c Release \
&& wget https://raw.githubusercontent.com/LANDIS-II-Foundation/Extension-Output-Cohort-Statistics/c64ad5b1e9a4916de7f608fa43d436ef6ec2e4e6/deploy/installer/Output%20Cohort%20Statistics%203.0.txt \
&& dotnet $LANDIS_EXTENSIONS_TOOL add "Output Cohort Statistics 3.0.txt" \
&& cd /bin/LANDIS_Linux/ && rm -r Extension-Output-Cohort-Statistics

# Landscape habitat Output
RUN cd /bin/LANDIS_Linux \
# Commit e5701c604d426807051f772dfe599b12a2685dd0 latest commit compatible with v7.
&& ./downloadSpecificGitCommitAndFolder.sh https://github.com/LANDIS-II-Foundation/Extension-Output-Landscape-Habitat.git e5701c604d426807051f772dfe599b12a2685dd0 /src \ 
&& cd /bin/LANDIS_Linux/Extension-Output-Landscape-Habitat/src \
&& python /bin/LANDIS_Linux/editing_csproj_LANDIS-II_files.py ./landscape-habitat-output.csproj \
# We remove the .sln
&& rm /bin/LANDIS_Linux/Extension-Output-Landscape-Habitat/src/landscape-habitat-output.sln \
# There is two .csproj files; one seems to be for the older name of the extension. We use the new one.
&& dotnet build -c Release landscape-habitat-output.csproj \
&& wget https://raw.githubusercontent.com/LANDIS-II-Foundation/Extension-Output-Landscape-Habitat/e5701c604d426807051f772dfe599b12a2685dd0/deploy/installer/Output%20Landscape%20Habitat.txt \
&& dotnet $LANDIS_EXTENSIONS_TOOL add "Output Landscape Habitat.txt" \
&& cd /bin/LANDIS_Linux/ && rm -r Extension-Output-Landscape-Habitat

# Local habitat suitability
RUN cd /bin/LANDIS_Linux \
# Commit fe3de71ec37f761c301cbbfea9eaae34c03d8dd7 latest commit compatible with v7.
&& ./downloadSpecificGitCommitAndFolder.sh https://github.com/LANDIS-II-Foundation/Extension-Local-Habitat-Suitability-Output.git fe3de71ec37f761c301cbbfea9eaae34c03d8dd7 /src \ 
&& cd /bin/LANDIS_Linux/Extension-Local-Habitat-Suitability-Output/src \
&& python /bin/LANDIS_Linux/editing_csproj_LANDIS-II_files.py ./local-habitat-output.csproj \
&& dotnet build -c Release \
&& wget https://raw.githubusercontent.com/LANDIS-II-Foundation/Extension-Local-Habitat-Suitability-Output/fe3de71ec37f761c301cbbfea9eaae34c03d8dd7/deploy/installer/Local%20Habitat%20Output.txt \
&& dotnet $LANDIS_EXTENSIONS_TOOL add "Local Habitat Output.txt" \
&& cd /bin/LANDIS_Linux/ && rm -r Extension-Local-Habitat-Suitability-Output

# PnET Output
RUN cd /bin/LANDIS_Linux \
# Commit 0855165e1601234357d088bda7cfe292e06e03a7 latest commit compatible with v7.
&& ./downloadSpecificGitCommitAndFolder.sh https://github.com/LANDIS-II-Foundation/Extension-Output-Biomass-PnET.git 0855165e1601234357d088bda7cfe292e06e03a7 /src \ 
&& cd /bin/LANDIS_Linux/Extension-Output-Biomass-PnET/src \
&& python /bin/LANDIS_Linux/editing_csproj_LANDIS-II_files.py ./output-biomass-pnet.csproj \
&& dotnet build -c Release \
&& wget https://raw.githubusercontent.com/LANDIS-II-Foundation/Extension-Output-Biomass-PnET/0855165e1601234357d088bda7cfe292e06e03a7/deploy/installer/PnET-Output.txt \
&& dotnet $LANDIS_EXTENSIONS_TOOL add "PnET-Output.txt" \
&& cd /bin/LANDIS_Linux/ && rm -r Extension-Output-Biomass-PnET

# Wildlife habitat Output
RUN cd /bin/LANDIS_Linux \
# Commit 027cefcf3cdfbfb70e2247531f514bf5fc0f1912 latest commit compatible with v7.
&& ./downloadSpecificGitCommitAndFolder.sh https://github.com/LANDIS-II-Foundation/Extension-Output-Wildlife-Habitat.git 027cefcf3cdfbfb70e2247531f514bf5fc0f1912 /src \ 
&& cd /bin/LANDIS_Linux/Extension-Output-Wildlife-Habitat/src \
&& python /bin/LANDIS_Linux/editing_csproj_LANDIS-II_files.py ./wildlife-habitat.csproj \
&& dotnet build -c Release \
&& wget https://raw.githubusercontent.com/LANDIS-II-Foundation/Extension-Output-Wildlife-Habitat/027cefcf3cdfbfb70e2247531f514bf5fc0f1912/deploy/installer/Wildlife%20Habitat%20Output%202.0.txt \
&& dotnet $LANDIS_EXTENSIONS_TOOL add "Wildlife Habitat Output 2.0.txt" \
&& cd /bin/LANDIS_Linux/ && rm -r Extension-Output-Wildlife-Habitat


################################################################
# TESTING THE EXTENSIONS
# (Optional; you can comment this whole section).
################################################################


# We create a temp file that is used as a time reference to delete the output files created by the tests,
# in case we don't want to keep them.
# RUN touch /tmp/tempFileForTestFileRemoval.txt
# ENV temp_file="/tmp/tempFileForTestFileRemoval.txt"

# RUN cd $LANDIS_FOLDER/Extension-Age-Only-Succession-Archived/testing/Corev7 \
# && dotnet $LANDIS_CONSOLE scenario_s1e1.txt

# # The version here is 6.1; and there are test files for v6.1. But these return an error where launched.
# # Seems like it's the precision of the biomass of a cohort in the initial community files that is the issue.
# # The test files for v6 works perfectly.
# RUN cd $LANDIS_FOLDER/Extension-Biomass-Succession/testings/CoreV7.0-BiomassSuccession6.0 \
# && dotnet $LANDIS_CONSOLE scenario.txt

# RUN cd $LANDIS_FOLDER/Extension-NECN-Succession/testing/Core7-NECN7-Landscape \
# && dotnet $LANDIS_CONSOLE scenario_landscape.txt

# For PnET, there are many test folders; I use one of the simplest ones.
# Problem : Many files are misnamed compared to how they are called in the parameter files.
# Need to rename them.
# RUN cd $LANDIS_FOLDER/Extension-PnET-Succession/test/PnET_Output \
# && mv PnEToutputsites.txt PNEToutputsites.txt \
# && mv Climate.txt climate.txt \
# && mv PnET-succession.txt pnet-succession.txt 
# RUN cd $LANDIS_FOLDER/Extension-PnET-Succession/test/PnET_Output \
# && dotnet $LANDIS_CONSOLE scenario.txt

# ForCS succession doesn't have test files for v7 in their repo. I've asked Caren Dymond about it.

# RUN cd $LANDIS_FOLDER/Extension-Base-Fire/testings/Core7.0-BaseFire4.0 \
# && dotnet $LANDIS_CONSOLE scenario_s1e1.txt

# BFOLDS Fire
# RUN cd $LANDIS_FOLDER/BFOLDS_v2.3_nocode/"examples 2.3/examples" \
# && dotnet $LANDIS_CONSOLE scenario.txt

# We delete all of the files generating during testing (more recent than the temporary file we created before.
# Comment this line to keep the files generated during testing.
# RUN find $LANDIS_FOLDER//Extension-Age-Only-Succession-Archived/testing/Corev7 -type f -newer "/tmp/tempFileForTestFileRemoval.txt" -exec rm {} \;


################################################################
# INSTALLING THE FOREST VEGETATION SIMULATOR
# This is not the latest version, as latest version as an issue that breaks the build; but should be good enough
# See https://github.com/USDAForestService/ForestVegetationSimulator/issues/44 
################################################################

# Install system dependencies and clear cache
RUN apt-get update && apt-get install -y \
    gfortran \
    cmake \
    unixodbc-dev \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/* \
    # Clone and compile FVS (3b09ae2 commit for stability)
	&& cd /bin && git clone https://github.com/USDAForestService/ForestVegetationSimulator.git \
    && cd /bin/ForestVegetationSimulator \
    && git checkout 3b09ae2 \
    # Compile the FVS binary, keeping this as a separate layer for caching
	&& cd /bin/ForestVegetationSimulator/bin && make

# Update PATH environment to include the FVS binary
# For example, FVSon can be called with just "FVSon" from everywhere.
ENV PATH="/bin/ForestVegetationSimulator/bin:$PATH"


################################################################
# FINISHING
################################################################

# Re-configure git for latest version of HTTP protocol
RUN git config --global --unset http.version

# Adding the Python packages I need to make Magic Harvest run
COPY ./files_to_help_compilation/pip_requirements.txt /tmp
RUN pip install -r /tmp/pip_requirements.txt

# Adding jupyter lab and book
RUN pip install jupyterlab jupyter-book

# Adding add-on "Variable explorer" for jupyter lab
RUN pip install lckr-jupyterlab-variableinspector

# Add-on to make markdown code look better
RUN pip install jupyterlab_myst

# Enable JupyterLab
ENV JUPYTER_ENABLE_LAB=yes

# Expose port 8888 for JupyterLab
EXPOSE 8888

# Make the Jupyterlab start with the container
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--allow-root"]