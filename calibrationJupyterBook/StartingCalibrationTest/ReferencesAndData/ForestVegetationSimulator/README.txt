VERY SHORT COURSE IN USING FVS

See https://www.fs.usda.gov/fmsc/ftp/fvs/docs/gtr/EssentialFVS.pdf for more


## Input files

There are three essential types of inputs :

- The Keyword file (which contains infos about where the other inputs are, the main parameters of the simulation run, what outputs should be made, etc.)
- Info about the trees to simulate (can be in a text file or a MySQL database)
- Info about the stands to simulate (can be in a text file or a MySQL database)

### Keyword file (text file, .key)

These can be pretty simple or very complex. They contain "keywords" that define everything in the simulation :

- Which stands should be harvested, and where are the informations about their initial conditions (Tree Data File; see below)
    - These initial conditions can be loaded from a .db file using the DATABASE keyword, and then the instruction DSNin.
- What is the beginning year, the time steps (cycles), the number of time steps to simulate etc.
- What forest management is supposed to be done in the stand during the simulation (e.g. THINBBA for thinning by basal area)
- What output files should be made : a Tree list file (TREELIST), a Summary Statistics file (ECHOSUM), etc.
    - Database output files can be made with DATABASE, followed by DSNOut.

Every "sequence" in the file start with a keyword, and end with "END".

### Stand data

Stand data must be given through a database (or at least, it no other ways are mentionned in the essential guide of FVS) or as a header in the tree data file (less recommanded ?).

⚠️ It seems that the model "fills" the stand with trees based on the composition of sample "plots" in the stand - in the same way that in reality, we measure sample plots rather than the whole stands. As such, information about these plots - their size, the cuttof for small trees, etc. - informs the model on the density of the stand, which is important. This is done through the "fixed plot area for large tree" (calculated through the Basal Area factor), and the firex plot area for small tree (calculated with the plot inventory size).

⚠️ Most outputs are in /Ha; but it seems that stands do have an area (in the outputs : SUMMARY STATISTICS (PER HA OR STAND BASED ON TOTAL   STAND AREA)).

Here are some attributes found in https://www.fs.usda.gov/fmsc/ftp/fvs/docs/gtr/DBSUserGuide.pdf :

- Stand_CN : Control number of stand. Not vital.
- Stand_ID : Stand ID. Necessary.
- Latitude, longitude : self-explanatory. This has impacts on the results.
- Region : A region code. Difficult to know what are the vailable region codes in the Ontario variant of FVS. Doesn't seem to change anything in the results whatever the value used.
- Forest : same as region, code for "national forest". Doesn't seem to change much.
- District, Compartment, location, ecoregion : don't seem useful in Ontario variant.
- Aspect, slope and elevation : Seem important. 0 can be used for no meaningful aspect. All three don't always have an influence on the result. Elevation is in 100's of feet ! Use ElevFt to input it in feet.
- Basal Area Factor : required. After research, I still don't understand what this means. It seems to be a parameter influencing the computation of stand density (page 169 of https://www.fs.usda.gov/fmsc/ftp/fvs/docs/gtr/EssentialFVS.pdf). Might change the number of large trees in the stand through the plot calculation (see above).
- Inventory plot size : Has a big effect on results; basically multiplies the numbers of trees ! (small trees only ?) If one wants to perfectly controll the amount of trees through the tree data file (see Tree_count), one can use 1.0.
- Site species and site index : The site index is a representation of the ressources in the stand (soil, etc.), and it is given in reference to a given species. Can both be put to Null to use default values. The site index has a certain importance on results according to some testings with the FVS Ontario variant I'm using. The values of the site index depend on the site curves used to build the variant model (normally described in a manual somewhere). Pretty nebulous. Can leave empty to use a default. It seems that the higher the value, the higher growth; so Site Index could be representative of soil nutrients or something like that. Values above 700 seems to create errors in the model. Enormous values (e.g. 600) can results in very high "plateaus" for the growth curve, that might be unrealistic. Difficult to know what value to use.
These site index seem to originally come from this document : https://www.fs.usda.gov/nrs/pubs/gtr/gtr_nc128.pdf . The document states " Forest Site Quality is an estimate of the capacity of forest land to grow trees, thus forest site quality corresponds to land capability for growing various agricultural crops". The document further states that it is possible to estimate the site index quality from tree inventories (I imagine that this is done by comparing their annual growth to what is predicted under different site index). They state "For most eastern forest species, site index is defined as total height of dominant or total height of dominant and codominant trees at 50 years age". Still according to this document, site index values often range from 40-80, but can go up to 130 for some species (the site index is species dependant). In the FVS Ontario, leaving the site index and the species for the stand at NULL input default values, which are close to 22 for all of the different species codes (see .out file).
- The rest are values related to the fuel (for fires) in the stand, and other optional variables.

### Tree data file

Apparently not needed if a .db file is used with the instruction DATABASE, and then DSNin in the Keyword file (see above), which loads the stand and tree data from a MySQL Database.

If a text file is used, I'm not certain that several stands can be simulated. At least, stand data txt files are not talked about in https://www.fs.usda.gov/fmsc/ftp/fvs/docs/gtr/EssentialFVS.pdf.

Trees are associated to a stand in any case; but they can also be applied to "plots", which are sub-units inside the stand (although that seems to be entirely optional). It's possible that they are used to "populate" the stands when the exact composition of stands is not known.

The variables that need to be associated to the individual trees are described in https://www.fs.usda.gov/fmsc/ftp/fvs/docs/gtr/EssentialFVS.pdf. One can find a MySQL database with tree data ready in https://github.com/USDAForestService/ForestVegetationSimulator/tree/main/tests/FVSon (see FVSDataHardwood).

Here are some of them (fields in the database) :

⚠️ The names of these variables are not the same as the column names that must be in the database ! See Table 4-2 page 44 of https://www.fs.usda.gov/fmsc/ftp/fvs/docs/gtr/EssentialFVS.pdf for the equivalence. For example, the variable ITH (Tree History) must be in a column named "history" in the database. I'll put the column name in parenthesis.

- PROB (Tree_count) : Allows to define several trees at once in once single record.
- ISP (Species) : Defines the tree species. Must be one of the species defined in the variant of FVS that is being used. For the ontario variant, the species seems to be indicated here in the code : https://github.com/USDAForestService/ForestVegetationSimulator/blob/5c29887e4168fd8182c1b2bad762f900b7d7e90c/canada/on/grinit.f#L43
- DBH (DBH) : Diameter at Breast Height for the tree. Can be 0, or 0.1 to represent buds (but height of tree should be bellow 4.5 feet). Somes species seem to be pretty sensitive to this parameter; for example, for the american beech (BE) in the FVS Ontario variant, putting a 0.1 value leads to no trees growing in 150 years; but a 0.3 value seems to work fine. It looks like a value of 0.5 can be useful to simulate a "empty" stand with only natural regeneration.
- ITH (History) : This one is a head scratcher. Looks like it indicates what trees have the same "history" - before the simulation runs. Codes 0 - 5 are for trees that are alive at the beginning of the simulation. Code 6, 7, 8 and 9 indicate forms of mortality. Seems like using values 0-5 for the same simple sim doesn't change anything. I just use 1.
- DG (DG) : Periodic diameter increment. I'm not certain how this interacts precisely with the equations already inside FVS; but it seems that this can "calibrate" these equations onto real periodic increments for a given tree record ? Things work if no value are precised.
- HT (Ht) : Tree height. If ommited, initial heights are calculated using allometric equations imbedded in the program. If given, this relation can be extrapolated from the data given.
- THT (HtTopK) : Height to point of top kill, above which the wood of the tree is dead. Only for trees still alive, but with a dead part.
- HTG (HTG) : Height increment. Same as DG, but for height.
- ICR (ICR) : Crown ratio code. When missing, a value is computed using imbedded equations.
- IDCD (IDCD) : Damage and severity code. Used for disturbance extensions (mistletoe, etc.).

THe minimum necessary data seems to be Stand_CN, Stand_ID, Plot_ID, Tree_ID, Tree_Count, History, Species, DBH, DG

## Output files


### Summary Output file

Most important one for us, as it contains the Gross Total Volume we will use to compute estimates biomass, which we can then compare to what's going on in PnET.

See page 82 of https://www.fs.usda.gov/fmsc/ftp/fvs/docs/gtr/EssentialFVS.pdf

First line (starting with -999) is a header record containing information about the stand, stand attributes, time, version of FVS used for the run. Here are the details :

- Header indicator: -999 indicates that a new table follows
- Number of rows of data in the table
- Stand ID
- Management ID
- Stand sampling weight
- FVS variant code
- FVS variant version number
- Date of simulation run
- Time of simulation run
- FVS variant revision date
- Parallel processing code (obsolete).

The lines after the header are made of several columns which indicate different variables related to the composition of the stand. To see what each variable correspond to, you should check the "SUMMARY STATISTICS" table in the MAIN OUTPUT FILE that you specified. Here is an example from a test run :

--------------------------------------------------------------------------------------------------------------------------------------
               START OF SIMULATION PERIOD                     REMOVALS             AFTER TREATMENT    GROWTH THIS PERIOD
         --------------------------------------------- ----------------------- ---------------------  ------------------    MAI ------
         NO OF              TOP        GTV   GMV   NMV NO OF   GTV   GMV   NMV              TOP  RES PERIOD  ACCRE MORT     GTV FOR SS
YEAR AGE TREES  BA  SDI CCF  HT  QMD  CU M  CU M  CU M TREES  CU M  CU M  CU M  BA  SDI CCF  HT  QMD  YEARS   PER  YEAR    CU M TYP ZT
---- --- ----- --- ---- --- --- ---- ----- ----- ----- ----- ----- ----- ----- --- ---- --- --- ----  ------ ---- -----   ----- ------

BA = Basal Area
SDI = Stand Density Index
CCF = Crown Competition Factor
TOP HT = Average Dominant Height (of the 40 largest diameter trees) ?
QMG = Quadratic Mean Diameter
GTV = Gross Total Volume (?) => Acronym is indicated nowhere, but see https://github.com/USDAForestService/ForestVegetationSimulator/blob/5c29887e4168fd8182c1b2bad762f900b7d7e90c/canada/on/volont.f#L249 .
GMV = Gross Merchantable Volume (?) => Makes sense.
NMV = Net Merchantable Volume => Again, see https://github.com/USDAForestService/ForestVegetationSimulator/blob/5c29887e4168fd8182c1b2bad762f900b7d7e90c/canada/on/varvol.f#L369 .
CU M = Cubic meter (m3).
MAI GTV CU M = Mean Annual Increment in GTV in m3
FOR TYP = Forest type
SS ZT = Stand site class ?
