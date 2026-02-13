# Load required packages
library(RSYC)
library(dplyr)

# Get all available species codes
species_list <- species_codes()

# Check the actual column names
print("Column names in species_codes():")
print(names(species_list))
print(head(species_list))

# Initialize results data frame
max_biomass_results <- data.frame(
  species = character(),
  max_biomass_tonnes_ha = numeric(),
  stringsAsFactors = FALSE
)

species_list[[1]] <- as.character(species_list[[1]])

# Loop through each species (adjust column names based on actual structure)
for (i in 1:nrow(species_list)) {
  
  # Use the correct column names from species_codes()
  species_code <- species_list[[1]][i]
  species_name <- species_list[[2]][i]
  
  cat("Processing species:", species_name, "(", species_code, ")\n")
  
  # Get all tiles available for this species
  tiles <- tiles_for_species(species_code)
  
  # Initialize max biomass for this species
  species_max_biomass <- 0
  
  # Loop through each tile for this species
  for (tile in tiles) {
    
    tryCatch({
      # Generate growth curve (age 0 to 250 years)
      ages <- 0:150
      biomass_predictions <- predict_rsyc(
        species = species_code,
        age = ages,
        tile = tile
      )
      
      # Find maximum biomass in this curve
      curve_max <- max(biomass_predictions, na.rm = TRUE)
      
      # Update species maximum if this curve has higher value
      if (curve_max > species_max_biomass) {
        species_max_biomass <- curve_max
      }
      
    }, error = function(e) {
      cat("  Warning: Error processing tile", tile, "for species", species_code, "\n")
    })
  }
  
  # Store result for this species
  max_biomass_results <- rbind(
    max_biomass_results,
    data.frame(
      species = species_name,
      max_biomass_tonnes_ha = species_max_biomass,
      stringsAsFactors = FALSE
    )
  )
  
  cat("  Max biomass:", species_max_biomass, "tonnes/ha\n\n")
}

# Save results to CSV
write.csv(
  max_biomass_results,
  file = "species_max_biomass_TompalskiEtAl2025.csv",
  row.names = FALSE
)

cat("Analysis complete! Results saved to species_max_biomass_TompalskiEtAl2025.csv\n")
print(max_biomass_results)
