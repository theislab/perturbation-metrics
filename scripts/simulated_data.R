library(splatter)
library(scater)
library(scuttle)
setwd("/dss/dsshome1/02/di93zoj/yuge")


base_params <- newSplatParams()
# You can estimate these from a real dataset
# There are some tips for this if you want to try it
# base_params <- splatEstimate(counts)

# Define the condition parameters
# Can be done in the function but this is a bit neater
#
# Prob - Probability cell is assigned to this group (affects group sizes)
# DEProb - Probability that a gene is DE in this group
# DownProb - Probability a DE gene is down-regulated
# FacLoc - Mean of DE factors (log-normal)
# FacScale - Variance of DE factors (log-normal)
# Steps - Number of steps in the path
# Skew - Distribution of cells along the path
#
# Together Steps and Skew create discrete clusters rather than continuous paths (except for the control)

# Example dataframe
# conditions <- tibble::tribble(
#     ~ Prob, ~ DEProb, ~ DownProb, ~ FacLoc, ~ FacScale, ~ Steps, ~ Skew,
#     # Control
#       0.20,     0.05,        0.5,     0.10,       0.40,      20,    0.5,
#     # Conditions
#       0.15,     0.05,       0.25,     0.20,       0.50,       2,    0.0,
#       0.08,     0.05,       0.50,     2.00,       0.02,       2,    0.0,
#       0.02,     0.05,       0.50,     0.50,       1.50,       2,    0.0,
#       0.08,     0.10,       0.25,     0.80,       0.80,       2,    0.0,
#       0.15,     0.80,       0.25,     1.20,       1.20,       2,    0.0,
#       0.01,     0.10,       0.50,     0.60,       0.90,       2,    0.0,
#       0.11,     0.15,       0.50,     1.50,       0.50,       2,    0.0,
#       0.11,     0.15,       0.25,     0.70,       2.00,       2,    0.0,
#   #    0.05,     0.50,       0.50,     0.50,       0.90,       2,    0.0,
#       0.09,     0.40,       0.75,     0.90,       0.60,       2,    0.0,
# )

control <-tibble::tribble(
  ~ Prob, ~ DEProb, ~ DownProb, ~ FacLoc, ~ FacScale, ~ Steps, ~ Skew,
  # Control
  0.40,     0.05,        0.5,     0.10,       0.40,      20,    0.5,
)

# same populations, different number of DE genes
perturbed <- tibble::tribble(
 ~ DEProb,  
   0.00001,  
   0.0001,   
   0.001,    
   0.005,    
   0.01,     
   0.05,     
   0.00001,  
   0.0001,   
   0.001,    
   0.005,    
   0.01,     
   0.05,     
   0.00001,  
   0.0001,   
   0.001,    
   0.005,    
   0.01,     
   0.05,     
   0.00001,  
   0.0001,   
   0.001,    
   0.005,    
   0.01,     
   0.05,     
)
n <- 24
perturbed$Prob = rep(.6/n, n)
perturbed$DownProb = rep(.50, n)
perturbed$FacLoc = c(rep(.2, n/4), rep(.5, n/4), rep(1, n/4), rep(1.5, n/4))  # adjust size of lognormfoldchange
perturbed$FacScale = c(rep(.2, n/4), rep(.5, n/4), rep(1, n/4), rep(1.5, n/4))  # adjust variance
perturbed$Steps = rep(2.0, n)
perturbed$Skew = rep(0.0, n)

conditions = rbind(control, perturbed)

sim <- splatSimulatePaths(
    params = base_params,
    batchCells = 40000,
    # BASIC PARAMETERS - CAN BE ESTIMATED
    # Base gene means (gamma distribution)
    mean.shape = 0.3,
    mean.rate = 0.6,
    # Library size (log-normal distribution)
    lib.loc = 9,
    lib.scale = 0.1,
    # Dispersion (bit hard to explain...)
    bcv.common = 0.1,
    bcv.df = 60,
    # PATH PARAMETERS - MUST BE SET
    group.prob = conditions$Prob,
    path.nSteps = conditions$Steps,
    path.skew = conditions$Skew,
    # RANDOM SEED
    seed = 1,
    # optional
    de.downProb = conditions$DownProb,
    de.prob = conditions$DEProb,
    de.facLoc = conditions$FacLoc,
    # de.facScale = conditions$FacScale
)

# Save the count matrix and then the observation labels
sce_counts <- as.matrix(counts(sim))
write.table(sce_counts, 'splatter_sim.csv', sep=",")
sce_obs <- colData(sim)
write.table(sce_obs, 'splatter_sim_obs.csv', sep=",")
write.table(conditions, 'splatter_sim_params.csv', sep=",")

##### Second simulation - sparsity ######
#########################################

# Here we keep the number of differentially expressed genes constant at __
# while we vary the library size to induce sparsity. Instead of comparing against
# a single control as before, we'll compare against a control of the same library
# size.

control <-tibble::tribble(
  ~ Prob, ~ DEProb, ~ DownProb, ~ FacLoc, ~ FacScale, ~ Steps, ~ Skew,
  # Control
  0.50,     0.05,        0.5,     0.10,       0.40,      20,    0.5,
)
perturbed <- tibble::tribble(
  ~ DEProb,  ~ FacLoc,
  0.05,    .1,     
  0.05,    .2,     
  0.05,    .3,   
  0.05,    .4,  
  0.05,    .5,  
)
n <- 5
perturbed$Prob = rep(.5/n, n)
perturbed$DownProb = rep(.50, n)
perturbed$FacScale = rep(.4, n)
perturbed$Steps = rep(2.0, n)
perturbed$Skew = rep(0.0, n)
conditions = rbind(control, perturbed)

for (LibLoc in c(8.3, 8.5, 8.7, 8.9, 9.1, 9.3)){
  sim <- splatSimulatePaths(
    params = base_params,
    batchCells = 12000,
    # BASIC PARAMETERS - CAN BE ESTIMATED
    # Base gene means (gamma distribution)
    mean.shape = 0.3,
    mean.rate = 0.6,
    # Library size (log-normal distribution), setting the parameters of hist(rlnorm(n = 1000, meanlog = 10, sdlog = 0.2))
    lib.loc = LibLoc,
    lib.scale = 0.1,
    # Dispersion (bit hard to explain...)
    bcv.common = 0.1,
    bcv.df = 60,
    # PATH PARAMETERS - MUST BE SET
    group.prob = conditions$Prob,
    path.nSteps = conditions$Steps,
    path.skew = conditions$Skew,
    # RANDOM SEED
    seed = 1,
    # optional
    de.downProb = conditions$DownProb,
    de.prob = conditions$DEProb,
    de.facLoc = conditions$FacLoc,
    # de.facScale = conditions$FacScale
  )

  # Save the count matrix and then the observation labels
  # name = paste(p.DE, pLFC, '0_data.csv', sep='_')
  sce_counts <- as.matrix(counts(sim))  
  write.table(sce_counts, paste(LibLoc, 'sparsity_sim.csv', sep='-'), sep=",")
  sce_obs <- colData(sim)
  write.table(sce_obs, paste(LibLoc, 'sparsity_sim_obs.csv', sep='-'), sep=",")
  write.table(conditions, paste(LibLoc, 'sparsity_sim_params.csv', sep='-'), sep=",")
}

# Check the UMAP
sim <- scuttle::logNormCounts(sim)
sim <- scater::runPCA(sim)
sim <- scater::runUMAP(sim)
scater::plotUMAP(sim, colour_by = "Group")
