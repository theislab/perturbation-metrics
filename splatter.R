library(splatter)
library(scater)

# quick start
vcf <- mockVCF()
gff <- mockGFF()

sim <- splatPopSimulate(vcf = vcf, gff = gff, sparsify = FALSE) 
sim <- logNormCounts(sim)
sim <- runPCA(sim, ncomponents = 5)
plotPCA(sim, colour_by = "Sample")

# conditional
params.cond <- newSplatPopParams(eqtl.n = 0.5, 
                                 batchCells = 50,
                                 similarity.scale = 5,
                                 condition.prob = c(0.5, 0.5),
                                 eqtl.condition.specific = 0.5,
                                 cde.facLoc = 1, 
                                 cde.facScale = 1)

sim.pop.cond <- splatPopSimulate(vcf = vcf, gff = gff, params = params.cond, 
                                 sparsify = FALSE)

sim.pop.cond <- logNormCounts(sim.pop.cond)
sim.pop.cond <- runPCA(sim.pop.cond, ncomponents = 10)
plotPCA(sim.pop.cond, colour_by = "Condition", shape_by = "Sample", point_size=3)