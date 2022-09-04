import time

import core
import fileioput as fio

# Download from https://doi.org/10.11577/1096906
patternIntensityFile = "./data/cxidb-4.cxi"
patternIntensityNode = "entry_1->image_1->data"
patternIntyErrorFile = "./data/cxidb-4.cxi"
patternIntyErrorNode = "entry_1->image_1->data_error"

patternIntensity = fio.ReadFile(patternIntensityFile, patternIntensityNode)
patternIntyError = fio.ReadFile(patternIntyErrorFile, patternIntyErrorNode)

phasing = core.Phasing(patternIntensity, patternIntyError)

phasing.SetPatternMask("match")
phasing.SetInitRealSpace("random")
phasing.SetInitSupport("auto")

phasing.SetOutputMethod("./yeast/HMG/", 10000)
phasing.SetOutputRealSpace()
phasing.SetOutputSupport()
phasing.SetOutputFourierSpace()

phasing.SetSupportUpdateMethod(50)
phasing.SetSupportUpdateBlur(50, 1, 0.006)
phasing.SetSupportUpdateThreshold(0.4, 0.2, 0.002)
phasing.SetSupportUpdateArea(0.1, 0.05, 0.002)

phasing.SetMultigridMethod(3)
phasing.SetPhasingMethod("HIO", 10000, beta=0.9)
phasing.SetPhasingMethod("ER", 2000, 0)

t=time.time()
phasing.Run(50)
print(time.time()-t)
