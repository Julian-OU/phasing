import core
import fileioput as fio

objectInputFile = "./data/vesicle.mrc"
oversamplingRatio = 5
patternCoverRatio = 0
patternCoverShape = "c"
patternCoverUpper = 10000
patternCoverLower = 0

objectInput = fio.ReadFile(objectInputFile)
objectInput = core.cp.sum(objectInput, axis=2)

for poissonNoiseRatio in [0.025, 0.05]:
    patternIntensity, patternMask, perfectSupport = core.Simulation(
        objectInput,
        oversamplingRatio,
        poissonNoiseRatio,
        patternCoverRatio,
        patternCoverShape,
        patternCoverUpper,
        patternCoverLower,
    )[:3]

    phasing = core.Phasing(patternIntensity)
    phasing.SetPatternMask("custom", patternMask=patternMask)
    phasing.SetInitRealSpace("random")
    phasing.SetInitSupport("auto", 0.02, 20)

    phasing.SetSupportUpdateMethod(50)
    phasing.SetSupportUpdateBlur(20, 0.5, 0.04)
    phasing.SetSupportUpdateThreshold(0.4, 0.2, 0.01)
    phasing.SetSupportUpdateArea(0.05, 0.025, 0.01)

    # phasing.SetOSSFramework(period=200)

    phasing.SetMultigridMethod(2)
    phasing.SetPhasingMethod("HIO", 4000, 1, beta=0.9)
    phasing.SetPhasingMethod("HIO", 2000, 0, beta=0.9)
    phasing.SetPhasingMethod("ER", 400, 0)
    phasing.SetOutputMethod("./vesicle/" + str(poissonNoiseRatio) + "/HMG/", 6000)
    phasing.SetOutputRealSpace()
    phasing.SetOutputSupport()
    phasing.Run(100)
