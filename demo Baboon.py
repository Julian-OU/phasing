import core
import fileioput as fio

objectInputFile = './data/Baboon.tif'
oversamplingRatio = 5
patternCoverRatio = 0.005
patternCoverShape = 'r'
patternCoverUpper = core.INF
patternCoverLower = 0

objectInput = fio.ReadFile(objectInputFile)

for poissonNoiseRatio in [0.1, 0.2, 0.3, 0.4]:
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
    phasing.SetInitSupport("custom", initSupport=perfectSupport)

    # phasing.SetOSSFramework(period=40)

    phasing.SetMultigridMethod(5)
    phasing.SetPhasingMethod("HIO", 100,(1,2,3,4), beta=0.9)
    phasing.SetPhasingMethod("HIO", 400,0, beta=0.9)
    phasing.SetPhasingMethod("ER", 100, 0)
    phasing.SetOutputMethod(
        "./Baboon/"+str(poissonNoiseRatio)+"/HMG/", 800)
    phasing.SetOutputRealSpace()
    phasing.SetOutputSupport()
    phasing.Run(50)
