import core
import fileioput as fio

objectInputFile = './data/Baboon.tif'
oversamplingRatio = 5
patternCoverRatio = 0.005
patternCoverShape = 'r'
patternCoverUpper = core.INF
patternCoverLower = 0

objectInput = fio.ReadFile(objectInputFile)

for emitPhotonsNumber in [1e10, 2e9]:
    (
        patternIntensity,
        patternNoiseFree,
        patternMask,
        perfectSupport,
        noiseRatio,
    ) = core.Simulation(
        objectInput,
        oversamplingRatio,
        emitPhotonsNumber,
        patternCoverRatio,
        patternCoverShape,
        patternCoverUpper,
        patternCoverLower,
    )[
        :5
    ]
    phasing = core.Phasing(patternIntensity)
    phasing.SetPatternMask("custom", patternMask=patternMask)
    phasing.SetInitRealSpace("random")
    phasing.SetInitSupport("custom", initSupport=perfectSupport)

    # phasing.SetOSSFramework(period=900)

    phasing.SetMultigridMethod(5)
    phasing.SetPhasingMethod("HIO", 100, (1, 2, 3, 4), beta=0.9)
    phasing.SetPhasingMethod("HIO", 400, 0, beta=0.9)
    phasing.SetPhasingMethod("ER", 100, 0)
    phasing.SetOutputMethod(
        "./Baboon/" + "%.2f-%.0e" % (noiseRatio,
                                     emitPhotonsNumber) + "/HMG/", 900
    )
    phasing.SetOutputRealSpace()
    phasing.SetOutputSupport()
    phasing.Run(50)
