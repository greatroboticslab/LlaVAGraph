# define our variables
MODELPATH=/projects/imo2d/LLaVAChartv2
IMAGEFOLDER=/home/imo2d/LLaVA/data/subset/trainData

# remove to test more
SUBSET=10

# random noise
python evaluateLLaVA.py --model-path $MODELPATH --image-folder $IMAGEFOLDER/NoiseData --output-file results/randomNoise.json --subset $SUBSET

# sine waves
python evaluateLLaVA.py --model-path $MODELPATH --image-folder $IMAGEFOLDER/SineData --output-file results/sineWave.json --subset $SUBSET

#square waves
python evaluateLLaVA.py --model-path $MODELPATH --image-folder $IMAGEFOLDER/SquareData --output-file results/squareWave.json --subset $SUBSET
