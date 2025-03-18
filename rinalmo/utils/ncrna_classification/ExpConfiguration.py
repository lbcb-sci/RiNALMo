# Adopted from https://github.com/bioinformatics-sannio/ncrna-deep

from rinalmo.utils.ncrna_classification.seqEncoders import *

nlayers=[0,1,2,3]  # deep architecture layers
bnoise=[200] # boundary noise simulation
t=[1,2,5,10] # number of random (non functional) seqs for each test seq
padds = ['random'] # how padding is done

# Encoders configurations 
seqEncoders = (
               {'enc' : seq2Kmer,
                'filename' : '1mer',
                'param0' : [[200],1,['A','T','C','G']],
                'param25' : [[250],1,['A','T','C','G']],
                'param50' : [[300],1,['A','T','C','G']],
                'param75' : [[350],1,['A','T','C','G']],
                'param100' : [[400],1,['A','T','C','G']],
                'param125' : [[450],1,['A','T','C','G']],
                'param150' : [[500],1,['A','T','C','G']],
                'param175' : [[550],1,['A','T','C','G']],
                'param200' : [[600],1,['A','T','C','G']]
                },
               )
