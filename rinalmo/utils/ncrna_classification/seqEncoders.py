# Adopted from https://github.com/bioinformatics-sannio/ncrna-deep

"""Provides sequence to 2D representation conversion utilities."""

from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
#from Bio.Alphabet import IUPAC
import numpy as np
import math
import itertools
from textwrap import wrap

IUPACletters = 'GATCRYWSMKHBVDN'

def checkRNAbind(l1,l2):
    if l1=='A' and l2=='T' or l1=='T' and l2=='A':
        return '1'
    if l1=='C' and l2=='G' or l1=='G' and l2=='C':
        return '2'
    return '0'

def seq2ContactMatrixLinear(seq: Seq, sdim=[[200],IUPACletters]):
    alphabet = [''.join(x) for x in itertools.product(sdim[1], repeat=2)]    
    char_to_int = dict((c, i) for i, c in enumerate(alphabet))
    seqr=np.random.randint(len(sdim[1])**2, size=sdim[0][0])
    outdiag = 1
    start_i = -1
    start_j = 1
    k=0
    while start_j<len(seq):
        if (outdiag):
            start_i += 1
            outdiag=0
        else:
            start_j += 1
            outdiag=1
        i = start_i
        j = start_j
        while i>=0 and j<len(seq):
            seqr[k] = char_to_int[seq[i]+seq[j]] #checkRNAbind(seq[i],seq[j])
            k += 1
            i -= 1
            j += 1
    return seqr

def seq2ContactMatrix(seq: Seq, sdim=[[16,16],['0','1','2']]):
    char_to_int = dict((c, i) for i, c in enumerate(sdim[1]))
    seqr=np.random.randint(len(sdim[1]), size=sdim[0])
    for i,c1 in enumerate(str(seq)):
        for j,c2 in enumerate(str(seq)):
            seqr[i,j] = char_to_int[checkRNAbind(c1,c2)]
    outdiag = 1
    start_i = -1
    start_j = 1
    k=0
    while start_j<len(seq):
        if (outdiag):
            start_i += 1
            outdiag=0
        else:
            start_j += 1
            outdiag=1
        i = start_i
        j = start_j
        while i>=0 and j<len(seq):
            if seqr[i,j]>0 and seqr[i+1,j-1]==0 and j<(len(seq)-1) and i>0 and seqr[i-1,j+1]==0:
                seqr[i,j] = 0
                seqr[j,i] = 0
            k += 1
            i -= 1
            j += 1
    return seqr


def seq2ContactMatrixMix(seq: Seq, sdim=[[16,16],IUPACletters]):
    alphabet = [''.join(x) for x in itertools.product(sdim[1], repeat=2)]
    alphabet = alphabet + sdim[1]
    char_to_int = dict((c, i) for i, c in enumerate(alphabet))
    seqr=np.random.randint(len(alphabet), size=sdim[0])
    for i,c1 in enumerate(str(seq)):
        for j,c2 in enumerate(str(seq)):
            if i==j:
                seqr[i,j] = char_to_int[c1]
            else:
                seqr[i,j] = char_to_int[c1 + c2]
    return seqr


def seq2Rnd2D(seq: Seq, sdim=[[16,16],IUPACletters]):
    char_to_int = dict((c, i) for i, c in enumerate(sdim[1]))
    #seqr=np.random.randint(len(sdim[1]), size=sdim[0])
    seqr=np.zeros(sdim[0])
    #ss = np.empty(sdim[0],dtype=np.str)
    k = 0
    j = 0
    w=list(str(seq))
    np.random.shuffle(w)
    for i,c in enumerate(w):
        if k==sdim[0][0]:
            k=0
            j +=1
        seqr[j,k] = char_to_int[c]
        #ss[k,j] = c
        k +=1
    return seqr

def seq2Xhot(seq: Seq, sdim=[[200],2,IUPACletters,2],padding='constant'):
    #seqr=np.random.randint(len(sdim[2]), size=sdim[0][0])
    seqr=np.zeros(sdim[0][0])
    data = [str(seq)[i:i+sdim[3]] for i in range(len(seq)-sdim[3]+1)]
    if (len(data[len(data)-1]) < sdim[1]):
        data = data[:(len(data)-1)]
    alphabet = [''.join(x) for x in itertools.product(sdim[2], repeat=sdim[1])]    
    char_to_int = dict((c, i+1) for i, c in enumerate(alphabet))
    integer_encoded = [char_to_int[char] for char in data]
    seqr[0:len(integer_encoded)] = integer_encoded
    return seqr


def seq2Kmer(seq: Seq, sdim=[[200],2,IUPACletters],padding='constant'):
    alphabet = [''.join(x) for x in itertools.product(sdim[2], repeat=sdim[1])]    
    # this is default constant padding
    char_to_int = dict((c, i) for i, c in enumerate(alphabet))
    seqr=np.zeros(sdim[0][0])
    if (padding=='random'):
        seqr=np.random.randint(len(sdim[2]), size=sdim[0][0])
    if (padding=='new'):
        char_to_int = dict((c, i+1) for i, c in enumerate(alphabet))
        seqr=np.zeros(sdim[0][0])

    data=wrap(str(seq),sdim[1])
    if (len(data[len(data)-1]) < sdim[1]):
        data = data[:(len(data)-1)]
    integer_encoded = [char_to_int[char] for char in data]
    seqr[0:len(integer_encoded)] = integer_encoded
    return seqr

def get_labels(fasta_file):
    seq_rec = list(SeqIO.parse(fasta_file, "fasta"))
    labels=[]
    for i,r in enumerate(seq_rec):
        labels.append(r.name)
    return labels

def write_seqs(fasta_file,seqs,ids):
    seqrecords = []
    for i,s in enumerate(seqs):
        sr = SeqRecord(s,id=ids[i],description='')
        seqrecords.append(sr)
    SeqIO.write(seqrecords, fasta_file, 'fasta')

def get_nonfunctional_seqs(fasta_file,t=1):
    seq_rec = list(SeqIO.parse(fasta_file, "fasta"))
    samples=[]
    for nt in range(t):
        for i,r in enumerate(seq_rec):
            sw = wrap(str(r.seq),2)
            np.random.shuffle(sw)
            samples.append(Seq(''.join(sw)))
    return samples

    
def get_seqs_with_bnoise(fasta_file,nperc=0,dinucleotide='preserve'):
    basi = ['A','T','G','C']
    seq_rec = list(SeqIO.parse(fasta_file, "fasta"))
    samples=[]
    for i,r in enumerate(seq_rec):
        stop=''
        sbottom=''
        if (nperc>0):
            if dinucleotide=='preserve':
                sw = wrap(str(r.seq),2)
                stop=np.random.choice(sw, int(0.25*len(r.seq)*nperc/100))
                sbottom=np.random.choice(sw, int(0.25*len(r.seq)*nperc/100))            
            else:
                stop=np.random.choice(basi, int(0.5*len(r.seq)*nperc/100), p=[0.25, 0.25, 0.25, 0.25])
                sbottom=np.random.choice(basi, int(0.5*len(r.seq)*nperc/100), p=[0.25, 0.25, 0.25, 0.25])
            stop=''.join(stop)
            sbottom=''.join(sbottom)
        samples.append(stop+r.seq+sbottom)
    return samples

def get_rnd_seqs(fasta_file,k=1):
    seq_rec = list(SeqIO.parse(fasta_file, "fasta"))
    samples=[]
    for i,r in enumerate(seq_rec):
        bs,cn = np.unique(r.seq,return_counts=True)
        for j in range(k):
            rndSeq=np.random.choice(bs, len(r.seq), p=cn/len(r.seq))
            rndSeq=''.join(rndSeq)
            samples.append(rndSeq)
    return samples

def encode_seqs(seqlist,enc=seq2Kmer,encparam=[[13,13]],padding='constant'):
    samples=np.zeros([len(seqlist)]+encparam[0],dtype=np.int16)
    for i,seq in enumerate(seqlist):
        samples[i]=enc(seq,sdim=encparam,padding=padding)
    print(' Done %d total records' % len(seqlist))
    return samples


if __name__ == "__main__":
    q=Seq('ABCDEFGHILMNOPQR')
    j = np.unique(q)
    w=seq2ContactMatrixLinear(q,sdim=[[8*15],j])
    print(w)
    
    w=seq2Rnd2D(q,sdim=[[4,4],j])
    print(w)
    
    q=Seq('ATGCGCA')
    w=seq2Xhot(q,sdim=[[[6]],2,['A','T','C','G'],2])
    print(w)
    
    q=Seq('ATACAGAGTGTAGACCGCAGCGACCTGCG')
    w=seq2ContactMatrixMix(q,sdim=[[30,30],['A','T','C','G']])
    print(w)
    
    print(len(w))
    w=seq2ContactMatrix(q,sdim=[[30,30],['0','1','2']])
    print(w)
