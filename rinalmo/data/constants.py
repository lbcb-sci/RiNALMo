# Tokens
ADENINE_TKN  = 'A'
CYTOSINE_TKN = 'C'
URACIL_TKN   = 'U'
GUANINE_TKN  = 'G'
THYMINE_TKN  = 'T'
INOSINE_TKN  = 'I'

ANY_NUCLEOTIDE_TKN = 'N'

# Tokens from https://en.wikipedia.org/wiki/FASTA_format
RNA_TOKENS = [ADENINE_TKN, CYTOSINE_TKN, GUANINE_TKN, THYMINE_TKN, INOSINE_TKN, "R", "Y", "K", "M", "S", "W", "B", "D", "H", "V", ANY_NUCLEOTIDE_TKN, "-"]

CLS_TKN  = "<cls>"
PAD_TKN  = "<pad>"
BOS_TKN  = "<bos>"
EOS_TKN  = "<eos>"
UNK_TKN  = "<unk>"
MASK_TKN = "<mask>"
