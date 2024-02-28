import numpy as np
from sklearn.metrics import precision_score, recall_score

from pathlib import Path

def _read_relevant_lines(file_path: Path):
    with open(file_path, 'r') as f:
        lines = f.readlines()

    lines = list(filter(lambda line: not line.lstrip().startswith("#"), lines)) # Ignore comment lines
    return lines

def parse_bpseq_file(bpseq_file_path: Path):
    lines = _read_relevant_lines(bpseq_file_path)

    seq_len = len(lines)
    seq = ''
    pair_mat = np.zeros((seq_len, seq_len), dtype=np.float32)

    for line in lines:
        res_idx, res_tkn, pair_idx = line.rstrip().split()
        seq += res_tkn

        if pair_idx != '0':
            pair_mat[int(res_idx) - 1, int(pair_idx) - 1] = 1.0

    return seq, pair_mat

def parse_ct_file(ct_file_path: Path):
    lines = _read_relevant_lines(ct_file_path)

    seq_len = int(lines[0].split()[0])
    pair_mat = np.zeros((seq_len, seq_len))
    seq = ""

    for line in lines[1:]:
        i, nuc, *_, j, _ = line.split()
        i, j = int(i), int(j)

        if j > 0:
            pair_mat[i - 1, j - 1] = 1.0
        seq += nuc

    return seq, pair_mat

def save_to_ct(ct_file_path: Path, sec_struct: np.array, seq: str):
    with open(ct_file_path, "w") as f:
        f.write(f"{len(seq)}\t{ct_file_path.stem}\n")

        for i in range(len(seq)):
            f.write(f"{i + 1}\t") # Base number: Index n
            f.write(f"{seq[i]}\t") # Base
            f.write(f"{i}\t") # Index n - 1
            f.write(f"{i + 2}\t") # Index n + 1
            f.write(f"{sec_struct[i].argmax() + 1 if sec_struct[i].sum() > 0.0 else 0}\t") # Pair index
            f.write(f"{i + 1}\n") # Natural numbering

ST_SEQ_LINE_IDX = 0
ST_DBN_LINE_IDX = 1
def parse_st_file(st_file_path: Path):
    lines = _read_relevant_lines(st_file_path)

    seq = lines[ST_SEQ_LINE_IDX].rstrip()
    db_notation = lines[ST_DBN_LINE_IDX].rstrip()
    pair_mat = dot_bracket_to_2d_mat(db_notation)

    return seq, pair_mat

def parse_sec_struct_file(sec_struct_file_path: Path):
    if sec_struct_file_path.suffix == ".ct":
        return parse_ct_file(sec_struct_file_path)
    elif sec_struct_file_path.suffix == ".bpseq":
        return parse_bpseq_file(sec_struct_file_path)
    elif sec_struct_file_path.suffix == ".st":
        return parse_st_file(sec_struct_file_path)
    else:
        raise NotImplementedError("Given secondary structure file type is not supported!")

def dot_bracket_to_2d_mat(db_notation: str):
    seq_len = len(db_notation)
    pair_mat = np.zeros((seq_len, seq_len))

    # Initialize bracket stacks
    stacks= {}
    stacks["("] = stacks[")"] = []
    stacks["["] = stacks["]"] = []
    stacks["{"] = stacks["}"] = []
    stacks["<"] = stacks[">"] = []

    # Iterate through the dot-bracket notation and fill the 2D matrix accordingly
    for i in range(seq_len):
        current_tkn = db_notation[i]

        if current_tkn in ("(", "[", "{", "<"):
            stacks[current_tkn].append(i)
        elif current_tkn in (")", "]", "}", ">"):
            j = stacks[current_tkn].pop()
            pair_mat[i, j] = 1.0
        elif db_notation[i] == ".":
            pass
        else:
            raise RuntimeError(f"Encountered unexpected symbol in dot-bracket notation string! (index {i}: '{db_notation[i]}')")

    # Symmetrize pairing matrix
    pair_mat = pair_mat + pair_mat.transpose()
    pair_mat = np.minimum(pair_mat, 1.0)

    return pair_mat

_SHARP_LOOP_DIST_THRESHOLD = 4
def _generate_sharp_loop_mask(seq_len):
    mask = np.eye(seq_len, k=0, dtype=bool)
    for i in range(1, _SHARP_LOOP_DIST_THRESHOLD):
        mask = mask + np.eye(seq_len, k=i, dtype=bool) + np.eye(seq_len, k=-i, dtype=bool)

    return mask

CANONICAL_PAIRS = ['AU', 'UA', 'GC', 'CG', 'GU', 'UG']
def _generate_canonical_pairs_mask(seq: str):
    seq = seq.replace('T', 'U')

    mask = np.zeros((len(seq), len(seq)), dtype=bool)
    for i, nt_i in enumerate(seq):
        for j, nt_j in enumerate(seq):
            if f'{nt_i}{nt_j}' in CANONICAL_PAIRS:
                mask[i, j] = True

    return mask

def _clean_sec_struct(sec_struct: np.array, probs: np.array):
    clean_sec_struct = np.copy(sec_struct)
    tmp_probs = np.copy(probs)
    tmp_probs[sec_struct < 1] = 0.0

    while np.sum(tmp_probs > 0.0) > 0:
        i, j = np.unravel_index(np.argmax(tmp_probs, axis=None), tmp_probs.shape)

        tmp_probs[i, :] = tmp_probs[j, :] = 0.0
        clean_sec_struct[i, :] = clean_sec_struct[j, :] = 0

        tmp_probs[:, i] = tmp_probs[:, j] = 0.0
        clean_sec_struct[:, i] = clean_sec_struct[:, j] = 0

        clean_sec_struct[i, j] = clean_sec_struct[j, i] = 1

    return clean_sec_struct

def prob_mat_to_sec_struct(probs: np.array, seq: str, threshold: float = 0.5, allow_nc_pairs: bool = False, allow_sharp_loops: bool = False):
    assert np.all(np.isclose(probs, np.transpose(probs))), "Probability matrix must be symmetric!"
    seq_len = probs.shape[-1]

    allowed_pairs_mask = np.logical_not(np.eye(seq_len, dtype=bool))

    if not allow_sharp_loops:  
        # Prevent pairings that would cause sharp loops
        allowed_pairs_mask = np.logical_and(allowed_pairs_mask, ~_generate_sharp_loop_mask(seq_len))

    if not allow_nc_pairs:
        # Prevent non-canonical pairings
        allowed_pairs_mask = np.logical_and(allowed_pairs_mask, _generate_canonical_pairs_mask(seq))

    probs[~allowed_pairs_mask] = 0.0

    sec_struct = np.greater(probs, threshold).astype(int)
    sec_struct = _clean_sec_struct(sec_struct, probs)

    return sec_struct

def _relax_ss(ss_mat: np.array) -> np.array:
    # Pad secondary structure (because of cyclical rolling)
    ss_mat = np.pad(ss_mat, ((1, 1), (1, 1)), mode='constant')

    # Create relaxed pairs matrix
    relax_pairs = \
            np.roll(ss_mat, shift=1, axis=-1) + np.roll(ss_mat, shift=-1, axis=-1) +\
            np.roll(ss_mat, shift=1, axis=-2) + np.roll(ss_mat, shift=-1, axis=-2)

    # Add relaxed pairs into original matrix
    relaxed_ss = ss_mat + relax_pairs

    # Ignore cyclical shift and clip values
    relaxed_ss = relaxed_ss[..., 1: -1, 1: -1]
    relaxed_ss = np.clip(relaxed_ss, 0, 1)

    return relaxed_ss

def ss_recall(target_ss: np.array, pred_ss: np.array, allow_flexible_pairings: bool = True) -> float:
    if allow_flexible_pairings:
        pred_ss = _relax_ss(pred_ss)
    
    seq_len = target_ss.shape[-1]
    upper_tri_idcs = np.triu_indices(seq_len, k=1)

    return recall_score(target_ss[upper_tri_idcs], pred_ss[upper_tri_idcs], zero_division=0.0)

def ss_precision(target_ss: np.array, pred_ss: np.array, allow_flexible_pairings: bool = True) -> float:
    if allow_flexible_pairings:
        target_ss = _relax_ss(target_ss)
    
    seq_len = target_ss.shape[-1]
    upper_tri_idcs = np.triu_indices(seq_len, k=1)

    return precision_score(target_ss[upper_tri_idcs], pred_ss[upper_tri_idcs], zero_division=0.0)

EPSILON = 1e-5
def ss_f1(target_ss: np.array, pred_ss: np.array, allow_flexible_pairings: bool = True) -> float:
    precision = ss_precision(target_ss, pred_ss, allow_flexible_pairings=allow_flexible_pairings)
    recall = ss_recall(target_ss, pred_ss, allow_flexible_pairings=allow_flexible_pairings)

    # Prevent division with 0.0
    if precision + recall < EPSILON:
        return 0.0

    return (2 * precision * recall) / (precision + recall)
