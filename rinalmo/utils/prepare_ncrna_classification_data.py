from rpy2.robjects import r
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter
import rpy2.robjects.packages as rpackages
from rpy2.robjects.vectors import StrVector

import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score, balanced_accuracy_score, confusion_matrix
import matplotlib.pyplot as plt

from Bio.SeqRecord import SeqRecord
from Bio.Seq import Seq
from Bio import SeqIO
import random

from rinalmo.utils.ncrna_classification.seqEncoders import *
from rinalmo.utils.ncrna_classification.ExpConfiguration import *

def is_canonical_sequence(sequence, canonical_bases):
    """Check if all characters in the sequence are canonical."""
    return set(sequence).issubset(canonical_bases)

def prepare_ncrna_classification_data(data_path):
    # Import Bioconductor installer
    utils = rpackages.importr('utils')
    utils.chooseCRANmirror(ind=1)  # Select a CRAN mirror

    # Install Bioconductor's BiocManager if not installed
    if not rpackages.isinstalled("BiocManager"):
        utils.install_packages(StrVector(["BiocManager"]))

    # Use BiocManager to install Biostrings
    r('''
    if (!requireNamespace("Biostrings", quietly = TRUE)) {
        BiocManager::install("Biostrings")
    }
    ''')

    # file name
    rda_file = str(data_path) + "/train_test_val_sets_177_families.RDa"
    dataf = r(f'load("{rda_file}", verbose = T)')

    # Load sequence data
    x_train = r['x_train'] # Populate with training sequences
    x_val = r['x_val'] # Populate with validation sequences
    x_test = r['x_test'] # Populate with testing sequences

    # Convert data to pandas DataFrame
    r('''
    library(Biostrings)
    x_train_df <- data.frame(
        name = names(x_train),
        width = width(x_train),
        sequence = as.character(x_train)
    )
    ''')
    r('''
    library(Biostrings)
    x_val_df <- data.frame(
        name = names(x_val),
        width = width(x_val),
        sequence = as.character(x_val)
    )
    ''')
    r('''
    library(Biostrings)
    x_test_df <- data.frame(
        name = names(x_test),
        width = width(x_test),
        sequence = as.character(x_test)
    )
    ''')
    x_train_df_r = r['x_train_df']
    x_val_df_r = r['x_val_df']
    x_test_df_r = r['x_test_df']

    with localconverter(pandas2ri.converter):
        x_train_df = pandas2ri.rpy2py(x_train_df_r)
        x_val_df = pandas2ri.rpy2py(x_val_df_r)
        x_test_df = pandas2ri.rpy2py(x_test_df_r)

    # Print lengths of the datasets
    print(f"Train: {len(x_train_df)}")
    print(f"Val: {len(x_val_df)}")
    print(f"Test: {len(x_test_df)}")
    print(f"Total: {len(x_train_df) + len(x_val_df) + len(x_test_df)}")

    print("Keep only common classes among sets")

    common_classes = set(x_test_df['name']).intersection(x_train_df['name'])
    common_classes = common_classes.intersection(x_val_df['name'])

    x_test_df = x_test_df[x_test_df['name'].isin(common_classes)]
    x_train_df = x_train_df[x_train_df['name'].isin(common_classes)]
    x_val_df = x_val_df[x_val_df['name'].isin(common_classes)]

    initial_total = len(x_train_df) + len(x_val_df) + len(x_test_df)

    print(f"Train: {len(x_train_df)}")
    print(f"Val: {len(x_val_df)}")
    print(f"Test: {len(x_test_df)}")
    print(f"Total: {initial_total}")
    print(f"Total classes: {len(common_classes)}")

    # Remove sequences with letters different from canonical A, T, C, and G
    print("Remove sequences with letters different from canonical A, T, C, and G")

    canonical_bases = set("ACTG")

    x_train_df = x_train_df[x_train_df['sequence'].apply(lambda seq: is_canonical_sequence(seq, canonical_bases))]
    x_val_df = x_val_df[x_val_df['sequence'].apply(lambda seq: is_canonical_sequence(seq, canonical_bases))]
    x_test_df = x_test_df[x_test_df['sequence'].apply(lambda seq: is_canonical_sequence(seq, canonical_bases))]

    # Final statistics
    print(f"Final Train: {len(x_train_df)}")
    print(f"Final Val: {len(x_val_df)}")
    print(f"Final Test: {len(x_test_df)}")
    initialtot = len(x_train_df) + len(x_val_df) + len(x_test_df)
    print(f"Final Total: {initialtot}")

    # test whether seq length predict seq family
    # Prepare the data
    w = pd.concat([x_train_df['width'], x_val_df['width'], x_test_df['width']])
    q = pd.concat([x_train_df['name'], x_val_df['name'], x_test_df['name']])
    sdata = pd.DataFrame({'length': w, 'family': q.astype('category')})

    # Classification setup
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    acc = []
    kappa = []
    f1_by_class = []
    bacc_by_class = []

    # Cross-validation loop
    for train_index, test_index in skf.split(sdata['length'], sdata['family']):
        dtrain = sdata.iloc[train_index]
        dtest = sdata.iloc[test_index]
        
        clf = DecisionTreeClassifier(random_state=42, max_depth=10)
        clf.fit(dtrain[['length']], dtrain['family'])
        
        predictions = clf.predict(dtest[['length']])
        
        # Calculate metrics
        acc.append(accuracy_score(dtest['family'], predictions))
        kappa.append(cohen_kappa_score(dtest['family'], predictions))
        
        # Confusion matrix for per-class metrics
        cm = confusion_matrix(dtest['family'], predictions, labels=dtest['family'].cat.categories)
        f1_scores = f1_score(dtest['family'], predictions, average=None)
        bacc_scores = balanced_accuracy_score(dtest['family'], predictions)
        
        f1_by_class.append(f1_scores)
        bacc_by_class.append(bacc_scores)

    # Overall metrics
    mean_acc = np.mean(acc)
    error_acc = 1.96 * np.std(acc) / np.sqrt(len(acc))
    mean_kappa = np.mean(kappa)
    error_kappa = 1.96 * np.std(kappa) / np.sqrt(len(kappa))

    print(f"Overall accuracy: {mean_acc:.4f} ± {error_acc:.4f}")
    print(f"Overall kappa: {mean_kappa:.4f} ± {error_kappa:.4f}")

    # Mean and range for per-class metrics
    f1_by_class = np.array(f1_by_class)
    mean_f1_by_class = np.nanmean(f1_by_class, axis=0)
    range_f1_by_class = (np.nanmin(mean_f1_by_class), np.nanmax(mean_f1_by_class))

    bacc_by_class = np.array(bacc_by_class)
    mean_bacc_by_class = np.nanmean(bacc_by_class, axis=0)
    range_bacc_by_class = (np.nanmin(mean_bacc_by_class), np.nanmax(mean_bacc_by_class))

    # Fake classes detection
    fake_classes = np.array(sdata['family'].cat.categories)[mean_f1_by_class > 0.8]

    # Remove fake classes from data
    x_test_df = x_test_df[~x_test_df['name'].isin(fake_classes)]
    x_train_df = x_train_df[~x_train_df['name'].isin(fake_classes)]
    x_val_df = x_val_df[~x_val_df['name'].isin(fake_classes)]

    # Summary of remaining sequences
    print(f"Train: {len(x_train_df)}")
    print(f"Val: {len(x_val_df)}")
    print(f"Test: {len(x_test_df)}")
    remaining_total = len(x_train_df) + len(x_val_df) + len(x_test_df)
    print(f"Total sequences: {remaining_total}")
    print(f"Removed %: {(1 - (remaining_total / initialtot))*100:.4f}")
    print(f"Total classes: {len(pd.concat([x_train_df['name'], x_val_df['name'], x_test_df['name']]).unique())}")

    # Remove sequences with length greater than 200 to exclude long non-coding RNA
    print('Remove sequences with length greater than 200')
    minlen = 0
    maxlen = 200

    # Filter sequences based on length
    x_train_df = x_train_df[(x_train_df['width'] <= maxlen) & (x_train_df['width'] >= minlen)]
    x_val_df = x_val_df[(x_val_df['width'] <= maxlen) & (x_val_df['width'] >= minlen)]
    x_test_df = x_test_df[(x_test_df['width'] <= maxlen) & (x_test_df['width'] >= minlen)]

    # Print the updated counts
    print(f"Train: {len(x_train_df)}")
    print(f"Val: {len(x_val_df)}")
    print(f"Test: {len(x_test_df)}")
    total_sequences = len(x_train_df) + len(x_val_df) + len(x_test_df)
    print(f"Total sequences: {total_sequences}")
    removed_percentage = 1 - (total_sequences / initialtot)
    print(f"Removed %: {removed_percentage * 100:.4f}")
    print(f"Total classes: {len(pd.concat([x_train_df['name'], x_val_df['name'], x_test_df['name']]).unique())}")

    # Exclude classes with fewer than 400 samples
    print('Keep classes with 400 or more samples')
    class_counts = x_train_df['name'].value_counts()

    # Keep classes with 400 or more samples
    classes_to_keep = class_counts[class_counts >= 400].index

    # Print the number of classes to keep
    print(f"Number of classes to keep: {len(classes_to_keep)}")

    # Filter the dataframes to keep only the classes with >= 400 samples
    x_train_df = x_train_df[x_train_df['name'].isin(classes_to_keep)]
    x_val_df = x_val_df[x_val_df['name'].isin(classes_to_keep)]
    x_test_df = x_test_df[x_test_df['name'].isin(classes_to_keep)]

    # Print the updated counts
    print(f"Train: {len(x_train_df)}")
    print(f"Val: {len(x_val_df)}")
    print(f"Test: {len(x_test_df)}")
    total_sequences = len(x_train_df) + len(x_val_df) + len(x_test_df)
    print(f"Total sequences: {total_sequences}")
    removed_percentage = 1 - (total_sequences / initialtot)
    print(f"Removed %: {removed_percentage * 100:.4f}")
    print(f"Total classes: {len(pd.concat([x_train_df['name'], x_val_df['name'], x_test_df['name']]).unique())}")

    # Balancing of training/validation sets
    print("Balancing of training/validation sets")
    minsize = min(x_train_df['name'].value_counts())
    minsize = minsize * 3

    # Initialize new balanced train and validation datasets
    x_train_new = []
    x_val_new = []
    x_test_new = []

    # Initialize new balanced train and validation dataframes and add new columns
    x_train_df_new = pd.DataFrame(columns=x_train_df.columns.tolist() + ['class_id'])
    x_val_df_new = pd.DataFrame(columns=x_val_df.columns.tolist() + ['class_id'])
    x_test_df_new = pd.DataFrame(columns=x_test_df.columns.tolist() + ['class_id'])

    # Balance training and validation sets by sampling
    for idx, class_name in enumerate(classes_to_keep):
        # Training set
        train_subset = x_train_df[x_train_df['name'] == class_name]
        train_subset['class_id'] = idx
        sampled_train = train_subset.sample(n=minsize, replace=True, random_state=42)
        # Append sampled_train to x_train_df_new
        x_train_df_new = pd.concat([x_train_df_new, sampled_train])
        x_train_new.extend([
            SeqRecord(Seq(row['sequence']), id=row['name'], description="") for _, row in sampled_train.iterrows()
        ])
        
        # Validation set
        val_subset = x_val_df[x_val_df['name'] == class_name]
        val_subset['class_id'] = idx
        if len(val_subset) >= minsize:
            sampled_val = val_subset.sample(n=minsize, replace=False, random_state=42)
        else:
            sampled_val = val_subset
        x_val_df_new = pd.concat([x_val_df_new, sampled_val])
        x_val_new.extend([
            SeqRecord(Seq(row['sequence']), id=row['name'], description="") for _, row in sampled_val.iterrows()
        ])

        # Test set
        test_subset = x_test_df[x_test_df['name'] == class_name]
        test_subset['class_id'] = idx
        x_test_df_new = pd.concat([x_test_df_new, test_subset])
        x_test_new.extend([
            SeqRecord(Seq(row['sequence']), id=row['name'], description="") for _, row in test_subset.iterrows()
        ])

    # Print results
    print(f"Train: {len(x_train_new)}")
    print(f"Val: {len(x_val_new)}")
    print(f"Test: {len(x_test_new)}")

    # Calculate proportions
    ttot = len(x_train_df) + len(x_test_df) + len(x_val_df)
    print(f"{len(x_train_df) / ttot:.4f}, {len(x_test_df) / ttot:.4f}, {len(x_val_df) / ttot:.4f}")

    # Save to FASTA files
    SeqIO.write(x_train_new, str(data_path) + "/train.fasta", "fasta")
    SeqIO.write(x_val_new, str(data_path) + "/val.fasta", "fasta")
    SeqIO.write(x_test_new, str(data_path) + "/test.fasta", "fasta")

    # Save datasets to CSV files
    x_train_df_new.to_csv(str(data_path) + "/train.csv", index=False)
    x_val_df_new.to_csv(str(data_path) + "/val.csv", index=False)
    x_test_df_new.to_csv(str(data_path) + "/test.csv", index=False)


def add_noise_to_ncrna_data(data_path):
    # Load CSV files with sequences and labels
    train = pd.read_csv(str(data_path) + '/train.csv')
    val = pd.read_csv(str(data_path) + '/val.csv')
    test = pd.read_csv(str(data_path) + '/test.csv')

    fastaTrain = str(data_path) + '/train.fasta'
    fastaVal = str(data_path) + '/val.fasta'
    fastaTest = str(data_path) + '/test.fasta'

    # go through all seqs in csv files and replace them with newly generated seqs
    for bn in bnoise:
        print("Noise = ", str(bn))
        seqTrain = get_seqs_with_bnoise(fastaTrain, nperc=bn)
        seqVal = get_seqs_with_bnoise(fastaVal, nperc=bn)
        seqTest = get_seqs_with_bnoise(fastaTest, nperc=bn)
        
        seqTrain = [str(s) for s in seqTrain]
        seqTrainWidth = [len(s) for s in seqTrain]
        train['sequence'] = seqTrain
        train['width'] = seqTrainWidth
        train.to_csv(str(data_path) + '/train_bn'+str(bn)+'.csv', index=False)

        seqVal = [str(s) for s in seqVal]
        seqValWidth = [len(s) for s in seqVal]
        val['sequence'] = seqVal
        val['width'] = seqValWidth
        val.to_csv(str(data_path) + '/val_bn'+str(bn)+'.csv', index=False)
        
        seqTest = [str(s) for s in seqTest]
        seqTestWidth = [len(s) for s in seqTest]
        test['sequence'] = seqTest
        test['width'] = seqTestWidth
        test.to_csv(str(data_path) + '/test_bn'+str(bn)+'.csv', index=False)
