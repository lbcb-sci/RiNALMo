from tqdm import tqdm, TqdmWarning
import ftplib

from pathlib import Path
import warnings
import itertools

import gzip
import shutil
import requests
from zipfile import ZipFile
import os
import tarfile
import json

# Ignore tqdm's clamping warnings
warnings.filterwarnings("ignore", category=TqdmWarning)


ONLINE_RESOURCES_CONFIG_PATH = Path(__file__).parent.parent / "resources" / "remote_data.json"
with open(ONLINE_RESOURCES_CONFIG_PATH, "r") as f:
    remote_data = json.load(f)

def _extract_gz(gz_file_path, default_extension="fasta", delete_archive=True):
    assert gz_file_path.suffix == ".gz"
    target_path = gz_file_path.with_suffix('') # Remove '.gz' suffix
    if target_path.suffix == '' and default_extension:
        target_path = target_path.with_suffix(f".{default_extension}") # Add default suffix in case there is no suffix

    if target_path.is_file():
        user_input = input(f"File '{target_path}' already exists. Are you sure you want to overwrite it? (y/n) : ")
        if user_input != 'y':
            return

    with gzip.open(gz_file_path, 'rb') as f_in, open(target_path, 'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)

    if delete_archive:
        gz_file_path.unlink()

def _extract_archives(paths, delete_archives=True):
    for p in paths:
        if p.suffix == ".gz":
            _extract_gz(p, delete_archive=delete_archives)

MEGABYTE = 1_000_000

def _write_and_update_progress_bar(file, progress, chunk):
    progress.update(len(chunk) / MEGABYTE)
    file.write(chunk)

def _get_download(url: str, local_dir_path: Path) -> Path:
    filename = url.split("/")[-1]
    local_file_path = local_dir_path / filename

    r = requests.get(url, stream=True)

    with open(local_file_path , 'wb') as f:
        with tqdm(total=int(r.headers['Content-Length']) / MEGABYTE, unit="MB") as progress_bar:
            for chunk in r.iter_content(chunk_size=1024):
                _write_and_update_progress_bar(f, progress_bar, chunk)

    return local_file_path

def _ftp_download(ftp_url, remote_paths, local_download_dir_path, extract_archives=False):
    local_download_dir_path.mkdir(parents=True, exist_ok=True)

    def _ftp_get_size_bytes(ftp, remote_paths):
        download_sizes_bytes = []
        for remote_path in remote_paths:
            download_sizes_bytes.append(ftp.size(remote_path))

        return sum(download_sizes_bytes)
    
    downloaded_file_paths = []
    with ftplib.FTP(ftp_url) as ftp:
        ftp.login()

        #Switch FTP session to binary mode
        ftp.sendcmd('TYPE I')

        print("Calculating total download size...")
        total_size = _ftp_get_size_bytes(ftp, remote_paths)
        print(f"Total download size is {total_size / MEGABYTE} MBs. Starting the download...")
        with tqdm(total=_ftp_get_size_bytes(ftp, remote_paths) / MEGABYTE, unit="MB") as progress_bar:
            for remote_path in remote_paths:
                local_path = local_download_dir_path / Path(remote_path).name

                if local_path.is_file():
                    print(f"File '{local_path}' already exists. Skipping the download...")
                    continue

                downloaded_file_paths.append(local_path)
                with open(local_path, 'wb') as f:
                    ftp.retrbinary(f"RETR {remote_path}", lambda chunk: _write_and_update_progress_bar(f, progress_bar, chunk), blocksize=262144)

    if downloaded_file_paths and extract_archives:
        print("Extracting downloaded archives...")
        _extract_archives(downloaded_file_paths)

def _ftp_dir_download(
    ftp_url, remote_dir_path, local_download_dir_path,
    file_extensions=None, remote_dir_tree_depth=0,
    extract_archives=False
):
    print("Fetching files list...")
    remote_paths = []
    with ftplib.FTP(ftp_url) as ftp:
        ftp.login()
        remote_paths = ftp.nlst(remote_dir_path)

        for i in range(remote_dir_tree_depth):
            remote_paths = list(itertools.chain(*[ftp.nlst(remote_path) for remote_path in remote_paths]))

    if file_extensions:
        remote_paths = list(filter(lambda p: any(p.endswith(ext) for ext in file_extensions), remote_paths))

    _ftp_download(ftp_url, remote_paths, local_download_dir_path, extract_archives)

# ------------------------------------------------------

def download_rnacentral_dataset(local_download_dir_path, download_active_seqs=True, download_inactive_seqs=False,
                                extract_archives=False):
    remote_paths = []
    
    if download_active_seqs:
        remote_paths += [remote_data["FTP"]["PATH"]["RNACENTRAL_ACTIVE_SEQS"]]
    if download_inactive_seqs:
        remote_paths += [remote_data["FTP"]["PATH"]["RNACENTRAL_INACTIVE_SEQS"]]

    print(f"Downloading RNACentral dataset...")
    _ftp_download(
        ftp_url=remote_data["FTP"]["ADDRESS"]["EMBL_EBI"],
        remote_paths=remote_paths,
        local_download_dir_path=local_download_dir_path,
        extract_archives=extract_archives,
    )

    print("RNACentral dataset download is complete!")

def download_rfam_dataset(local_download_dir_path, extract_archives=False):
    print(f"Downloading Rfam dataset...")
    _ftp_download(
        ftp_url=remote_data["FTP"]["ADDRESS"]["EMBL_EBI"],
        remote_paths=[remote_data["FTP"]["PATH"]["RFAM_SEQS"]],
        local_download_dir_path=local_download_dir_path,
        extract_archives=extract_archives,
    )

def download_nt_dataset(local_download_dir_path, extract_archives=False):
    print(f"Downloading nt dataset...")
    _ftp_download(
        ftp_url=remote_data["FTP"]["ADDRESS"]["NCBI"],
        remote_paths=[remote_data["FTP"]["PATH"]["NT_SEQS"]],
        local_download_dir_path=local_download_dir_path,
        extract_archives=extract_archives,
    )

def download_ensembl_dataset(local_download_dir_path, extract_archives=False):
    print(f"Downloading Ensembl dataset...")
    _ftp_dir_download(
        ftp_url=remote_data["FTP"]["ADDRESS"]["ENSEMBL"],
        remote_dir_path=remote_data["FTP"]["PATH"]["ENSEMBL_SEQS_DIR"],
        file_extensions=[".ncrna.fa.gz"],
        remote_dir_tree_depth=2,
        local_download_dir_path=local_download_dir_path,
        extract_archives=extract_archives,
    )

def download_ensembl_bacteria_dataset(local_download_dir_path, extract_archives=False):
    # TODO This download is much slower than other ones because of deeper directory tree, optimize it somehow?
    print(f"Downloading Ensembl Bacteria dataset...")
    _ftp_dir_download(
        ftp_url=remote_data["FTP"]["ADDRESS"]["ENSEMBL_GENOMES"],
        remote_dir_path=remote_data["FTP"]["PATH"]["ENSEMBL_BACTERIA_SEQS_DIR"],
        file_extensions=[".ncrna.fa.gz"],
        remote_dir_tree_depth=3,
        local_download_dir_path=local_download_dir_path,
        extract_archives=extract_archives,
    )

SPOT_RNA_BPRNA_ROOT_DIR = "bpRNA_dataset"
SPOT_RNA_BPRNA_TRAIN_DIR = "TR0"
SPOT_RNA_BPRNA_VAL_DIR = "VL0"
SPOT_RNA_BPRNA_TEST_DIR = "TS0"
def download_spot_rna_bprna(local_dir_path: Path, train_dir_name: str = "train", val_dir_name: str = "valid", test_dir_name: str = "test") -> None:
    local_dir_path.mkdir(parents=True, exist_ok=True)

    print(f"Downloading SPOT-RNA's bpRNA dataset...")
    archive_path = _get_download(remote_data["GET"]["URL"]["SPOTRNA_BPRNA"], local_dir_path)

    # "Unzip" and delete the archive
    with ZipFile(archive_path, 'r') as f:
        f.extractall(local_dir_path)

    archive_path.unlink()

    # Move downloaded content out of the root directory
    root_dir_path = local_dir_path / SPOT_RNA_BPRNA_ROOT_DIR
    for data_dir_path in root_dir_path.glob("*"):
        shutil.move(data_dir_path, local_dir_path / data_dir_path.name)

    # Delete the root directory
    shutil.rmtree(root_dir_path)

    # Rename data directories
    train_dir_path = local_dir_path / SPOT_RNA_BPRNA_TRAIN_DIR
    val_dir_path = local_dir_path / SPOT_RNA_BPRNA_VAL_DIR
    test_dir_path = local_dir_path / SPOT_RNA_BPRNA_TEST_DIR

    train_dir_path.rename(local_dir_path / train_dir_name)
    val_dir_path.rename(local_dir_path / val_dir_name)
    test_dir_path.rename(local_dir_path / test_dir_name)

ARCHIVEII_SPLITS_ROOT_DIR = "ct"
ARCHIVEII_FAM_SPLITS_DIR = "fam-fold"
ARCHIVEII_TRAIN_DIR = "train"
ARCHIVEII_VAL_DIR = "valid"
ARCHIVEII_TEST_DIR = "test"
def download_archiveII_fam_splits(local_dir_path: Path, train_dir_name: str = "train", val_dir_name: str = "valid", test_dir_name: str = "test") -> None:
    local_dir_path.mkdir(parents=True, exist_ok=True)

    # Download data archive
    print(f"Downloading ArchiveII inter-family dataset splits...")
    archive_path = _get_download(remote_data["GET"]["URL"]["ARCHIVEII_SPLITS"], local_dir_path)

    # Extract from the downloaded archive
    with tarfile.open(archive_path, mode="r") as f:
        f.extractall(local_dir_path)

    # Delete the archive after extraction
    archive_path.unlink()

    # Move relevant directories to the root directory
    fam_splits_dir = local_dir_path / ARCHIVEII_SPLITS_ROOT_DIR / ARCHIVEII_FAM_SPLITS_DIR
    for split_dir_path in fam_splits_dir.glob("*"):
        shutil.move(split_dir_path, local_dir_path / split_dir_path.name)

    # Delete unnecessary files and directories
    shutil.rmtree(local_dir_path / ARCHIVEII_SPLITS_ROOT_DIR)

    # Rename data directories
    for data_split_dir_path in local_dir_path.glob("*"):
        train_dir_path = data_split_dir_path / ARCHIVEII_TRAIN_DIR
        val_dir_path = data_split_dir_path / ARCHIVEII_VAL_DIR
        test_dir_path = data_split_dir_path / ARCHIVEII_TEST_DIR

        train_dir_path.rename(data_split_dir_path / train_dir_name)
        val_dir_path.rename(data_split_dir_path / val_dir_name)
        test_dir_path.rename(data_split_dir_path / test_dir_name)

def download_kaggle_competition_data_file(competition_name: str, file_name: str, local_path: Path):
    local_path.parent.mkdir(parents=True, exist_ok=True)
    local_zip = local_path.parent / (file_name + ".zip")

    try:
        import kaggle
    except OSError:
        os.environ['KAGGLE_USERNAME'] = input("Enter Kaggle username: ")
        os.environ['KAGGLE_KEY'] = input("Enter Kaggle API key: ")
        import kaggle

    kaggle.api.competition_download_file(
        competition=competition_name,
        file_name=file_name,
        path=local_zip.parent
    )

    with ZipFile(local_zip, 'r') as f_z:
        f_z.extract(file_name, path=local_path.parent)

    (local_path.parent / file_name).rename(local_path)
    local_zip.unlink()

def download_ribosome_loading_data(local_download_dir_path: Path):
    local_download_dir_path.mkdir(parents=True, exist_ok=True)

    print(f"Downloading synthetic Human 5'UTR library...")
    archive_path = _get_download(remote_data["GET"]["URL"]["HUMAN_5UTR_LIB"], local_download_dir_path)

    with tarfile.open(archive_path, 'r') as tar:
        tar.extractall(local_download_dir_path)

    archive_path.unlink()
