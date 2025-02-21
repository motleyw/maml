import os
import glob
import random
from typing import List, Dict


def get_csv_files(root_location: str, subject_ids: List[str], keywords=['walk'], specific_files=None,
                  shuffle=True) -> List[str]:
    """
    Fetch specific CSV files matching the given subject IDs and keywords, with an option to filter by exact filenames.

    Parameters:
    - root_location (str): Root directory where CSV files are located.
    - subject_ids (List[str]): List of subject identifiers to filter files.
    - keywords (List[str]): List of keywords that should be present in the filename.
    - specific_files (List[str], optional): List of exact filenames to include.
    - shuffle (bool): Whether to shuffle the result.

    Returns:
    - List[str]: List of matching file paths.
    """
    search_pattern = os.path.join(root_location, '**', '*.csv')
    files = glob.glob(search_pattern, recursive=True)

    if not files:
        print(f"[WARNING] No CSV files found in: {root_location}")

    csv_files = []
    for f in files:
        basename = os.path.basename(f)

        # Check if specific file filtering is enabled
        if specific_files:
            if basename in specific_files:
                csv_files.append(f)
        else:
            # Standard filtering by subject ID and keyword
            if any(f'_{subject_id}_' in basename for subject_id in subject_ids):
                if any(f'_{kw}_' in basename for kw in keywords):
                    csv_files.append(f)

    if shuffle:
        random.shuffle(csv_files)

    return csv_files


def get_diff_csv_files(root_location: str, subject_ids: List[str], keywords=['walk','SD','SA'], specific_files_dict: Dict[str, List[str]] = None, shuffle=True) -> List[str]:
    """
    Fetch CSV files for given subject IDs and keywords. Allows specifying exact files for certain subjects.

    Parameters:
    - root_location (str): Root directory where CSV files are located.
    - subject_ids (List[str]): List of subject identifiers to filter files.
    - keywords (List[str]): Keywords that should be present in the filename.
    - specific_files_dict (Dict[str, List[str]], optional): A dictionary where keys are subject IDs and values are specific filenames.
    - shuffle (bool): Whether to shuffle the result.

    Returns:
    - List[str]: List of matching file paths.
    """
    search_pattern = os.path.join(root_location, '**', '*.csv')
    files = glob.glob(search_pattern, recursive=True)

    if not files:
        print(f"[WARNING] No CSV files found in: {root_location}")

    csv_files = []
    subject_files_map = {subject_id: [] for subject_id in subject_ids}  # Store subject-specific files

    for f in files:
        basename = os.path.basename(f)
        matched_subject = next((subject_id for subject_id in subject_ids if f'_{subject_id}_' in basename), None)

        if matched_subject:
            # Store all files for the subject
            subject_files_map[matched_subject].append(f)

            # If specific files are defined for this subject, only allow those
            if specific_files_dict and matched_subject in specific_files_dict:
                if basename in specific_files_dict[matched_subject]:
                    csv_files.append(f)
            else:
                # Otherwise, apply the normal keyword filter
                if any(f'_{kw}_' in basename for kw in keywords):
                    csv_files.append(f)

    # Handle subjects that do not have keyword-matching files
    for subject_id, files in subject_files_map.items():
        if subject_id in subject_ids and subject_id not in (specific_files_dict or {}):  # Skip specific-file subjects
            subject_has_keyword_files = any(f'_{kw}_' in os.path.basename(f) for f in files for kw in keywords)

            # If no keyword files found, include all subject files
            if not subject_has_keyword_files:
                print(f"[INFO] No '{keywords}' files found for subject {subject_id}, including all files.")
                csv_files.extend(files)

    if shuffle:
        random.shuffle(csv_files)

    return csv_files