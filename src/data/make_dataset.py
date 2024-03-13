import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import os, csv 
from diff_processing import get_diff_files_frag


def write_to_csv(data, out_path):
    try:
        with open(out_path, "w") as csvfile:
            fieldnames = ["dataset", "tool", "buggy", "patch", "label"]
            writer = csv.DictWriter(csvfile, fieldnames= fieldnames)

            writer.writeheader()
            writer.writerows(data)
    except Exception as e:
        logger.error("Exception", exc_info=True)

def main(mainpath):
    logger = logging.getLogger(__name__)

    if os.path.basename(mainpath) == "Large": # if large then we perform on operation
        logger.info('Processing: Large patch dataset')
        data = []
        for root, dirname, filenames in os.walk(mainpath):
            for filename in filenames:
                try:
                    components = root.split("/")
                    if "correct" in components:
                        correct_idx = components.index("correct")
                        label = "correct"
                        tool = components[correct_idx + 1]
                        dataset = components[correct_idx + 2]
                        buggy_component = get_diff_files_frag(os.path.join(root, filename), type="buggy")
                        patch_compnent = get_diff_files_frag(os.path.join(root, filename), type="patched")

                    elif "overfitting" in components:
                        overfitting_idx = components.index("overfitting")
                        label = "overfitting"
                        tool = components[overfitting_idx + 1]
                        dataset = components[overfitting_idx + 2]
                        buggy_component = get_diff_files_frag(os.path.join(root, filename), type="buggy")
                        patch_compnent = get_diff_files_frag(os.path.join(root, filename), type="patched")

                    else: # if "overfitting" or "correct" dir not found in path then we skip
                        logger.warn(f"Path with no 'overfitting' or 'correct' dir: {root}")
                    
                    data.append({
                        "dataset": dataset,
                        "tool": tool,
                        "buggy": buggy_component,
                        "patch": patch_compnent,
                        "label": label
                    })
                except Exception as e:
                    logger.error("Exception", exc_info=True)
                    continue
        
        return data


if __name__ == '__main__':
    load_dotenv(find_dotenv())
    project_root = os.environ.get("PROJECT_ROOT")
    data_path = os.path.join(project_root, os.environ.get("DATA_DIR"))
    output_path = os.path.join(data_path, "output")

    log_write_file = "write_data_to_csv.log"
    open(log_write_file, "w").close()
    log_writefile_path = os.path.join(project_root, log_write_file)
    
    logger = logging.getLogger(__name__)
    log_fmt = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_fmt); console_handler.setLevel(logging.INFO)

    file_handler = logging.FileHandler(log_write_file)
    file_handler.setFormatter(log_fmt); file_handler.setLevel(logging.ERROR)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    load_dotenv(find_dotenv())

    small_subset_datapath =  os.path.join(data_path, "all_patches", "Small")
    large_subset_datapath = os.path.join(data_path, "all_patches", "Large")

    data = main(large_subset_datapath)
    csv_file_path = os.path.join(output_path, "large-patches.csv")
    open(os.path.join(output_path, "large-patches.csv"), "w").close()
    write_to_csv(data, csv_file_path)