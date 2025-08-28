from pathlib import Path
import glob
import re
import yaml


def increment_path(path, exist_ok=False, sep="", mkdir=False):
    # Increment file or directory path, i.e.
    # trained_models/resnet/exp{sep}1 --> trained_models/resnet/exp{sep}2 ... etc
    path = Path(path)
    if path.exists() and not exist_ok:
        path, suffix = (
            (path.with_suffix(""), path.suffix)
            if path.is_file()
            else (path, "")
        )
        dirs = glob.glob(f"{path}{sep}*")  # similar paths
        matches = [re.search(rf"%s{sep}(\d+)" % path.stem, d) for d in dirs]
        i = [int(m.groups()[0]) for m in matches if m]  # indices
        n = max(i) + 1 if i else 2
        path = Path(f"{path}{sep}{n}{suffix}")  # increment path
    if mkdir:
        path.mkdir(parents=True, exist_ok=True)  # make directory
    return path


def export(JOBID, path="./history/information.yaml", **kwargs):
    args_dict = {f"JOBID_{JOBID}": JOBID}
    args_dict[f"JOBID_{JOBID}"] = {"JOBID": JOBID}
    for key, value in kwargs.items():
        args_dict[f"JOBID_{JOBID}"].update({key: value})

    dump = yaml.dump(args_dict, indent=4)

    # if os.path.isfile(path):
    with open(path, "a") as f:
        f.write(dump)
        f.write("\n")
