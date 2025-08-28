import yaml
from yaml.loader import SafeLoader

path = "training_information.yaml"

with open(path) as f:
    yaml_data = yaml.load(f, Loader=SafeLoader)

with open(path, "r") as f:
    data = f.read()


def replacer(data, pattern):
    counter = 0
    idx = 0
    while True:
        file_size = len(data)
        if pattern in data[idx : idx + len(pattern)]:
            replaced_pattern = pattern + "_" + str(counter)
            data = data[0:idx] + replaced_pattern + data[idx + len(pattern) :]
            counter += 1
        idx += 1
        if idx == file_size:
            break
    return data


for header in yaml_data:
    data = replacer(data, header)

with open("training_information_seperated.yaml", "w") as f:
    f.write(data)
