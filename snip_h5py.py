import h5py
import numpy as np
from collections import defaultdict


def save_nested_dictionary_to_hdf(dictionary, group):
    for key, value in dictionary.items():
        if isinstance(value, dict):
            subgroup = group.create_group(key)
            save_nested_dictionary_to_hdf(value, subgroup)
        else:
            group.create_dataset(key, data=value)


def save_dictionary_to_hdf(dictionary, filename):
    with h5py.File(filename, "w") as f:
        save_nested_dictionary_to_hdf(dictionary, f)


def load_nested_dictionary_from_hdf(group):
    dictionary = {}
    for key, item in group.items():
        if isinstance(item, h5py.Dataset):
            dictionary[key] = item[...]
        elif isinstance(item, h5py.Group):
            dictionary[key] = load_nested_dictionary_from_hdf(item)
    return dictionary


def load_dictionary_from_hdf(filename):
    with h5py.File(filename, "r") as f:
        return load_nested_dictionary_from_hdf(f)


# Example usage:
if __name__ == "__main__":
    toto = np.array([1, 2, 3])
    tata = np.array([4, 5, 6])
    new_dictionary = defaultdict(dict)

    new_dictionary["Camera_01"]["Point_01"] = toto
    new_dictionary["Camera_01"]["Point_02"] = tata
    new_dictionary["Camera_02"]["Point_01"] = toto
    new_dictionary["Camera_02"]["Point_02"] = tata

    filename = "test.hdf5"
    save_dictionary_to_hdf(new_dictionary, filename)

    loaded_dict = load_dictionary_from_hdf(filename)
    print(loaded_dict)
    print(loaded_dict["Camera_01"]["Point_01"])
