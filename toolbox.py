import copy


def grid_expand(conditions):
    expanded = [{}]
    for key, value_list in conditions.items():
        new_expanded = []
        for args_dict in expanded:
            for value in value_list:
                new_args_dict = copy.copy(args_dict)
                new_args_dict[key] = value
                new_expanded.append(new_args_dict)
        expanded = new_expanded
    return expanded
