def get_shortest_pair_paths(pairs: list[list[str]], ref: str) -> dict[str, list[list[str]]]:
    """
    For a list of pairs of "nodes" A-B and a given target node "ref" (ref should appear in at least one pair),
    returns a dictionary which for each node N in all pairs gives a chain of pairs with equal adjacent nodes which
    eventually leads from N to ref. For ref, the output will contain an empty chain.

    Example: A-B, B-C, C-X, X-ref

    :param pairs: list of pairs (Nodes are strings)
    :param ref: the reference node
    :return: dictionary with
    """
    pairs = pairs.copy()  # we remove items in this function
    paths = {
        # from: path to ref
        ref: []
    }
    found = True
    while found:
        found = False
        toremove = []
        for p0 in pairs:
            for p in [[p0[0], p0[1]], [p0[1], p0[0]]]:
                # p iterates over all pairs in both directions
                if p[0] in paths:
                    paths[p[1]] = [[p[1], p[0]]] + paths[p[0]]
                    toremove.append(p0)
                    found = True
                    break
            if found:
                break
        for t in toremove:
            pairs.remove(t)
    return paths


if __name__ == "__main__":
    test_pairs = [
        ["a", "b"],
        ["a", "c"],
        ["d", "b"],
        ["e", "c"],
        ["d", "f"],
        ["a", "f"],
        ["g", "h"]
    ]
    print(get_shortest_pair_paths(test_pairs, "b"))
