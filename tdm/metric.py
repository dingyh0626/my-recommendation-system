
def recall(ground_truth, fetch):
    return len(set(ground_truth).intersection(fetch)) / len(ground_truth)


def precision(ground_truth, fetch):
    return len(set(ground_truth).intersection(fetch)) / len(fetch)


def f1(ground_truth, fetch):
    p = precision(ground_truth, fetch)
    r = recall(ground_truth, fetch)
    return 2 * p * r / (p + r + 1e-10)


def novelty(fetch, history):
    return len(set(fetch).difference(history)) / len(fetch)


if __name__ == '__main__':
    ground_truth = [0, 1, 2, 3, 4]

    history = [6, 300, 301, 203, 302]

    fetch = list(range(0, 200))

    print(precision(ground_truth, fetch))
    print(recall(ground_truth, fetch))
    print(f1(ground_truth, fetch))
    print(novelty(fetch, history))