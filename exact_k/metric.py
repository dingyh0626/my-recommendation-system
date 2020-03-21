def precision_at_k(card_infer, item_pos):
    res = 0
    for i in range(len(card_infer)):
        if item_pos[i] in set(card_infer[i]):
            res += 1
    return res / len(card_infer)


def hit_ratio_at_k(card_infer, card):
    res = 0
    for i in range(len(card_infer)):
        tmp = 0
        for x in card_infer[i]:
            if x in set(card[i]):
                tmp += 1
        res += (tmp / (1.0 * len(card_infer[i])))
    return res / len(card_infer)