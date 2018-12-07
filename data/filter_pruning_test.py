def _get_l1_norm(layer):
    pass


pruning = []
while True:

    # *load model with weights *

    ranking = []
    for i in range(len(model.layers)):
        layer = model.layers[i]
        l1_norm = _get_l1_norm(layer)
        ranking.append({'layer_idx': i, 'l1_norm': l1_norm})  # ascending -> first n will have the lowest l1 norms

    from operator import itemgetter

    ranking = sorted(ranking, key=itemgetter('l1_norm'))
    ranking = [layer['layer_idx'] for layer in ranking]

    ranking_no_repeat = [x for x in ranking if x not in already_pruned]

    lowest_n = ranking_no_repeat[:n]
    pruning.expand(lowest_n)

    # *load model with pruning=pruning *

    # *training for n epochs + early stopping *

    # *evaluation *

    if accuracy < accuracy_thr or recall < recall_thr:
        break

    # *save weights + h5 filename and pruning array into file *