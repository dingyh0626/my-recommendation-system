import os
os.environ["CUDA_VISIBLE_DEVICES"] = '4'
from torch.optim import Adam
from torch.utils.data import DataLoader
from hyperparams import Hyperparams as hp
from data_load_ml import Dataset, load_item_vocab, load_user_vocab
from modules.pointer_rnn import PointerRNN
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm
import torch
from metric import precision_at_k, hit_ratio_at_k
import numpy as np

if __name__ == '__main__':
    user2idx, idx2user = load_user_vocab()
    item2idx, idx2item = load_item_vocab()
    train_loader = DataLoader(Dataset(), hp.batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(Dataset(False), hp.batch_size, shuffle=False, num_workers=4)
    net = PointerRNN(len(user2idx), hp.hidden_units, len(item2idx), hp.hidden_units, hp.num_heads, hp.num_layers, rnn_cell=nn.GRUCell)
    net = net.cuda()
    optimizer = Adam(net.parameters(), lr=1e-3, weight_decay=1e-6)

    net.train()

    for epoch in range(hp.gen_num_epochs):
        tbar = tqdm(train_loader)
        for user, card, card_idx, item_cand, item_pos, mask_dec in tbar:
            # user, card, card_idx, item_cand, item_pos, mask_dec = next(train_loader)
            optimizer.zero_grad()
            user = user.cuda()
            item_cand = item_cand.cuda()
            card_idx = card_idx.cuda()
            mask_dec = mask_dec.cuda()
            if hp.schedule_sampling:
                # SUPERVISE with Policy-sampling
                # with torch.autograd.no_grad():
                sampled_card_idx, logits = net.sample(user, item_cand)
                # logits = net(user, item_cand, sampled_card_idx, mask_dec=mask_dec)
                sampled_card = torch.gather(item_cand, 1, sampled_card_idx)
                sampled_card = sampled_card.data.cpu().numpy()
                item_pos = item_pos.data.cpu().numpy()
                rewards = []
                for item, card_set in zip(item_pos, sampled_card):
                    if item in set(card_idx):
                        rewards.append(1)
                    else:
                        rewards.append(-1)
                rewards = torch.from_numpy(np.array(rewards)).float().cuda()
            else:
                # Learning from Demonstrations
                logits = net(user, item_cand, card_idx, mask_dec=mask_dec)
            batch_size, seq_len, _ = logits.size()
            logits = logits.view(-1, logits.size(-1))
            card_idx = card_idx.view(-1)
            loss_sup = F.cross_entropy(logits, card_idx, reduction='none')
            loss_sup = loss_sup.view(batch_size, seq_len).sum(1)
            loss_sup = loss_sup.mean()
            if hp.schedule_sampling:
                sampled_card_idx = sampled_card_idx.view(-1)
                logits_sampled = logits.view(-1, logits.size(-1))
                loss_rl = F.cross_entropy(logits_sampled, sampled_card_idx, reduction='none')
                loss_rl = loss_rl.view(batch_size, seq_len).sum(1)
                loss_rl = loss_rl * rewards
                loss_rl = loss_rl.mean()
                loss = (1 - hp.supervised_coe) * loss_rl + hp.supervised_coe * loss_sup
                loss = loss_sup
            else:
                loss = loss_sup
            loss.backward()
            optimizer.step()
            tbar.set_description('epoch: %d, loss: %.3f' % (epoch + 1, loss.item()))
            # print(loss.item())

        tbar_test = tqdm(test_loader)
        net.eval()
        precision_at_k_total = 0
        hit_ratio_total = 0
        batches = 0
        for i, (user, card, card_idx, item_cand, item_pos, mask_dec) in enumerate(tbar_test):
            with torch.autograd.no_grad():
                batch_size = user.size(0)
                batches += batch_size
                item_cand = item_cand.cuda()
                user = user.cuda()
                item_fetch = net.beam_search(user, item_cand, beam=hp.beam_size, depth=hp.res_length)
                item_fetch = item_fetch[:, 0].data.cpu().numpy()
                item_pos = item_pos.data.numpy()
                card = card.data.numpy()
                precision_at_k_total += precision_at_k(item_fetch, item_pos) * batch_size
                hit_ratio_total += hit_ratio_at_k(item_fetch, card) * batch_size
            tbar_test.set_description('epoch: %d, precision@10: %.3f, hit-ratio@10: %.3f' % (epoch + 1,
                                                                                          precision_at_k_total / batches,
                                                                                          hit_ratio_total / batches))

