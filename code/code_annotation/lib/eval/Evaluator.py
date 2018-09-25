from __future__ import division
import lib
import sys
import torch
import pdb
import time


class Evaluator(object):
    def __init__(self, model, metrics, dicts, opt):
        self.model = model
        self.loss_func = metrics["xent_loss"]
        self.sent_reward_func = metrics.get("sent_reward", None)
        self.corpus_reward_func = metrics.get("corp_reward", None)
        self.dicts = dicts
        self.max_length = opt.max_predict_length
        self.opt = opt

    def eval(self, data, pred_file=None):
        with torch.no_grad():
            self.model.eval()

            total_loss = 0
            total_words = 0
            total_sents = 0
            total_sent_reward = 0

            all_preds = []
            all_targets = []
            all_srcs = []
            all_qts = []
            for i in range(len(data)): #
                batch = data[i]

                if self.opt.data_type == 'code':
                    targets = batch[2]
                    # attention_mask = batch[2][0].data.eq(lib.Constants.PAD).t()
                    attention_mask = batch[1][2][0].data.eq(lib.Constants.PAD).t()
                elif self.opt.data_type == 'text':
                    targets = batch[2]
                    # attention_mask = batch[0][0].data.eq(lib.Constants.PAD).t()
                    attention_mask = batch[0][0].data.eq(lib.Constants.PAD).t()
                elif self.opt.data_type == 'hybrid':
                    targets = batch[2]
                    attention_mask_code = batch[1][2][0].data.eq(lib.Constants.PAD).t()
                    attention_mask_txt = batch[0][0].data.eq(lib.Constants.PAD).t()
                else:
                    raise Exception("Invalid data_type %s" % self.opt.data_type)

                qts = batch[-1]

                if self.opt.has_attn:
                    if self.opt.data_type == 'code' or self.opt.data_type == 'text':
                        self.model.decoder.attn.applyMask(attention_mask)
                    elif self.opt.data_type == 'hybrid':
                        self.model.decoder.attn.applyMask(attention_mask_code, attention_mask_txt)

                outputs = self.model(batch, True)

                weights = targets.ne(lib.Constants.PAD).float()
                num_words = weights.data.sum()
                _, loss = self.model.predict(outputs, targets, weights, self.loss_func)

                preds = self.model.translate(batch, self.max_length)
                preds = preds.t().tolist()
                srcs = batch[0][0]
                srcs = srcs.data.t().tolist()
                targets = targets.data.t().tolist()
                qts = [item.tolist() for item in qts]

                if self.sent_reward_func is not None:
                    s0 = time.time()
                    rewards, _ = self.sent_reward_func(srcs, preds, qts, targets)
                    print("Eval one batch time: %.2f" % (time.time() - s0))
                else:
                    rewards = [0.0] * len(preds)

                all_preds.extend(preds)
                all_targets.extend(targets)
                all_srcs.extend(srcs)
                all_qts.extend(qts)

                total_loss += loss
                total_words += num_words
                total_sent_reward += sum(rewards)

                if self.opt.data_type == 'code':
                    total_sents += batch[2].size(1)
                elif self.opt.data_type == 'text':
                    total_sents += batch[2].size(1)
                elif self.opt.data_type == 'hybrid':
                    total_sents += batch[2].size(1)

            loss = total_loss / total_words
            sent_reward = total_sent_reward / total_sents
            if self.corpus_reward_func is not None:
                corpus_reward = self.corpus_reward_func(all_preds, all_targets)
            else:
                corpus_reward = 0.0

            if pred_file is not None:
                self._convert_and_report(data, pred_file, all_preds, all_targets, all_srcs, all_qts,
                                         (loss, sent_reward, corpus_reward))

            return loss, sent_reward, corpus_reward

    def _convert_and_report(self, data, pred_file, preds, targets, srcs, qts, metrics):
        with open(pred_file, "w") as f:
            for i in range(len(preds)):
                pred = preds[i]
                target = targets[i]
                src = srcs[i]
                qt = qts[i]

                src = lib.Reward.clean_up_sentence(src, remove_unk=False, remove_eos=True)
                pred = lib.Reward.clean_up_sentence(pred, remove_unk=False, remove_eos=True)
                target = lib.Reward.clean_up_sentence(target, remove_unk=False, remove_eos=True)

                src = [self.dicts["src"].getLabel(w) for w in src]
                pred = [self.dicts["tgt"].getLabel(w) for w in pred]
                tgt = [self.dicts["tgt"].getLabel(w) for w in target]
                qt = [self.dicts["qt"].getLabel(w) for w in qt]

                f.write(str(i) + ": src: "+ " ".join(src).encode('utf-8', 'ignore') + '\n')
                f.write(str(i) + ": pre: " + " ".join(pred).encode('utf-8', 'ignore') + '\n')
                f.write(str(i) + ": tgt: "+ " ".join(tgt).encode('utf-8', 'ignore') + '\n')
                f.write(str(i) + ": qt: " + " ".join(qt).encode('utf-8', 'ignore') + '\n')

        loss, sent_reward, corpus_reward = metrics
        print("")
        print("Loss: %.6f" % loss)
        print("Sentence reward: %.2f" % (sent_reward * 100))
        print("Corpus reward: %.2f" % (corpus_reward * 100))
        print("Predictions saved to %s" % pred_file)

