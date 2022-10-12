"""
Apply mask on logical form
"""

from datetime import datetime
import numpy as np
import os
import logging
import shutil
from tqdm import tqdm
from pyrouge import Rouge155
import time
import re
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from dataloader import DataLoader, OrgDataMng

from utils import encoder
from utils.metric import rouge_score, check_res, bleu_score
from BLEC.eval.blec import logic_matching

from Model import SequencePipeline

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--mode', type=str)
parser.add_argument('--load-ckpt', type=str)
parser.add_argument('--save-ckpt', type=str, default='checkpoints')
parser.add_argument('--output-path', type=str, default='output')

parser.add_argument('--data-path', type=str, default='logic2text/data')

parser.add_argument('--beam-size', type=int, default=2)
parser.add_argument('--max-input-len', type=int, default=500)
parser.add_argument('--max-text-len', type=int, default=180)
parser.add_argument('--max-table-len', default=200, type=int)
parser.add_argument('--pretrained-path', type=str, default='./gpt2/')

parser.add_argument('--val-batch-size', type=int, default=10)
parser.add_argument('--batch-size', type=int, default=2)
parser.add_argument('--epochs', type=int, default=1000)
parser.add_argument('--lr', type=float, default=3e-4)
parser.add_argument('--alpha', type=float, default=0)
parser.add_argument('--beta', type=float, default=0)
parser.add_argument('--prefix-mode', type=str, default='')
parser.add_argument('--edit-strategy', type=str, choices=['rand', 'rep', 'expand', 'mix', 'dtype', ''], default='')
parser.add_argument('--edit-prob', type=float, default=1)
parser.add_argument('--froze', action='store_true')
parser.add_argument('--fd', action='store_true')

parser.add_argument('--inspection', action='store_true')

parser.add_argument('--display-batch', type=float, default=1000)
parser.add_argument('--seed', type=int, default=17)

args = parser.parse_args()

np.random.seed(args.seed)
torch.manual_seed(args.seed)


@torch.no_grad()
def evaluate(enc, pipeline: SequencePipeline, output_path, save_ckpt, valid_iterator, best_score, writer, step_id, epoch, data_path, mode='valid'):
    pipeline.model.eval()
    r = Rouge155()
    # model prediction
    system_dir = os.path.join(output_path, 'load')
    if os.path.exists(system_dir):
        shutil.rmtree(system_dir)
    os.makedirs(system_dir)
    r.system_dir = system_dir
    r.system_filename_pattern = '(\d+)_pred.txt'
    # gold sentence
    r.model_dir = os.path.join(data_path, 'test_split_for_%s/' % mode)
    r.model_filename_pattern = '#ID#_gold.txt'

    valid_iterator.count = 0
    valid_t = tqdm(valid_iterator, total=valid_iterator.num_batches)
    if epoch is not None:
        valid_t.set_description_str('validation on epoch %d' % epoch)
    generated_lines = []
    valid_clean_path = os.path.join(output_path, '%s_clean.txt' % mode)

    golds = []
    logics = []
    prefix_mode = args.prefix_mode + '_' if args.prefix_mode != '' else ''

    with torch.no_grad():
        for batch_dict in valid_t:
            # model.gpt.zero_grad()
            generated = pipeline.decode_batch(batch_dict, args.prefix_mode).tolist()
            # decode
            for b_text in generated:
                seq = enc.decode(b_text)
                seq = seq.split('<|endoftext|>')[0]
                seq = seq.replace('\n', ' ')
                if not check_res(seq):
                    seq = 'empty .'
                generated_lines.append(seq)
            
            for inputs in batch_dict['enc_in'].tolist():
                seq = enc.decode(inputs)
                match = re.search(r"\. (.+ = true)?", seq)
                logic = match.group(1)
                logics.append(logic)
                
            # if prefix_mode != '':
            for gold in batch_dict[f'{prefix_mode}dec_out'].tolist():
                seq = enc.decode(gold)
                seq = seq.split('<|endoftext|>')[0]
                golds.append(seq)

            # bleu score
    with open(valid_clean_path, 'w') as f:
        f.write('\n'.join(generated_lines))
    
    if prefix_mode != '':
        gold_path = os.path.join(output_path, '%s_gold.txt' % mode)
        with open(gold_path, 'w') as f:
            f.write('\n'.join(golds))
        correct = 0
        for p, t in zip(generated_lines, golds):
            if p == t:
                correct += 1
        acc = correct / len(generated_lines)
        writer.add_scalars(f'{mode}', {
            'acc': acc,
        }, global_step=step_id)
        writer.flush()
        if best_score < acc:
            best_score = acc
            if save_ckpt is not None:
                save_f = os.path.join(save_ckpt, '%d-%.4f.pt' % (epoch, acc))
                logging.info('save model checkpoint for %.4f to %s' % (acc, save_f))
                torch.save(pipeline.model.state_dict(), save_f)
        return best_score
    
    blec_res = []
    for pred_line, logic_line, gold_line in zip(generated_lines, logics, golds):
        errs = logic_matching(logic_line, pred_line, gold_line)
        blec_res.append(len(errs) == 0)
    blec_score = np.mean(blec_res)

    if epoch is not None:
        logging.info('validation on epoch %d' % epoch)
    bleu_ = bleu_score(os.path.join(
        data_path, 'original_data', '%s.text' % mode), valid_clean_path)
    logging.info('bleu: %.4f' % bleu_)
    logging.info('blec: %.4f' % blec_score)

    # rouge score
    for line_id, line in enumerate(generated_lines):
        # write into system dir
        gen_path = os.path.join(system_dir, f'{line_id}_pred.txt')
        with open(gen_path, 'w') as f:
            f.write(line + '\n')
    try:
        start_time = time.time()
        output, results_dict = rouge_score(r)
        rouge_result = "\n".join([output.split("\n")[3], output.split(
            "\n")[7], output.split("\n")[15], output.split("\n")[19]])
        logging.info(rouge_result)
        logging.info('time cost: %.4f' % (time.time() - start_time))
    except Exception as e:
        logging.warning(e)
    if writer is not None:
        writer.add_scalars(f'{mode} language metric', {
            'bleu': bleu_,
            'blec': blec_score,
            'rouge_1': results_dict['rouge_1_f_score'] * 100,
            'rouge_2': results_dict['rouge_2_f_score'] * 100,
            'rouge_4': results_dict['rouge_4_f_score'] * 100,
            'rouge_l': results_dict['rouge_l_f_score'] * 100
        }, global_step=step_id)
        writer.flush()

        # save checkpoint
        if blec_score > best_score:
            if save_ckpt is not None:
                save_f = os.path.join(save_ckpt, '%d-%.4f.pt' % (epoch, blec_score))
                logging.info('save model weights to %s' % save_f)
                torch.save(pipeline.model.state_dict(), save_f)
            best_score = blec_score

        return best_score
    return best_score


def main():
    output_path = os.path.join(
        args.output_path, datetime.today().strftime('%mM%dD-%H-%M-%S'))
    os.makedirs(output_path, exist_ok=True)
    # log file
    log_file = os.path.join(output_path, 'output.log')

    logging.basicConfig(filename=log_file, filemode='w', level=logging.INFO,
                        format='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s')
    
    logging.info('this is experiment of gpt2')

    logging.info('#' * 50 + '\n')
    logging.info('\n'.join([f'{k}: {v}' for k, v in args._get_kwargs()]))

    if torch.cuda.is_available():
        logging.info('using cuda, count: %d' % torch.cuda.device_count())
    else:
        raise ValueError('no gpu to use')
    
    # pad used to pad to max length
    empty = 2
    # end of speech
    eos = 50256
    # tokenizer.decode('</s>')
    logging.info('loading tokenizer')
    enc = encoder.get_encoder('gpt2', '')
    
    save_ckpt = os.path.join(
        output_path, args.save_ckpt
    )
    os.makedirs(save_ckpt, exist_ok=True)

    logging.info('initializing pipeline')
    pipeline = SequencePipeline(
        ptm_path=args.pretrained_path,
        beam_size=args.beam_size,
        empty_token=empty,
        stop_token=eos,
        max_length=args.max_text_len
    )
    # logging.info('\n'.join([name for name, _ in pipeline.model.named_parameters()]))
    org_data = OrgDataMng(args.data_path)
    
    if args.mode == 'resume' or args.mode == 'test':
        # load checkpoint
        logging.info('loading from checkpoint: %s' % args.load_ckpt)
        pipeline.model.load_state_dict(torch.load(args.load_ckpt))
    
    if args.mode in ('train', 'resume'):
        # train processing
        summary_path = os.path.join(output_path, 'summary')

        # make dir
        if os.path.exists(summary_path):
            shutil.rmtree(summary_path)
        os.makedirs(summary_path, exist_ok=True)

        train_iterator = DataLoader(
            org_data.train,
            enc,
            args.batch_size,
            True,
            args.max_text_len,
            args.max_input_len,
            args.max_table_len,
            eos,
            empty,
            True,
            args.edit_strategy,
            args.edit_prob
        )

        if args.fd:
            fd_train_iterator = DataLoader(
                org_data.train,
                enc,
                args.batch_size,
                True,
                args.max_text_len,
                args.max_input_len,
                args.max_table_len,
                eos,
                empty,
                True
            )

        valid_iterator = DataLoader(
            org_data.valid,
            enc,
            args.val_batch_size,
            False,
            args.max_text_len,
            args.max_input_len,
            args.max_table_len,
            eos,
            empty
        )

        if args.inspection:
            cd_org_data = OrgDataMng('CD/data')
            cd_iterator = DataLoader(
                cd_org_data.test,
                enc,
                args.val_batch_size,
                False,
                args.max_text_len,
                args.max_input_len,
                args.max_table_len,
                eos,
                empty)

        best_score = 0
        writer = SummaryWriter(summary_path)

        optimizer = torch.optim.SGD(pipeline.model.parameters(), args.lr)
        if args.fd:
            fd_optimizer = torch.optim.SGD(pipeline.model.parameters(), args.lr * 0.5)
        pipeline.model = pipeline.model.to('cuda')

        if args.froze:
            for n, p in pipeline.model.named_parameters():
                if 'lm_head' in n or 'ln_f' in n:
                    p.requires_grad = True
                else:
                    p.requires_grad = False
        logging.info('\n'.join(f'{n}: {p.requires_grad}' for n, p in pipeline.model.named_parameters()))

        step_id = 0
        display_batch = args.display_batch or train_iterator.num_batches
        best_cd_score = 0
        
        logging.info('start training')
        for epoch in range(args.epochs):
            # train
            train_iterator.reset()
            train_t = tqdm(train_iterator, total=train_iterator.num_batches)
            train_t.set_description_str('training on epoch %d' % epoch)
            pipeline.model.train()
            for batch_dict in train_t:
                # check if time to evaluate
                if (step_id + 1) % display_batch == 0:
                    # evaluate
                    tmp_score = evaluate(enc, pipeline, output_path, save_ckpt, valid_iterator,
                                          best_score, writer, step_id, epoch, args.data_path)
                    if args.inspection:
                        tmp_cd_score = evaluate(enc, pipeline, output_path, None, cd_iterator, best_cd_score, writer, step_id, epoch, 'CD/data/', mode='test')
                        if tmp_score <= best_score:
                            # saved in evaluation
                            if tmp_cd_score > best_cd_score:
                                save_f = os.path.join(save_ckpt, 'CDs-%d-%.4f.pt' % (epoch, tmp_cd_score))
                                logging.info('save model weights to %s' % save_f)
                                torch.save(pipeline.model.state_dict(), save_f)
                        best_cd_score = max(best_cd_score, tmp_cd_score)

                    best_score = max(best_score, tmp_score)

                    pipeline.model.train()
                # clear grad
                pipeline.model.zero_grad()
                loss_dict = pipeline.train_batch(batch_dict)
                display_loss_dict = {
                    k: v.item() for k, v in loss_dict.items()
                }

                writer.add_scalars('train loss', display_loss_dict, global_step=step_id)
                
                # train_t.set_postfix_str('loss: %.4f' % )
                writer.flush()
                # total_loss = (1 - args.alpha - args.beta) * loss_dict['lm_loss'] \
                #     + args.alpha * loss_dict['level_loss'] \
                #     + args.beta * loss_dict['chain_loss']
                # randomly select one loss to optimize, instead of optimizing sum of them
                rdnm = np.random.rand()
                if rdnm < args.alpha:
                    use_loss = 'level_loss'
                elif rdnm < args.alpha + args.beta:
                    use_loss = 'chain_loss'
                else:
                    use_loss = 'lm_loss'
                total_loss = loss_dict[use_loss]
                total_loss.backward()
                optimizer.step()
                step_id += 1
                torch.cuda.empty_cache()
            
            # for fd:
            if args.fd:
                fd_train_iterator.reset()
                train_t = tqdm(fd_train_iterator, total=fd_train_iterator.num_batches)
                train_t.set_description_str('training on epoch %d full data' % epoch)
                pipeline.model.train()
                for batch_dict in train_t:
                    # check if time to evaluate
                    if (step_id + 1) % display_batch == 0:
                        # evaluate
                        tmp_score = evaluate(enc, pipeline, output_path, save_ckpt, valid_iterator,
                                            best_score, writer, step_id, epoch, args.data_path)
                        if args.inspection:
                            tmp_cd_score = evaluate(enc, pipeline, output_path, None, cd_iterator, best_cd_score, writer, step_id, epoch, 'CD/data/', mode='test')
                            if tmp_score <= best_score:
                                # saved in evaluation
                                if tmp_cd_score > best_cd_score:
                                    save_f = os.path.join(save_ckpt, 'CDs-%d-%.4f.pt' % (epoch, tmp_cd_score))
                                    logging.info('save model weights to %s' % save_f)
                                    torch.save(pipeline.model.state_dict(), save_f)
                            best_cd_score = max(best_cd_score, tmp_cd_score)

                        best_score = max(best_score, tmp_score)

                        pipeline.model.train()
                    # clear grad
                    pipeline.model.zero_grad()
                    loss_dict = pipeline.train_batch(batch_dict)
                    display_loss_dict = {
                        k: v.item() for k, v in loss_dict.items()
                    }

                    writer.add_scalars('train loss', display_loss_dict, global_step=step_id)
                    
                    # train_t.set_postfix_str('loss: %.4f' % )
                    writer.flush()
                    # total_loss = (1 - args.alpha - args.beta) * loss_dict['lm_loss'] \
                    #     + args.alpha * loss_dict['level_loss'] \
                    #     + args.beta * loss_dict['chain_loss']
                    # randomly select one loss to optimize, instead of optimizing sum of them
                    rdnm = np.random.rand()
                    if rdnm < args.alpha:
                        use_loss = 'level_loss'
                    elif rdnm < args.alpha + args.beta:
                        use_loss = 'chain_loss'
                    else:
                        use_loss = 'lm_loss'
                    total_loss = loss_dict[use_loss]
                    total_loss.backward()
                    fd_optimizer.step()
                    step_id += 1
                    torch.cuda.empty_cache()

    else:
        assert args.mode == 'test'
        test_iterator = DataLoader(
            org_data.test,
            enc,
            args.val_batch_size,
            False,
            args.max_text_len,
            args.max_input_len,
            args.max_table_len,
            eos,
            empty
        )
        evaluate(enc, pipeline, output_path, None, test_iterator, 100, None, None, None, args.data_path, mode='test')

if __name__ == '__main__':
    main()
