import os
import re
import numpy as np
import torch
from scipy import linalg
from tqdm import tqdm
import spacy
# os.environ['MOVERSCORE_MODEL'] = 'path/to/distill-bert/'
from nlgmetricverse import NLGMetricverse, load_metric
from bert_score import score


@torch.no_grad()
def evaluation_m2t(val_loader, cfg, model, tokenizer, w_vectorizer, eval_wrapper, instruction, max_new_tokens, base_token_num=0):
    model.eval()

    nb_sample = 0

    generated_texts_list = []
    all_captions_list = []

    R_precision_real = 0
    R_precision = 0
    matching_score_real = 0
    matching_score_pred = 0

    max_text_len = 20

    nlp = spacy.load('en_core_web_sm')

    for batch in tqdm(val_loader):
        word_embeddings, pos_one_hots, clip_text, sent_len, pose, m_length, _, name, all_captions, guo_motion, guo_m_length = batch

        bs, seq = pose.shape[:2]
        pred_pos_one_hots = torch.zeros(pos_one_hots.size())
        pred_word_embeddings = torch.zeros(word_embeddings.size())
        pred_sent_len_list = torch.zeros(sent_len.size())


        for k in range(bs):
            # tokenize motion
            motion = pose[k].clone()
            motion = motion#[:m_length[k]]
            motion = motion.unsqueeze(0).cuda()


            # --- 生成预测 ---
            preds = model.generate(
                motion_seq=motion,
                prompt='Describe this motion:',
                num_beams=1,
                max_new_tokens=40,
                repetition_penalty=1.2,
                do_sample=False,
            )
            #preds = ['a person is running']

            output_text = preds[0]
            generated_texts_list.append(preds[0])
            all_captions_list.append([all_captions[0][k], all_captions[1][k], all_captions[2][k]])

            # Generated Text needs to be processed as word_embeddings and pos_one_hots
            word_list, pos_list = _process_text(output_text.strip(), nlp)
            t_tokens = ['%s/%s' % (word_list[i], pos_list[i]) for i in range(len(word_list))]

            if len(t_tokens) < max_text_len:
                # pad with "unk"
                tokens = ['sos/OTHER'] + t_tokens + ['eos/OTHER']
                pred_sent_len = len(tokens)
                tokens = tokens + ['unk/OTHER'] * (max_text_len + 2 - pred_sent_len)
            else:
                # crop
                tokens = t_tokens[:max_text_len]
                tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
                pred_sent_len = len(tokens)
            pred_pos_one_hots_a_sample = []
            pred_word_embeddings_a_sample = []
            for token in tokens:
                word_emb, pos_oh = w_vectorizer[token]
                pred_pos_one_hots_a_sample.append(pos_oh[None, :])
                pred_word_embeddings_a_sample.append(word_emb[None, :])
            pred_pos_one_hots_a_sample = np.concatenate(pred_pos_one_hots_a_sample, axis=0)
            pred_word_embeddings_a_sample = np.concatenate(pred_word_embeddings_a_sample, axis=0)

            pred_pos_one_hots[k] = torch.from_numpy(pred_pos_one_hots_a_sample)
            pred_word_embeddings[k] = torch.from_numpy(pred_word_embeddings_a_sample)
            pred_sent_len_list[k] = pred_sent_len



        pred_pos_one_hots = pred_pos_one_hots.cuda()
        pred_word_embeddings = pred_word_embeddings.cuda()
        pred_sent_len_list = pred_sent_len_list.cuda()

        sorted_pred_sent_len_list, indices = pred_sent_len_list.sort(descending=True)
        sorted_pred_pos_one_hots = pred_pos_one_hots[indices]
        sorted_pred_word_embeddings = pred_word_embeddings[indices]

        sorted_pose = guo_motion.clone().cuda()[indices]
        sorted_m_length = guo_m_length.clone().cuda()[indices]

        et_pred, em_pred = eval_wrapper.get_co_embeddings(sorted_pred_word_embeddings, sorted_pred_pos_one_hots, sorted_pred_sent_len_list, sorted_pose, sorted_m_length)

        pose = guo_motion.cuda().float()

        et, em = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len, pose, guo_m_length)
        
        temp_R, temp_match = calculate_R_precision(em.cpu().numpy(), et.cpu().numpy(), top_k=3, sum_all=True)
        R_precision_real += temp_R
        matching_score_real += temp_match
        temp_R, temp_match = calculate_R_precision(em_pred.cpu().numpy(), et_pred.cpu().numpy(),  top_k=3, sum_all=True)
        R_precision += temp_R
        matching_score_pred += temp_match

        nb_sample += bs

    R_precision_real = R_precision_real / nb_sample
    R_precision = R_precision / nb_sample

    matching_score_real = matching_score_real / nb_sample
    matching_score_pred = matching_score_pred / nb_sample

    try:
        bleu1, bleu2, rouge, cider = calculate_bleu_rouge_cider(ref_text_list=all_captions_list, hyp_text_list=generated_texts_list)
    except:
        bleu1, bleu2, rouge, cider = 0.0, 0.0, 0.0, 0.0
    
    #try:
    bert_score = 0.0 #evaluate_bert_score(generated_texts_list, all_captions_list)
    #except:
    #    bert_score = 0.0
        
    msg = f"R_precision_real. {R_precision_real}, R_precision. {R_precision}," \
          f" MM_Dist_real. {matching_score_real}, MM_Dist_pred. {matching_score_pred}, " \
          f"Bleu@1. {bleu1}, Bleu@2. {bleu2}, " \
          f"Rouge. {rouge}, Cider. {cider}, " \
          f"BertScore. {bert_score}."
    print(msg)
    model.train()
    return R_precision[0], R_precision[1], R_precision[2], matching_score_pred, bleu1, bleu2, rouge, cider, bert_score, msg



@torch.no_grad()
def evaluation_m2dt(val_loader, vqvae, model, logger, tokenizer, instruction, max_new_tokens):
    model.eval()

    text_gt_list = []
    text_pred_list = []
    text_gt_snippet_list = []
    text_pred_snippet_list = []

    for batch in tqdm(val_loader):
        gt_detailed_text, pose, m_length, name = batch

        bs, seq = pose.shape[:2]

        for k in range(bs):
            # tokenize motion
            motion = pose[k].clone()
            motion = motion[:m_length[k]]
            motion = motion.unsqueeze(0).cuda()
            tokenized_motion = vqvae.encode(motion)
            tokenized_motion = tokenized_motion.cpu().numpy()[0]
            tokenized_motion = tokenized_motion.reshape(-1).tolist()

            motion_string = '<Motion Tokens>'
            for token in tokenized_motion:
                motion_string += ('<' + str(token) + '>')
            motion_string += '</Motion Tokens>'

            prompt = instruction + motion_string

            input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to('cuda', dtype=torch.long)
            outputs = model.generate(
                input_ids,
                max_length=max_new_tokens,
                num_beams=1,
                do_sample=False,
            )
            output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            output_text = output_text.split('### Motion Script ###')[1].strip()

            gt_detailed_text_k_snippet_list = gt_detailed_text[k].split(" <SEP> ")
            output_text_k_snippet_list = output_text.split(" <SEP> ")

            # sequence-level
            text_pred_list.append(output_text)
            text_gt_list.append(gt_detailed_text[k])

            # snippet-level
            for item in gt_detailed_text_k_snippet_list:
                text_gt_snippet_list.append(item)

            for item in output_text_k_snippet_list[:len(gt_detailed_text_k_snippet_list)]:
                text_pred_snippet_list.append(item)

            while len(text_pred_snippet_list) < len(text_gt_snippet_list):
                text_pred_snippet_list.append("")

            while len(text_gt_snippet_list) < len(text_pred_snippet_list):
                text_gt_snippet_list.append("")

    bleu1, bleu2, bleu7, rouge, cider = calculate_bleu147_rouge_cider(ref_text_list=text_gt_list,
                                                                      hyp_text_list=text_pred_list)
    s_bleu1, s_bleu2, s_bleu7, s_rouge, s_cider = calculate_bleu147_rouge_cider(ref_text_list=text_gt_snippet_list,
                                                                                hyp_text_list=text_pred_snippet_list)

    bert_score = evaluate_bert_score(text_pred_list, text_gt_list)
    s_bert_score = evaluate_bert_score(text_pred_snippet_list, text_gt_snippet_list)

    logger.info('Sequence-level:')
    msg = f"Bleu@1. {bleu1}, Bleu@2. {bleu2}, Bleu@7. {bleu7}, " \
          f"Rouge. {rouge}, Cider. {cider}, BertScore. {bert_score}"
    print(msg)
    logger.info(msg)

    logger.info('Snippet-level:')
    msg = f"Bleu@1. {s_bleu1}, Bleu@2. {s_bleu2}, Bleu@7. {s_bleu7}, " \
          f"Rouge. {s_rouge}, Cider. {s_cider}, BertScore. {s_bert_score}"
    print(msg)
    logger.info(msg)

    model.train()
    return bleu1, bleu2, bleu7, rouge, cider, bert_score, \
           s_bleu1, s_bleu2, s_bleu7, s_rouge, s_cider, s_bert_score, \
           logger



def euclidean_distance_matrix(matrix1, matrix2):
    """
        Params:
        -- matrix1: N1 x D
        -- matrix2: N2 x D
        Returns:
        -- dist: N1 x N2
        dist[i, j] == distance(matrix1[i], matrix2[j])
    """
    assert matrix1.shape[1] == matrix2.shape[1]
    d1 = -2 * np.dot(matrix1, matrix2.T)    # shape (num_test, num_train)
    d2 = np.sum(np.square(matrix1), axis=1, keepdims=True)    # shape (num_test, 1)
    d3 = np.sum(np.square(matrix2), axis=1)     # shape (num_train, )
    dists = np.sqrt(d1 + d2 + d3)  # broadcasting
    return dists



def calculate_top_k(mat, top_k):
    size = mat.shape[0]
    gt_mat = np.expand_dims(np.arange(size), 1).repeat(size, 1)
    bool_mat = (mat == gt_mat)
    correct_vec = False
    top_k_list = []
    for i in range(top_k):
#         print(correct_vec, bool_mat[:, i])
        correct_vec = (correct_vec | bool_mat[:, i])
        # print(correct_vec)
        top_k_list.append(correct_vec[:, None])
    top_k_mat = np.concatenate(top_k_list, axis=1)
    return top_k_mat



def calculate_R_precision(embedding1, embedding2, top_k, sum_all=False):
    dist_mat = euclidean_distance_matrix(embedding1, embedding2)
    matching_score = dist_mat.trace()
    argmax = np.argsort(dist_mat, axis=1)
    top_k_mat = calculate_top_k(argmax, top_k)
    if sum_all:
        return top_k_mat.sum(axis=0), matching_score
    else:
        return top_k_mat, matching_score



def calculate_diversity(activation, diversity_times):
    assert len(activation.shape) == 2
    assert activation.shape[0] > diversity_times
    num_samples = activation.shape[0]

    first_indices = np.random.choice(num_samples, diversity_times, replace=False)
    second_indices = np.random.choice(num_samples, diversity_times, replace=False)
    dist = linalg.norm(activation[first_indices] - activation[second_indices], axis=1)
    return dist.mean()



def calculate_multimodality(activation, multimodality_times):
    assert len(activation.shape) == 3
    assert activation.shape[1] > multimodality_times
    num_per_sent = activation.shape[1]

    first_dices = np.random.choice(num_per_sent, multimodality_times, replace=False)
    second_dices = np.random.choice(num_per_sent, multimodality_times, replace=False)
    dist = linalg.norm(activation[:, first_dices] - activation[:, second_dices], axis=2)
    return dist.mean()




def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1)
            + np.trace(sigma2) - 2 * tr_covmean)



def calculate_activation_statistics(activations):
    mu = np.mean(activations, axis=0)
    cov = np.cov(activations, rowvar=False)
    return mu, cov



def calculate_frechet_feature_distance(feature_list1, feature_list2):
    feature_list1 = np.stack(feature_list1)
    feature_list2 = np.stack(feature_list2)

    # normalize the scale
    mean = np.mean(feature_list1, axis=0)
    std = np.std(feature_list1, axis=0) + 1e-10
    feature_list1 = (feature_list1 - mean) / std
    feature_list2 = (feature_list2 - mean) / std

    dist = calculate_frechet_distance(
        mu1=np.mean(feature_list1, axis=0),
        sigma1=np.cov(feature_list1, rowvar=False),
        mu2=np.mean(feature_list2, axis=0),
        sigma2=np.cov(feature_list2, rowvar=False),
    )
    return dist



def calculate_bleu_rouge_cider(ref_text_list, hyp_text_list):
    # 删除了 bleu 的加载，只保留 rouge 和 cider
    metrics = [
        load_metric("rouge"),
        load_metric("cider"),
    ]
    nlg_evaluator = NLGMetricverse(metrics)
    scores = nlg_evaluator(predictions=hyp_text_list,
                           references=ref_text_list)

    # 用 0.0 替代原本的 bleu_1 和 bleu_2 返回值，防止外部解包报错
    return 0.0, 0.0, \
           scores['rouge']['rougeL'], scores['cider']['score']


def calculate_bleu147_rouge_cider(ref_text_list, hyp_text_list):
    # 同样删除 bleu 相关的指标加载
    metrics = [
        load_metric("rouge"),
        load_metric("cider"),
    ]
    nlg_evaluator = NLGMetricverse(metrics)
    scores = nlg_evaluator(predictions=hyp_text_list,
                           references=ref_text_list)

    # 用 0.0 占位 bleu_1, bleu_2, bleu_7
    return 0.0, 0.0, 0.0, \
           scores['rouge']['rougeL'], scores['cider']['score']


def evaluate_bert_score(text_list1, text_list2):
    print('========== Evaluating BERT Score ==========')
    P, R, F1 = score(text_list1,
                     text_list2,
                     lang='en',
                     rescale_with_baseline=True,
                     #model_type='/mnt/netdisk2/zhangjh/Code/MG-MotionLLM-main/checkpoints/roberta-large',
                     idf=True,
                     device='cuda:0',
                     verbose=True)

    return F1.mean().item()



def _process_text(sentence, nlp):
    sentence = sentence.replace('-', '')
    doc = nlp(sentence)
    word_list = []
    pos_list = []
    for token in doc:
        word = token.text
        if not word.isalpha():
            continue
        if (token.pos_ == 'NOUN' or token.pos_ == 'VERB') and (word != 'left'):
            word_list.append(token.lemma_)
        else:
            word_list.append(word)
        pos_list.append(token.pos_)
    return word_list, pos_list

