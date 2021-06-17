import argparse
import os

import torch
from torch.utils.data import DataLoader
from transformers import T5ForConditionalGeneration, T5Tokenizer

from tqdm.auto import tqdm
from utils import TsvDataset

parser = argparse.ArgumentParser()
parser.add_argument('-visible_gpus', default=0, type=int)
parser.add_argument('-data_path', default='../data/', type=str)
parser.add_argument('-model_path', default='../model/', type=str)
parser.add_argument('-beam_size', default=1, type=int)
ag = parser.parse_args()

# トークナイザー（SentencePiece）
tokenizer = T5Tokenizer.from_pretrained(ag.model_path, is_fast=True)

# 学習済みモデル
trained_model = T5ForConditionalGeneration.from_pretrained(ag.model_path)

# GPUの利用有無
USE_GPU = torch.cuda.is_available()
if USE_GPU:
    trained_model.cuda(ag.visible_gpus)

# 各種ハイパーパラメータ
args_dict = dict(
    data_dir=ag.data_path,
    max_input_length=512,
    max_target_length=64,
    eval_batch_size=8,
)
args = argparse.Namespace(**args_dict)

# テストデータの読み込み
test_dataset = TsvDataset(tokenizer, args_dict["data_dir"], "test.tsv",
                          input_max_len=args.max_input_length,
                          target_max_len=args.max_target_length)

test_loader = DataLoader(test_dataset, batch_size=args.eval_batch_size, num_workers=4)

trained_model.eval()

inputs = []
outputs = []
targets = []

# beamが複数のときのパラメータ
beam_params = dict()
if ag.beam_size != 1:
    cnt = 0
    beam_params.update({
        "num_beams": ag.beam_size,  # ビームサーチの探索幅
        "diversity_penalty": 1.0,  # 生成結果の多様性を生み出すためのペナルティ
        "num_beam_groups": ag.beam_size,  # ビームサーチのグループ数
        "num_return_sequences": ag.beam_size,  # 生成する文の数
    })

for batch in tqdm(test_loader):
    input_ids = batch['source_ids']
    input_mask = batch['source_mask']

    if USE_GPU:
        input_ids = input_ids.cuda(ag.visible_gpus)
        input_mask = input_mask.cuda(ag.visible_gpus)

    output = trained_model.generate(input_ids=input_ids,
                                    attention_mask=input_mask,
                                    max_length=args.max_target_length,
                                    temperature=1.0,  # 生成にランダム性を入れる温度パラメータ
                                    repetition_penalty=1.5,  # 同じ文の繰り返し（モード崩壊）へのペナルティ
                                    **beam_params
                                    )

    output_text = [tokenizer.decode(ids, skip_special_tokens=True,
                                    clean_up_tokenization_spaces=False)
                   for ids in output]
    target_text = [tokenizer.decode(ids, skip_special_tokens=True,
                                    clean_up_tokenization_spaces=False)
                   for ids in batch["target_ids"]]
    input_text = [tokenizer.decode(ids, skip_special_tokens=True,
                                   clean_up_tokenization_spaces=False)
                  for ids in input_ids]

    inputs.extend(input_text)
    outputs.extend(output_text)
    targets.extend(target_text)

with open(os.path.join(ag.data_path, "output.txt"), "w") as out, open(os.path.join(ag.data_path, "target.txt"),
                                                                      "w") as tar, open(
    os.path.join(ag.data_path, "input.txt"), "w") as inp:
    for i in range(len(inputs)):
        print("generated: " + "\n\t".join(outputs[i * ag.beam_size:i * ag.beam_size + ag.beam_size]))
        print("target:    " + targets[i])
        print("src:       " + inputs[i])
        print()
        out.write(", ".join(outputs[i * ag.beam_size:i * ag.beam_size + ag.beam_size]) + "\n")
        tar.write(targets[i] + "\n")
        inp.write(inputs[i] + "\n")
