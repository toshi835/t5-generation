import argparse
import warnings

import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer

warnings.simplefilter('ignore')

parser = argparse.ArgumentParser()
parser.add_argument('-visible_gpus', default=0, type=int)
parser.add_argument('-model_path', default='../model/', type=str)
parser.add_argument('-beam_size', default=10, type=int)
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
    max_input_length=512,
    max_target_length=64,
    eval_batch_size=8,
)
args = argparse.Namespace(**args_dict)

trained_model.eval()

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

print("入力をどうぞ")

while 1:
    # テストデータの読み込み
    input_raw = input()
    if input_raw == "q":
        exit()

    tokenized_inputs = tokenizer.batch_encode_plus(
        [input_raw], max_length=512, truncation=True,
        padding="max_length", return_tensors="pt")

    source_ids = tokenized_inputs["input_ids"]
    source_mask = tokenized_inputs["attention_mask"]

    if USE_GPU:
        source_ids = source_ids.cuda(ag.visible_gpus)
        source_mask = source_mask.cuda(ag.visible_gpus)

    output = trained_model.generate(input_ids=source_ids,
                                    attention_mask=source_mask,
                                    max_length=args.max_target_length,
                                    temperature=1.0,  # 生成にランダム性を入れる温度パラメータ
                                    repetition_penalty=1.5,  # 同じ文の繰り返し（モード崩壊）へのペナルティ
                                    **beam_params
                                    )

    output_text = [tokenizer.decode(ids, skip_special_tokens=True,
                                    clean_up_tokenization_spaces=False)
                   for ids in output]

    print("\n".join(output_text) + "\n")
