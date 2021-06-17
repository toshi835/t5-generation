import argparse
import torch
import pytorch_lightning as pl
from utils import TsvDataset, T5FineTuner, set_seed
from transformers import T5Tokenizer

set_seed(42)

parser = argparse.ArgumentParser()
parser.add_argument('-visible_gpus', default=0, type=int)
parser.add_argument('-data_path', default='../data/', type=str)
parser.add_argument('-model_path', default='../model/', type=str)
ag = parser.parse_args()

# 事前学習済みモデル
PRETRAINED_MODEL_NAME = "sonoisa/t5-base-japanese"

# GPU利用有無
USE_GPU = torch.cuda.is_available()

# 各種ハイパーパラメータ
args_dict = dict(
    data_dir=ag.data_path,  # データセットのディレクトリ
    model_name_or_path=PRETRAINED_MODEL_NAME,
    tokenizer_name_or_path=PRETRAINED_MODEL_NAME,

    learning_rate=3e-4,
    weight_decay=0.0,
    adam_epsilon=1e-8,
    warmup_steps=0,
    gradient_accumulation_steps=1,

    n_gpu=1 if USE_GPU else 0,
    early_stop_callback=False,
    fp_16=False,
    opt_level='O1',
    max_grad_norm=1.0,
    seed=42,
)

# トークナイザー（SentencePiece）モデルの読み込み
tokenizer = T5Tokenizer.from_pretrained(PRETRAINED_MODEL_NAME, is_fast=True)

# テストデータセットの読み込み
train_dataset = TsvDataset(tokenizer, args_dict["data_dir"], "train.tsv",
                           input_max_len=512, target_max_len=64)

# 学習に用いるハイパーパラメータを設定する
args_dict.update({
    "max_input_length": 512,  # 入力文の最大トークン数
    "max_target_length": 64,  # 出力文の最大トークン数
    "train_batch_size": 4,  # 訓練時のバッチサイズ
    "eval_batch_size": 8,  # テスト時のバッチサイズ
    "num_train_epochs": 9,  # 訓練するエポック数
})
args = argparse.Namespace(**args_dict)

train_params = dict(
    accumulate_grad_batches=args.gradient_accumulation_steps,
    gpus=ag.visible_gpus,
    max_epochs=args.num_train_epochs,
    precision=16 if args.fp_16 else 32,
    amp_level=args.opt_level,
    gradient_clip_val=args.max_grad_norm,
)

# 転移学習の実行（GPUを利用すれば1エポック10分程度）
model = T5FineTuner(args)
trainer = pl.Trainer(**train_params)
trainer.fit(model)

# 最終エポックのモデルを保存
model.tokenizer.save_pretrained(ag.model_path)
model.model.save_pretrained(ag.model_path)

print("学習終了")
