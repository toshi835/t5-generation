# T5-generation in Japanese

## 日本語T5で転移学習
データを用意すれば日本語T5モデルの転移学習を行い、文章生成を行うことができます

## 使用例・解説
[日本語T5を使って文章生成 〜ヒロアカのキャラ名生成〜](https://qiita.com/toshi0427/items/61f2f4a6fb385b31c638)

## 使用方法
Google Colaboratoryで試す場合  
[t5-generation.ipynb](https://colab.research.google.com/drive/1eFtsJ2UgD0hvYU2ROsS2FaSm8QY9PlEv?usp=sharing)


コマンドラインで実行する場合
```
# data/data.csvを作成

# 前処理
python preprocess.py

# 訓練
python train.py -data_path ../data -model_path ../model

# テスト
python test.py -data_path ../data -model_path ../model -beam_size 1

# 生の文を入力してテスト
python test_raw.py -data_path ../data -model_path ../model -beam_size 1
```

- data_pathとmodel_pathは必要に応じて変更してください
- beam_sizeを変更することで生成する文章数を変更できます
- -visible_gpusを使用してgpuの番号を指定できます