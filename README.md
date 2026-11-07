# Gemma 4 12B × MHEALTH：技術書典21の章の実験コード
## これはなに？
MHEALTHの身体センサ波形（100時点×15ch）を10パッチへ分け、線形投影でGemmaの入力埋め込みに挿入し、projectorとLoRAで12活動の回答コードを学習するColabノートブックです。

## 環境と実行（Colab）

1. `Gemma4_MHEALTH_PyTorch.ipynb` をColabで開き、BF16対応のCUDA GPUを選びます。章で報告した環境はG4 / RTX PRO 6000 Blackwell 96GBです。他のGPUの必要VRAMや所要時間は保証しません。
2. 先頭の依存導入と `torchao` 削除セルを実行します。インストール後はColabのランタイムを再起動し、依存導入・削除セルを飛ばして認証へ進みます。Colab標準のPyTorch/CUDA・scikit-learnは維持します。
3. [モデルカード](https://huggingface.co/google/gemma-4-12B-it)と[Gemma 4のApache-2.0ライセンス](https://ai.google.dev/gemma/apache_2)を確認します。認証セルではColab Secretsの `HF_TOKEN` にread tokenを登録し、このノートブックからのアクセスを許可します。tokenをコードへ直接記載しないでください。モデルは `google/gemma-4-12B-it`、revision `707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7` 固定です。
4. 残りを上から実行します。MHEALTH zipは[UCIの配布元](https://archive.ics.uci.edu/static/public/319/mhealth%2Bdataset.zip)から自動取得され、SHA256 `16ad0ce709f3f00df18f348610d15bce0884b79e2143f57f446493673f02b8e0` を照合します。既存zipを使う場合は `archive` をそのパスに変更します。
5. モデル読込セルは12B重みを取得し、`trainer.train()` は実際に学習を開始します。最大10epoch、validation lossでbest選択・patience2の早期停止です。生成評価セルの再実行は再推論します。
6. `gemma4_mhealth_run/` にbest adapter/projector/tokenizer、前処理統計が保存され、評価指標はPolarsの集計表としてColabの出力欄に標準出力されます。

## 出典とライセンス

- **本リポジトリの独自部分**：Apache License 2.0（ルートの `LICENSE` を参照）。第三者由来のコード・データ・モデル重みには以下の各ライセンスが適用されます。
- [MHEALTH](https://archive.ics.uci.edu/dataset/319/mhealth%2Bdataset)：CC BY 4.0。Banos, O., Garcia, R., & Saez, A. (2014). *MHEALTH* [Dataset]. UCI Machine Learning Repository. DOI: [10.24432/C5TW22](https://doi.org/10.24432/C5TW22)。配布READMEの引用依頼：Banos et al. (2014), “mHealthDroid: A Novel Framework for Agile Development of Mobile Health Applications,” IWAAL 2014。データは同梱せず利用者が原典から取得します。
- [SensorLLM](https://github.com/cruiseresearchgroup/SensorLLM/blob/ddd17fecb508fcfa5ae156dbd37902e3ddbf652a/mhealth_stage2.ipynb)：MIT License（Copyright (c) 2024 cruiseresearchgroup）。列選択・連続ラベル区間・窓切りを改変利用。末尾窓を追加せず、subject10をvalへ留保、評価stride100に変更しています。全文は `third_party/SensorLLM-LICENSE.txt`。
- [Google Gemma cookbook](https://github.com/google-gemma/cookbook/blob/5c52c7250feff52e0a7cc526bf32edd646bc3d6b/docs/core/huggingface_text_finetune_qlora.ipynb)：Apache-2.0（Copyright 2025 Google LLC）。PEFT＋TrainerのFT例を参考に、4bitを使わず独自波形入力・回答lossへ変更。全文は `third_party/Gemma-cookbook-LICENSE.txt`。
