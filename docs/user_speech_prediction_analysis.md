# Moshi推論プロセスとユーザー発話予測の分析レポート

## 概要

本ドキュメントは、Moshiの推論プロセスを詳細に分析し、システム発話だけでなく未来のユーザー発話も予測する方法の実装可能性を検討したものです。

## 1. 現在のMoshiアーキテクチャ

### 1.1 デュアルストリームアーキテクチャ

Moshiは以下の2つの音声ストリームを同時に処理します:

- **ユーザーストリーム**: ユーザーの発話（入力）
- **システムストリーム**: Moshiの応答（生成出力）

加えて、**インナーモノローグ（内部独白）**として、システムの発話に対応するテキストトークンを予測します。

### 1.2 推論プロセスフロー

```
ユーザー音声 (PCM 24kHz)
    ↓
[Mimi Encoder] → オーディオコード [B, 32, T']  (12.5Hz)
    ↓
[LMGen.step()]
    ├─ [Main Transformer] → テキストロジット
    │   └─ sample_token() → テキストトークン（インナーモノローグ）
    └─ [Depformer] → オーディオロジット
        └─ sample_token() → オーディオトークン（8 codebooks）
    ↓
[キャッシュ管理] → max_delay待機後に出力
    ↓
[Mimi Decoder] → システム音声 (PCM 24kHz)
```

### 1.3 コードブック構造

参照: `moshi/models/lm.py:668-783`

```python
# Codebook構成:
# - Index 0: テキストストリーム（インナーモノローグ）
# - Index 1-8: システムのオーディオコードブック（Depformerで生成）
# - Index 9+: ユーザーのオーディオコードブック（入力から）

delays = [0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1]
#         ^text  ^システムaudio    ^ユーザーaudio
```

## 2. 現在の予測メカニズム

### 2.1 システム発話の予測（実装済み）

参照: `moshi/models/lm.py:726-772`

**テキスト予測** (インナーモノローグ):
```python
transformer_out, text_logits = state.graphed_main(input_, ...)
text_token = sample_token(text_logits, temp=0.7, top_k=25)
```

**オーディオ予測** (Depformer):
```python
audio_tokens = state.graphed_depth(text_token, transformer_out)
# 8つのオーディオコードブックを順次生成
```

### 2.2 ユーザー発話の扱い（現状）

参照: `moshi/models/lm.py:683-696`

```python
# ユーザーのオーディオコードは「入力」として扱われる
needed_tokens = lm_model.num_codebooks - lm_model.dep_q - 1

# ユーザーコードをキャッシュに書き込み
scatter_with_mask_(state.cache[:, lm_model.dep_q + 1:], -1,
                   write_positions, input_tokens, state.exec_mask)
```

**現在の動作**:
- ユーザーのオーディオコードは**リアルタイムで入力**される
- モデルは**予測せず、実際の入力値を使用**
- これらは**コンテキスト**としてのみ機能

## 3. ユーザー発話予測の可能性

### 3.1 理論的基盤

参照: `moshi/models/lm.py:322-377`

訓練時の処理:
```python
def forward(self, codes: torch.Tensor, ...) -> LMOutput:
    # codes [B, K, T]: 全コードブック（テキスト+全オーディオ）
    # 訓練時は、ユーザーとシステムの両方の発話が含まれる

    delayed_codes = _delay_sequence(self.delays, codes, initial)
    transformer_out, text_logits = self.forward_text(delayed_codes[:, :, :-1], ...)
    logits = self.forward_depformer_training(delayed_codes[:, :, 1:], transformer_out)
    # ↑ 全コードブックのロジットを計算（ユーザーストリームを含む）
```

**重要な発見**:
- 訓練時、モデルは**全コードブックの次トークン予測**を学習
- これには**ユーザーストリーム**のトークンも含まれる
- つまり、モデルは理論的に**ユーザー発話を予測する能力**を持つ

### 3.2 現在の制約

1. **推論時の設計**: `LMGen._step()`は、ユーザーコードを入力パラメータとして受け取る
2. **コードブックの分離**: Depformerはシステムストリーム用の`dep_q`個のみ生成
3. **実用的課題**: ユーザー発話は予測不可能な要素が多い

## 4. 実装アプローチ

### アプローチ1: 投機的デコーディング（Speculative Prediction）

**概念**: 未来のユーザー発話を仮説として予測し、実際の入力と比較

**実装案**:

```python
class LMGen:
    def step_with_user_prediction(
        self,
        input_tokens: torch.Tensor,
        predict_user_future: bool = True
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Returns:
            output_tokens: 通常の出力（テキスト+システムオーディオ）
            user_predictions: 未来のユーザートークン予測 [B, K_user, T_future]
        """
        out = self._step(input_tokens)

        if predict_user_future and out is not None:
            _, transformer_out = out
            user_predictions = self._predict_user_stream(transformer_out)
            return out[0], user_predictions

        return out[0] if out else None, None

    def _predict_user_stream(self, transformer_out: torch.Tensor) -> torch.Tensor:
        """ユーザーストリームのコードブックを予測"""
        # 新しいDepformer分岐でユーザーストリームを予測
        ...
```

**必要な変更**:

1. モデルアーキテクチャへの追加 (`moshi/models/lm.py`):
```python
def __init__(self, ...):
    # ユーザー予測用のDepformerを追加
    self.user_depformer = StreamingTransformer(
        d_model=depformer_dim,
        ...
    )
    self.user_linears = nn.ModuleList([
        nn.Linear(depformer_dim, card, bias=bias_proj)
        for _ in range(n_q_user)
    ])
```

2. 訓練時の損失関数:
```python
# ユーザーストリーム予測の損失を追加
user_loss = F.cross_entropy(
    user_logits.reshape(-1, card),
    user_codes.reshape(-1),
    ignore_index=pad_token_id
)
total_loss = system_loss + alpha * user_loss  # alpha: 重み係数
```

### アプローチ2: Extra Headsの活用

**概念**: 既存の`extra_heads`機能を補助タスクとして利用

参照: `moshi/models/lm.py:794-807`

```python
def step_with_user_prediction_heads(
    self,
    input_tokens: torch.Tensor
) -> tuple[torch.Tensor | None, list[torch.Tensor]]:
    """extra_headsを使ってユーザー発話特徴を予測"""
    out = self._step(input_tokens)
    if out is None:
        return None, []

    _, transformer_out = out

    user_predictions = [
        F.softmax(extra_head(transformer_out), dim=-1)
        for extra_head in self.lm_model.user_extra_heads
    ]

    return out[0], user_predictions
```

**メリット**:
- 既存のインフラを活用
- モデルの大幅な変更不要
- 段階的な実装が可能

### アプローチ3: バックチャネル予測（限定的ユーザー予測）

**概念**: 完全な発話ではなく、バックチャネル（相槌、間投詞など）のみを予測

```python
def predict_backchannel(
    self,
    transformer_out: torch.Tensor
) -> torch.Tensor:
    """
    バックチャネルの有無と種類を予測
    Returns: [B, num_backchannel_types]
    """
    # 簡単な分類タスク
    # 例: [silence, "uh-huh", "yeah", "hmm", ...]
    return self.backchannel_classifier(transformer_out.mean(dim=1))
```

**メリット**:
- 限定的で予測しやすいタスク
- 対話の自然さ向上に貢献
- 誤予測のリスクが低い

## 5. 技術的課題と解決策

### 課題1: 予測精度の低さ

**問題**: ユーザー発話は本質的に予測困難

**解決策**:
- 信頼度スコアの導入
- アンサンブル予測
- 短期予測に限定（1-2ステップ先のみ）

### 課題2: レイテンシの増加

**問題**: ユーザー発話予測は計算コストが高い

**解決策**:
- 並列処理: システム発話生成と並行して予測
- 軽量モデル: 専用の小型予測モデル
- 選択的予測: 重要なターンポイントでのみ実行

### 課題3: 訓練データの要件

**問題**: ユーザー予測を明示的に訓練していない

**解決策**:
- 自己教師あり学習: 未来のユーザートークンをマスクして予測
- マルチタスク学習: システム生成とユーザー予測を同時訓練

## 6. 実装の優先順位

### Phase 1: 基礎調査（1-2週間）
1. `extra_heads`を使った簡単な実験
2. ユーザートークンの予測可能性の検証
3. 既存モデルの潜在能力の評価

### Phase 2: 限定的実装（2-4週間）
1. バックチャネル予測の実装
2. 1-2ステップ先の短期予測
3. 信頼度スコアの追加

### Phase 3: 完全実装（1-3ヶ月）
1. 専用のユーザー予測Depformerの追加
2. モデルのファインチューニング
3. エンドツーエンドの評価

## 7. コード実装例

### LMModelの拡張

```python
# moshi/models/lm.py への追加

class LMModelWithUserPrediction(LMModel):
    """ユーザー発話予測機能を追加したLMModel"""

    def __init__(self, *args, predict_user_stream: bool = False,
                 user_prediction_steps: int = 1, **kwargs):
        super().__init__(*args, **kwargs)
        self.predict_user_stream = predict_user_stream
        self.user_prediction_steps = user_prediction_steps

        if predict_user_stream:
            # ユーザー予測用のヘッドを追加
            self.user_prediction_heads = nn.ModuleList([
                nn.Linear(self.dim, self.card, bias=False)
                for _ in range(self.n_q - self.dep_q - 1)
            ])

    def forward_user_prediction(
        self,
        transformer_out: torch.Tensor,
        current_user_codes: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            transformer_out: [B, T, dim]
            current_user_codes: [B, K_user, T]
        Returns:
            user_logits: [B, K_user, T_future, card]
        """
        predictions = []
        for head in self.user_prediction_heads:
            logits = head(transformer_out)
            predictions.append(logits)

        return torch.stack(predictions, dim=1)
```

### LMGenの拡張

```python
# moshi/models/lm.py の LMGen クラスへの追加

class LMGen:
    def __init__(self, *args, enable_user_prediction: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.enable_user_prediction = enable_user_prediction

        if enable_user_prediction and isinstance(self.lm_model, LMModelWithUserPrediction):
            self.user_prediction_enabled = True
        else:
            self.user_prediction_enabled = False

    def step_with_user_prediction(
        self,
        input_tokens: torch.Tensor
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """
        通常のstepに加えてユーザー発話予測を返す

        Returns:
            output_tokens: 通常の出力 [B, K_out, 1] or None
            user_predictions: ユーザー予測 [B, K_user, 1, card] or None
        """
        # 通常の推論
        result = self._step(input_tokens)

        if result is None or not self.user_prediction_enabled:
            return None, None

        output_tokens, transformer_out = result

        # ユーザー発話予測
        if hasattr(self.lm_model, 'forward_user_prediction'):
            user_pred_logits = self.lm_model.forward_user_prediction(
                transformer_out,
                input_tokens
            )
            return output_tokens, user_pred_logits

        return output_tokens, None
```

## 8. 評価方法

### 8.1 予測精度の評価

```python
def evaluate_user_prediction(model, test_data):
    """
    ユーザー発話予測の精度を評価

    Metrics:
    - Token-level accuracy
    - Sequence-level accuracy (1-5 steps ahead)
    - Confidence calibration
    """
    total_correct = 0
    total_tokens = 0

    for batch in test_data:
        user_codes, system_codes = batch

        # T-k時点でT時点のユーザーコードを予測
        for k in [1, 2, 3, 5]:
            predictions = model.predict_user_at_offset(
                user_codes[:, :, :-k],
                k_steps_ahead=k
            )
            targets = user_codes[:, :, k:]

            correct = (predictions.argmax(-1) == targets).sum()
            total_correct += correct
            total_tokens += targets.numel()

    accuracy = total_correct / total_tokens
    return accuracy
```

### 8.2 対話品質への影響評価

```python
def evaluate_dialogue_quality_with_prediction(model, dialogue_data):
    """
    ユーザー予測が対話品質に与える影響を評価

    Metrics:
    - Response appropriateness
    - Turn-taking naturalness
    - Interruption handling
    """
    metrics = {
        'latency_reduction': [],
        'response_relevance': [],
        'turn_taking_score': []
    }

    for dialogue in dialogue_data:
        # ユーザー予測ありの推論
        with_pred = model.generate_with_user_prediction(dialogue)

        # ユーザー予測なしの推論（ベースライン）
        without_pred = model.generate_without_user_prediction(dialogue)

        # 比較
        metrics['latency_reduction'].append(
            compute_latency_diff(with_pred, without_pred)
        )
        metrics['response_relevance'].append(
            compute_relevance_score(with_pred, dialogue)
        )

    return metrics
```

## 9. 関連ファイル

### 主要ファイル

- `moshi/models/lm.py`: 言語モデル本体（LMModel, LMGen）
- `moshi/models/compression.py`: Mimiオーディオコーデック
- `moshi/run_inference.py`: 推論エントリーポイント
- `moshi/models/tts.py`: TTS状態機械
- `moshi/utils/sampling.py`: トークンサンプリング

### 関連箇所

- `moshi/models/lm.py:668-783`: `LMGen._step()` メソッド
- `moshi/models/lm.py:322-377`: `LMModel.forward()` メソッド
- `moshi/models/lm.py:224-225, 794-807`: extra_heads実装

## 10. 結論

### 現状の理解

1. Moshiは**デュアルストリーム**でユーザーとシステムの発話を処理
2. 現在は**システム発話のみを予測**（テキスト+オーディオ）
3. ユーザー発話は**入力として受け取り、予測しない**
4. 訓練時には全ストリームを学習しており、**潜在的な予測能力**は存在

### 実装可能性

- **技術的に可能**: モデルアーキテクチャは拡張可能
- **実用性は限定的**: ユーザー発話の予測精度は限界がある
- **有望な用途**:
  - バックチャネル予測
  - 短期予測（1-2ステップ先）
  - 対話戦略の計画
  - ターンテイキングの改善

### 推奨アプローチ

1. **Phase 1**: `extra_heads`で簡単な実験から開始
2. **Phase 2**: バックチャネルや短期予測に焦点を当てる
3. **Phase 3**: 必要に応じて完全な実装を検討

### 次のステップ

1. 既存モデルでのユーザートークン予測可能性の検証
2. バックチャネル予測のプロトタイプ実装
3. ファインチューニング用のデータ準備
4. 評価フレームワークの構築

## 参考文献

- Moshi論文: https://arxiv.org/abs/2410.00037
- ファインチューニングリポジトリ: https://github.com/kyutai-labs/moshi-finetune
- 関連プロジェクト: Hibiki (同時音声翻訳), Delayed Streams Modeling

---

作成日: 2025-11-10
分析者: Claude (Anthropic)
