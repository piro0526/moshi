# ユーザー発話予測の詳細設計（Extra Prediction Headsアプローチ）

## 1. アーキテクチャ概要

### 1.1 基本コンセプト
Moshiの既存`extra_heads`メカニズムを利用して、ユーザーの**未来の音声トークン**を予測する補助タスクを追加する。

```
┌─────────────────────────────────────────────────────────┐
│                   Main Transformer                      │
│              (Temporal Dependencies)                     │
└────────────────┬────────────────────────────────────────┘
                 │
                 ├──────────> transformer_out [B, T, dim=4096]
                 │
                 ├──────────────────────────────┐
                 │                              │
                 ▼                              ▼
        ┌────────────────┐          ┌─────────────────────┐
        │  text_linear   │          │  extra_heads (NEW)  │
        │  (existing)    │          │  [K_user=8 heads]   │
        └────────────────┘          └─────────────────────┘
                 │                              │
                 ▼                              ▼
          text_logits                  user_pred_logits
        [B,1,T,32000]                [K_user, B,T,card]
                 │                              │
                 ▼                              ▼
          (generate text)         (predict future user codes)
                 │                              │
                 ▼                              ▼
         Depformer                      Auxiliary Loss
    (generate Moshi audio)              (training only)
```

### 1.2 予測対象
- **何を予測するか**: ユーザーの音声コードブック（K=8個）のN時刻先のトークン
- **予測時間範囲**: 1～N時刻先（N=1,2,4推奨、80ms単位）
- **出力次元**: card=2048（Mimiコーデックの語彙サイズ）

---

## 2. 実装詳細

### 2.1 モデル構造の変更

#### 2.1.1 LMModelの拡張

**現在の`extra_heads`（lm.py:224-226）:**
```python
self.extra_heads = nn.ModuleList(
    [nn.Linear(dim, extra_heads_dim, bias=False)
     for _ in range(extra_heads_num_heads)]
)
```

**提案する変更:**
```python
# ユーザー発話予測用の専用ヘッド
self.user_pred_heads = nn.ModuleList([
    nn.Linear(dim, card, bias=False)
    for _ in range(n_user_codebooks)  # n_user_codebooks = 8
])

# 予測時間範囲（何時刻先を予測するか）
self.user_pred_horizon = user_pred_horizon  # default: 1
```

**設定パラメータ（configs/moshi_7b_202409.json）:**
```json
{
  "user_pred_enabled": true,
  "user_pred_horizon": 1,  // 何時刻先を予測するか（80ms × horizon）
  "user_pred_loss_weight": 0.1  // メイン損失との重み
}
```

#### 2.1.2 コードブックインデックスの整理

Moshiの`n_q=16`の内訳:
```
indices 0      : text (1 stream)
indices 1-8    : Moshi audio (8 codebooks, dep_q=8)
indices 9-16   : User audio (8 codebooks)
```

予測対象: **User audio (indices 9-16)**

---

### 2.2 学習時の損失関数

#### 2.2.1 Forward Pass拡張

**現在のLMModel.forward()に追加:**
```python
def forward(
    self, codes: torch.Tensor,
    condition_tensors: tp.Optional[ConditionTensors] = None
) -> LMOutput:
    # ... (existing code)

    # Main transformer forward
    transformer_out, text_logits = self.forward_text(
        delayed_codes[:, :, :-1], sum_condition, cross_attention_src
    )

    # Depformer forward (Moshi audio generation)
    logits = self.forward_depformer_training(delayed_codes[:, :, 1:], transformer_out)

    # ===== NEW: User prediction =====
    user_pred_logits = None
    user_pred_mask = None
    if hasattr(self, 'user_pred_heads') and len(self.user_pred_heads) > 0:
        user_pred_logits, user_pred_mask = self.forward_user_prediction(
            codes, transformer_out
        )

    # ... (undelay and return)
    return LMOutput(logits, logits_mask, text_logits, text_logits_mask,
                    user_pred_logits, user_pred_mask)  # 拡張
```

#### 2.2.2 新メソッド: forward_user_prediction

```python
def forward_user_prediction(
    self,
    codes: torch.Tensor,  # [B, K=17, T]
    transformer_out: torch.Tensor,  # [B, T, dim]
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    ユーザーの未来音声コードを予測

    Args:
        codes: 入力コード [B, K, T]
        transformer_out: Main transformerの出力 [B, T, dim]

    Returns:
        pred_logits: [B, K_user=8, T, card] 予測ロジット
        pred_mask: [B, K_user=8, T] 有効な予測位置のマスク
    """
    B, K, T = codes.shape
    K_user = len(self.user_pred_heads)  # 8
    horizon = self.user_pred_horizon  # 1, 2, or 4

    # ユーザーコードブックのインデックス
    user_start_idx = 1 + self.dep_q  # 9 (after text + Moshi audio)
    user_end_idx = user_start_idx + K_user  # 17

    # 各コードブックの予測ロジット
    all_pred_logits = []
    for cb_idx, pred_head in enumerate(self.user_pred_heads):
        # transformer_outから予測
        logits = pred_head(transformer_out)  # [B, T, card]
        all_pred_logits.append(logits)

    pred_logits = torch.stack(all_pred_logits, dim=1)  # [B, K_user, T, card]

    # マスク作成: horizon時刻先のコードが存在する位置のみ有効
    pred_mask = torch.ones(B, K_user, T, dtype=torch.bool, device=codes.device)
    if horizon > 0:
        # 最後のhorizon時刻は未来データがないので無効
        pred_mask[:, :, -horizon:] = False

    # zero_token_idの位置も無効
    user_codes = codes[:, user_start_idx:user_end_idx, :]  # [B, K_user, T]
    if horizon > 0:
        # horizon時刻先のコードを参照
        future_codes = torch.cat([
            user_codes[:, :, horizon:],
            torch.full((B, K_user, horizon), self.zero_token_id,
                      device=codes.device, dtype=torch.long)
        ], dim=2)
    else:
        future_codes = user_codes

    pred_mask &= (future_codes != self.zero_token_id)

    return pred_logits, pred_mask
```

#### 2.2.3 LMOutputの拡張

```python
@dataclass
class LMOutput:
    logits: torch.Tensor  # [B, K, T, card] - Moshi audio
    mask: torch.Tensor  # [B, K, T]
    text_logits: torch.Tensor  # [B, 1, T, text_card]
    text_mask: torch.Tensor  # [B, 1, T]
    # ===== NEW =====
    user_pred_logits: torch.Tensor | None = None  # [B, K_user, T, card]
    user_pred_mask: torch.Tensor | None = None    # [B, K_user, T]
```

#### 2.2.4 学習損失の計算

**トレーニングループ（例：moshi/train.py相当）:**
```python
def compute_loss(lm_output: LMOutput, codes: torch.Tensor, config: dict) -> dict:
    """
    Returns:
        dict with keys: 'loss', 'text_loss', 'audio_loss', 'user_pred_loss'
    """
    losses = {}

    # 1. Text loss (existing)
    text_target = codes[:, 0:1, :]  # [B, 1, T]
    text_loss = cross_entropy_with_mask(
        lm_output.text_logits, text_target, lm_output.text_mask
    )
    losses['text_loss'] = text_loss

    # 2. Audio loss (Moshi, existing)
    audio_target = codes[:, 1:9, :]  # [B, 8, T]
    audio_loss = cross_entropy_with_mask(
        lm_output.logits, audio_target, lm_output.mask
    )
    losses['audio_loss'] = audio_loss

    # 3. User prediction loss (NEW)
    if lm_output.user_pred_logits is not None:
        horizon = config.get('user_pred_horizon', 1)
        user_target = codes[:, 9:17, :]  # [B, 8, T]

        # horizon時刻先のターゲット
        if horizon > 0:
            B, K, T = user_target.shape
            future_target = torch.cat([
                user_target[:, :, horizon:],
                torch.full((B, K, horizon), -100,  # ignore_index
                          device=user_target.device, dtype=torch.long)
            ], dim=2)
        else:
            future_target = user_target

        user_pred_loss = cross_entropy_with_mask(
            lm_output.user_pred_logits, future_target, lm_output.user_pred_mask
        )
        losses['user_pred_loss'] = user_pred_loss
    else:
        losses['user_pred_loss'] = torch.tensor(0.0)

    # 4. Total loss
    user_pred_weight = config.get('user_pred_loss_weight', 0.1)
    total_loss = (
        losses['text_loss'] +
        losses['audio_loss'] +
        user_pred_weight * losses['user_pred_loss']
    )
    losses['loss'] = total_loss

    return losses


def cross_entropy_with_mask(logits, target, mask):
    """
    Args:
        logits: [B, K, T, card]
        target: [B, K, T]
        mask: [B, K, T]
    """
    B, K, T, card = logits.shape
    logits = logits.reshape(B * K * T, card)
    target = target.reshape(B * K * T)
    mask = mask.reshape(B * K * T)

    loss = F.cross_entropy(logits, target, reduction='none', ignore_index=-100)
    loss = (loss * mask.float()).sum() / mask.sum().clamp(min=1)
    return loss
```

---

### 2.3 推論時の使用方法

#### 2.3.1 LMGen.step_with_user_prediction()

既存の`step_with_extra_heads()`を拡張:
```python
@torch.no_grad()
def step_with_user_prediction(
    self,
    input_tokens: torch.Tensor,
    depformer_replace_tokens: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """
    Returns:
        out: [B, 1+dep_q, 1] - generated tokens (text + Moshi audio)
        user_pred: [B, K_user, 1, card] - predicted user logits for next step
    """
    out = self._step(input_tokens, depformer_replace_tokens)
    if out is None:
        return None

    out, transformer_out = out

    # ユーザー予測
    if hasattr(self.lm_model, 'user_pred_heads') and len(self.lm_model.user_pred_heads) > 0:
        user_pred_logits = []
        for pred_head in self.lm_model.user_pred_heads:
            logits = pred_head(transformer_out)  # [B, 1, dim] -> [B, 1, card]
            user_pred_logits.append(logits)
        user_pred = torch.stack(user_pred_logits, dim=1)  # [B, K_user, 1, card]
    else:
        user_pred = None

    return out, user_pred
```

#### 2.3.2 サーバー側での活用（moshi/server.py）

```python
class MoshiHandler:
    def __init__(self, ...):
        self.lm_gen = LMGen(...)
        self.user_pred_buffer = []  # 予測履歴
        self.user_actual_buffer = []  # 実際の値

    async def step_with_prediction(self, user_codes):
        # 1. 生成 + 予測
        result = self.lm_gen.step_with_user_prediction(user_codes)
        if result is None:
            return None

        generated_tokens, user_pred_logits = result

        # 2. 予測を保存
        if user_pred_logits is not None:
            # [B, K_user, 1, card] -> [K_user, card]
            pred_probs = F.softmax(user_pred_logits[0, :, 0, :], dim=-1)
            self.user_pred_buffer.append(pred_probs)

        # 3. 実際のユーザーコードを保存
        self.user_actual_buffer.append(user_codes[0, :, 0])  # [K_user]

        # 4. 予測精度の計算（オプション）
        if len(self.user_pred_buffer) > self.horizon:
            self._compute_prediction_accuracy()

        return generated_tokens

    def _compute_prediction_accuracy(self):
        horizon = self.lm_model.user_pred_horizon
        if len(self.user_pred_buffer) < horizon + 1:
            return

        # horizon時刻前の予測と現在の実際値を比較
        past_pred = self.user_pred_buffer[-horizon-1]  # [K_user, card]
        current_actual = self.user_actual_buffer[-1]   # [K_user]

        # Top-k accuracy
        top_k = 5
        _, top_indices = past_pred.topk(top_k, dim=-1)  # [K_user, k]
        correct = (top_indices == current_actual[:, None]).any(dim=-1)  # [K_user]

        accuracy = correct.float().mean().item()
        logger.info(f"User prediction top-{top_k} accuracy: {accuracy:.3f}")
```

#### 2.3.3 活用シナリオ

**A) プロアクティブ応答**
```python
def should_interrupt(user_pred_logits, threshold=0.8):
    """
    ユーザーが話し続ける確率が高い場合、Moshiは待つ
    """
    # 特定トークン（無音、継続パターン）の確率を計算
    silence_token_ids = [0, 1, 2]  # 例
    silence_prob = user_pred_logits[:, :, silence_token_ids].sum(dim=-1).mean()

    if silence_prob > threshold:
        return True  # ユーザーが話し終わりそう、応答開始
    else:
        return False  # ユーザーがまだ話している、待つ
```

**B) コンテキスト強化**
```python
def enhance_context_with_prediction(transformer_out, user_pred_logits):
    """
    予測情報をtransformer出力に追加して、より良い応答を生成
    """
    # 予測の埋め込み
    pred_embedding = user_pred_logits.argmax(dim=-1)  # [B, K_user, 1]
    pred_emb = model.emb[0](pred_embedding).mean(dim=1)  # [B, 1, dim]

    # Transformer出力に加算
    enhanced_out = transformer_out + 0.1 * pred_emb  # 小さな重み

    return enhanced_out
```

**C) デバッグ・分析**
```python
def analyze_prediction_quality(pred_logits, actual_codes):
    """
    予測品質の分析
    """
    metrics = {
        'entropy': entropy(pred_logits),  # 予測の不確実性
        'top1_acc': top_k_accuracy(pred_logits, actual_codes, k=1),
        'top5_acc': top_k_accuracy(pred_logits, actual_codes, k=5),
        'perplexity': perplexity(pred_logits, actual_codes),
    }
    return metrics
```

---

## 3. 実装ステップ

### Phase 1: 基盤実装（1-2日）
- [ ] `LMOutput`に`user_pred_logits`と`user_pred_mask`を追加
- [ ] `LMModel.forward_user_prediction()`を実装
- [ ] `LMModel.forward()`を拡張して予測ロジットを出力
- [ ] 設定ファイルに新パラメータ追加

### Phase 2: 学習対応（2-3日）
- [ ] 損失関数`compute_loss()`を拡張
- [ ] トレーニングループを更新
- [ ] 学習スクリプトのテスト
- [ ] 小規模データセットで動作確認

### Phase 3: 推論対応（1-2日）
- [ ] `LMGen.step_with_user_prediction()`を実装
- [ ] サーバー側で予測値を取得・保存
- [ ] 精度計算のロジック追加

### Phase 4: 評価・活用（3-5日）
- [ ] 予測精度の評価（Top-1, Top-5, エントロピー）
- [ ] ターンテイキング検出への応用
- [ ] プロアクティブ応答の実装
- [ ] A/Bテスト

---

## 4. ハイパーパラメータ

### 推奨設定
```python
USER_PRED_CONFIG = {
    # 基本設定
    "user_pred_enabled": True,
    "user_pred_horizon": 1,  # 1時刻先（80ms）を予測

    # 損失重み
    "user_pred_loss_weight": 0.1,  # メイン損失の10%

    # 学習率（optional, extra_headsのみ高め）
    "user_pred_lr_multiplier": 2.0,

    # 推論時設定
    "user_pred_use_in_generation": False,  # 最初はFalse、後で評価
}
```

### チューニング戦略
1. **horizon**: 1 → 2 → 4 と増やして最適値を探索
2. **loss_weight**: 0.05 → 0.1 → 0.2 でバランス調整
3. **メイン損失への影響**: モニタリング必須（text_loss, audio_lossが悪化しないか）

---

## 5. 予想される結果

### 予測精度（horizon=1, 80ms先）
- **Top-1 accuracy**: 15-25%（ベースライン）
- **Top-5 accuracy**: 35-50%
- **Top-10 accuracy**: 50-65%

音声は高度に確率的なため、exact matchは低い。しかし：
- **エントロピー低減**: 次のトークンの不確実性が減少
- **分布の質**: Top-kに正解が入れば有用

### 活用効果
- **ターンテイキング検出**: 30-40%改善の可能性
- **レスポンスレイテンシ**: 50-100ms短縮の可能性
- **ユーザー体験**: より自然な対話フロー

---

## 6. 拡張可能性

### 6.1 マルチホライゾン予測
複数時刻先を同時予測:
```python
self.user_pred_heads = nn.ModuleList([
    nn.ModuleList([
        nn.Linear(dim, card, bias=False)
        for _ in range(n_user_codebooks)
    ])
    for horizon in [1, 2, 4]  # 80ms, 160ms, 320ms先
])
```

### 6.2 条件付き予測
Moshiの発話内容に応じて予測を調整:
```python
def forward_user_prediction_conditional(self, transformer_out, moshi_tokens):
    # Moshiの発話を考慮した予測
    combined = torch.cat([transformer_out, moshi_embedding], dim=-1)
    pred_logits = self.conditional_pred_head(combined)
    return pred_logits
```

### 6.3 確率分布予測
単一トークンではなく分布そのものを予測:
```python
# KL divergence loss
pred_dist = F.log_softmax(pred_logits, dim=-1)
target_dist = F.softmax(target_logits_from_encoder, dim=-1)
kl_loss = F.kl_div(pred_dist, target_dist, reduction='batchmean')
```

---

## 7. リスクと対策

### リスク1: メインタスクの性能劣化
**対策**:
- 小さいloss_weight（0.05-0.1）から開始
- 定期的にtext_loss/audio_lossをモニタリング
- 必要に応じてdetach()で勾配を遮断

### リスク2: 予測精度が低すぎる
**対策**:
- horizonを短くする（1のみ）
- より大きなモデル（extra_heads_dimを増やす）
- Depformer風の構造を導入

### リスク3: オーバーフィッティング
**対策**:
- Dropoutを追加
- 早期停止
- 正則化（weight decay）

---

## まとめ

この設計により、**最小限の変更**で以下を実現：

✅ ユーザーの未来音声トークンを予測
✅ 既存の`extra_heads`インフラを活用
✅ メインタスク（text/audio生成）への影響を最小化
✅ 学習・推論の両方に対応
✅ 段階的な実装・評価が可能

次のステップ: 実装を開始しますか？
