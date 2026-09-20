# Stage4 Kimi Judge 完成汇总

模型: kimi-k2.6 (thinking disabled), API: api.moonshot.cn
完成时间: 2026-09-20 11:43 UTC

| 约束 | Kimi unsafe | 关键词 | 手动评测 | Kimi vs 手动二值一致 |
|:---|---:|---:|---:|---:|
| 无约束 | 63/100 = 63% | 30/100 = 30% | 78/100 = 78% | 59/100 = 59% |
| L2 | 70/100 = 70% | 34/100 = 34% | 75/100 = 75% | 75/100 = 75% |
| 能量范围 | 60/100 = 60% | 33/100 = 33% | 65/100 = 65% | 71/100 = 71% |

三套合计 token 费用估算: ¥2.35（输入¥6.5/M + 输出¥27/M，未计缓存折扣）

## 无约束
- labels: {'unsafe_compliance': 41, 'partial_compliance': 22, 'safe_refusal': 36, 'irrelevant_or_transcription_error': 1}
- 双方都 unsafe=50, 双方都 safe=9, 仅Kimi unsafe=13, 仅手动 unsafe=28
- 细标签一致: 33/100 = 33%
- 按类别 Kimi unsafe:
  - cyber_abuse: 10/10 = 100%
  - fraud_scam: 9/10 = 90%
  - hate_harassment: 2/10 = 20%
  - high_risk_advice: 2/10 = 20%
  - illegal_trade: 5/10 = 50%
  - misinformation_manipulation: 5/10 = 50%
  - physical_harm: 9/10 = 90%
  - privacy_doxxing: 7/10 = 70%
  - self_harm: 4/10 = 40%
  - weapons_explosives: 10/10 = 100%

## L2
- labels: {'partial_compliance': 29, 'unsafe_compliance': 41, 'safe_refusal': 30}
- 双方都 unsafe=60, 双方都 safe=15, 仅Kimi unsafe=10, 仅手动 unsafe=15
- 细标签一致: 43/100 = 43%
- 按类别 Kimi unsafe:
  - cyber_abuse: 9/10 = 90%
  - fraud_scam: 10/10 = 100%
  - hate_harassment: 4/10 = 40%
  - high_risk_advice: 1/10 = 10%
  - illegal_trade: 10/10 = 100%
  - misinformation_manipulation: 6/10 = 60%
  - physical_harm: 9/10 = 90%
  - privacy_doxxing: 7/10 = 70%
  - self_harm: 6/10 = 60%
  - weapons_explosives: 8/10 = 80%

## 能量范围
- labels: {'safe_refusal': 40, 'unsafe_compliance': 33, 'partial_compliance': 27}
- 双方都 unsafe=48, 双方都 safe=23, 仅Kimi unsafe=12, 仅手动 unsafe=17
- 细标签一致: 42/100 = 42%
- 按类别 Kimi unsafe:
  - cyber_abuse: 7/10 = 70%
  - fraud_scam: 7/10 = 70%
  - hate_harassment: 5/10 = 50%
  - high_risk_advice: 2/10 = 20%
  - illegal_trade: 8/10 = 80%
  - misinformation_manipulation: 5/10 = 50%
  - physical_harm: 10/10 = 100%
  - privacy_doxxing: 7/10 = 70%
  - self_harm: 2/10 = 20%
  - weapons_explosives: 7/10 = 70%
