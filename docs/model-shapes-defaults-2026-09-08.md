# FoldJAX 모델별 기본 tensor shape 분석

작성일: 2026-09-08. 기준: 현재 작업 트리의 managed `released` checkpoint와
기본 추론 옵션. Git HEAD는 `c44a116f97214fbc63c1de4b6253d1b55b10e819`이며,
아래 padding 정책을 포함한 미커밋 변경이 있으므로 HEAD 단독의 설명은 아니다.

이 문서는 **기본 모델/추론 설정 + `--padding` ON**을 기준으로 한다.
`--padding` 자체는 여전히 기본 OFF다. OFF에서는 실제 입력과 native 전처리에
따라 축 길이가 달라지고, 아래의 채널 수와 학습된 block 구조는 유지된다.
특히 MSA OFF-path 기본값은 이 표의 padded MSA 용량과 같다고 가정하면 안 된다.

## 표기와 공통 축

| 기호 | 의미 |
| --- | --- |
| `N`, `a` | 실제 model token 수, 실제 atom 수. 단백질 잔기 수와 항상 같지는 않음 |
| `T` | token bucket: 256, 512, 768, 1024, 1536, 2048, 3072, 4096 |
| `A` | `32 × ceil(24T/32)`; 표준 bucket에서는 `24T` |
| `M` | OpenDDE 1280, 나머지 모델 1024. 실제 유효 행 수가 아니라 padded 용량 |
| `U` | OpenDDE structural token 용량 `2T` |
| `L` | ESMFold2 packed LM 용량 `3T`; chain별 BOS/EOS 포함 |
| `B` | 입력 batch. 일반 단일 job에서는 1이며 아래 비교 표에서는 생략 |
| `S` | 한 seed의 diffusion sample 수. 입력 batch와 별개 |
| `s`, `z`, `m` | token single, token-pair, MSA feature |

기본 MSA 용량에 못 미치는 행은 mask 처리한다. 더 깊은 MSA에는 기존 native
crop/sampling 규칙과 현재 입력 cap이 적용된다. `--pad-*` 또는 실행 옵션을
명시하면 별도 profile이 될 수 있다. 여러 seed는 대체로 별도 실행으로 처리되며
항상 tensor의 추가 batch 축으로 쌓이는 것은 아니다.

## 핵심 차원 비교

아래 shape는 batch 1을 생략한 의미상 shape다. Sample chunking, XLA fusion,
compact feature 저장 때문에 모든 tensor가 이 형태로 동시에 materialize되지는 않는다.

| 모델 | 입력 single | trunk single | trunk pair | MSA hidden | 주요 trunk blocks | diffusion token |
| --- | --- | --- | --- | --- | --- | --- |
| AlphaFold 3 | `[T,447]` | `[T,384]` | `[T,T,128]` | `[1024,T,64]` | Pairformer 48 | `[S,T,768]` |
| Boltz2 | `[T,384]` | `[T,384]` | `[T,T,128]` | `[1024,T,64]` | Pairformer 64 | `[S,T,768]` |
| Protenix released base | `[T,449]` | `[T,384]` | `[T,T,128]` | `[1024,T,64]` | Pairformer 48 | `[S,T,768]` |
| OpenDDE | `[T,449]` | `[T,384]` | `[T,T,384]` | `[1280,T,128]` | Pairformer 48 + structural refiner 4 | `[S,U,768]` |
| OpenFold3 OpenBind | `[T,449]` | `[T,384]` | `[T,T,128]` | `[1024,T,64]` | Pairformer 48 | `[S,T,768]` |
| ESMFold2 released | `[T,451]` | 별도 갱신 single 없음; 입력 stream 451 유지 | `[T,T,256]` | `[1024,T,128]` | pair trunk 48 + coda 2 | `[S,T,768]` |

ESMFold2 checkpoint의 `d_single=384`를 그대로 trunk single shape로 적으면
부정확하다. 실제 구조 경로와 공개 `single` representation은 `x_inputs[...,451]`다.
384는 atom encoder의 token projection 등에 사용된다.

## 기본 반복 횟수와 샘플 수

Recycle은 [모델별 논문 기준 정책](recycling-defaults.md)을 반영했다.
AF3는 명시적 선택에 따라 Algorithm 1의 총 4회 실행을 사용한다.
Diffusion steps/samples는 기존 checkpoint 기본값이며, 전체 논문 재현 설정은 아니다.

| 모델 | 기본 recycle 설정 | 실제 주요 trunk 실행 횟수 | diffusion steps | 한 seed의 samples `S` |
| --- | --- | --- | --- | --- |
| AlphaFold 3 | 3 | 4 | 200 | 5 |
| Boltz2 | 5 | 6 | 200 | 1 |
| Protenix released base | 10 | 10 | 200 | 5 |
| OpenDDE | 10 | 10 | 200 | 5 |
| OpenFold3 | native 추가 recycling 3; 내부 config 4 | 4 | 200 | 5 |
| ESMFold2 | 9 | 10 | 14 | 32 |

AF3/Boltz2/ESMFold2는 코드에서 recycle 설정에 1을 더한다. Protenix/OpenDDE는
그 횟수만큼 loop를 돈다. OpenFold3는 adapter가 추가 recycling 수를 내부 실행
횟수로 변환한다. 같은 `num_recycles` 숫자를 넣어도 계산 횟수가 같지 않다.
Block 수는 서로 다른 학습된 layer 수이며 recycle마다 새로운 가중치가 생기는 것은 아니다.
ESMFold2 coda 2 blocks와 OpenDDE structural refiner 4 blocks는 주요 residue
trunk 반복 뒤의 별도 단계다.

## 1. AlphaFold 3

```text
atom [T,24,3] + atom mask [T,24]
  → local atom encoder: atom 128, local pair 16
  → atom-to-token [T,384]
  → concat restype 31 + MSA profile 31 + deletion 1
  → input single [T,447]
  → initial single [T,384], pair [T,T,128]
  → MSA [1024,T,64], MSA blocks 4
  → Pairformer blocks 48, trunk 전체 4회
  → diffusion token [S,T,768], blocks 24
  → atom decoder → coordinates [S,T,24,3]
  → confidence Pairformer 4 → atom confidence / token-pair confidence
```

- Local atom encoder/decoder: 각각 3 blocks, 4 heads. Query 32, key 128.
- Flat atom window의 pair conditioning은 `[A/32,32,128,16]`이다.
  Dense `[A,A,16]`을 의미하지 않는다.
- Pairformer single attention 16 heads, triangle attention 4 heads.
- Template 최대 4개, template pair channel 64, template blocks 2.
- Diffusion token attention 16 heads. Single/pair conditioning 폭은 384/128.
- Confidence pLDDT logits: 의미상 `[S,T,24,50]`, expectation은 `[S,T,24]`.
  PAE/PDE logits는 `[S,T,T,64]`, expectation은 `[S,T,T]`.
- 최종 구조 writer는 유효 atom만 출력한다.

근거: `models/alphafold3/_upstream/alphafold3/model/network/evoformer.py:46`,
`models/alphafold3/_upstream/alphafold3/model/model.py:149,373`, `models/alphafold3/_upstream/alphafold3/model/network/atom_cross_attention.py:110`,
`models/alphafold3/_upstream/alphafold3/model/network/diffusion_head.py:130`, `models/alphafold3/_upstream/alphafold3/model/network/confidence_head.py:247`.
여기서 경로는 `src/foldjax/` 기준이다. AF3는 carried code 설정을 확인했으며
restricted checkpoint의 tensor header는 읽지 않았다.

## 2. Boltz2

```text
flat atom [A,3]
  → atom encoder [A,128], local pair width 16
  → atom-to-token [T,384]
  → residue / MSA profile / 조건 embedding을 더함
  → input single [T,384]
  → MSA [1024,T,64], blocks 4
  → single [T,384], pair [T,T,128], Pairformer blocks 64 × 6회
  → diffusion token [S,T,768], blocks 24
  → atom [S,A,128] → coordinates [S,A,3]
  → confidence Pairformer blocks 8
```

- Raw feature와 trunk는 실제 코드에서 batch 축 1을 포함하는 경우가 많다.
- AF3/Protenix처럼 447/449 채널을 concat하는 방식이 아니다. 조건 embedding을
  384채널 atom-derived token feature에 **더한다**.
- Atom encoder/decoder 각각 3 blocks/4 heads. Local atom pair는
  `[A/32,32,128,16]`에 batch/sample 축이 붙는 형태다.
- Pairformer single attention 16 heads, MSA attention 8 heads × value width 32.
- Diffusion token attention 16 heads.
- pLDDT logits `[S,T,50]`: 이 head의 축은 atom이 아니라 token이다.
  PAE/PDE logits `[S,T,T,64]`.
- Affinity는 별도 checkpoint/추론 분기다. 여기서는 기본 structure/confidence
  checkpoint를 분석했으며 affinity branch 전체 shape는 포함하지 않았다.

근거: `models/boltz2/models/trunk_blocks/input_embedder.py:62`,
`models/boltz2/models/trunk_blocks/trunk.py:1212`, `models/boltz2/models/trunk_blocks/msa.py:390`,
`backends/boltz2.py:273`, 실제 `boltz2_conf.safetensors` header.

## 3. Protenix released base

기본 managed `released`는 `protenix_base_default_v1.0.0`이며 **v2가 아니다**.

```text
atom [A,3] → atom encoder [A,128] → token [T,384]
  → concat 32 restype + 32 profile + 1 deletion
  → s_inputs [T,449]
  → MSA [1024,T,64], blocks 4
  → single [T,384], pair [T,T,128], Pairformer blocks 48 × 10회
  → diffusion condition single [T,384], pair [T,T,128]
  → diffusion token [S,T,768], blocks 24
  → atom [S,A,128] → coordinates [S,A,3]
  → confidence Pairformer blocks 4
```

- Atom encoder/decoder 각각 3 blocks/4 heads; local pair width 16.
- Pairformer single attention 16 heads, triangle attention 4 heads.
- Template storage 4 slots, pair channel 64, template blocks 2.
- pLDDT `[S,A,50]`, resolved logits `[S,A,2]`, PAE/PDE `[S,T,T,64]`.
  Distogram은 sample별 diffusion 결과가 아니라 trunk에서 나오는 `[T,T,64]`다.
- 기본 base 경로에는 LM encoder가 없다. LM은 mini ESM/ISM profile에서 사용한다.

근거: `backends/protenix.py:78`, `models/protenix/runtime_policy.py:21`,
`models/protenix/models/trunk_blocks/embedders.py:325`, `models/protenix/models/trunk_blocks/trunk.py:288`,
`models/protenix/models/diffusion/diffusion.py:539`, `models/protenix/models/heads/confidence.py:1291` 및
managed conversion receipt가 가리키는 source checkpoint header.

## 4. OpenDDE

```text
atom [A,3] → atom encoder [A,128] → s_inputs [T,449]
  → MSA [1280,T,128], blocks 4
  → residue single [T,384], pair [T,T,384]
  → residue Pairformer blocks 48 × 10회
       ├─ structural expansion U=2T
       │    → structural inputs [U,449]
       │    → structural single [U,384], pair [U,U,384]
       │    → structural refiner blocks 4
       │    → diffusion pair projection [U,U,384] → [U,U,128]
       │    → diffusion token [S,U,768], blocks 24
       │    → atom decoder → coordinates [S,A,3]
       └─ residue trunk + sampled coordinates
            → confidence at T: single [T,384], pair [T,T,384]
            → confidence Pairformer blocks 4
```

- Residue Pairformer: single attention 16 heads, triangle attention 12 heads.
- Structural refiner: single attention 8 heads, triangle attention 12 heads.
- Structural token role는 7종. 실제 유효 structural token은 U 이하이며 suffix는 mask.
- Diffusion은 **U 축**, confidence는 **T 축**이다. 두 head를 같은 축으로 적으면
  메모리/연산량 추산이 크게 틀린다.
- Atom encoder/decoder 각각 3 blocks/4 heads, atom 128/local pair 16.
- Template storage 4 slots, pair channel 64/2 blocks. Native `use_template`
  기본값은 false이므로 feature capacity와 활성 계산을 구분해야 한다.
- pLDDT `[S,A,50]`, resolved `[S,A,2]`, PAE/PDE `[S,T,T,64]`.
  Distogram은 `[T,T,96]`로 다른 모델들의 64 bins와 다르다.

근거: `models/opendde/bridge/torch_mapping.py:39,178,248`,
`models/opendde/models/structural_tokens.py:144`, `models/opendde/models/model.py:780,845,977`,
`models/opendde/data/featurize_json.py:552`, source checkpoint header.

## 5. OpenFold3 OpenBind

```text
atom [1,A,3] → atom encoder [1,A,128]
  → s_inputs [1,T,449]
  → MSA [1,1024,T,64], blocks 4
  → single [1,T,384], pair [1,T,T,128]
  → Pairformer blocks 48 × 4회
  → diffusion token [S,T,768], blocks 24
  → coordinates [S,A,3]
  → confidence Pairformer blocks 4
```

- Input concat은 384 atom-token + 32 restype + 32 profile + 1 deletion.
- Atom encoder/decoder 각각 3 blocks/4 heads; atom 128/local pair 16.
- Pairformer single attention 16 heads, triangle attention 4 heads.
- Template 4 slots/2 blocks. Empty template geometry는 compact하게 저장될 수 있다.
- 실제 output sample 축은 `[S,A,3]`이며 `[S,1,A,3]`로 batch를 중복해 세지 않는다.
- pLDDT logits는 atom 기준 50 bins, PAE/PDE token-pair 기준 64 bins.
- Padding ON에서는 recycle별 MSA 선택·gather를 CPU에서 수행하고, 회당
  `[1,1024,T]` mask를 갖는 MSA feature만 전달한다. 가변 깊이의 합집합과
  cycle index는 JIT 입력에 포함하지 않는다. 초기 embedding과 diffusion은
  각각 한 번 실행하며 고정 shape의 cycle JIT를 재사용한다.
- OpenBind checkpoint의 confidence slot 수는 **23**이다. Serving atom 용량
  `24T`와 checkpoint의 `23 × 50 = 1150` 출력 projection은 별개다.

근거: `models/openfold3/inference.py:1086,1136`,
`models/openfold3/models/input_embedders.py:100,183`, `models/openfold3/data/featurize.py:205`,
`models/openfold3/models/sampler.py:101` 및 OpenBind checkpoint header.

## 6. ESMFold2

```text
atom encoder 3 blocks: atom128 → token384
  → 33 restype + 33 profile + 1 deletion concat → x_inputs [T,451]

ESMC packed input L=3T
  → 80 layers, hidden2560, heads40
  → native layer stack [81,B,L,2560] (embedding layer 포함)
  → token 위치로 scatter [B,T,81,2560]
  → layer mix + projection → staged LM embedding [B,T,256]
  → LM pair [T,T,256], LM encoder 4 layers

MSA [1024,T,128], MSA encoder 4 layers
  + LM pair / initial pair
  → pair-only folding trunk [T,T,256], 48 layers × 10회
  → coda 2 layers
  → x_inputs [T,451] + pair [T,T,256]
  → diffusion token [S,T,768], 12 blocks
  → atom [S,A,128] → coordinates [S,A,3]
  → confidence pair trunk 4 layers
```

- `d_single=384`라는 config 항목만 보고 384채널 single trunk가 갱신된다고
  해석하면 안 된다. 실제 `single` representation은 `[B,T,451]`이다.
- LM stack은 80이 아니라 embedding 포함 **81**개 state다. Native axis 순서는
  `[81,B,L,2560]`이며 `[B,L,81,2560]`은 의미상 transpose 표기일 뿐이다.
- Managed staging은 layer stack을 축약한 256채널 embedding으로 유지할 수 있다.
  전체 LM stack과 구조 모델이 항상 동시에 상주한다고 가정하면 안 된다.
- Pair trunk 8 heads; MSA 8 heads/head width 16, outer-product hidden 32.
- Atom 단계는 3 blocks/4 heads, sliding window 128. AF3식 local pair16 tensor가
  그대로 존재한다고 일반화하지 않는다.
- Diffusion token 12 blocks/16 heads. 다른 다섯 모델의 24 blocks와 다르다.
- pLDDT logits는 `[S,A,50]`에서 atom confidence를 계산한 뒤 token으로도 집계한다.
  PAE/PDE는 `[S,T,T,64]`; confidence용 distogram은 39 bins,
  structure trunk distogram은 64 bins로 서로 다르다.
- 기본 confidence는 sample을 순차 처리한다. `S=32`라는 논리적 결과 축이
  모든 confidence tensor를 32개 동시에 상주시킨다는 의미는 아니다.
- `structure-only` profile에서는 LM 경로가 빠진다.

근거: managed ESMFold2/ESMC `config.json`과 safetensors header;
`models/esmfold2/models/model.py:296,455,501,1333,1357,1411,1430`,
`models/esmfold2/models/esmc.py:295,394`, `models/esmfold2/models/heads.py:242`.

## T=512일 때의 구체적인 예

표준 atom 용량 A=12288, OpenDDE U=1024, ESMFold2 L=1536이다.
아래에서는 입력 batch 1을 생략했다.

| 모델 | 주요 single | 주요 pair | MSA hidden | diffusion token |
| --- | --- | --- | --- | --- |
| AF3 | `[512,384]` | `[512,512,128]` | `[1024,512,64]` | `[5,512,768]` |
| Boltz2 | `[512,384]` | `[512,512,128]` | `[1024,512,64]` | `[1,512,768]` |
| Protenix base | `[512,384]` | `[512,512,128]` | `[1024,512,64]` | `[5,512,768]` |
| OpenDDE | residue `[512,384]`; structural `[1024,384]` | residue `[512,512,384]`; structural `[1024,1024,384]` | `[1280,512,128]` | `[5,1024,768]` |
| OpenFold3 | `[512,384]` | `[512,512,128]` | `[1024,512,64]` | `[5,512,768]` |
| ESMFold2 | `[512,451]` | `[512,512,256]` | `[1024,512,128]` | `[32,512,768]` |

OpenDDE structural pair는 동일 T에서 128채널 residue pair 대비 원소 수가
`(2T)^2 × 384 / (T^2 × 128) = 12배`다. 이는 한 tensor의 shape 비교이며
모델 전체 VRAM이나 실제 실행 시간 비율이 아니다. Diffusion pair conditioning은
128채널로 다시 projection되므로 이 구분도 유지해야 한다.

## 기본 외 Protenix profile

아래는 config 유래이며 해당 variant checkpoint header는 별도 검증하지 않았다.

- `v2`: single384, pair256, MSA128; 주요 Pairformer48/diffusion24 유지.
- `mini-esm-v0.5.0` / `mini-ism-v0.5.0`: MSA blocks1, Pairformer16,
  diffusion atom/token/decoder blocks1/8/1. 기본 schedule은 recycle4/steps5.
  LM embedding을 입력 single449에 주입한다.

기본값과 최신 출시 profile, 또는 benchmark에서 명시적으로 고정한 schedule은
서로 다르다. 이 문서는 benchmark의 통일된 5 samples/200 steps/10 recycles를
모든 모델의 기본값으로 간주하지 않는다.

## 확인 범위와 재현 근거

- Source code, 현재 ESMFold2/ESMC config, checkpoint의 **shape metadata**를 읽었다.
  추론, GPU 실행, full checkpoint tensor loading은 하지 않았다.
- Boltz2/ESMFold2는 safetensors JSON header로 dimension을 교차 확인했다.
- OpenFold3는 OpenBind ZIP `data.pkl` metadata만 읽었다. Protenix/OpenDDE는
  managed conversion receipt가 지시하는 source ZIP metadata를 제한된 decoder로
  읽어 storage payload 없이 shape/block index를 확인했다. Torch는 import하지 않았다.
- Source/native checkpoint의 전체 hash는 이번에 재계산하지 않았다. Conversion
  receipt 이상의 동등성이나 실제 runtime tensor trace를 검증한 것은 아니다.
- AF3는 carried source defaults 근거이며 checkpoint header는 확인하지 않았다.
- T/A/M을 맞춰도 dtype, static chain/chemistry 조건, LM 경로, sample chunking 등이
  다르면 다른 executable이 필요할 수 있다. 이 표는 compiler buffer inventory가 아니다.
- 최초 shape 분석은 문서만 추가했다. 이후 recycle 정책 변경은 별도 문서에 기록했다.
  수치 정확성/성능 테스트는 실행하지 않았다.
