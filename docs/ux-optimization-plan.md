# FoldJAX 사용성·최적화 제안

작성 근거: `src/foldjax` 전체 오케스트레이션 레이어(약 10k 줄)와 6개 백엔드 어댑터,
`README.md`, `PROJECT.md` 검토. 모든 항목은 **모델 내부와 upstream 기본값을 건드리지
않는 선**에서만 제안한다 (PROJECT.md "Prediction and preprocessing stay
scientifically native to each model").

## 3차 구현 (2026-08-23, 요청 범위 세션)

2.1(c)의 공통 `Backend.session()` 계약과 ESMFold2·AlphaFold3·Boltz-2 구현을 추가했다.
세션 재사용은 명시적으로 opt-in한 백엔드만 적용되며, 나머지 백엔드는 기존처럼 scalar
run마다 새 인스턴스를 받는다. ESMFold2는 같은 요청의 seed/input 사이에서 checkpoint를
한 번만 로드하고, 정규화·패딩된 입력이 같은 동안 seed-independent ESMC hidden state를 한
번만 계산한다. 한 input의 hidden만 보유하며 model group이 끝나면 모두 해제한다.

재사용은 weight 경로만으로 판단하지 않는다. 실제 loader가 읽는 structure checkpoint,
ESMC config/index/shard와 device placement를 고정하고 resume·load·predict·manifest
경계마다 다시 확인한다. 중간에 파일 세대가 바뀌면 과거 seed와 새 seed를 섞지 않고
세션을 실패시킨다. all-resume은 stat 검증만 하고 26 GB bundle을 로드하지 않는다.

Boltz-2도 같은 request session에 opt-in했다. 실제 loader가 선택한 confidence/affinity
checkpoint와 scalar sidecar, device·CP topology, compile cache, trace-time option이
모두 같을 때만 파라미터와 JIT owner를 재사용한다. role별 JIT cache는 최대 8개 shape로
제한하고, continued failure 뒤에는 다음 seed가 fresh load/trace하며, all-resume은
native runtime을 import하지 않는다. CPU mock에서 같은 2개 seed의 primary load/trace가
2회에서 1회로, affinity job은 primary/affinity 각각 2회에서 1회로 줄었다.

Boltz-2 loader는 nested parameter tree 생성 직후 flat checkpoint mapping도 놓아,
pre-stack/device transfer 동안 두 소유자가 unstacked 배열을 붙잡지 않게 했다. 로컬
released confidence checkpoint에서 max RSS는 6,687,764 KiB에서 4,848,852 KiB로
1,838,912 KiB(27.5%) 줄었고 최종 parameter bytes와 값은 동일했다. 두 변경 모두 CPU
계약 검증 결과이며, released-weight CUDA latency/peak/parity는 별도 배치 검증 전까지
주장하지 않는다.

## 2차 구현 (2026-08-15, 후속)

1차 이후 코드를 다시 훑어 찾은 결함과 남은 기능 공백을 전부 처리했다.

**결함** — 시드 단위 `--resume`(5시드 중 4번째에서 죽으면 0~3을 다시 돌리던 문제) /
Ctrl-C 트레이스백 → exit 130 한 줄 / `PermissionError`·`OSError`가 트레이스백으로
새던 것 / 다운로드 전 디스크 여유 미확인 / 실패한 run이 아무 기록도 안 남기던 것
(`foldjax_failures.json`) / `py.typed` 부재 / `foldjax --version` 부재 / CHANGELOG 부재.

**기능 공백** — 에러에 다음 행동 힌트(누락 extra, 디스크 가득, AF3 런타임) /
공통 스키마의 templates·affinity / 로컬·오프라인 MSA와 RNA 검색 배선 /
구조 파일(.pdb/.mmcif/`structure:`) 입력.

실행 정책(`resume`, `on_error`)은 CLI 플래그가 아니라 **요청 필드**로 넣었다.
README가 스스로 정한 "CLI 전용 철자는 없다" 규칙을 지키기 위해서이고, 덕분에
배치 루프가 `cli.py`에서 `api.predict_batch`로 내려가 시드 단위까지 같은 정책이
적용된다. `foldjax.predict`의 반환 타입은 그대로 두고, 결과·건너뜀·실패를 함께
돌려주는 `predict_batch`를 따로 추가했다.

## 구현 상태 (2026-08-15)

아래 항목은 이 문서를 쓴 뒤 실제로 구현되어 CPU 스위트를 통과했다. 각 절의 제목 옆
표시를 참고할 것.

**완료** — 1.1 `--msa auto`(스텁 검색기로 배선 검증, 실서버 왕복은 미검증) /
1.2 `--sequence`·FASTA·디렉토리 입력(구조 파일·PDB ID·stdin·CSV는 미구현) /
1.3 서열 정규화·알파벳 검증·자동 chain id / 1.4 capability 정직화
(`common_schema_features`, `native_only_features`) / 1.5 `models --for` /
1.6 오타 제안(`foldjax validate` 명령은 미구현) / 2.1(a) 단계별 비용 계측
(`cost.phases`) / 2.5 `--resume`·`--keep-going` / 3.1 진행 표시 /
3.2 결과 요약 표·`foldjax show` / 4.1 `doctor` / 4.2 `cache gc` /
4.4 `--help` 그룹화 / 5 README 정정.

3.3 에러에 다음 행동 붙이기와 4.3 설정 파일은 손대지 않았다.

`--msa auto`는 **서열을 외부 서버로 보낸다**. 그래서 opt-in이고, 기본값은 바뀌지
않았다. 사내 서열은 `FOLDJAX_MSA_SERVER_URL`로 자체 인스턴스를 가리킬 것.

**미구현 (근거 부족 또는 포트 수술 필요)** — 2.1(b) 네이티브 시드·잡 팬아웃 /
2.1(c)의 ESMFold2 외 백엔드 / 2.2 버킷 정렬 / 2.3 `warm --grid` /
2.4 `plan --estimate` / 2.6 feature 캐시 / 2.7 장치 샤딩. 이유는 각 절 끝에 적었다.

---

## 0. 이미 되어 있는 것 — 다시 만들지 말 것

제안을 읽기 전에, 이 프로젝트가 이미 해결한 문제 목록. 아래 제안들은 전부 이 위에
얹는 것이다.

| 영역 | 현재 상태 | 위치 |
|---|---|---|
| 중립 vocabulary | sampling 4종 + execution 4종이 6모델로 번역, 이중 지정은 에러 | `execution.py`, `backends/base.py:48` |
| 계획/해석 분리 | `resolve_request` / `plan`이 실행 없이 전부 확정 | `api.py:65`, `cli.py:732` |
| 출력 정규화 | 6가지 레이아웃 → `seed-N_sample-NN/` 하나 | `output.py:163` |
| 재현성 | `foldjax_run.json`에 입력 sha256·가중치 identity·knob·cost | `manifest.py:77` |
| 컴파일 캐시 | 모델/가중치/런타임/옵션 네임스페이스, 프로세스 전역 설정 복원 | `cache.py`, `api.py:656` |
| 패딩 계약 | 6축, 마스크·크롭·RNG prefix, 기본 off, overflow 정책 | `padding.py`, `schema.py:206` |
| 가중치 스토어 | 검증·변환·진행률·구조화 이벤트, 암묵 다운로드 없음 | `assets.py`, `cli.py:542` |
| 실패 진단 | OOM 분류, `SystemExit` 봉쇄, 비밀값 리댁션 | `oom.py`, `redaction.py` |
| 결과 검증 | 좌표/구조 없는 "성공"을 거부 | `api.py:512` |

즉 **엔진과 계약은 성숙**했다. 남은 격차는 전부 *진입 비용*(첫 성공까지),
*반복 비용*(배치·다중 시드), *관측성*(지금 뭐 하는 중인지)에 있다.

---

## 1. 입력을 더 쉽게

### 1.1 ★ MSA를 FoldJAX가 직접 채운다 — `--msa auto`

**문제.** 공통 스키마는 `unpaired_msa` **파일 경로만** 받는다(`input.py:254`). 경로가
없으면 조용히 단일서열 예측이 된다:

- boltz2 → `msa: "empty"` 자동 삽입 (`input.py:352`)
- alphafold3 → `unpairedMsa: ""` 자동 삽입 (`input.py:305`)

정확도가 크게 떨어지지만 exit code는 0이고 경고도 없다. 이건 이 모듈이 스스로 세운
원칙("silently discarding an MSA … changes the science without changing the exit
code", `input.py:1-8`)과 실질적으로 어긋난다.

**그런데 검색기는 이미 패키지 안에 있다.** `foldjax/search/msa.py` — 서열 기반
content-addressed 캐시, provenance JSON, sha256 검증, HTTP 429 지수 백오프,
로컬 래퍼/원격 MMseqs2 백엔드, RNA 경로까지. 그런데 top-level에서 아무도 쓰지 않는다.
사용처는 protenix 자체 CLI(`models/protenix/cli/predict.py:394`)와 boltz2 native
옵션(`use_msa_server`, `backends/boltz2.py:143`)뿐 — 즉 **6모델 중 1개만** MSA를
자동으로 얻을 수 있고, 그것도 native 옵션 이름을 아는 사람만 쓸 수 있다.

**제안.**

```python
PredictionRequest(model=..., input=..., msa="auto")   # "none"(기본) | "auto" | "required"
```
```bash
foldjax predict --model openfold3 --input job.yaml --msa auto
foldjax msa --input job.yaml --out alignments/        # 검색만 따로 (재현 파이프라인 분리)
```

- 구현 지점은 `input.materialize_native_input` **한 곳**. MSA가 비어 있는
  protein/RNA 체인에 대해 `MsaSearchPipeline`을 호출하고, 결과 a3m 경로를 각 dialect의
  기존 필드(`unpairedMsaPath` / `msa` / `main_msa_file_paths` / …)에 주입한다.
  **모델 코드는 한 줄도 안 바뀐다.**
- 캐시는 `$FOLDJAX_HOME/msa/<sha>/`에 두어 **6모델이 같은 정렬을 공유**한다. 배치에서
  이게 가장 큰 절감(같은 서열을 3모델로 돌리면 검색 1회).
- 기본값은 `none` 유지 — 기본을 바꾸면 과거 결과의 의미가 바뀐다. 대신 **경고 한 줄**:
  `chain A has no alignment; predicting from a single sequence (--msa auto to search)`.
- `required`는 검색 실패 시 에러 — 배치 스크립트가 조용히 단일서열로 떨어지는 것을 막는다.
- OpenFold3는 stem으로 DB를 식별하므로(`input.py:437-463`) 링크 이름 규칙이 이미
  구현돼 있다. 재사용하면 된다.

**비용/위험.** 중간. 네트워크 의존이 예측 경로에 처음 들어오므로 `--msa auto`는
명시적 opt-in이어야 하고, provenance(`provenance.json`)를 매니페스트에 기록해야 한다.

---

### 1.2 파일 없이 시작하기 — 서열/FASTA/구조/PDB ID

**문제.** `PredictionRequest.__post_init__`가 input을 **반드시 존재하는 파일**로 요구한다
(`schema.py:391`). 처음 쓰는 사람은 job 파일 문법부터 배워야 첫 예측을 돌린다.
흥미롭게도 boltz2 port는 이미 `seq=`, `ligand_ccd=`를 직접 받는다
(`models/boltz2/api.py:120`) — 나머지 5개엔 없으니, 공통 레이어에서 Job 문서를
만들어 주는 것이 6모델 모두에 통하는 유일한 길이다.

**제안 (모두 기존 경로로 합류, 새 실행 경로 없음).**

```bash
foldjax predict --model boltz2 --sequence MKTAYIAKQRQISFVK --ligand ATP
foldjax predict --model protenix --input target.fasta        # 레코드 = 체인
foldjax predict --model boltz2 --input 1ubq.cif              # 구조에서 서열 추출
foldjax predict --model openfold3 --input pdb:1UBQ           # RCSB에서 fetch
cat job.yaml | foldjax predict --model opendde --input -
foldjax predict --model boltz2 --input jobs/                 # 디렉토리 = 배치
foldjax predict --model boltz2 --input screen.csv            # name,sequence,ligand
```

- `.cif/.pdb` 파싱은 `gemmi`가 이미 의존성(`output.py:107`)이라 추가 설치가 없다.
  "이 PDB의 서열을 리간드와 함께 다시 접어라"는 가장 흔한 워크플로다.
- `detect_input_format`(`api.py:47`)에 `fasta`/`structure` 분기를 추가하고, 내부적으로
  `Job` 문서를 만들어 `output_dir/inputs/`에 **써 둔다** — 재현성을 잃지 않는다.
- Python: `PredictionRequest(job=Job(...))`로 Job 객체 직접 수용. 지금은
  `job.write(path)`가 강제된다(`job.py:181`).
- 한 줄 API: `foldjax.fold("MKT...", model="boltz2")` — 첫 성공까지 5초.

**비용/위험.** 낮음. 전부 "공통 문서를 만들어 기존 경로에 넣기"이고, 생성된 문서를
디스크에 남기므로 감사 가능성이 유지된다.

---

### 1.3 서열 정규화와 알파벳 검증 (작지만 실제 버그)

**문제.** 서열은 양끝만 `strip()` 된다(`input.py:253`, `job.py:59`). YAML에서 서열을
붙여넣는 가장 흔한 방법인 블록 스칼라(`|`, `>`)는 **내부 개행/공백을 남기고**, 그대로
featurizer까지 내려간다. 잘못된 잔기 문자도 공통 레이어에서 검사하지 않아, 오타 하나가
모델 깊은 곳의 인덱스 에러로 나타난다.

**제안.** 공통 레이어에서 (a) 모든 공백 제거 + 대문자화, (b) 엔티티 타입별 알파벳 검증,
(c) 위치를 짚는 에러 메시지:

```
protein entity 'A' has an unsupported residue 'J' at position 42
  MKTAYIAKQRQ...VKJLE...
                  ^
```
(d) `id` 미지정 시 A, B, C… 자동 부여 — 지금은 에러(`input.py:93`)지만, 단일 체인
job에서 체인 이름을 강제할 이유가 없다.

**비용/위험.** 매우 낮음. 단, 공백 제거는 동작 변화이므로 "지금 실패하던 것이
성공하게" 되는 방향뿐이어야 한다(반대 방향 없음).

---

### 1.4 광고된 능력인데 공통 입력으로 도달 불가: templates / affinity

**문제.**

- `ModelCapabilities.supports_templates` 기본값이 `True`(`schema.py:97`)라 6모델 중
  5개가 템플릿 지원을 광고한다. 그런데 공통 스키마의 폴리머 키는
  `{type, id, sequence, unpaired_msa, paired_msa, modifications}`뿐
  (`input.py:26`) — **템플릿을 넣을 자리가 없다.**
- boltz2는 `supports_affinity=True`(`backends/boltz2.py:215`)지만 `_JOB_KEYS`는
  `{name, entities, bonds}`(`input.py:25`) — **친화도 job도 native YAML로만 가능**.

PROJECT.md의 공통 스키마 범위에 templates/affinity가 없으므로 이는 *의도된 한계*이지
버그는 아니다. 문제는 **capabilities가 그렇게 말하지 않는다**는 것이다.

**제안 (둘 중 하나, 반반은 안 됨).**

1. 스키마 확장: 폴리머에 `templates: [{mmcif: path, chain: A}]`, job에
   `properties: [{affinity: {binder: L}}]`. 표현 못 하는 모델은 기존
   reject-don't-drop 규칙(`input.py:116`)으로 거부. 또는
2. capability를 정직하게: `supports_templates`를 `"native-input-only"` 같은 3-state로
   바꾸거나 `input_requirements`에 "common schema로는 도달 불가"를 명시.

최소한 (2)는 지금 당장 해야 한다 — 지금은 `foldjax capabilities`가 사실상 사실이
아닌 값을 반환한다.

---

### 1.5 모델 선택을 돕는 `foldjax models --for job.yaml`

`foldjax models`는 이름만 나열한다(`cli.py:696`). 그런데 어떤 모델이 이 job을 돌릴 수
있는지는 **가중치 없이 순수 계산으로 알 수 있다** — `_TARGETS` 표(`input.py:59-81`)에
모델별 표현 가능 기능이 전부 들어 있다.

```
$ foldjax models --for complex_with_ligand.json
model       runs?  why
boltz2      yes
protenix    yes
opendde     yes
alphafold3  yes    (weights not installed)
openfold3   no     cannot express: bonds
esmfold2    no     cannot express: ligand entity
```

**비용.** 매우 낮음(반나절). 첫 사용자가 가장 자주 막히는 지점이다.

---

### 1.6 `foldjax validate` / 오타 제안

`plan`은 모델이 필수이고 해석 결과를 뱉는다. 그와 별개로 "이 문서가 맞나"만 보는
명령이 필요하다. 그리고 알 수 없는 키 에러(`input.py:220`)에 `difflib`로 가까운 이름을
붙인다: `unsupported protein entity fields: ['unpared_msa'] (did you mean
'unpaired_msa'?)`.

---

## 2. 실행 최적화

### 2.1 ★ 가중치 상주 세션 — 배치/다중 시드의 지배적 낭비

**문제.** `_predict_once`가 매 실행마다 `backend.predict()`를 부르고, 각 백엔드는
체크포인트를 **처음부터 다시 읽는다**. `api.py:229-236`의 docstring이 이미 인정한다:
> "Each pass reloads the weights; the compiled program is read back from the cache,
> so the repeat cost is the load, not the compile."

- `--num-seeds 5` → 가중치 5회 로드
- `--model A B --input x y` → 4회 로드
- ESMFold2는 ESMC-6B **25 GB** — 로드가 예측보다 오래 걸릴 수 있다.

**제안 (3단계, 각각 독립적으로 가치 있음).**

**(a) 먼저 측정.** 매니페스트 `cost`에 `weight_load_seconds` / `featurize_seconds` /
`compile_seconds`를 추가한다. 이 프로젝트의 방식대로, 최적화 전에 숫자부터.
비용 거의 0, 이후 모든 판단의 근거가 된다.

**(b) 네이티브 시드/잡 팬아웃 활용 (저비용, 근거 확실).** 코드를 읽어 확인한 사실:

- opendde의 native `main`은 **가중치를 루프 밖에서 한 번 로드**하고
  (`models/opendde/cli/predict.py:391` `params = _load_weights(...)`),
  그 다음 `for job in jobs:` → `for seed in _job_seeds(...):`로 돈다(:400-401).
  즉 **여러 job × 여러 seed를 한 번의 로드로 처리하는 능력이 이미 있다.**
- protenix도 동일하게 `--seeds`를 받고(`models/protenix/cli/predict.py:30`)
  job×seed를 in-process로 돈다(:703, :728).
- 출력 충돌도 없다. protenix는
  `<root>/<job>/seed_<seed>/predictions/<job>_sample_<rank>.cif`로 쓴다
  (`models/protenix/data/output.py:89,105`) — seed와 job이 이미 경로에 있다.
- 그런데 FoldJAX는 native 문서를 항상 **단일 job / 단일 시드**로 쓴다
  (`input.py:413` `"modelSeeds": [seed]`, `input.py:431` `return [native]`),
  그리고 시드마다 어댑터를 다시 호출한다(`api.py:296-314`).

**중립성은 API 표면의 성질이지 구현의 성질이 아니다.** 표면(`--seeds`, `--num-seeds`,
plural `inputs`)은 그대로 두고, 팬아웃을 네이티브에 위임할 수 있는 백엔드에서는
`modelSeeds`에 전체 리스트를, 그리고 같은 모델의 여러 입력을 하나의 job 배열로 넘기면
**가중치 로드가 N회 → 1회**가 된다. 포트는 한 줄도 안 고친다.

필요한 오케스트레이션 변경은 두 가지뿐이고 둘 다 국소적이다:
1. 어댑터가 결과 샘플의 seed를 `request.seed`로 하드코딩하지 말고 경로의 `seed_<n>`에서
   읽을 것(`backends/protenix.py:292`).
2. `_validate_result`의 "모든 샘플의 seed == 실행 seed" 단언(`api.py:567`)을
   "요청된 seed 집합에 속할 것"으로 완화.

featurization은 seed에 의존하므로(opendde `_featurize(..., seed=seed)`, :404) 시드마다
다시 도는 것이 **맞다** — 이 최적화가 없애는 것은 가중치 로드이지 전처리가 아니다.

**(c) `Backend.session()` 프로토콜 (본격).**

```python
class Backend:
    def session(self, request) -> AbstractContextManager[Session] | None:
        """가중치/컴파일 실행체를 붙잡고 여러 요청을 처리. 미구현이면 None."""
```

`predict()`가 (model, weights, compile-profile)이 동일한 연속 run에 대해 세션을
재사용한다. 미구현 백엔드는 지금 동작 그대로(fallback) — 계약 파괴 없음.
난이도는 백엔드마다 다르다:

- boltz2: **쉬움**. `load_params` + `boltz2_predict`가 이미 분리돼 있다
  (`models/boltz2/api.py:248-281`).
- protenix / opendde / openfold3 / alphafold3: **어려움**. `module.main(argv)`로
  in-process CLI를 부르는 구조라(`backends/protenix.py:276`), 포트 쪽에
  load-once/predict-many 진입점을 새로 열어야 한다.

따라서 (a) → (b) → boltz2·esmfold2부터 (c), 나머지는 측정 결과가 정당화할 때.

**현재 구현 상태.** 공통 프로토콜과 ESMFold2·AlphaFold3·Boltz-2 세션은 구현됐다.
dispatcher는 모델별로 한 세션만 순서대로 열고, ESMFold2는 verified checkpoint
snapshot이 같은 동안 model과 ESMC hidden state를 재사용한다. AlphaFold3는 managed runtime의 weights, runner,
vendored source generation이 고정된 동안 하나의 `ModelRunner`와 그 JIT owner를
재사용한다. 명시적 외부 `source=`와 단일 scalar 호출은 기존 fresh-instance 경로를
유지한다. Boltz-2는 confidence/affinity parameter tree와 role별 bounded JIT owner를
실제 checkpoint bundle·device·CP·compile identity가 고정된 동안 재사용한다. 나머지
backend는 아직 opt-in하지 않았다. 위의 native
fan-out 제안 (b)는 출력/seed 계약을 바꾸지 않고도 같은 지배적 낭비를 제거했기 때문에
당장 필요하지 않다.

---

### 2.2 배치 스케줄러: 패딩 버킷으로 정렬

**문제.** 교차곱은 선언 순서대로 실행된다(`api.py:195-217`). 패딩을 켜도 토큰 수가
번갈아 오면 실행체가 매번 바뀐다. 캐시 덕에 재컴파일은 피하지만 디스크 로드와
오토튠은 반복된다.

**제안.** `--padding`이 켜진 배치에서 (모델, 해석된 패딩 프로파일) 키로 정렬해 실행하고,
**결과는 선언 순서로 되돌려 반환**한다 → 관측 가능한 계약 불변.
`plan`은 이미 프로파일을 계산할 수 있으니, 출력에 "이 배치 12개 job은 실행체 3개로
커버됨"을 미리 보여준다. 세션(2.1)과 결합하면 곱셈으로 이득이 커진다.

---

### 2.3 `cache warm --grid`

**문제.** warm은 프로파일 **하나**만 데운다(`warmup.py:1-8`, README도 명시). 서빙에서는
보통 여러 버킷을 미리 데운다.

**제안.** `--pad-tokens 384 512 768`처럼 축별 리스트를 받아 데카르트 곱을 순차 warm.
실제 실행이므로 정직함이 유지된다. 총 예상 시간을 먼저 출력하고 `--yes`로 확인.
리포트는 프로파일당 한 행.

---

### 2.4 사전 자원 예측 — `plan --estimate`

**문제.** OOM은 몇 분 실행한 뒤에 발견된다. `oom.py`는 **사후** 진단이다.

**제안.** `foldjax-bench`와 `docs/benchmark.md`에 이미 모델별 토큰수 대비 peak/시간
곡선이 있다. 이를 작은 계수 테이블로 패키지에 담아:

```
$ foldjax plan --model opendde --input big.yaml --estimate
tokens 1531 (padded 1536)
estimated peak   ~46 GiB   device has 48 GiB   ⚠ 여유 4%
estimated time   ~11 min (warm cache)
suggestions: --option trunk_dtype=bf16 (≈ -35% peak, upstream fp32와 결과가 다름)
             --max-msa-depth 1024      (opendde에서는 정확도 손실이 있다)
```

- 반드시 **추정치**로 표기하고, 실행이 끝나면 매니페스트의 실측 `cost`를
  `$FOLDJAX_HOME/estimates.json`에 append해 **자기 기계에서 자동으로 정확해지게** 한다.
- 자동 적용은 **금지**. 제안만. 과학적 결과를 바꾸는 knob을 도구가 몰래 켜면 안 된다
  (프로젝트의 기존 원칙과 동일).

---

### 2.5 부분 실패와 재개

**문제.** `predict`의 튜플 comprehension(`api.py:248`)은 한 run이 실패하면 전체가
예외로 끝난다. 20개 배치의 3번째가 OOM이면 나머지 17개를 잃는다.

**제안.**

- `on_error="stop"(기본, 호환) | "continue"`. continue면 결과 자리에
  `PredictionFailure(model, input, error)`를 넣고 CLI는 부분 실패 시 exit 3.
- `--resume`: 출력 디렉토리에 `foldjax_run.json`이 있으면 건너뛴다. 매니페스트는
  이미 "run finished"의 증거로 정의돼 있다(`manifest.py:12`) — 새 상태 파일이 필요 없다.

**비용.** 낮음. 배치를 실제로 돌리는 사람에게는 체감이 가장 큰 항목 중 하나.

---

### 2.6 featurization 캐시 공통화

boltz2에만 `feature_cache`가 있다(`models/boltz2/api.py:135`). MSA 캐시와 같은 자리에
`(모델, job sha256, msa sha256, seed, 전처리 옵션)` 키로 공통 feature 캐시를 둔다.

**키에 seed가 들어가는 것이 핵심이다.** opendde/protenix는 MSA 서브샘플링이 seed에
의존하므로 시드마다 featurize를 다시 한다(`models/opendde/cli/predict.py:404`).
따라서 이 캐시는 *다중 시드*를 빠르게 하지 못한다 — 빠르게 하는 것은
**같은 job·같은 seed로 샘플 수/스텝 수/dtype만 바꿔 다시 돌리는 경우**, 즉 튜닝
루프와 벤치마크다. 그쪽이 실제로 사람이 하루에 열 번 하는 일이다.

---

### 2.7 장치 단위 병렬 (모델 샤딩 아님)

PROJECT.md는 multi-device 실행을 주장하지 않는다 — 그 선은 유지한다. 하지만 **배치의
서로 다른 job을 서로 다른 GPU에 배정**하는 것은 모델을 전혀 건드리지 않는다.
`--devices 0,1`로 워커 프로세스를 띄워 교차곱을 샤딩한다. 각 워커는 지금과 100%
동일한 단일 장치 실행. 카드가 여러 장인 환경에서 선형 처리량.

---

## 3. 결과·관측성

### 3.1 진행 표시

**문제.** 예측 중 아무것도 출력되지 않는다(가중치 fetch만 진행률이 있다, `cli.py:382`).
254 토큰에서는 전체 시간의 대부분이 컴파일이라 사용자는 멈춘 줄 안다.

**제안.** stderr에 단계 라인 — stdout의 JSON 계약은 그대로 둔다.

```
[foldjax] boltz2 · job.yaml · seed 0
  featurize          12.3s
  weights            4.1s   (1.2 GB)
  compile            cache hit
  sample 3/5         ████████░░  1m42s
  write              out/seed-0_sample-00/…
```
TTY면 갱신, 파이프면 라인 단위. `--quiet` / `FOLDJAX_PROGRESS=0`.

### 3.2 사람이 읽는 결과 요약

CLI는 JSON만 뱉는다(`cli.py:745`). `best_sample`은 구현돼 있는데
(`output.py:249`) 매니페스트에만 들어가고 화면에는 안 보인다.

```
$ foldjax predict --model protenix --input job.yaml --num-seeds 3
model     protenix        weights  protenix-v0.5.0
samples   15              time     6m21s      peak  18.4 GiB
best      seed 102 / sample 01     ranking_score 0.873
          out/seed-102_sample-01/job_seed-102_sample-01.cif

 seed  sample  ranking_score  plddt   file
  101      00          0.812   84.2   out/seed-101_sample-00/…
  102      01          0.873   88.9   out/seed-102_sample-01/…  ← best
```

TTY 기본은 표, `--json`으로 현재 출력. 추가로 `foldjax show <output_dir>`가 끝난
디렉토리의 매니페스트를 같은 표로 렌더 — 배치 결과 비교에 바로 쓰인다.
**모델 간 점수 비교는 여전히 하지 않는다**(`output.py:20-24`의 원칙 유지).

### 3.3 에러에 "다음 행동" 붙이기

`_USER_ERRORS`가 한 줄로 줄여주는 것은 좋다(`cli.py:756`). 여기에 다음 명령을 붙인다:
가중치 없음 → 정확한 `weights fetch` 한 줄, AF3 런타임 → `runtime prepare`,
OOM → 이 job에 **실제로 유효한** knob만 나열(모델별로 다르다). 일부는 이미
`assets.py` / `oom.py`에 문장이 있으니 `hint` 필드로 통일하면 된다.

---

## 4. 설치·운영

- **`foldjax doctor`** — GPU/드라이버/CUDA extra 일치, JAX 백엔드, 가중치 준비 상태,
  AF3 런타임, 캐시 크기, kalign·템플릿 상태(`cli.py:416`의 `_template_report`가 이미
  있는데 `setup`에만 묶여 있다), 디스크 여유를 한 번에. 이슈 리포트 첨부용 `--json`.
- **`foldjax cache gc --older-than 30d --max-size 50G`** — 컴파일 캐시는 지금 무한히
  자란다. `cache` 하위에 `warm`만 있다(`cli.py:277`).
- **설정 파일/프로파일** — `~/.config/foldjax/defaults.toml` 또는 `--config run.toml`로
  랩 단위 기본값(`num_samples`, `padding`, `mem_fraction`)을 고정. 매니페스트에 어떤
  설정 파일이 적용됐는지 기록.
- **`--help` 구조화** — 지금 `predict`는 20개 넘는 플래그가 평평하게 나열된다
  (`cli.py:27-191`). argparse group으로 input / sampling / padding / execution /
  output / cache 6개로 나누면 읽을 수 있게 된다. 비용 1시간.
- **셸 완성**(`argcomplete`)과 `foldjax --version`.

---

## 5. 문서 정정 (즉시)

`README.md:450`은 `--include-raw`를 FoldJAX CLI 플래그처럼 적었지만, 실제로는 OpenDDE
네이티브 플래그다(`backends/opendde.py:65`). FoldJAX에서는
`--option include_raw=true`로 써야 한다.

---

## 6. 우선순위

| # | 항목 | 체감 | 비용 | 위험 | 모델 불변 |
|---|---|---|---|---|---|
| P0 | 1.1 `--msa auto` | 매우 큼 | 중 | 네트워크 opt-in | ✔ |
| P0 | 2.1(a) 비용 계측 | (근거) | 매우 낮음 | 없음 | ✔ |
| P0 | 3.1 진행 표시 | 매우 큼 | 낮음 | 없음 | ✔ |
| P0 | 3.2 결과 요약 표 | 큼 | 낮음 | 없음 | ✔ |
| P1 | 1.2 서열/FASTA/구조 입력 | 매우 큼 | 낮음~중 | 없음 | ✔ |
| P1 | 1.3 서열 정규화·검증 | 중 | 매우 낮음 | 완화 방향만 | ✔ |
| P1 | 2.5 부분 실패·`--resume` | 큼(배치) | 낮음 | 없음 | ✔ |
| P1 | 1.5 `models --for` | 중 | 매우 낮음 | 없음 | ✔ |
| P1 | 1.4 templates/affinity 정직화 | 중 | 낮음(문서) | 없음 | ✔ |
| P1 | 4.1 `doctor` | 중 | 낮음 | 없음 | ✔ |
| P1 | 2.1(b) 네이티브 시드/잡 팬아웃 | 큼 | 낮음 | 어댑터 2곳 국소 수정 | ✔ |
| P2 | 2.1(c) 세션 프로토콜 | 큼 | 높음 | 포트 수정 필요 | ✔ |
| P2 | 2.3 `warm --grid` | 중(서빙) | 낮음 | 없음 | ✔ |
| P2 | 2.2 버킷 정렬 | 중(배치) | 중 | 순서 계약 유지 필요 | ✔ |
| P2 | 2.4 `plan --estimate` | 중 | 중 | 추정치 표기 필수 | ✔ |
| P2 | 2.6 feature 캐시 | 중 | 중 | 키 설계 주의 | ✔ |
| P3 | 2.7 장치 샤딩, 4.2 cache gc, 4.3 config | 소~중 | 중 | 없음 | ✔ |

---

## 6.5 미구현 항목을 남긴 이유

전부 "할 수 있지만 지금 검증할 수 없다"에 해당한다. 검증 없이 넣으면 이 프로젝트가
지켜온 증거 기준을 깨는 쪽이라 남겼다.

- **2.1(b) 네이티브 시드/잡 팬아웃** — 어댑터가 샘플의 seed를 경로에서 읽도록 바꾸고
  `_validate_result`의 seed 단언을 집합 비교로 완화해야 한다. 둘 다 출력 계약을
  건드리므로 실제 가중치로 GPU에서 한 번은 돌려봐야 하고, 이 세션에서는 GPU 작업을
  큐에 넣지 않았다. 선행 조건인 2.1(a) 계측이 방금 들어갔으니, 다음 실행에서
  `cost.phases`가 로드 비용을 얼마로 보고하는지 보고 판단하는 것이 순서다.
- **2.1(c)의 나머지 백엔드** — 공통 프로토콜과 ESMFold2·AlphaFold3·Boltz-2는
  구현했다.
  AlphaFold3는 upstream `ModelRunner`를 직접 보유하는 adapter라 native API를 바꾸지
  않고 managed route에 한해 안전한 load-once 경계를 만들 수 있었다.
  protenix/opendde/openfold3는 `module.main(argv)` 구조라 포트마다 load-once 진입점을
  새로 열어야 한다. 실제 반복 비용이 정당화할 때 백엔드별로 추가한다.
- **2.2 버킷 정렬** — 실행 순서를 바꾸는 최적화라, 결과 순서 계약이 유지되는지
  패딩을 켠 실제 배치로 확인해야 의미가 있다.
- **2.3 `warm --grid`** — 구현은 짧지만 각 격자점이 진짜 실행이라, 검증에 GPU 시간이
  필요하다.
- **2.4 `plan --estimate`** — 계수 테이블을 `foldjax-bench` 결과에서 뽑아야 한다.
  검증되지 않은 추정치를 "48 GiB 중 46 GiB"처럼 출력하면 없느니만 못하다.
- **2.6 feature 캐시** — 키 설계에 백엔드별 전처리 감사(무엇이 seed·옵션에
  의존하는지)가 선행되어야 한다. opendde/protenix가 seed 의존이라는 것까지는
  확인했다.
- **2.7 장치 샤딩** — 워커 프로세스 관리가 필요하고, 이 기계의 단일 GPU 큐에서는
  동작을 검증할 수 없다.

## 7. 침범하면 안 되는 불변식

제안이 이 선을 넘으면 그 제안은 틀린 것이다.

1. **모델 내부와 upstream 기본값 불변.** knob은 번역하되 강요하지 않는다.
2. **조용한 대체 금지.** `--msa auto`도 opt-in, `--estimate`의 제안도 자동 적용 금지.
   표현 못 하는 필드는 거부하지 드롭하지 않는다.
3. **Torch-free.** 새 입력 경로(FASTA/구조/PDB fetch)도 gemmi·NumPy 범위 안에서.
4. **패딩 기본 off**, 출력 레이아웃·매니페스트 스키마는 추가만(제거·의미변경 없음).
5. **컴파일에 영향을 주는 새 옵션은 반드시 캐시 네임스페이스에 포함**
   (`Backend.compile_options`).
6. **모델 간 점수 비교 금지.** 요약 표는 모델 내부 랭킹만 표시한다.
7. stdout의 JSON 계약 불변 — 새 출력은 stderr나 `--json` 반대편으로.
