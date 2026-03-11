# M3.10 OOD Accuracy Probe

- run_id: `m3_10_20260307_r2`
- generated_utc: `2026-03-07T04:09:59.020727+00:00`

## Headline Table

| bucket | n | accuracy | mean_ce | active_tokens | scope | identity |
|---|---:|---:|---:|---:|---:|---:|
| `legacy` | 20 | 0.4500 | 0.1069 | 24.9000 | 0.3379 | 0.0000 |
| `easy` | 20 | 0.2000 | 0.2106 | 26.8000 | 0.3318 | 0.0000 |
| `medium` | 20 | 0.7000 | 0.1871 | 32.3500 | 0.3117 | 0.0000 |
| `hard` | 20 | 1.0000 | 0.2790 | 45.8000 | 0.2714 | 0.0000 |

## Structural-Performance Linkage

### `legacy`
- corr(active_token_count, correct): `0.816241`
- corr(scope, correct): `-0.804115`
- corr(primitive_usage, correct): `0.000000`
### `easy`
- corr(active_token_count, correct): `-0.291121`
- corr(scope, correct): `0.322360`
- corr(primitive_usage, correct): `0.000000`
### `medium`
- corr(active_token_count, correct): `0.887039`
- corr(scope, correct): `-0.915109`
- corr(primitive_usage, correct): `0.000000`
### `hard`
- corr(active_token_count, correct): `0.000000`
- corr(scope, correct): `0.000000`
- corr(primitive_usage, correct): `0.000000`

## Representative Examples

### `legacy`
- correct:
  - id=0 gold=`box` pred=`box
Explanation: Dana puts the red` active=37 scope=0.2892
  - id=4 gold=`shelf` pred=`shelf
Explanation: Dana moves the red` active=39 scope=0.2824
  - id=8 gold=`desk` pred=`desk
Explanation: Alice moves the red` active=38 scope=0.2824
- incorrect:
  - id=1 gold=`A=knight,B=knave,C=knight` pred=`A is a knave.
Explanation:` active=12 scope=0.3934
  - id=2 gold=`A=knight,B=knave,C=knight` pred=`A is a knave.
Explanation:` active=12 scope=0.3934
  - id=3 gold=`councilmen` pred=`demonstrators
Explanation: The city councilmen` active=24 scope=0.3380
### `easy`
- correct:
  - id=1 gold=`demonstrators` pred=`demonstrators
Explanation: The city councilmen` active=24 scope=0.3380
  - id=5 gold=`no` pred=`No.
Explanation: The blue box is` active=25 scope=0.3380
  - id=10 gold=`no` pred=`No.
Explanation: The logical sequence is` active=12 scope=0.4068
- incorrect:
  - id=0 gold=`yes` pred=`No.
Explanation: The red box is` active=28 scope=0.3288
  - id=2 gold=`suitcase` pred=`The trophy.
Explanation: The question asks` active=28 scope=0.3200
  - id=3 gold=`yes` pred=`No.
Explanation: The red box is` active=37 scope=0.2892
### `medium`
- correct:
  - id=1 gold=`desk` pred=`desk
Explanation: Eve puts the red` active=36 scope=0.2892
  - id=2 gold=`desk` pred=`desk
Explanation: Dana puts the red` active=37 scope=0.2892
  - id=3 gold=`shelf` pred=`shelf
Explanation: Alice moves the red` active=38 scope=0.2824
- incorrect:
  - id=0 gold=`yes` pred=`No tester missed a build.
Explanation:` active=19 scope=0.3810
  - id=4 gold=`A=knight,B=knave,C=knight` pred=`Knight, Knave, Knight
Explanation` active=12 scope=0.3934
  - id=9 gold=`A=knight,B=knight,C=knave` pred=`Knight: A, Knave: B` active=13 scope=0.3934
### `hard`
- correct:
  - id=0 gold=`no` pred=`No.
Explanation: The policy states that` active=49 scope=0.2474
  - id=1 gold=`no` pred=`No.
Explanation: The logic question asks` active=62 scope=0.2243
  - id=2 gold=`no` pred=`No.
Explanation: The policy states that` active=49 scope=0.2474
- incorrect: