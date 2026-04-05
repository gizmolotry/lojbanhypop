# M3.10 OOD Accuracy Probe

- run_id: `m3_10_20260310_refactor_small`
- generated_utc: `2026-03-11T01:06:14.351788+00:00`

## Headline Table

| bucket | relation | n | accuracy | mean_ce | active_tokens | scope | identity |
|---|---|---:|---:|---:|---:|---:|---:|
| `legacy` | `ood` | 16 | 0.4375 | 0.4193 | 12.0000 | 0.4020 | 0.0000 |
| `easy` | `ood` | 16 | 0.6250 | 0.2818 | 12.3750 | 0.3988 | 0.0000 |
| `medium` | `ood` | 16 | 0.6875 | 0.2179 | 13.3750 | 0.3932 | 0.0000 |
| `hard` | `in_distribution` | 16 | 0.3125 | 0.1608 | 16.3750 | 0.3734 | 0.0000 |

## Structural-Performance Linkage

### `legacy`
- corr(active_token_count, correct): `0.813157`
- corr(scope, correct): `-0.806228`
- corr(primitive_usage, correct): `0.000000`
### `easy`
- corr(active_token_count, correct): `0.009794`
- corr(scope, correct): `0.024608`
- corr(primitive_usage, correct): `0.000000`
### `medium`
- corr(active_token_count, correct): `0.726993`
- corr(scope, correct): `-0.738664`
- corr(primitive_usage, correct): `0.000000`
### `hard`
- corr(active_token_count, correct): `-0.940019`
- corr(scope, correct): `0.947933`
- corr(primitive_usage, correct): `0.000000`

## Representative Examples

### `legacy`
- correct:
  - id=0 gold=`box` pred=`box
Explanation: Dana puts the red ball in the box. Bob moves it` active=15 scope=0.3810
  - id=4 gold=`shelf` pred=`shelf
Explanation: Dana moves the red ball to the shelf while Alice is watching` active=17 scope=0.3692
  - id=8 gold=`desk` pred=`desk
Explanation: Alice moves the red ball to the drawer while Bob is outside` active=17 scope=0.3692
- incorrect:
  - id=1 gold=`A=knight,B=knave,C=knight` pred=`knave, knave, knight, knight
Explanation: The answer is "` active=7 scope=0.4364
  - id=2 gold=`A=knight,B=knave,C=knight` pred=`knave, knave, knight, knight
Explanation: The answer is "` active=7 scope=0.4364
  - id=3 gold=`councilmen` pred=`demonstrators
Explanation: The city councilmen feared violence because they refused the demonstrators a` active=13 scope=0.3934
### `easy`
- correct:
  - id=0 gold=`yes` pred=`yes
Explanation: The red box is inside the amber chest, so it must` active=13 scope=0.3934
  - id=1 gold=`demonstrators` pred=`demonstrators
Explanation: The city councilmen refused the demonstrators a permit because they advocated` active=13 scope=0.3934
  - id=3 gold=`yes` pred=`yes
Explanation: The red box is inside the amber chest, so the silver` active=17 scope=0.3692
- incorrect:
  - id=2 gold=`suitcase` pred=`small
Explanation: The trophy does not fit in the suitcase because it is too` active=13 scope=0.3934
  - id=4 gold=`trophy` pred=`suitcase
Explanation: The question asks for the answer to the logic question, which` active=13 scope=0.3934
  - id=5 gold=`no` pred=`yes
Explanation: The blue box is left of the green crate, so the` active=13 scope=0.3934
### `medium`
- correct:
  - id=0 gold=`yes` pred=`yes
Explanation: The first statement says that no tester missed a build. This` active=9 scope=0.4211
  - id=1 gold=`desk` pred=`desk
Explanation: Eve puts the red ball in the desk. Charlie moves it` active=15 scope=0.3810
  - id=2 gold=`desk` pred=`desk
Explanation: Dana puts the red ball in the shelf. Frank moves it` active=17 scope=0.3692
- incorrect:
  - id=4 gold=`A=knight,B=knave,C=knight` pred=`knight, knave, knight
Explanation: The knight says "A is a` active=7 scope=0.4364
  - id=9 gold=`A=knight,B=knight,C=knave` pred=`knight, knight, knave
Explanation: The knight says 'A is a` active=7 scope=0.4364
  - id=10 gold=`A=knight,B=knave,C=knight` pred=`knight, knave, knight
Explanation: The knight says "A is a` active=7 scope=0.4364
### `hard`
- correct:
  - id=3 gold=`no` pred=`no
Explanation: The question asks if the downtime must exceed one hour because the` active=13 scope=0.3934
  - id=6 gold=`no` pred=`no
Explanation: The question asks if the downtime must exceed one hour because the` active=13 scope=0.3934
  - id=10 gold=`no` pred=`no
Explanation: The question asks if the downtime must exceed one hour because the` active=13 scope=0.3934
- incorrect:
  - id=0 gold=`no` pred=`yes
Explanation: The policy states that approval is required unless an emergency waiver exists` active=19 scope=0.3582
  - id=1 gold=`no` pred=`yes
Explanation: The logic question asks whether it is possible for all regions to` active=17 scope=0.3692
  - id=2 gold=`no` pred=`yes
Explanation: The policy states that approval is required unless an emergency waiver exists` active=19 scope=0.3582