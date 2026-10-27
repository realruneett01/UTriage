# Failure Modes and Generalization Boundaries


## Plain-Language Summary

Evaluation summary derived strictly from measurable counts and evaluation metrics:

- **Fold crack_1.6**: 3435 misses (top cause: small_flaw with 1865 cases, 54.3%), 0 false calls; top clean scores listed as near misses.
- **Fold crack_4.0**: 3407 misses (top cause: unclassified with 1750 cases, 51.4%), 0 false calls; top clean scores listed as near misses.
- **Fold crack_8.6**: 0 misses (top cause: none with 0 cases, 0.0%), 0 false calls; top clean scores listed as near misses.
- **Synthetic Shift Summary**: ML 1D false-call rate shifts from 13.4% in-distribution to 60.0% at 10 MHz and 40.7% under elevated noise.
- **Artifact Storage**: 100 PNG plots saved; cause counts cover all cases while files are kept for the top 20 of each kind.

## LOCO Folds Failure Breakdown

| Fold | Misses | Near Edge | Low SNR | Small Flaw | Miss Unclass | False Calls | Artefact-like | FC Unclass |
|---|---|---|---|---|---|---|---|---|
| crack_1.6 | 3435 | 0 | 0 | 1865 | 1570 | 0 | 0 | 0 |
| crack_4.0 | 3407 | 0 | 0 | 1657 | 1750 | 0 | 0 | 0 |
| crack_8.6 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

## Evaluated Failure Cases

| Case ID | Fold | Kind | Score | Cause | Evidence |
|---|---|---|---|---|---|
| miss_crack_1.6_829F2A1A-AC69-4E3F-9092-FABBD75F3A1D_40 | crack_1.6 | miss | 0.498 | unclassified | size=1.6 eq=1.3494509721059926 amp=6028 noise=804 |
| miss_crack_1.6_199BD889-BF94-4EE6-8E55-5B5BD58ED8CE_21 | crack_1.6 | miss | 0.497 | unclassified | size=1.6 eq=1.3500845190472854 amp=6028 noise=804 |
| miss_crack_1.6_728D32B4-D1A3-4CF2-A530-615412183F42_55 | crack_1.6 | miss | 0.496 | unclassified | size=1.6 eq=1.3108099255722263 amp=6028 noise=804 |
| miss_crack_1.6_3D3BC24B-3E1A-4CE4-BF8E-F039AE5DE417_72 | crack_1.6 | miss | 0.496 | unclassified | size=1.6 eq=1.2957708618099866 amp=6028 noise=804 |
| miss_crack_1.6_4A409E8B-26B6-470A-8463-3B464443B81B_42 | crack_1.6 | miss | 0.495 | unclassified | size=1.6 eq=1.5855813067051487 amp=6028 noise=804 |
| miss_crack_1.6_07F57255-BE18-49DB-9E60-A6C43F9F527E_25 | crack_1.6 | miss | 0.494 | unclassified | size=1.6 eq=1.491246283999287 amp=6028 noise=804 |
| miss_crack_1.6_24837367-47F9-4E38-B71B-409405C74D4E_84 | crack_1.6 | miss | 0.493 | unclassified | size=1.6 eq=1.4958147367359609 amp=6028 noise=804 |
| miss_crack_1.6_76E63756-0E55-4148-AF22-5DC7DF24E3F7_78 | crack_1.6 | miss | 0.493 | small_flaw | size=1.6 eq=1.1782681685897174 amp=6028 noise=804 |
| miss_crack_1.6_F1823AFF-436E-4B64-AE46-9CC3EBBA100D_23 | crack_1.6 | miss | 0.493 | unclassified | size=1.6 eq=1.5405638736003455 amp=6028 noise=804 |
| miss_crack_1.6_8BD1288C-8103-4171-A2D7-111DF98EE1E6_95 | crack_1.6 | miss | 0.492 | unclassified | size=1.6 eq=1.5547077030797145 amp=6028 noise=804 |
| miss_crack_1.6_2CECF70F-40A6-4838-BFDF-AD8DE8ED2149_51 | crack_1.6 | miss | 0.491 | unclassified | size=1.6 eq=1.5258922487621427 amp=6028 noise=804 |
| miss_crack_1.6_7266E2A1-7CC0-46A2-BF77-5480B778BD23_56 | crack_1.6 | miss | 0.491 | small_flaw | size=1.6 eq=1.226560414218433 amp=6028 noise=804 |
| miss_crack_1.6_9A770208-645C-41AE-9328-5F7C2E0CEE2C_92 | crack_1.6 | miss | 0.490 | unclassified | size=1.6 eq=1.248696571249635 amp=6028 noise=804 |
| miss_crack_1.6_30B333B6-D988-4D87-BB21-7C8134FCB067_82 | crack_1.6 | miss | 0.488 | unclassified | size=1.6 eq=1.5666540678831784 amp=6028 noise=804 |
| miss_crack_1.6_DA0E6EA7-891A-4CAD-B175-CEF7468A9C19_91 | crack_1.6 | miss | 0.487 | unclassified | size=1.6 eq=1.277157582325549 amp=6028 noise=804 |
| miss_crack_1.6_8C4C5FC1-C515-459D-ADEF-60B6EB85F3BB_91 | crack_1.6 | miss | 0.487 | unclassified | size=1.6 eq=1.3925738799831628 amp=6028 noise=804 |
| miss_crack_1.6_2139604D-0B14-4B87-8B71-CFF857F92A46_98 | crack_1.6 | miss | 0.486 | unclassified | size=1.6 eq=1.5201819704935335 amp=6028 noise=804 |
| miss_crack_1.6_10281F86-4508-402D-910F-8ED105F2166B_39 | crack_1.6 | miss | 0.484 | unclassified | size=1.6 eq=1.313742833876831 amp=6028 noise=804 |
| miss_crack_1.6_1B58783C-A437-4552-9C10-82B66F332074_20 | crack_1.6 | miss | 0.484 | unclassified | size=1.6 eq=1.5792705269579796 amp=6028 noise=804 |
| miss_crack_1.6_42EA6651-1891-4F98-8E7A-67CF33919A6E_11 | crack_1.6 | miss | 0.484 | unclassified | size=1.6 eq=1.3473733202349871 amp=6028 noise=804 |
| near_miss_clean_crack_1.6_818B8CB4-5C86-4396-AAF5-B48F41302D8F_27 | crack_1.6 | near_miss_clean | 0.000 | clean_near_miss | amp=4050 noise=973 |
| near_miss_clean_crack_1.6_0275DE06-7EB3-4D99-B71D-39F259B64862_4 | crack_1.6 | near_miss_clean | 0.000 | clean_near_miss | amp=3734 noise=1068 |
| near_miss_clean_crack_1.6_4D9E9185-588A-433B-BB2C-8E87144A324A_72 | crack_1.6 | near_miss_clean | 0.000 | clean_near_miss | amp=3734 noise=1068 |
| near_miss_clean_crack_1.6_818B8CB4-5C86-4396-AAF5-B48F41302D8F_28 | crack_1.6 | near_miss_clean | 0.000 | clean_near_miss | amp=3734 noise=1068 |
| near_miss_clean_crack_1.6_9394DF9F-28AD-46C4-82B1-083843062055_59 | crack_1.6 | near_miss_clean | 0.000 | clean_near_miss | amp=3734 noise=1068 |
| near_miss_clean_crack_1.6_5EC43978-E60B-475E-B136-6B505FB7BE1E_15 | crack_1.6 | near_miss_clean | 0.000 | clean_near_miss | amp=3734 noise=1068 |
| near_miss_clean_crack_1.6_FE57FD4B-E7A8-4307-BB77-65F85DA4E962_91 | crack_1.6 | near_miss_clean | 0.000 | clean_near_miss | amp=4050 noise=973 |
| near_miss_clean_crack_1.6_DA68E5F8-9C58-4343-86DB-5D53586D9A70_13 | crack_1.6 | near_miss_clean | 0.000 | clean_near_miss | amp=4050 noise=973 |
| near_miss_clean_crack_1.6_728D32B4-D1A3-4CF2-A530-615412183F42_68 | crack_1.6 | near_miss_clean | 0.000 | clean_near_miss | amp=3734 noise=1068 |
| near_miss_clean_crack_1.6_5E031CFB-E25C-4AAD-A65F-F91A7419F0CF_28 | crack_1.6 | near_miss_clean | 0.000 | clean_near_miss | amp=3734 noise=1068 |
| near_miss_clean_crack_1.6_D2D163FD-047D-4AE0-A84E-5DE59C681961_96 | crack_1.6 | near_miss_clean | 0.000 | clean_near_miss | amp=3734 noise=1068 |
| near_miss_clean_crack_1.6_3BEC7C3C-1441-46B0-8442-ABC5BBD185A7_96 | crack_1.6 | near_miss_clean | 0.000 | clean_near_miss | amp=3734 noise=1068 |
| near_miss_clean_crack_1.6_268A3193-0491-40CF-BA54-1B7061557AC1_93 | crack_1.6 | near_miss_clean | 0.000 | clean_near_miss | amp=3734 noise=1068 |
| near_miss_clean_crack_1.6_3BEC7C3C-1441-46B0-8442-ABC5BBD185A7_62 | crack_1.6 | near_miss_clean | 0.000 | clean_near_miss | amp=3734 noise=1068 |
| near_miss_clean_crack_1.6_5E031CFB-E25C-4AAD-A65F-F91A7419F0CF_26 | crack_1.6 | near_miss_clean | 0.000 | clean_near_miss | amp=4050 noise=973 |
| near_miss_clean_crack_1.6_744784D3-1AB4-4E4B-95D1-49573EE1F8F9_2 | crack_1.6 | near_miss_clean | 0.000 | clean_near_miss | amp=3734 noise=1068 |
| near_miss_clean_crack_1.6_605F522F-8CBF-4E22-811D-60D594F283D2_33 | crack_1.6 | near_miss_clean | 0.000 | clean_near_miss | amp=4050 noise=973 |
| near_miss_clean_crack_1.6_3BEC7C3C-1441-46B0-8442-ABC5BBD185A7_82 | crack_1.6 | near_miss_clean | 0.000 | clean_near_miss | amp=3734 noise=1068 |
| near_miss_clean_crack_1.6_CB5DDE36-318E-4C1B-B831-614E91996167_88 | crack_1.6 | near_miss_clean | 0.000 | clean_near_miss | amp=3734 noise=1068 |
| near_miss_clean_crack_1.6_9394DF9F-28AD-46C4-82B1-083843062055_88 | crack_1.6 | near_miss_clean | 0.000 | clean_near_miss | amp=3734 noise=1068 |
| miss_crack_4.0_4B35EF47-6730-4C4D-A1AF-29C9999C01F3_93 | crack_4.0 | miss | 0.498 | small_flaw | size=4.0 eq=2.8190437392839276 amp=7414 noise=934 |
| miss_crack_4.0_FE767298-15D2-493D-ABA0-919F9EEC5886_82 | crack_4.0 | miss | 0.497 | unclassified | size=4.0 eq=3.7640183461028847 amp=7414 noise=934 |
| miss_crack_4.0_38560032-7BCA-4FCD-A2CC-5C2598964162_19 | crack_4.0 | miss | 0.495 | unclassified | size=4.0 eq=3.3246679623765623 amp=7414 noise=934 |
| miss_crack_4.0_EEA99100-A5CF-4D08-A0E5-073F648FA77B_5 | crack_4.0 | miss | 0.495 | small_flaw | size=4.0 eq=2.8782204560371705 amp=7414 noise=934 |
| miss_crack_4.0_61CBA605-CD97-4983-9757-16418D70504A_77 | crack_4.0 | miss | 0.494 | small_flaw | size=4.0 eq=2.8884922554769004 amp=7414 noise=934 |
| miss_crack_4.0_4ED2AF43-7D1A-4BA6-B785-E8F4910EF5A3_94 | crack_4.0 | miss | 0.494 | unclassified | size=4.0 eq=3.4681495701545724 amp=7414 noise=934 |
| miss_crack_4.0_4D9E9185-588A-433B-BB2C-8E87144A324A_79 | crack_4.0 | miss | 0.494 | unclassified | size=4.0 eq=3.096768456028311 amp=7414 noise=934 |
| miss_crack_4.0_91890000-2F8D-40C4-8564-B71F4CFE66EC_92 | crack_4.0 | miss | 0.493 | unclassified | size=4.0 eq=3.389490568433598 amp=7414 noise=934 |
| miss_crack_4.0_1A48018B-F1B5-428E-B6AF-67E8F6B02C2E_6 | crack_4.0 | miss | 0.491 | unclassified | size=4.0 eq=3.6906009118405234 amp=7414 noise=934 |
| miss_crack_4.0_CE36580B-B0C4-4611-8FEA-0578ABEAAE97_8 | crack_4.0 | miss | 0.491 | unclassified | size=4.0 eq=3.007176743407759 amp=7414 noise=934 |
| miss_crack_4.0_C4E68802-6FAF-48E3-B8D4-CA53F445E66C_82 | crack_4.0 | miss | 0.490 | unclassified | size=4.0 eq=3.6126525832223613 amp=7414 noise=934 |
| miss_crack_4.0_9036CAF6-63B6-41F5-924F-2A33561198BF_61 | crack_4.0 | miss | 0.489 | unclassified | size=4.0 eq=3.6885727413955838 amp=7414 noise=934 |
| miss_crack_4.0_20A0BBD4-6C3C-470A-82A8-A3A221704182_50 | crack_4.0 | miss | 0.489 | small_flaw | size=4.0 eq=2.6341463025903007 amp=7414 noise=934 |
| miss_crack_4.0_3F348C28-448D-4BB3-AF52-211B1234068A_76 | crack_4.0 | miss | 0.486 | unclassified | size=4.0 eq=3.363577625396984 amp=7414 noise=934 |
| miss_crack_4.0_9394DF9F-28AD-46C4-82B1-083843062055_7 | crack_4.0 | miss | 0.485 | small_flaw | size=4.0 eq=2.8254731718810318 amp=7414 noise=934 |
| miss_crack_4.0_6DE6BDD7-3029-4436-AB34-EF15DD854231_6 | crack_4.0 | miss | 0.485 | unclassified | size=4.0 eq=3.324094046796024 amp=7414 noise=934 |
| miss_crack_4.0_07220132-CE4D-4FCF-8B6B-1B2A12E7D9D4_99 | crack_4.0 | miss | 0.485 | small_flaw | size=4.0 eq=2.554329941064851 amp=7414 noise=934 |
| miss_crack_4.0_BA90E6E7-46D1-4FDF-933A-DCB071E465C6_42 | crack_4.0 | miss | 0.485 | unclassified | size=4.0 eq=3.0404254390596366 amp=7414 noise=934 |
| miss_crack_4.0_DA322B67-6D33-45E9-9C38-20B358DC79E4_98 | crack_4.0 | miss | 0.484 | unclassified | size=4.0 eq=3.096808914097919 amp=7414 noise=934 |
| miss_crack_4.0_3EA26DBB-D534-4A47-804F-B08ED8F6E38F_34 | crack_4.0 | miss | 0.484 | unclassified | size=4.0 eq=3.970936544844759 amp=7414 noise=934 |
| near_miss_clean_crack_4.0_DA68E5F8-9C58-4343-86DB-5D53586D9A70_13 | crack_4.0 | near_miss_clean | 0.000 | clean_near_miss | amp=4050 noise=973 |
| near_miss_clean_crack_4.0_268A3193-0491-40CF-BA54-1B7061557AC1_19 | crack_4.0 | near_miss_clean | 0.000 | clean_near_miss | amp=4050 noise=973 |
| near_miss_clean_crack_4.0_6E6E8837-D467-4350-8A29-70B58BCA9066_44 | crack_4.0 | near_miss_clean | 0.000 | clean_near_miss | amp=4050 noise=973 |
| near_miss_clean_crack_4.0_6E6E8837-D467-4350-8A29-70B58BCA9066_3 | crack_4.0 | near_miss_clean | 0.000 | clean_near_miss | amp=4050 noise=973 |
| near_miss_clean_crack_4.0_6A706C1D-2067-470F-A053-8E91FBFEDBED_23 | crack_4.0 | near_miss_clean | 0.000 | clean_near_miss | amp=4050 noise=973 |
| near_miss_clean_crack_4.0_818B8CB4-5C86-4396-AAF5-B48F41302D8F_58 | crack_4.0 | near_miss_clean | 0.000 | clean_near_miss | amp=4050 noise=973 |
| near_miss_clean_crack_4.0_605F522F-8CBF-4E22-811D-60D594F283D2_33 | crack_4.0 | near_miss_clean | 0.000 | clean_near_miss | amp=4050 noise=973 |
| near_miss_clean_crack_4.0_3BEC7C3C-1441-46B0-8442-ABC5BBD185A7_96 | crack_4.0 | near_miss_clean | 0.000 | clean_near_miss | amp=3734 noise=1068 |
| near_miss_clean_crack_4.0_6E6E8837-D467-4350-8A29-70B58BCA9066_32 | crack_4.0 | near_miss_clean | 0.000 | clean_near_miss | amp=4050 noise=973 |
| near_miss_clean_crack_4.0_A61932EB-A1E0-4E35-B1CD-C5DBA69F4B7E_73 | crack_4.0 | near_miss_clean | 0.000 | clean_near_miss | amp=4050 noise=973 |
| near_miss_clean_crack_4.0_6A706C1D-2067-470F-A053-8E91FBFEDBED_76 | crack_4.0 | near_miss_clean | 0.000 | clean_near_miss | amp=4050 noise=973 |
| near_miss_clean_crack_4.0_6E6E8837-D467-4350-8A29-70B58BCA9066_89 | crack_4.0 | near_miss_clean | 0.000 | clean_near_miss | amp=4050 noise=973 |
| near_miss_clean_crack_4.0_744784D3-1AB4-4E4B-95D1-49573EE1F8F9_49 | crack_4.0 | near_miss_clean | 0.000 | clean_near_miss | amp=4050 noise=973 |
| near_miss_clean_crack_4.0_818B8CB4-5C86-4396-AAF5-B48F41302D8F_18 | crack_4.0 | near_miss_clean | 0.000 | clean_near_miss | amp=4050 noise=973 |
| near_miss_clean_crack_4.0_6A706C1D-2067-470F-A053-8E91FBFEDBED_94 | crack_4.0 | near_miss_clean | 0.000 | clean_near_miss | amp=4050 noise=973 |
| near_miss_clean_crack_4.0_FE57FD4B-E7A8-4307-BB77-65F85DA4E962_91 | crack_4.0 | near_miss_clean | 0.000 | clean_near_miss | amp=4050 noise=973 |
| near_miss_clean_crack_4.0_DA68E5F8-9C58-4343-86DB-5D53586D9A70_0 | crack_4.0 | near_miss_clean | 0.000 | clean_near_miss | amp=3734 noise=1068 |
| near_miss_clean_crack_4.0_6E6E8837-D467-4350-8A29-70B58BCA9066_34 | crack_4.0 | near_miss_clean | 0.000 | clean_near_miss | amp=4050 noise=973 |
| near_miss_clean_crack_4.0_FE57FD4B-E7A8-4307-BB77-65F85DA4E962_37 | crack_4.0 | near_miss_clean | 0.000 | clean_near_miss | amp=4050 noise=973 |
| near_miss_clean_crack_4.0_A61932EB-A1E0-4E35-B1CD-C5DBA69F4B7E_22 | crack_4.0 | near_miss_clean | 0.000 | clean_near_miss | amp=4050 noise=973 |
| near_miss_clean_crack_8.6_0275DE06-7EB3-4D99-B71D-39F259B64862_78 | crack_8.6 | near_miss_clean | 0.001 | clean_near_miss | amp=4050 noise=973 |
| near_miss_clean_crack_8.6_5EC43978-E60B-475E-B136-6B505FB7BE1E_15 | crack_8.6 | near_miss_clean | 0.000 | clean_near_miss | amp=3734 noise=1068 |
| near_miss_clean_crack_8.6_5E031CFB-E25C-4AAD-A65F-F91A7419F0CF_47 | crack_8.6 | near_miss_clean | 0.000 | clean_near_miss | amp=3734 noise=1068 |
| near_miss_clean_crack_8.6_3BEC7C3C-1441-46B0-8442-ABC5BBD185A7_96 | crack_8.6 | near_miss_clean | 0.000 | clean_near_miss | amp=3734 noise=1068 |
| near_miss_clean_crack_8.6_6A706C1D-2067-470F-A053-8E91FBFEDBED_69 | crack_8.6 | near_miss_clean | 0.000 | clean_near_miss | amp=3734 noise=1068 |
| near_miss_clean_crack_8.6_9394DF9F-28AD-46C4-82B1-083843062055_59 | crack_8.6 | near_miss_clean | 0.000 | clean_near_miss | amp=3734 noise=1068 |
| near_miss_clean_crack_8.6_5E031CFB-E25C-4AAD-A65F-F91A7419F0CF_28 | crack_8.6 | near_miss_clean | 0.000 | clean_near_miss | amp=3734 noise=1068 |
| near_miss_clean_crack_8.6_FE57FD4B-E7A8-4307-BB77-65F85DA4E962_91 | crack_8.6 | near_miss_clean | 0.000 | clean_near_miss | amp=4050 noise=973 |
| near_miss_clean_crack_8.6_DA68E5F8-9C58-4343-86DB-5D53586D9A70_24 | crack_8.6 | near_miss_clean | 0.000 | clean_near_miss | amp=4050 noise=973 |
| near_miss_clean_crack_8.6_D2D163FD-047D-4AE0-A84E-5DE59C681961_96 | crack_8.6 | near_miss_clean | 0.000 | clean_near_miss | amp=3734 noise=1068 |
| near_miss_clean_crack_8.6_818B8CB4-5C86-4396-AAF5-B48F41302D8F_58 | crack_8.6 | near_miss_clean | 0.000 | clean_near_miss | amp=4050 noise=973 |
| near_miss_clean_crack_8.6_0275DE06-7EB3-4D99-B71D-39F259B64862_25 | crack_8.6 | near_miss_clean | 0.000 | clean_near_miss | amp=3734 noise=1068 |
| near_miss_clean_crack_8.6_9394DF9F-28AD-46C4-82B1-083843062055_88 | crack_8.6 | near_miss_clean | 0.000 | clean_near_miss | amp=3734 noise=1068 |
| near_miss_clean_crack_8.6_268A3193-0491-40CF-BA54-1B7061557AC1_19 | crack_8.6 | near_miss_clean | 0.000 | clean_near_miss | amp=4050 noise=973 |
| near_miss_clean_crack_8.6_59345583-4DB7-44D1-8FDF-11CA79F044B5_58 | crack_8.6 | near_miss_clean | 0.000 | clean_near_miss | amp=4050 noise=973 |
| near_miss_clean_crack_8.6_5E031CFB-E25C-4AAD-A65F-F91A7419F0CF_26 | crack_8.6 | near_miss_clean | 0.000 | clean_near_miss | amp=4050 noise=973 |
| near_miss_clean_crack_8.6_DA68E5F8-9C58-4343-86DB-5D53586D9A70_52 | crack_8.6 | near_miss_clean | 0.000 | clean_near_miss | amp=3734 noise=1068 |
| near_miss_clean_crack_8.6_5EC43978-E60B-475E-B136-6B505FB7BE1E_38 | crack_8.6 | near_miss_clean | 0.000 | clean_near_miss | amp=3734 noise=1068 |
| near_miss_clean_crack_8.6_268A3193-0491-40CF-BA54-1B7061557AC1_30 | crack_8.6 | near_miss_clean | 0.000 | clean_near_miss | amp=3734 noise=1068 |
| near_miss_clean_crack_8.6_59345583-4DB7-44D1-8FDF-11CA79F044B5_88 | crack_8.6 | near_miss_clean | 0.000 | clean_near_miss | amp=3734 noise=1068 |

## Synthetic Held-Out Regions Generalization

Evaluation results extracted from `results/sim_generalization.json` demonstrate false-call blow-up under parameter shifts without re-running simulation:

| Region | Rules FCR | ML 1D FCR | Union FCR | ML False / Total Scans |
|---|---|---|---|---|
| in_distribution | 2.93% | 13.41% | 14.63% | 55 / 410 |
| freq_10mhz | 74.76% | 60.00% | 80.24% | 252 / 420 |
| material_third | 3.95% | 4.42% | 6.74% | 19 / 430 |
| noise_above_range | 34.29% | 40.71% | 52.86% | 171 / 420 |

- **10 MHz Frequency Shift**: ML false-call rate = 60.0% (252/420 scans), union FCR = 80.2%.
- **Noise Above Range**: ML false-call rate = 40.7% (171/420 scans), union FCR = 52.9%.
