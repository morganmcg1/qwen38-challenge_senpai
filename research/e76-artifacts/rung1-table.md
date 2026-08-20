| variant | NA | rows_per_simd | g16s regs / spill | g17s regs / spill | bit-identical? | machine-code digest changed? | max threads / threadgroup |
|---|---:|---:|---:|---:|:--:|:--:|---:|
| `plain` | 2 | 4 | 70 / 0 | 83 / 0 | reference | no | - |
| `lazysb` | 2 | 4 | 73 / 0 | 89 / 0 | not run | yes | - |
| `lazyw` | 2 | 4 | 79 / 0 | 84 / 0 | not run | yes | - |
| `lazy` | 2 | 4 | 75 / 0 | 84 / 0 | not run | yes | - |
| `rps2` | 2 | 2 | 50 / 0 | 56 / 0 | not run | yes | - |
| `rps2lazysb` | 2 | 2 | 50 / 0 | 57 / 0 | not run | yes | - |
| `rps2lazyw` | 2 | 2 | 50 / 0 | 54 / 0 | not run | yes | - |
| `rps2lazy` | 2 | 2 | 50 / 0 | 58 / 0 | not run | yes | - |
| `rps1` | 2 | 1 | 45 / 0 | 55 / 0 | not run | yes | - |
| `rps1lazysb` | 2 | 1 | 44 / 0 | 55 / 0 | not run | yes | - |
| `rps1lazyw` | 2 | 1 | 45 / 0 | 55 / 0 | not run | yes | - |
| `rps1lazy` | 2 | 1 | 44 / 0 | 55 / 0 | not run | yes | - |
| `rps2nu` | 2 | 2 | 50 / 0 | 56 / 0 | not run | yes | - |
| `rps1nu` | 2 | 1 | 45 / 0 | 55 / 0 | not run | yes | - |
| `facc` | 2 | 4 | 70 / 0 | 83 / 0 | not run | no | - |
| `lazyfacc` | 2 | 4 | 75 / 0 | 81 / 0 | not run | yes | - |
| `rps1facc` | 2 | 1 | 45 / 0 | 55 / 0 | not run | yes | - |
| `rps1lazyfacc` | 2 | 1 | 44 / 0 | 55 / 0 | not run | yes | - |
| `fall` | 2 | 4 | 81 / 0 | 95 / 0 | not run | yes | - |
| `lazyfall` | 2 | 4 | 78 / 0 | 82 / 0 | not run | yes | - |
| `rps1fall` | 2 | 1 | 43 / 0 | 50 / 0 | not run | yes | - |
| `rps1lazyfall` | 2 | 1 | 43 / 0 | 50 / 0 | not run | yes | - |
| `mc4` | 2 | 4 | 70 / 0 | 83 / 0 | not run | no | - |
| `mc3` | 2 | 4 | 70 / 0 | 83 / 0 | not run | no | - |
| `mc2` | 2 | 4 | 70 / 0 | 83 / 0 | not run | no | - |
| `plain` | 3 | 4 | 93 / 0 | 90 / 0 | reference | no | - |
| `lazysb` | 3 | 4 | 94 / 0 | 91 / 0 | not run | yes | - |
| `lazyw` | 3 | 4 | 88 / 0 | 96 / 0 | not run | yes | - |
| `lazy` | 3 | 4 | 85 / 0 | 95 / 0 | not run | yes | - |
| `rps2` | 3 | 2 | 62 / 0 | 68 / 0 | not run | yes | - |
| `rps2lazysb` | 3 | 2 | 63 / 0 | 76 / 0 | not run | yes | - |
| `rps2lazyw` | 3 | 2 | 63 / 0 | 69 / 0 | not run | yes | - |
| `rps2lazy` | 3 | 2 | 62 / 0 | 68 / 0 | not run | yes | - |
| `rps1` | 3 | 1 | 50 / 0 | 71 / 0 | not run | yes | - |
| `rps1lazysb` | 3 | 1 | 51 / 0 | 71 / 0 | not run | yes | - |
| `rps1lazyw` | 3 | 1 | 49 / 0 | 55 / 0 | not run | yes | - |
| `rps1lazy` | 3 | 1 | 48 / 0 | 54 / 0 | not run | yes | - |
| `rps2nu` | 3 | 2 | 62 / 0 | 68 / 0 | not run | yes | - |
| `rps1nu` | 3 | 1 | 50 / 0 | 71 / 0 | not run | yes | - |
| `facc` | 3 | 4 | 93 / 0 | 90 / 0 | not run | yes | - |
| `lazyfacc` | 3 | 4 | 85 / 0 | 94 / 0 | not run | yes | - |
| `rps1facc` | 3 | 1 | 48 / 0 | 72 / 0 | not run | yes | - |
| `rps1lazyfacc` | 3 | 1 | 48 / 0 | 54 / 0 | not run | yes | - |
| `fall` | 3 | 4 | 85 / 0 | 98 / 0 | not run | yes | - |
| `lazyfall` | 3 | 4 | 62 / 0 | 70 / 0 | not run | yes | - |
| `rps1fall` | 3 | 1 | 49 / 0 | 56 / 0 | not run | yes | - |
| `rps1lazyfall` | 3 | 1 | 46 / 0 | 54 / 0 | not run | yes | - |
| `mc4` | 3 | 4 | 93 / 0 | 90 / 0 | not run | no | - |
| `mc3` | 3 | 4 | 93 / 0 | 90 / 0 | not run | no | - |
| `mc2` | 3 | 4 | 96 / 144 | 116 / 0 | not run | yes | - |
| `plain` | 4 | 4 | 94 / 0 | 91 / 0 | reference | no | - |
| `lazysb` | 4 | 4 | 95 / 0 | 93 / 0 | not run | yes | - |
| `lazyw` | 4 | 4 | 96 / 32 | 113 / 0 | not run | yes | - |
| `lazy` | 4 | 4 | 96 / 16 | 110 / 0 | not run | yes | - |
| `rps2` | 4 | 2 | 73 / 0 | 82 / 0 | not run | yes | - |
| `rps2lazysb` | 4 | 2 | 73 / 0 | 83 / 0 | not run | yes | - |
| `rps2lazyw` | 4 | 2 | 71 / 0 | 83 / 0 | not run | yes | - |
| `rps2lazy` | 4 | 2 | 70 / 0 | 82 / 0 | not run | yes | - |
| `rps1` | 4 | 1 | 60 / 0 | 78 / 0 | not run | yes | - |
| `rps1lazysb` | 4 | 1 | 60 / 0 | 78 / 0 | not run | yes | - |
| `rps1lazyw` | 4 | 1 | 58 / 0 | 65 / 0 | not run | yes | - |
| `rps1lazy` | 4 | 1 | 54 / 0 | 64 / 0 | not run | yes | - |
| `rps2nu` | 4 | 2 | 73 / 0 | 82 / 0 | not run | yes | - |
| `rps1nu` | 4 | 1 | 60 / 0 | 78 / 0 | not run | yes | - |
| `facc` | 4 | 4 | 94 / 0 | 93 / 0 | not run | yes | - |
| `lazyfacc` | 4 | 4 | 96 / 16 | 109 / 0 | not run | yes | - |
| `rps1facc` | 4 | 1 | 58 / 0 | 83 / 0 | not run | yes | - |
| `rps1lazyfacc` | 4 | 1 | 54 / 0 | 64 / 0 | not run | yes | - |
| `fall` | 4 | 4 | 89 / 0 | 101 / 0 | not run | yes | - |
| `lazyfall` | 4 | 4 | 85 / 0 | 93 / 0 | not run | yes | - |
| `rps1fall` | 4 | 1 | 57 / 0 | 68 / 0 | not run | yes | - |
| `rps1lazyfall` | 4 | 1 | 53 / 0 | 64 / 0 | not run | yes | - |
| `mc4` | 4 | 4 | 94 / 0 | 91 / 0 | not run | no | - |
| `mc3` | 4 | 4 | 96 / 224 | 126 / 144 | not run | yes | - |
| `mc2` | 4 | 4 | 96 / 176 | 122 / 0 | not run | yes | - |
| `plain` | 5 | 4 | 95 / 0 | 98 / 0 | reference | no | 1024 |
| `lazysb` | 5 | 4 | 94 / 0 | 100 / 0 | yes | yes | 1024 |
| `lazyw` | 5 | 4 | 89 / 0 | 98 / 0 | yes | yes | 1024 |
| `lazy` | 5 | 4 | 86 / 0 | 97 / 0 | yes | yes | 1024 |
| `rps2` | 5 | 2 | 84 / 0 | 96 / 0 | yes | yes | 1024 |
| `rps2lazysb` | 5 | 2 | 86 / 0 | 96 / 0 | yes | yes | 1024 |
| `rps2lazyw` | 5 | 2 | 86 / 0 | 94 / 0 | yes | yes | 1024 |
| `rps2lazy` | 5 | 2 | 83 / 0 | 98 / 0 | yes | yes | 1024 |
| `rps1` | 5 | 1 | 71 / 0 | 93 / 0 | yes | yes | 1024 |
| `rps1lazysb` | 5 | 1 | 70 / 0 | 88 / 0 | yes | yes | 1024 |
| `rps1lazyw` | 5 | 1 | 65 / 0 | 76 / 0 | yes | yes | 1024 |
| `rps1lazy` | 5 | 1 | 64 / 0 | 75 / 0 | yes | yes | 1024 |
| `rps2nu` | 5 | 2 | 84 / 0 | 96 / 0 | yes | yes | 1024 |
| `rps1nu` | 5 | 1 | 71 / 0 | 93 / 0 | yes | yes | 1024 |
| `facc` | 5 | 4 | 95 / 0 | 98 / 0 | not run | yes | - |
| `lazyfacc` | 5 | 4 | 86 / 0 | 96 / 0 | not run | yes | - |
| `rps1facc` | 5 | 1 | 69 / 0 | 94 / 0 | not run | yes | - |
| `rps1lazyfacc` | 5 | 1 | 64 / 0 | 75 / 0 | not run | yes | - |
| `fall` | 5 | 4 | 91 / 0 | 100 / 0 | not run | yes | - |
| `lazyfall` | 5 | 4 | 87 / 0 | 93 / 0 | not run | yes | - |
| `rps1fall` | 5 | 1 | 69 / 0 | 80 / 0 | not run | yes | - |
| `rps1lazyfall` | 5 | 1 | 63 / 0 | 75 / 0 | not run | yes | - |
| `mc4` | 5 | 4 | 96 / 320 | 126 / 240 | not run | yes | - |
| `mc3` | 5 | 4 | 96 / 272 | 126 / 176 | not run | yes | - |
| `mc2` | 5 | 4 | 96 / 320 | 126 / 240 | not run | yes | - |
| `plain` | 6 | 4 | 96 / 16 | 111 / 0 | reference | no | 1024 |
| `lazysb` | 6 | 4 | 96 / 16 | 108 / 0 | yes | yes | 1024 |
| `lazyw` | 6 | 4 | 96 / 16 | 110 / 0 | yes | yes | 1024 |
| `lazy` | 6 | 4 | 96 / 16 | 107 / 0 | yes | yes | 1024 |
| `rps2` | 6 | 2 | 96 / 0 | 100 / 0 | yes | yes | 1024 |
| `rps2lazysb` | 6 | 2 | 96 / 16 | 100 / 0 | yes | yes | 1024 |
| `rps2lazyw` | 6 | 2 | 74 / 0 | 88 / 0 | yes | yes | 1024 |
| `rps2lazy` | 6 | 2 | 72 / 0 | 86 / 0 | yes | yes | 1024 |
| `rps1` | 6 | 1 | 79 / 0 | 99 / 0 | yes | yes | 1024 |
| `rps1lazysb` | 6 | 1 | 79 / 0 | 99 / 0 | yes | yes | 1024 |
| `rps1lazyw` | 6 | 1 | 58 / 0 | 70 / 0 | yes | yes | 1024 |
| `rps1lazy` | 6 | 1 | 57 / 0 | 70 / 0 | yes | yes | 1024 |
| `rps2nu` | 6 | 2 | 96 / 0 | 100 / 0 | yes | yes | 1024 |
| `rps1nu` | 6 | 1 | 79 / 0 | 99 / 0 | yes | yes | 1024 |
| `facc` | 6 | 4 | 96 / 16 | 111 / 0 | not run | yes | - |
| `lazyfacc` | 6 | 4 | 96 / 16 | 106 / 0 | not run | yes | - |
| `rps1facc` | 6 | 1 | 77 / 0 | 104 / 0 | not run | yes | - |
| `rps1lazyfacc` | 6 | 1 | 57 / 0 | 70 / 0 | not run | yes | - |
| `fall` | 6 | 4 | 95 / 48 | 108 / 48 | not run | yes | - |
| `lazyfall` | 6 | 4 | 93 / 0 | 99 / 0 | not run | yes | - |
| `rps1fall` | 6 | 1 | 76 / 0 | 90 / 0 | not run | yes | - |
| `rps1lazyfall` | 6 | 1 | 56 / 0 | 68 / 0 | not run | yes | - |
| `mc4` | 6 | 4 | 96 / 352 | 126 / 288 | not run | yes | - |
| `mc3` | 6 | 4 | 96 / 320 | 126 / 224 | not run | yes | - |
| `mc2` | 6 | 4 | 96 / 352 | 126 / 288 | not run | yes | - |

| variant | NA | seconds per verify round | vs `plain` |
|---|---:|---:|---:|
| `plain` | 5 | 0.084630 | +0.00 % |
| `lazy` | 5 | 0.097942 | +15.73 % |
| `rps2` | 5 | 0.096616 | +14.16 % |
| `rps2lazyw` | 5 | 0.108823 | +28.59 % |
| `rps2lazy` | 5 | 0.107767 | +27.34 % |
| `rps1` | 5 | 0.124884 | +47.56 % |
| `rps1lazyw` | 5 | 0.137363 | +62.31 % |
| `rps1lazy` | 5 | 0.139395 | +64.71 % |
| `plain` | 6 | 0.111994 | +0.00 % |
| `lazy` | 6 | 0.118393 | +5.71 % |
| `rps2` | 6 | 0.117795 | +5.18 % |
| `rps2lazyw` | 6 | 0.132007 | +17.87 % |
| `rps2lazy` | 6 | 0.131192 | +17.14 % |
| `rps1` | 6 | 0.149836 | +33.79 % |
| `rps1lazyw` | 6 | 0.170631 | +52.36 % |
| `rps1lazy` | 6 | 0.167719 | +49.76 % |
